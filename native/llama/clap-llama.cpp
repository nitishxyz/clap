#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

// Reads stdin on a dedicated thread so the main thread can poll for cancel
// messages while a generation is in progress.
class StdinReader {
 public:
  StdinReader() : thread_([this] { run(); }) {}

  ~StdinReader() {
    if (thread_.joinable()) thread_.detach();
  }

  bool next(std::string& out) {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !lines_.empty() || eof_; });
    if (lines_.empty()) return false;
    out = std::move(lines_.front());
    lines_.pop_front();
    return true;
  }

  bool poll(std::string& out) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (lines_.empty()) return false;
    out = std::move(lines_.front());
    lines_.pop_front();
    return true;
  }

 private:
  void run() {
    std::string line;
    while (std::getline(std::cin, line)) {
      {
        std::lock_guard<std::mutex> lock(mutex_);
        lines_.push_back(line);
      }
      cv_.notify_all();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      eof_ = true;
    }
    cv_.notify_all();
  }

  std::deque<std::string> lines_;
  std::mutex mutex_;
  std::condition_variable cv_;
  bool eof_ = false;
  std::thread thread_;
};

struct LoadedLlama {
  std::string model_path;
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  // KV cache slots: one llama sequence per slot so multiple concurrent
  // sessions (agent loops, chats, side requests) each keep a warm prefix.
  struct CacheSlot {
    std::vector<llama_token> tokens;
    uint64_t last_used = 0;
    bool busy = false;  // an active request is generating on this slot
    bool is_anchor = false;  // holds a shared prefix snapshot, never generates
  };
  std::vector<CacheSlot> slots;
  uint64_t use_counter = 0;
  // Hybrid/recurrent models (e.g. Gated DeltaNet) only support whole-sequence
  // KV state copies; attention-only models support partial-range copies.
  bool hybrid = false;
};

struct ChatEntry {
  std::string role;
  std::string content;
};

struct SamplingParams {
  int max_tokens = 4096;
  double temperature = 0.7;
  double top_p = 0.95;
  int top_k = 0;
  uint32_t seed = LLAMA_DEFAULT_SEED;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::vector<std::string> stops;
};

void emit(const std::string& id, json fields) {
  if (!id.empty()) fields["id"] = id;
  // error_handler_t::replace: never throw on stray invalid UTF-8 bytes from
  // byte-level BPE pieces; substitute U+FFFD instead of killing the request.
  std::cout << fields.dump(-1, ' ', false, json::error_handler_t::replace) << "\n";
  std::cout.flush();
}

void emit_error(const std::string& id, const std::string& message) {
  emit(id, json{{"error", message}});
}

void emit_error(const std::string& id, const std::string& message, const std::string& code) {
  if (code.empty()) {
    emit_error(id, message);
  } else {
    emit(id, json{{"error", message}, {"code", code}});
  }
}

// A request rejected at admission (before any ingest) with a machine-readable
// code the server maps to a client error instead of a backend failure.
struct RequestError : std::runtime_error {
  std::string code;
  RequestError(std::string error_code, const std::string& message)
      : std::runtime_error(message), code(std::move(error_code)) {}
};

// Number of trailing bytes forming an incomplete (but so far valid) UTF-8
// sequence. Byte-level BPE tokens can split a multi-byte character across
// pieces; those bytes must be held back until the character completes.
std::size_t utf8_incomplete_suffix(const std::string& text) {
  const std::size_t size = text.size();
  std::size_t cont = 0;
  while (cont < 3 && cont < size &&
         (static_cast<unsigned char>(text[size - 1 - cont]) & 0xC0) == 0x80) {
    cont += 1;
  }
  if (cont >= size) return 0;
  const unsigned char lead = static_cast<unsigned char>(text[size - 1 - cont]);
  std::size_t need = 0;
  if ((lead & 0x80) == 0) need = 1;
  else if ((lead & 0xE0) == 0xC0) need = 2;
  else if ((lead & 0xF0) == 0xE0) need = 3;
  else if ((lead & 0xF8) == 0xF0) need = 4;
  else return 0;
  const std::size_t have = cont + 1;
  return have < need ? have : 0;
}

int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  try {
    return std::stoi(raw);
  } catch (...) {
    return fallback;
  }
}

std::vector<ChatEntry> messages_from_request(const json& request) {
  std::vector<ChatEntry> messages;
  if (!request.contains("messages") || !request["messages"].is_array()) return messages;
  for (const auto& message : request["messages"]) {
    if (!message.is_object()) continue;
    const std::string role = message.value("role", "");
    std::string content;
    if (message.contains("content") && message["content"].is_string()) {
      content = message["content"].get<std::string>();
    }
    if (role.empty() || content.empty()) continue;
    messages.push_back({role, content});
  }
  return messages;
}

SamplingParams sampling_from_request(const json& request) {
  SamplingParams params;
  params.max_tokens = request.value("max_tokens", params.max_tokens);
  params.temperature = request.value("temperature", params.temperature);
  params.top_p = request.value("top_p", params.top_p);
  params.top_k = request.value("top_k", params.top_k);
  params.presence_penalty = request.value("presence_penalty", params.presence_penalty);
  params.frequency_penalty = request.value("frequency_penalty", params.frequency_penalty);
  if (request.contains("seed") && request["seed"].is_number_integer()) {
    params.seed = static_cast<uint32_t>(request["seed"].get<int64_t>());
  }
  if (request.contains("stop")) {
    const auto& stop = request["stop"];
    if (stop.is_string()) {
      params.stops.push_back(stop.get<std::string>());
    } else if (stop.is_array()) {
      for (const auto& value : stop) {
        if (value.is_string()) params.stops.push_back(value.get<std::string>());
      }
    }
  }
  if (params.max_tokens < 1) params.max_tokens = 1;
  return params;
}

std::string templated_prompt(llama_model* model, const std::vector<ChatEntry>& entries) {
  if (entries.empty()) return "";

  const char* tmpl = llama_model_chat_template(model, nullptr);
  if (tmpl && std::string(tmpl).find("<|turn>") != std::string::npos) {
    std::string prompt = "<bos>";
    for (const auto& entry : entries) {
      std::string role = entry.role == "assistant" ? "model" : entry.role;
      if (role == "tool") role = "user";
      prompt += "<|turn>" + role + "\n" + entry.content + "<turn|>\n";
    }
    prompt += "<|turn>model\n";
    return prompt;
  }

  std::vector<ChatEntry> normalized;
  normalized.reserve(entries.size());
  for (const auto& entry : entries) {
    // llama.cpp chat templates generally support system/user/assistant only.
    normalized.push_back({entry.role == "tool" ? "user" : entry.role, entry.content});
  }

  std::vector<llama_chat_message> chat;
  chat.reserve(normalized.size());
  for (const auto& entry : normalized) {
    chat.push_back({entry.role.c_str(), entry.content.c_str()});
  }

  int32_t length = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, nullptr, 0);
  if (length <= 0) return "";
  std::vector<char> buffer(static_cast<std::size_t>(length) + 1);
  int32_t written = llama_chat_apply_template(tmpl, chat.data(), chat.size(), true, buffer.data(), buffer.size());
  if (written <= 0) return "";
  return std::string(buffer.data(), static_cast<std::size_t>(written));
}

std::string fallback_prompt(const std::vector<ChatEntry>& entries) {
  std::string prompt;
  for (const auto& entry : entries) {
    prompt += entry.content;
    prompt += "\n";
  }
  return prompt;
}

void unload(LoadedLlama& loaded) {
  if (loaded.ctx) {
    llama_free(loaded.ctx);
    loaded.ctx = nullptr;
  }
  if (loaded.model) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
  }
  loaded.model_path.clear();
  loaded.slots.clear();
}

void load_model(LoadedLlama& loaded, const std::string& model_path) {
  if (loaded.model && loaded.ctx && loaded.model_path == model_path) return;
  unload(loaded);

  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("GGUF model not found: " + model_path);
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = env_int("CLAP_LLAMA_GPU_LAYERS", 999);
  loaded.model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!loaded.model) throw std::runtime_error("failed to load GGUF model: " + model_path);

  llama_context_params ctx_params = llama_context_default_params();
  // Use the model's own trained context by default (Qwen3.6 = 262k, gemma =
  // 32k, ...). CLAP_LLAMA_CONTEXT overrides explicitly; otherwise start from
  // train_ctx (capped at 262144 to bound KV allocation) and let the fallback
  // loop below halve it if the KV cache does not fit in memory.
  const int32_t env_ctx = env_int("CLAP_LLAMA_CONTEXT", 0);
  int32_t n_ctx = env_ctx;
  if (n_ctx <= 0) {
    const int32_t train_ctx = llama_model_n_ctx_train(loaded.model);
    n_ctx = std::min(train_ctx > 0 ? train_ctx : 8192, 262144);
    n_ctx = std::max(n_ctx, 8192);
  }
  // Match llama.cpp server defaults; small batches make long-prompt prefill
  // several times slower on Metal.
  ctx_params.n_batch = env_int("CLAP_LLAMA_BATCH", 2048);
  ctx_params.n_ubatch = env_int("CLAP_LLAMA_UBATCH", 512);
  // Multiple KV cache slots (one llama sequence each) so any number of
  // concurrent sessions keep warm prefixes; slots are LRU-recycled. With a
  // unified KV pool, slots beyond the working set cost only bookkeeping, so
  // default high enough that N concurrent agent sessions do not thrash
  // (working set > slots means near-zero prefix reuse).
  const int32_t n_slots = std::max(1, env_int("CLAP_LLAMA_SLOTS", 16));
  ctx_params.n_seq_max = n_slots;
  // Without a unified KV buffer, llama.cpp splits n_ctx into n_seq_max
  // per-sequence streams (32k ctx / 4 slots = only 8k per session), so long
  // agent prompts fail to find a memory slot even when the cache is mostly
  // empty. A unified buffer lets any session use the full context.
  ctx_params.kv_unified = env_int("CLAP_LLAMA_KV_UNIFIED", 1) != 0;
  // Opt-in KV cache quantization (CLAP_LLAMA_KV_TYPE=q8_0|q4_0|f16). Halves
  // (q8_0) or quarters (q4_0) KV memory per token at a small quality cost;
  // default stays f16 until enabled by policy.
  if (const char* kv_type = std::getenv("CLAP_LLAMA_KV_TYPE"); kv_type && *kv_type) {
    const std::string requested(kv_type);
    if (requested == "q8_0") {
      ctx_params.type_k = GGML_TYPE_Q8_0;
      ctx_params.type_v = GGML_TYPE_Q8_0;
    } else if (requested == "q4_0") {
      ctx_params.type_k = GGML_TYPE_Q4_0;
      ctx_params.type_v = GGML_TYPE_Q4_0;
    } else if (requested != "f16") {
      fprintf(stderr, "clap-llama: unknown CLAP_LLAMA_KV_TYPE '%s'; using f16\n", kv_type);
    }
  }
  ctx_params.no_perf = true;
  // KV cache allocation grows with n_ctx and can exceed device memory for
  // long-context models. Retry with halved context until it fits (unless the
  // user pinned CLAP_LLAMA_CONTEXT, which we honor verbatim).
  while (true) {
    ctx_params.n_ctx = n_ctx;
    loaded.ctx = llama_init_from_model(loaded.model, ctx_params);
    if (loaded.ctx || env_ctx > 0 || n_ctx <= 8192) break;
    n_ctx = std::max(n_ctx / 2, 8192);
    fprintf(stderr, "clap-llama: context allocation failed; retrying with n_ctx=%d\n", n_ctx);
  }
  if (!loaded.ctx) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
    throw std::runtime_error("failed to create llama context for: " + model_path);
  }
  loaded.model_path = model_path;
  loaded.slots.assign(static_cast<std::size_t>(n_slots), {});
  loaded.use_counter = 0;
  loaded.hybrid = llama_model_is_recurrent(loaded.model) || llama_model_is_hybrid(loaded.model);
}

void batch_add(llama_batch& batch, llama_token token, llama_pos pos, bool logits, llama_seq_id seq) {
  const int32_t index = batch.n_tokens;
  batch.token[index] = token;
  batch.pos[index] = pos;
  batch.n_seq_id[index] = 1;
  batch.seq_id[index][0] = seq;
  batch.logits[index] = logits ? 1 : 0;
  batch.n_tokens += 1;
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
  char buffer[256];
  int n = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, true);
  if (n >= 0) return std::string(buffer, n);
  // Negative return is the required size; retry instead of dropping the piece.
  std::vector<char> big(static_cast<std::size_t>(-n));
  n = llama_token_to_piece(vocab, token, big.data(), static_cast<int32_t>(big.size()), 0, true);
  if (n < 0) return "";
  return std::string(big.data(), n);
}

std::size_t max_stop_length(const std::vector<std::string>& stops) {
  std::size_t max = 0;
  for (const auto& stop : stops) max = std::max(max, stop.size());
  return max;
}

// Returns the byte index of the earliest stop match in text, or npos.
std::size_t find_stop(const std::string& text, const std::vector<std::string>& stops) {
  std::size_t earliest = std::string::npos;
  for (const auto& stop : stops) {
    if (stop.empty()) continue;
    const std::size_t index = text.find(stop);
    if (index != std::string::npos && index < earliest) earliest = index;
  }
  return earliest;
}

// Longest suffix of text that is a prefix of any stop sequence.
std::size_t partial_stop_suffix(const std::string& text, const std::vector<std::string>& stops) {
  const std::size_t max = std::min(text.size(), max_stop_length(stops));
  for (std::size_t length = max; length > 0; --length) {
    const std::string suffix = text.substr(text.size() - length);
    for (const auto& stop : stops) {
      if (stop.size() > suffix.size() && stop.compare(0, suffix.size(), suffix) == 0) return length;
    }
  }
  return 0;
}

llama_sampler* make_sampler(const SamplingParams& params) {
  auto sparams = llama_sampler_chain_default_params();
  sparams.no_perf = true;
  llama_sampler* sampler = llama_sampler_chain_init(sparams);
  if (params.presence_penalty != 0.0 || params.frequency_penalty != 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(
      64,
      1.0f,
      static_cast<float>(params.frequency_penalty),
      static_cast<float>(params.presence_penalty)));
  }
  if (params.temperature <= 0.0) {
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
  } else {
    if (params.top_k > 0) llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(static_cast<float>(params.top_p), 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(static_cast<float>(params.temperature)));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));
  }
  return sampler;
}

// One in-flight chat request bound to a KV slot. The continuous-batching
// scheduler advances all active requests together: each decoding request
// contributes one token per step and prefilling requests fill the remaining
// batch budget with prompt chunks, so long ingests never stall other streams.
struct ActiveRequest {
  std::string id;
  SamplingParams params;
  llama_sampler* sampler = nullptr;
  llama_seq_id seq = 0;
  LoadedLlama::CacheSlot* slot = nullptr;

  std::vector<llama_token> prompt_tokens;       // un-ingested remainder
  std::vector<llama_token> full_prompt_tokens;  // for the one ingest retry
  std::size_t ingested = 0;
  int32_t n_pos = 0;
  int prompt_token_count = 0;
  int cached_prompt_tokens = 0;

  enum class Phase { Prefill, Decode };
  Phase phase = Phase::Prefill;
  llama_token pending_token = 0;  // sampled but not yet decoded
  std::string held;               // tail held for stop/UTF-8 boundaries
  int completion_tokens = 0;
  std::string finish_reason = "stop";
  bool cancelled = false;
  bool retried = false;
  bool done = false;

  // When >= 0: during full prefill of a hybrid model, snapshot a shared-prefix
  // anchor into an empty slot once ingestion reaches exactly this position.
  int32_t anchor_at = -1;

  // per-step scratch
  int32_t logits_index = -1;
  int32_t step_tokens = 0;
};

// Snapshots the current (mid-prefill) state of req.seq into an empty slot as
// a shared-prefix anchor. Hybrid models cannot branch a partial range off a
// longer sequence, but they can whole-copy a sequence that holds exactly the
// shared prefix — so future sessions with the same system prompt + tools
// branch from the anchor and skip that prefill entirely. Never evicts a warm
// session; anchors only claim currently-empty slots.
void maybe_create_anchor(LoadedLlama& loaded, ActiveRequest& req) {
  const std::size_t count = static_cast<std::size_t>(req.anchor_at);
  if (count < 16 || count > req.full_prompt_tokens.size()) return;
  // The sequence state must hold exactly the prefix (chunking lands a
  // boundary on anchor_at; anything else means the plan was disrupted).
  if (req.ingested != count || req.cached_prompt_tokens != 0) return;
  // Skip when an equivalent prefix holder already exists (concurrent prefills
  // race to plant the same shared boundary; one anchor serves everyone).
  for (const auto& s : loaded.slots) {
    if (s.tokens.size() != count) continue;  // only an exact holder can donate this prefix
    std::size_t p = 0;
    while (p < count && s.tokens[p] == req.full_prompt_tokens[p]) p += 1;
    if (p == count) return;
  }
  std::size_t target = SIZE_MAX;
  for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
    if (!loaded.slots[index].busy && loaded.slots[index].tokens.empty()) {
      target = index;
      break;
    }
  }
  if (target == SIZE_MAX) return;
  llama_memory_t mem = llama_get_memory(loaded.ctx);
  llama_memory_seq_rm(mem, static_cast<llama_seq_id>(target), -1, -1);
  llama_memory_seq_cp(mem, req.seq, static_cast<llama_seq_id>(target), -1, -1);
  auto& slot = loaded.slots[target];
  slot.tokens.assign(req.full_prompt_tokens.begin(), req.full_prompt_tokens.begin() + static_cast<std::ptrdiff_t>(count));
  slot.is_anchor = true;
  slot.last_used = ++loaded.use_counter;
  fprintf(stderr, "clap-llama: created prefix anchor (%zu tokens) in slot %zu\n", count, target);
}

void finalize(ActiveRequest& req) {
  if (req.sampler) {
    llama_sampler_free(req.sampler);
    req.sampler = nullptr;
  }
  if (req.slot) req.slot->busy = false;
  req.done = true;
  emit(req.id, json{
    {"done", true},
    {"finish_reason", req.finish_reason},
    {"cancelled", req.cancelled},
    {"usage", json{
      {"prompt_tokens", req.prompt_token_count},
      {"completion_tokens", req.completion_tokens},
    }},
    {"cache", json{
      {"hit", req.cached_prompt_tokens > 0},
      {"reused_tokens", req.cached_prompt_tokens},
      {"slot", static_cast<int>(req.seq)},
    }},
  });
}

void fail_request(LoadedLlama& loaded, ActiveRequest& req, const std::string& message) {
  if (req.sampler) {
    llama_sampler_free(req.sampler);
    req.sampler = nullptr;
  }
  if (req.slot) {
    req.slot->busy = false;
    req.slot->tokens.clear();
  }
  llama_memory_seq_rm(llama_get_memory(loaded.ctx), req.seq, -1, -1);
  req.done = true;
  emit_error(req.id, message);
}

void flush_held(ActiveRequest& req) {
  req.held.resize(req.held.size() - utf8_incomplete_suffix(req.held));
  if (!req.held.empty()) emit(req.id, json{{"token", req.held}});
  req.held.clear();
}

// Emits the visible portion of req.held, holding back partial stop sequences
// and incomplete UTF-8 tails. Returns true when a stop sequence completed.
bool emit_visible(ActiveRequest& req) {
  if (!req.params.stops.empty()) {
    const std::size_t stop_index = find_stop(req.held, req.params.stops);
    if (stop_index != std::string::npos) {
      std::string visible = req.held.substr(0, stop_index);
      visible.resize(visible.size() - utf8_incomplete_suffix(visible));
      if (!visible.empty()) emit(req.id, json{{"token", visible}});
      req.held.clear();
      return true;
    }
    const std::size_t stop_hold = partial_stop_suffix(req.held, req.params.stops);
    const std::size_t utf8_hold = utf8_incomplete_suffix(req.held);
    const std::size_t hold = std::max(stop_hold, utf8_hold);
    const std::string visible = req.held.substr(0, req.held.size() - hold);
    if (!visible.empty()) {
      emit(req.id, json{{"token", visible}});
      req.held = req.held.substr(visible.size());
    }
  } else {
    const std::size_t hold = utf8_incomplete_suffix(req.held);
    const std::string visible = req.held.substr(0, req.held.size() - hold);
    if (!visible.empty()) {
      emit(req.id, json{{"token", visible}});
      req.held = req.held.substr(visible.size());
    }
  }
  return false;
}

// Handles one sampled token: EOS, stop sequences, budget checks, streaming
// emission. Finalizes the request when generation ends.
void process_sampled(LoadedLlama& loaded, ActiveRequest& req, const llama_vocab* vocab, llama_token token) {
  if (llama_vocab_is_eog(vocab, token)) {
    flush_held(req);
    finalize(req);
    return;
  }
  req.completion_tokens += 1;
  const std::string piece = token_to_piece(vocab, token);
  if (!piece.empty()) {
    req.held += piece;
    if (emit_visible(req)) {
      req.finish_reason = "stop";
      finalize(req);
      return;
    }
  }
  if (req.completion_tokens >= req.params.max_tokens) {
    req.finish_reason = "length";
    flush_held(req);
    finalize(req);
    return;
  }
  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.ctx));
  if (req.n_pos + 1 >= n_ctx) {
    req.finish_reason = "length";
    flush_held(req);
    finalize(req);
    return;
  }
  req.pending_token = token;
  req.phase = ActiveRequest::Phase::Decode;
}

// Advances one request after its slice of the batch decoded successfully.
void post_decode(LoadedLlama& loaded, ActiveRequest& req, const llama_vocab* vocab) {
  if (req.phase == ActiveRequest::Phase::Prefill) {
    const std::size_t chunk = static_cast<std::size_t>(req.step_tokens);
    if (req.slot) {
      req.slot->tokens.insert(
        req.slot->tokens.end(),
        req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(req.ingested),
        req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(req.ingested + chunk));
    }
    req.n_pos += req.step_tokens;
    req.ingested += chunk;
    if (req.anchor_at >= 0 && req.ingested >= static_cast<std::size_t>(req.anchor_at)) {
      maybe_create_anchor(loaded, req);
      req.anchor_at = -1;
    }
    if (req.prompt_tokens.size() > 1024 && req.ingested < req.prompt_tokens.size()) {
      emit(req.id, json{{"prefill", {
        {"done", req.cached_prompt_tokens + static_cast<int>(req.ingested)},
        {"total", req.prompt_token_count},
      }}});
    }
    if (req.logits_index >= 0) {
      const llama_token token = llama_sampler_sample(req.sampler, loaded.ctx, req.logits_index);
      process_sampled(loaded, req, vocab, token);
    }
    return;
  }
  req.n_pos += 1;
  if (req.slot) req.slot->tokens.push_back(req.pending_token);
  const llama_token token = llama_sampler_sample(req.sampler, loaded.ctx, req.logits_index);
  process_sampled(loaded, req, vocab, token);
}

void handle_decode_failure(LoadedLlama& loaded, ActiveRequest& req, bool sole_active) {
  if (req.phase == ActiveRequest::Phase::Prefill && !req.retried) {
    // Self-heal: wipe this sequence (or everything when alone, which also
    // clears fragmented unified-KV state) and re-ingest the prompt once.
    req.retried = true;
    if (sole_active) {
      for (auto& s : loaded.slots) {
        s.tokens.clear();
        s.is_anchor = false;
      }
      llama_memory_clear(llama_get_memory(loaded.ctx), true);
    } else {
      llama_memory_seq_rm(llama_get_memory(loaded.ctx), req.seq, -1, -1);
      if (req.slot) req.slot->tokens.clear();
    }
    req.prompt_tokens = req.full_prompt_tokens;
    req.ingested = 0;
    req.n_pos = 0;
    req.cached_prompt_tokens = 0;
    fprintf(stderr, "clap-llama: ingest failed for request %s; retrying from scratch\n", req.id.c_str());
    return;
  }
  fail_request(loaded, req,
    "llama_decode failed; this often indicates llama.cpp GPU memory pressure. "
    "Check the llama worker log. Try a smaller GGUF quant such as Q4_K_M, reduce "
    "CLAP_LLAMA_CONTEXT/CLAP_LLAMA_BATCH/CLAP_LLAMA_UBATCH, set CLAP_LLAMA_GPU_LAYERS "
    "to a lower value, or use CPU fallback with CLAP_LLAMA_GPU_LAYERS=0.");
}

// Adds one request's contribution to a batch. Returns tokens added.
int32_t add_contribution(llama_batch& batch, ActiveRequest& req, int32_t budget) {
  if (req.phase == ActiveRequest::Phase::Decode) {
    req.logits_index = batch.n_tokens;
    req.step_tokens = 1;
    batch_add(batch, req.pending_token, req.n_pos, true, req.seq);
    return 1;
  }
  std::size_t remaining = req.prompt_tokens.size() - req.ingested;
  // Land a chunk boundary exactly on a pending anchor position so the
  // sequence state can be snapshotted at the shared-prefix boundary.
  if (req.anchor_at >= 0 && static_cast<std::size_t>(req.anchor_at) > req.ingested) {
    remaining = std::min(remaining, static_cast<std::size_t>(req.anchor_at) - req.ingested);
  }
  const int32_t chunk = static_cast<int32_t>(std::min<std::size_t>(remaining, static_cast<std::size_t>(budget)));
  req.logits_index = -1;
  req.step_tokens = chunk;
  for (int32_t i = 0; i < chunk; ++i) {
    const bool last = req.ingested + static_cast<std::size_t>(i) + 1 == req.prompt_tokens.size();
    if (last) req.logits_index = batch.n_tokens;
    batch_add(batch, req.prompt_tokens[req.ingested + static_cast<std::size_t>(i)], req.n_pos + i, last, req.seq);
  }
  return chunk;
}

// Runs one request's contribution as its own batch so a failing sequence can
// be isolated (and healed) without erroring every stream in the mixed batch.
void step_single(LoadedLlama& loaded, ActiveRequest& req, const llama_vocab* vocab, bool sole_active) {
  const int32_t size = std::max<int32_t>(req.step_tokens, 1);
  llama_batch batch = llama_batch_init(size, 0, 1);
  batch.n_tokens = 0;
  add_contribution(batch, req, size);
  const int result = llama_decode(loaded.ctx, batch);
  llama_batch_free(batch);
  if (result == 0) {
    post_decode(loaded, req, vocab);
  } else {
    handle_decode_failure(loaded, req, sole_active);
  }
}

// One scheduler step: cancellations, then a mixed batch of decode tokens and
// prefill chunks, then per-request sampling/emission.
void step(LoadedLlama& loaded, std::vector<std::unique_ptr<ActiveRequest>>& active) {
  const llama_vocab* vocab = llama_model_get_vocab(loaded.model);
  const int32_t n_batch = static_cast<int32_t>(llama_n_batch(loaded.ctx));

  for (auto& req : active) {
    if (!req->done && req->cancelled) {
      req->finish_reason = "cancel";
      finalize(*req);
    }
  }

  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  batch.n_tokens = 0;
  std::vector<ActiveRequest*> contributors;
  int32_t budget = n_batch;

  // Decode streams first: one token each keeps every session moving.
  for (auto& req : active) {
    if (req->done || req->phase != ActiveRequest::Phase::Decode) continue;
    if (budget <= 0) break;
    budget -= add_contribution(batch, *req, budget);
    contributors.push_back(req.get());
  }
  // Prefill fills the remaining budget in admission order.
  for (auto& req : active) {
    if (req->done || req->phase != ActiveRequest::Phase::Prefill) continue;
    if (budget <= 0) break;
    if (req->prompt_tokens.size() == req->ingested) continue;
    budget -= add_contribution(batch, *req, budget);
    contributors.push_back(req.get());
  }

  if (contributors.empty()) {
    llama_batch_free(batch);
    return;
  }

  const bool sole = contributors.size() == 1 && active.size() == 1;
  if (llama_decode(loaded.ctx, batch) == 0) {
    llama_batch_free(batch);
    for (auto* req : contributors) post_decode(loaded, *req, vocab);
    return;
  }
  llama_batch_free(batch);
  if (contributors.size() == 1) {
    handle_decode_failure(loaded, *contributors.front(), sole);
    return;
  }
  fprintf(stderr, "clap-llama: mixed batch decode failed; isolating %zu sequences\n", contributors.size());
  for (auto* req : contributors) step_single(loaded, *req, vocab, false);
}

std::unique_ptr<ActiveRequest> prepare_request(LoadedLlama& loaded, const std::string& id, const json& request) {
  const llama_vocab* vocab = llama_model_get_vocab(loaded.model);
  const std::vector<ChatEntry> entries = messages_from_request(request);
  std::string prompt = templated_prompt(loaded.model, entries);
  if (prompt.empty()) prompt = fallback_prompt(entries);
  if (prompt.empty() && request.contains("prompt") && request["prompt"].is_string()) {
    prompt = request["prompt"].get<std::string>();
  }
  if (prompt.empty()) throw std::runtime_error("chat request contains no messages or prompt");

  auto req = std::make_unique<ActiveRequest>();
  req->id = id;
  req->params = sampling_from_request(request);

  const bool add_special = prompt.find("<bos>") == std::string::npos && prompt.find("<|turn>") == std::string::npos;
  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_special, true);
  if (n_prompt <= 0) throw std::runtime_error("failed to tokenize prompt");
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), add_special, true) < 0) {
    throw std::runtime_error("failed to tokenize prompt");
  }

  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.ctx));
  const int32_t output_reserve = std::max(1, std::min(req->params.max_tokens, 256));
  if (static_cast<int32_t>(prompt_tokens.size()) + output_reserve >= n_ctx) {
    throw RequestError("context_length_exceeded",
      "prompt exceeds context window; prompt tokens=" + std::to_string(prompt_tokens.size()) +
      ", context=" + std::to_string(n_ctx) +
      ", reserved output tokens=" + std::to_string(output_reserve) +
      ". Increase CLAP_LLAMA_CONTEXT or reduce the prompt/tool history."
    );
  }
  // Per-session context cap: bounds one session's share of the unified KV
  // pool so a single conversation cannot promise itself the full window on a
  // box shared by many sessions. Admin policy, not a physical limit.
  const int32_t session_cap = env_int("CLAP_LLAMA_MAX_SESSION_CTX", 0);
  if (session_cap > 0 && static_cast<int32_t>(prompt_tokens.size()) + output_reserve >= session_cap) {
    throw RequestError("context_length_exceeded",
      "prompt exceeds the per-session context cap; prompt tokens=" + std::to_string(prompt_tokens.size()) +
      ", max_session_ctx=" + std::to_string(session_cap) +
      ", reserved output tokens=" + std::to_string(output_reserve) +
      ". Reduce the prompt/tool history or raise max_session_ctx / CLAP_LLAMA_MAX_SESSION_CTX."
    );
  }

  req->prompt_token_count = static_cast<int>(prompt_tokens.size());
  req->full_prompt_tokens = prompt_tokens;

  if (llama_model_has_encoder(loaded.model)) {
    // Encoder-decoder models run alone (admission guarantees no other active
    // request) and reset all cache state.
    for (auto& s : loaded.slots) {
      s.tokens.clear();
      s.is_anchor = false;
    }
    llama_memory_clear(llama_get_memory(loaded.ctx), true);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_encode(loaded.ctx, batch)) throw std::runtime_error("llama_encode failed");
    llama_token decoder_start = llama_model_decoder_start_token(loaded.model);
    if (decoder_start == LLAMA_TOKEN_NULL) decoder_start = llama_vocab_bos(vocab);
    req->prompt_tokens = { decoder_start };
    req->full_prompt_tokens = req->prompt_tokens;
    req->seq = 0;
    req->slot = &loaded.slots[0];
    req->slot->busy = true;
    req->slot->last_used = ++loaded.use_counter;
    req->sampler = make_sampler(req->params);
    return req;
  }

  llama_memory_t mem = llama_get_memory(loaded.ctx);
  auto common_prefix = [&](const std::vector<llama_token>& candidate) {
    std::size_t p = 0;
    const std::size_t limit = std::min(candidate.size(), prompt_tokens.size());
    while (p < limit && candidate[p] == prompt_tokens[p]) p += 1;
    return p;
  };

  // Donor scan across ALL slots — busy sessions and anchors included — for
  // the longest shared prefix. A tiny prefix (e.g. just <bos>) is noise.
  std::size_t donor = SIZE_MAX;
  std::size_t donor_prefix = 0;
  for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
    if (loaded.slots[index].tokens.empty()) continue;
    const std::size_t p = common_prefix(loaded.slots[index].tokens);
    if (p > donor_prefix) {
      donor_prefix = p;
      donor = index;
    }
  }
  if (donor_prefix < 16) {
    donor = SIZE_MAX;
    donor_prefix = 0;
  }
  const bool donor_exact = donor != SIZE_MAX && donor_prefix == loaded.slots[donor].tokens.size();
  const bool donor_idle_session =
      donor != SIZE_MAX && !loaded.slots[donor].busy && !loaded.slots[donor].is_anchor;
  // In-place continuation: exact idle donors always (cheapest, no copy); for
  // hybrid models also non-exact idle donors — llama.cpp recurrent state
  // checkpoints can often rewind them, and branching mid-stream is impossible
  // anyway, so trying the rewind beats an unconditional full re-prefill.
  // Attention models with a non-exact donor branch instead (preserves the
  // donor's longer suffix for its own session).
  const bool donor_usable_in_place = donor_idle_session && (donor_exact || loaded.hybrid);

  // Fresh slot pick: empty first, then oldest idle session; anchors are
  // recycled last because they serve every future session of their prefix.
  auto pick_fresh_slot = [&]() -> std::size_t {
    std::size_t best = SIZE_MAX;
    uint64_t oldest = UINT64_MAX;
    for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
      if (loaded.slots[index].busy || index == donor) continue;
      const auto& candidate = loaded.slots[index];
      uint64_t age = candidate.tokens.empty() ? 0 : candidate.last_used;
      if (candidate.is_anchor && !candidate.tokens.empty()) age = (UINT64_MAX / 2) + candidate.last_used;
      if (age < oldest) {
        oldest = age;
        best = index;
      }
    }
    return best;
  };

  if (donor_usable_in_place) {
    // Same-session continuation: extend the donor slot in place (no copy).
    auto* slot = &loaded.slots[donor];
    slot->last_used = ++loaded.use_counter;
    req->slot = slot;
    req->seq = static_cast<llama_seq_id>(donor);
    std::size_t prefix = donor_prefix;
    if (prefix == prompt_tokens.size()) prefix -= 1;  // always re-decode the last token for logits
    // Hybrid/recurrent models (e.g. Gated DeltaNet) cannot remove a partial
    // range from their recurrent state; llama_memory_seq_rm returns false. In
    // that case clear the sequence and re-ingest the whole prompt.
    if (prefix > 0 && llama_memory_seq_rm(mem, req->seq, static_cast<llama_pos>(prefix), -1)) {
      slot->tokens.resize(prefix);
      req->n_pos = static_cast<int32_t>(prefix);
      prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
      req->cached_prompt_tokens = static_cast<int>(prefix);
    } else {
      llama_memory_seq_rm(mem, req->seq, -1, -1);
      slot->tokens.clear();
      req->n_pos = 0;
    }
  } else {
    const std::size_t target = pick_fresh_slot();
    if (target == SIZE_MAX) throw std::runtime_error("no idle KV slot available");
    auto* slot = &loaded.slots[target];
    llama_memory_seq_rm(mem, static_cast<llama_seq_id>(target), -1, -1);
    slot->tokens.clear();
    slot->is_anchor = false;
    slot->last_used = ++loaded.use_counter;
    req->slot = slot;
    req->seq = static_cast<llama_seq_id>(target);
    req->n_pos = 0;

    if (donor != SIZE_MAX) {
      // Shared-prefix dedup: branch the common prefix off the donor without
      // disturbing it, so sessions sharing a system prompt + tool schema skip
      // that prefill entirely. In the unified KV pool a sequence copy shares
      // cells (no duplicate memory). Attention KV supports partial-range
      // copies from any donor; hybrid/recurrent state only supports
      // whole-sequence copies (exact-prefix donors, e.g. anchors).
      const std::size_t want = std::min(donor_prefix, prompt_tokens.size() - 1);
      if (!loaded.hybrid && want >= 16) {
        llama_memory_seq_cp(mem, static_cast<llama_seq_id>(donor), req->seq, 0, static_cast<llama_pos>(want));
        slot->tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(want));
        req->n_pos = static_cast<int32_t>(want);
        prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(want));
        req->cached_prompt_tokens = static_cast<int>(want);
      } else if (loaded.hybrid && donor_exact && donor_prefix < prompt_tokens.size()) {
        llama_memory_seq_cp(mem, static_cast<llama_seq_id>(donor), req->seq, -1, -1);
        slot->tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(donor_prefix));
        req->n_pos = static_cast<int32_t>(donor_prefix);
        prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(donor_prefix));
        req->cached_prompt_tokens = static_cast<int>(donor_prefix);
      } else if (loaded.hybrid && !donor_exact) {
        // Cannot branch mid-stream off hybrid state. Prefill fully, but
        // snapshot an anchor at the shared boundary so every future session
        // with this prefix branches instantly.
        req->anchor_at = static_cast<int32_t>(donor_prefix);
      }
    }
  }

  // Proactive KV pressure relief: evict idle slots (oldest first) when the
  // resident sessions plus this prompt would overflow the unified pool.
  // Note: in the unified pool, copied prefixes share cells, so this sum
  // over-counts shared tokens — a conservative (early) eviction trigger.
  std::size_t resident = 0;
  for (const auto& s : loaded.slots) resident += s.tokens.size();
  const std::size_t incoming = prompt_tokens.size() + static_cast<std::size_t>(output_reserve);
  const std::size_t capacity = static_cast<std::size_t>(n_ctx);
  while (resident + incoming > capacity) {
    std::size_t victim = loaded.slots.size();
    uint64_t oldest = UINT64_MAX;
    for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
      if (&loaded.slots[index] == req->slot || loaded.slots[index].busy || loaded.slots[index].tokens.empty()) continue;
      if (loaded.slots[index].last_used < oldest) {
        oldest = loaded.slots[index].last_used;
        victim = index;
      }
    }
    if (victim >= loaded.slots.size()) break;
    llama_memory_seq_rm(llama_get_memory(loaded.ctx), static_cast<llama_seq_id>(victim), -1, -1);
    resident -= loaded.slots[victim].tokens.size();
    loaded.slots[victim].tokens.clear();
    loaded.slots[victim].is_anchor = false;
    fprintf(stderr, "clap-llama: evicted KV slot %zu to fit incoming prompt\n", victim);
  }

  req->prompt_tokens = std::move(prompt_tokens);
  req->sampler = make_sampler(req->params);
  req->slot->busy = true;
  return req;
}

}  // namespace

int main() {
  LoadedLlama loaded;
  std::vector<std::unique_ptr<ActiveRequest>> active;
  std::deque<std::pair<std::string, json>> waiting;

  try {
    ggml_backend_load_all();

    StdinReader reader;
    bool running = true;

    // Returns false on shutdown.
    auto handle_message = [&](const std::string& line) -> bool {
      if (line.empty()) return true;
      std::string id;
      try {
        const json message = json::parse(line);
        id = message.value("id", "");
        const std::string type = message.value("type", "");

        if (type == "shutdown") {
          emit(id, json{{"done", true}});
          return false;
        }
        if (type == "cancel") {
          const std::string target = message.value("id", "");
          for (auto& req : active) {
            if (!req->done && (target.empty() || req->id == target)) req->cancelled = true;
          }
          for (auto it = waiting.begin(); it != waiting.end(); ++it) {
            if (!target.empty() && it->first == target) {
              emit(target, json{{"done", true}, {"finish_reason", "cancel"}, {"cancelled", true}});
              waiting.erase(it);
              break;
            }
          }
          return true;
        }
        if (type == "unload") {
          if (!active.empty()) throw std::runtime_error("cannot unload while requests are active");
          unload(loaded);
          emit(id, json{{"unloaded", true}, {"done", true}});
          return true;
        }
        if (type == "load") {
          const std::string model = message.value("model", "");
          if (model.empty()) throw std::runtime_error("load.model is required");
          if (!active.empty() && loaded.model && loaded.model_path != model) {
            throw std::runtime_error("cannot switch models while requests are active");
          }
          load_model(loaded, model);
          emit(id, json{{"loaded", true}, {"done", true}});
          return true;
        }
        // Anything else is a chat request; it queues until a slot frees up.
        waiting.emplace_back(id, message);
        return true;
      } catch (const RequestError& error) {
        emit_error(id, error.what(), error.code);
        return true;
      } catch (const std::exception& error) {
        emit_error(id, error.what());
        return true;
      }
    };

    while (running) {
      std::string line;
      if (active.empty() && waiting.empty()) {
        if (!reader.next(line)) break;  // idle: block until work or EOF
        running = handle_message(line);
        if (!running) break;
      }
      while (reader.poll(line)) {
        running = handle_message(line);
        if (!running) break;
      }
      if (!running) break;

      // Admit queued requests to free slots.
      while (!waiting.empty()) {
        const std::string req_model = waiting.front().second.value("model", "");
        if (req_model.empty()) {
          emit_error(waiting.front().first, "chat.model is required");
          waiting.pop_front();
          continue;
        }
        // Drain in-flight requests before switching models.
        if (!active.empty() && loaded.model && loaded.model_path != req_model) break;
        try {
          load_model(loaded, req_model);
        } catch (const std::exception& error) {
          emit_error(waiting.front().first, error.what());
          waiting.pop_front();
          continue;
        }
        if (llama_model_has_encoder(loaded.model) && !active.empty()) break;
        std::size_t busy = 0;
        for (const auto& s : loaded.slots) busy += s.busy ? 1 : 0;
        if (busy >= loaded.slots.size()) break;
        auto [wid, wreq] = std::move(waiting.front());
        waiting.pop_front();
        try {
          auto prepared = prepare_request(loaded, wid, wreq);
          emit(wid, json{{"started", true}});
          active.push_back(std::move(prepared));
        } catch (const RequestError& error) {
          emit_error(wid, error.what(), error.code);
        } catch (const std::exception& error) {
          emit_error(wid, error.what());
        }
      }

      if (!active.empty()) {
        step(loaded, active);
        active.erase(
          std::remove_if(active.begin(), active.end(), [](const std::unique_ptr<ActiveRequest>& r) { return r->done; }),
          active.end());
      }
    }

    for (auto& req : active) {
      if (!req->done) {
        req->finish_reason = "cancel";
        req->cancelled = true;
        finalize(*req);
      }
    }
    unload(loaded);
    return 0;
  } catch (const std::exception& error) {
    emit_error("", error.what());
    unload(loaded);
    return 1;
  }
}
