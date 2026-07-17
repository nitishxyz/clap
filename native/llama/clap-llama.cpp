#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
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
  };
  std::vector<CacheSlot> slots;
  uint64_t use_counter = 0;
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
  std::cout << fields.dump() << "\n";
  std::cout.flush();
}

void emit_error(const std::string& id, const std::string& message) {
  emit(id, json{{"error", message}});
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
  // Default to the model's trained context (capped at 32k for memory). Agent
  // harnesses routinely send 5-15k-token prompts, so a fixed 4096 default
  // rejects them. CLAP_LLAMA_CONTEXT still overrides explicitly.
  int32_t n_ctx = env_int("CLAP_LLAMA_CONTEXT", 0);
  if (n_ctx <= 0) {
    const int32_t train_ctx = llama_model_n_ctx_train(loaded.model);
    n_ctx = std::min(train_ctx > 0 ? train_ctx : 8192, 32768);
    n_ctx = std::max(n_ctx, 8192);
  }
  ctx_params.n_ctx = n_ctx;
  // Match llama.cpp server defaults; small batches make long-prompt prefill
  // several times slower on Metal.
  ctx_params.n_batch = env_int("CLAP_LLAMA_BATCH", 2048);
  ctx_params.n_ubatch = env_int("CLAP_LLAMA_UBATCH", 512);
  // Multiple KV cache slots (one llama sequence each) so any number of
  // concurrent sessions keep warm prefixes; slots are LRU-recycled.
  const int32_t n_slots = std::max(1, env_int("CLAP_LLAMA_SLOTS", 4));
  ctx_params.n_seq_max = n_slots;
  ctx_params.no_perf = true;
  loaded.ctx = llama_init_from_model(loaded.model, ctx_params);
  if (!loaded.ctx) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
    throw std::runtime_error("failed to create llama context for: " + model_path);
  }
  loaded.model_path = model_path;
  loaded.slots.assign(static_cast<std::size_t>(n_slots), {});
  loaded.use_counter = 0;
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

void decode_tokens(LoadedLlama& loaded, const std::vector<llama_token>& tokens, int32_t& n_pos, llama_seq_id seq,
                   const std::function<void(std::size_t)>& progress = nullptr) {
  const int32_t n_batch = static_cast<int32_t>(llama_n_batch(loaded.ctx));
  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.ctx));
  llama_batch batch = llama_batch_init(n_batch, 0, 1);
  try {
    for (std::size_t offset = 0; offset < tokens.size();) {
      batch.n_tokens = 0;
      const std::size_t remaining = tokens.size() - offset;
      const int32_t available = n_ctx - n_pos;
      if (available <= 0) {
        llama_batch_free(batch);
        throw std::runtime_error("prompt exceeds context window; increase CLAP_LLAMA_CONTEXT or reduce prompt size");
      }
      const int32_t chunk = static_cast<int32_t>(std::min<std::size_t>(remaining, static_cast<std::size_t>(std::min(n_batch, available))));
      for (int32_t i = 0; i < chunk; ++i) {
        const bool logits = offset + i + 1 == tokens.size();
        batch_add(batch, tokens[offset + i], n_pos + i, logits, seq);
      }
      if (llama_decode(loaded.ctx, batch)) {
        llama_batch_free(batch);
        throw std::runtime_error("llama_decode failed while ingesting prompt; prompt was chunked to fit n_batch but llama.cpp rejected the batch");
      }
      n_pos += chunk;
      offset += static_cast<std::size_t>(chunk);
      if (progress && offset < tokens.size()) progress(offset);
    }
  } catch (...) {
    llama_batch_free(batch);
    throw;
  }
  llama_batch_free(batch);
}

std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
  char buffer[256];
  int n = llama_token_to_piece(vocab, token, buffer, sizeof(buffer), 0, true);
  if (n < 0) return "";
  return std::string(buffer, n);
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

void generate(LoadedLlama& loaded, const std::string& id, const json& request, const std::function<bool()>& check_cancel) {
  const std::string model_path = request.value("model", "");
  if (model_path.empty()) throw std::runtime_error("chat.model is required");
  load_model(loaded, model_path);

  const llama_vocab* vocab = llama_model_get_vocab(loaded.model);
  const std::vector<ChatEntry> entries = messages_from_request(request);
  std::string prompt = templated_prompt(loaded.model, entries);
  if (prompt.empty()) prompt = fallback_prompt(entries);
  if (prompt.empty() && request.contains("prompt") && request["prompt"].is_string()) {
    prompt = request["prompt"].get<std::string>();
  }
  if (prompt.empty()) throw std::runtime_error("chat request contains no messages or prompt");

  const SamplingParams params = sampling_from_request(request);

  const bool add_special = prompt.find("<bos>") == std::string::npos && prompt.find("<|turn>") == std::string::npos;
  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_special, true);
  if (n_prompt <= 0) throw std::runtime_error("failed to tokenize prompt");
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), add_special, true) < 0) {
    throw std::runtime_error("failed to tokenize prompt");
  }

  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.ctx));
  const int32_t output_reserve = std::max(1, std::min(params.max_tokens, 256));
  if (static_cast<int32_t>(prompt_tokens.size()) + output_reserve >= n_ctx) {
    throw std::runtime_error(
      "prompt exceeds context window; prompt tokens=" + std::to_string(prompt_tokens.size()) +
      ", context=" + std::to_string(n_ctx) +
      ", reserved output tokens=" + std::to_string(output_reserve) +
      ". Increase CLAP_LLAMA_CONTEXT or reduce the prompt/tool history."
    );
  }

  const int prompt_token_count = static_cast<int>(prompt_tokens.size());

  int32_t n_pos = 0;
  llama_seq_id seq = 0;
  int cached_prompt_tokens = 0;
  LoadedLlama::CacheSlot* slot = nullptr;
  if (llama_model_has_encoder(loaded.model)) {
    for (auto& s : loaded.slots) s.tokens.clear();
    llama_memory_clear(llama_get_memory(loaded.ctx), true);
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_encode(loaded.ctx, batch)) throw std::runtime_error("llama_encode failed");
    llama_token decoder_start_token_id = llama_model_decoder_start_token(loaded.model);
    if (decoder_start_token_id == LLAMA_TOKEN_NULL) decoder_start_token_id = llama_vocab_bos(vocab);
    prompt_tokens = { decoder_start_token_id };
  } else {
    // Multi-session KV slots: pick the slot with the longest common prefix
    // against this prompt so any number of interleaved sessions keep warm
    // caches. Ties/misses recycle the least-recently-used slot.
    std::size_t best_slot = 0;
    std::size_t best_prefix = 0;
    for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
      const auto& candidate = loaded.slots[index].tokens;
      std::size_t prefix = 0;
      const std::size_t max_prefix = std::min(candidate.size(), prompt_tokens.size());
      while (prefix < max_prefix && candidate[prefix] == prompt_tokens[prefix]) prefix += 1;
      if (prefix > best_prefix) {
        best_prefix = prefix;
        best_slot = index;
      }
    }
    // A tiny shared prefix (e.g. just <bos>) is not a session match; prefer
    // the LRU empty-or-oldest slot instead so we do not evict a warm session.
    if (best_prefix < 16) {
      best_prefix = 0;
      best_slot = 0;
      uint64_t oldest = UINT64_MAX;
      for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
        const auto& candidate = loaded.slots[index];
        const uint64_t age = candidate.tokens.empty() ? 0 : candidate.last_used;
        if (age < oldest) {
          oldest = age;
          best_slot = index;
        }
      }
    }
    slot = &loaded.slots[best_slot];
    slot->last_used = ++loaded.use_counter;
    seq = static_cast<llama_seq_id>(best_slot);
    std::size_t prefix = best_prefix;
    if (prefix == prompt_tokens.size()) prefix -= 1;  // always re-decode the last token to get logits
    if (prefix > 0) {
      llama_memory_seq_rm(llama_get_memory(loaded.ctx), seq, static_cast<llama_pos>(prefix), -1);
      slot->tokens.resize(prefix);
      n_pos = static_cast<int32_t>(prefix);
      prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() + static_cast<std::ptrdiff_t>(prefix));
      cached_prompt_tokens = static_cast<int>(prefix);
    } else {
      llama_memory_seq_rm(llama_get_memory(loaded.ctx), seq, -1, -1);
      slot->tokens.clear();
      n_pos = 0;
    }
  }

  const bool track_cache = slot != nullptr;

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

  try {
    // Report prefill progress for long prompts so the server dashboard can
    // show real ingestion state instead of an opaque wait.
    const std::size_t progress_threshold = 1024;
    if (prompt_tokens.size() > progress_threshold) {
      decode_tokens(loaded, prompt_tokens, n_pos, seq, [&](std::size_t ingested) {
        emit(id, json{{"prefill", {
          {"done", cached_prompt_tokens + static_cast<int>(ingested)},
          {"total", prompt_token_count},
        }}});
      });
    } else {
      decode_tokens(loaded, prompt_tokens, n_pos, seq);
    }
  } catch (...) {
    if (slot) slot->tokens.clear();
    llama_sampler_free(sampler);
    throw;
  }
  if (track_cache) slot->tokens.insert(slot->tokens.end(), prompt_tokens.begin(), prompt_tokens.end());

  std::string finish_reason = "stop";
  std::string held;  // tail that may be a partial stop sequence
  int completion_tokens = 0;
  bool stopped = false;
  bool cancelled = false;

  for (int n_decode = 0; n_decode < params.max_tokens && !stopped; ) {
    if (check_cancel && check_cancel()) {
      finish_reason = "cancel";
      cancelled = true;
      break;
    }
    llama_token token = llama_sampler_sample(sampler, loaded.ctx, -1);
    if (llama_vocab_is_eog(vocab, token)) break;

    completion_tokens += 1;
    const std::string piece = token_to_piece(vocab, token);
    if (!piece.empty()) {
      held += piece;
      if (!params.stops.empty()) {
        const std::size_t stop_index = find_stop(held, params.stops);
        if (stop_index != std::string::npos) {
          const std::string visible = held.substr(0, stop_index);
          if (!visible.empty()) emit(id, json{{"token", visible}});
          finish_reason = "stop";
          stopped = true;
          break;
        }
        const std::size_t hold = partial_stop_suffix(held, params.stops);
        const std::string visible = held.substr(0, held.size() - hold);
        if (!visible.empty()) {
          emit(id, json{{"token", visible}});
          held = held.substr(visible.size());
        }
      } else {
        emit(id, json{{"token", held}});
        held.clear();
      }
    }

    std::vector<llama_token> next = { token };
    try {
      decode_tokens(loaded, next, n_pos, seq);
    } catch (...) {
      if (slot) slot->tokens.clear();
      llama_sampler_free(sampler);
      throw std::runtime_error("llama_decode failed; this often indicates llama.cpp/Metal GPU memory pressure. Check the llama worker log for kIOGPUCommandBufferCallbackErrorOutOfMemory. Try a smaller GGUF quant such as Q4_K_M, reduce CLAP_LLAMA_CONTEXT/CLAP_LLAMA_BATCH/CLAP_LLAMA_UBATCH, set CLAP_LLAMA_GPU_LAYERS to a lower value, or use CPU fallback with CLAP_LLAMA_GPU_LAYERS=0.");
    }
    if (track_cache) slot->tokens.push_back(token);
    n_decode += 1;
    if (n_decode >= params.max_tokens) finish_reason = "length";
    if (n_pos + 1 >= n_ctx) {
      finish_reason = "length";
      break;
    }
  }

  if (!stopped && !cancelled && !held.empty()) emit(id, json{{"token", held}});

  llama_sampler_free(sampler);
  emit(id, json{
    {"done", true},
    {"finish_reason", finish_reason},
    {"cancelled", cancelled},
    {"usage", json{
      {"prompt_tokens", prompt_token_count},
      {"completion_tokens", completion_tokens},
    }},
    {"cache", json{
      {"hit", cached_prompt_tokens > 0},
      {"reused_tokens", cached_prompt_tokens},
      {"slot", static_cast<int>(seq)},
    }},
  });
}

}  // namespace

int main() {
  LoadedLlama loaded;
  try {
    ggml_backend_load_all();

    StdinReader reader;
    std::deque<std::string> deferred;

    // Non-blockingly drains stdin during generation; returns true when a
    // cancel for the active request arrives. Other messages are deferred.
    auto make_cancel_check = [&](const std::string& active_id) {
      return [&reader, &deferred, active_id]() {
        std::string pending;
        while (reader.poll(pending)) {
          if (pending.empty()) continue;
          try {
            const json message = json::parse(pending);
            if (message.value("type", "") == "cancel") {
              const std::string target = message.value("id", "");
              if (target.empty() || target == active_id) return true;
              continue;  // cancel for an unknown request: drop it
            }
          } catch (...) {
            // Defer unparsable lines to the main loop for error reporting.
          }
          deferred.push_back(pending);
        }
        return false;
      };
    };

    std::string line;
    while (true) {
      if (!deferred.empty()) {
        line = std::move(deferred.front());
        deferred.pop_front();
      } else if (!reader.next(line)) {
        break;
      }
      if (line.empty()) continue;

      std::string id;
      try {
        const json request = json::parse(line);
        id = request.value("id", "");
        const std::string type = request.value("type", "");

        if (type == "shutdown") {
          emit(id, json{{"done", true}});
          unload(loaded);
          return 0;
        }
        if (type == "cancel") {
          continue;  // request already finished; nothing to cancel
        }
        if (type == "unload") {
          unload(loaded);
          emit(id, json{{"unloaded", true}, {"done", true}});
          continue;
        }
        if (type == "load") {
          const std::string model = request.value("model", "");
          if (model.empty()) throw std::runtime_error("load.model is required");
          load_model(loaded, model);
          emit(id, json{{"loaded", true}, {"done", true}});
          continue;
        }
        generate(loaded, id, request, make_cancel_check(id));
      } catch (const std::exception& error) {
        emit_error(id, error.what());
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
