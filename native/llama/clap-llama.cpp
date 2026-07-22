#include "llama.h"
#include "active-concurrency.h"
#include "cache-adapter.h"
#include "clap/llama/environment.h"
#include "clap/llama/protocol.h"
#include "native-characterization.h"
#include "stable-boundary.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using json = nlohmann::json;

namespace {

using clap::llama::available_memory_bytes;
using clap::llama::env_enabled;
using clap::llama::env_int;
using clap::llama::env_u64;
using clap::llama::emit;
using clap::llama::emit_error;
using clap::llama::RequestError;
using clap::llama::StdinReader;

struct LoadedLlama {
  std::string model_path;
  llama_model* model = nullptr;
  llama_context* ctx = nullptr;
  int32_t model_context_window = 0;
  int32_t backend_allocation_cap = 0;
  int32_t context_override = 0;
  int32_t max_output_tokens = 0;
  int32_t max_active = 0;
  int32_t retained_max = 0;
  clap::llama_active::Decision active_policy{};
  uint64_t startup_available_bytes = 0;
  uint64_t model_file_bytes = 0;
  std::string last_eviction_reason;
  int32_t previous_max_active = 0;
  std::string last_adjustment_reason;
  std::string last_adjustment_at;
  uint64_t retained_growth_reserve_bytes = 0;
  uint64_t global_resident_memory_bytes = 0;
  std::string pressure_state;
  // KV cache slots: one llama sequence per slot so multiple concurrent
  // sessions (agent loops, chats, side requests) each keep a warm prefix.
  struct CacheSlot {
    std::vector<llama_token> tokens;
    uint64_t last_used = 0;
    uint64_t coordinator_generation = 0;
    bool busy = false;  // an active request is generating on this slot
    bool is_anchor = false;  // holds a shared prefix snapshot, never generates
  };
  std::vector<CacheSlot> slots;
  uint64_t use_counter = 0;
  std::unique_ptr<clap::llama_cache::Coordinator> coordinator;
  std::string cache_domain;
  // Hybrid/recurrent models (e.g. Gated DeltaNet) only support whole-sequence
  // KV state copies; attention-only models support partial-range copies.
  bool hybrid = false;
};

struct ChatEntry {
  std::string role;
  std::string content;
};

struct SamplingParams {
  // Zero means omitted by the caller; admission derives a safe request-local
  // default from the loaded model's remaining effective context.
  int max_tokens = 0;
  double temperature = 0.7;
  double top_p = 0.95;
  int top_k = 0;
  uint32_t seed = LLAMA_DEFAULT_SEED;
  double presence_penalty = 0.0;
  double frequency_penalty = 0.0;
  std::vector<std::string> stops;
};

struct CacheBackpressure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

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
    if (role.empty()) continue;
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
  return params;
}

std::string templated_prompt(llama_model* model, const std::vector<ChatEntry>& entries,
                             bool add_generation_prompt = true) {
  if (entries.empty()) return "";

  const char* tmpl = llama_model_chat_template(model, nullptr);
  if (tmpl && std::string(tmpl).find("<|turn>") != std::string::npos) {
    std::string prompt = "<bos>";
    for (const auto& entry : entries) {
      std::string role = entry.role == "assistant" ? "model" : entry.role;
      if (role == "tool") role = "user";
      prompt += "<|turn>" + role + "\n" + entry.content + "<turn|>\n";
    }
    if (add_generation_prompt) prompt += "<|turn>model\n";
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

  int32_t length = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
      add_generation_prompt, nullptr, 0);
  if (length <= 0) return "";
  std::vector<char> buffer(static_cast<std::size_t>(length) + 1);
  int32_t written = llama_chat_apply_template(tmpl, chat.data(), chat.size(),
      add_generation_prompt, buffer.data(), buffer.size());
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
  if (loaded.coordinator) loaded.coordinator->reset();
  loaded.coordinator.reset();
  if (loaded.ctx) {
    llama_free(loaded.ctx);
    loaded.ctx = nullptr;
  }
  if (loaded.model) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
  }
  loaded.model_path.clear();
  loaded.cache_domain.clear();
  loaded.slots.clear();
  loaded.model_context_window = 0;
  loaded.backend_allocation_cap = 0;
  loaded.context_override = 0;
  loaded.max_output_tokens = 0;
  loaded.max_active = 0;
  loaded.retained_max = 0;
  loaded.active_policy = {};
  loaded.startup_available_bytes = 0;
  loaded.model_file_bytes = 0;
  loaded.last_eviction_reason.clear();
}

void load_model(LoadedLlama& loaded, const std::string& model_path) {
  if (loaded.model && loaded.ctx && loaded.model_path == model_path) return;
  unload(loaded);

  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("GGUF model not found: " + model_path);
  }
  const uint64_t startup_available = available_memory_bytes();
  std::error_code file_size_error;
  const uint64_t model_file_bytes = std::filesystem::file_size(model_path, file_size_error);
  // The pinned llama.cpp memory API exposes the allocated context-cell count,
  // but no authoritative used/free cell or KV byte telemetry. A byte budget
  // therefore cannot be converted to cells without fabricating a bytes/token
  // ratio (shared branches make logical token sums invalid). Fail explicit
  // byte policy closed rather than silently overallocating or reporting fake
  // byte pressure.
  if (env_enabled("CLAP_LLAMA_KV_BUDGET_BYTES") ||
      env_enabled("CLAP_LLAMA_KV_BUDGET_PERCENT")) {
    throw std::runtime_error(
        "CLAP_LLAMA_KV_BUDGET_BYTES/PERCENT is unsupported by the pinned llama.cpp API: "
        "authoritative KV used/free cells and bytes are unavailable; use CLAP_LLAMA_CONTEXT");
  }

  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = env_int("CLAP_LLAMA_GPU_LAYERS", 999);
  loaded.model = llama_model_load_from_file(model_path.c_str(), model_params);
  if (!loaded.model) throw std::runtime_error("failed to load GGUF model: " + model_path);

  llama_context_params ctx_params = llama_context_default_params();
  // Allocate no more than the model declares. An explicit admin override is a
  // cap, not permission to manufacture a larger model context window.
  const int32_t env_ctx = env_int("CLAP_LLAMA_CONTEXT", 0);
  const int32_t train_ctx = llama_model_n_ctx_train(loaded.model);
  int32_t n_ctx = train_ctx > 0 && env_ctx > 0 ? std::min(train_ctx, env_ctx)
      : (env_ctx > 0 ? env_ctx : train_ctx);
  // Match llama.cpp server defaults; small batches make long-prompt prefill
  // several times slower on Metal.
  ctx_params.n_batch = env_int("CLAP_LLAMA_BATCH", 2048);
  ctx_params.n_ubatch = env_int("CLAP_LLAMA_UBATCH", 512);
  // Active scheduling and retained sequence identity are independent. Sequence
  // IDs are cheap labels over the unified KV engine; they do not partition or
  // multiply the physical context-cell allocation.
  const int32_t retained_override = env_int("CLAP_LLAMA_RETAINED_MAX", 0);
  const int32_t derived_retained = std::min(128, std::max(32, std::max(n_ctx, 1) / 256));
  const int32_t retained_max = retained_override > 0 ? retained_override : derived_retained;
  ctx_params.n_seq_max = retained_max;
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
  // A zero n_ctx asks llama.cpp to use its model-derived allocation. The
  // actual successful allocation below is authoritative for enforcement.
  ctx_params.n_ctx = std::max(n_ctx, 0);
  loaded.ctx = llama_init_from_model(loaded.model, ctx_params);
  if (!loaded.ctx) {
    llama_model_free(loaded.model);
    loaded.model = nullptr;
    throw std::runtime_error("failed to create llama context for: " + model_path);
  }
  loaded.model_path = model_path;
  loaded.model_context_window = train_ctx > 0 ? train_ctx : 0;
  loaded.backend_allocation_cap = static_cast<int32_t>(llama_n_ctx(loaded.ctx));
  loaded.context_override = env_ctx > 0 ? env_ctx : 0;
  loaded.max_output_tokens = std::max(0, env_int("CLAP_LLAMA_MAX_OUTPUT", 0));
  loaded.retained_max = retained_max;
  n_ctx = loaded.backend_allocation_cap;
  loaded.slots.assign(static_cast<std::size_t>(retained_max), {});
  loaded.use_counter = 0;
  loaded.hybrid = llama_model_is_recurrent(loaded.model) || llama_model_is_hybrid(loaded.model);
  loaded.startup_available_bytes = startup_available;
  loaded.model_file_bytes = file_size_error ? 0 : model_file_bytes;
  loaded.active_policy = clap::llama_active::select({
      env_int("CLAP_MAX_ACTIVE", 0), startup_available, loaded.model_file_bytes,
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency())), n_ctx,
      retained_max, loaded.hybrid, llama_model_has_encoder(loaded.model)});
  loaded.max_active = loaded.active_policy.selected_max;
  const char* kv_type = std::getenv("CLAP_LLAMA_KV_TYPE");
  loaded.cache_domain = model_path + "|llama|ctx=" + std::to_string(n_ctx) +
      "|kv=" + (kv_type && *kv_type ? kv_type : "f16") +
      "|unified=" + (ctx_params.kv_unified ? "1" : "0") +
      "|layout=1";
  try {
    loaded.coordinator = std::make_unique<clap::llama_cache::Coordinator>(
      1, 16, static_cast<uint64_t>(n_ctx),
      static_cast<uint32_t>(retained_max),
      static_cast<uint32_t>(retained_max), 0, 0, 0,
      env_int("CLAP_CACHE_CHECKPOINTS_ENABLED", 1) != 0,
      static_cast<uint64_t>(env_int("CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS", 2048)),
      static_cast<uint64_t>(env_int("CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS", 2048)),
      static_cast<uint32_t>(env_int("CLAP_CACHE_CHECKPOINT_MAX", 8)),
      static_cast<uint32_t>(env_int("CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS", 2500)),
      env_u64("CLAP_CACHE_CHECKPOINT_BUDGET_BYTES", 0));
    for (int32_t expected = 1; expected < retained_max; ++expected) {
      const auto registered = loaded.coordinator->register_slot();
      if (registered.slot != static_cast<uint32_t>(expected) || registered.generation == 0) {
        throw std::runtime_error("cache coordinator returned unstable slot registration");
      }
    }
  } catch (const std::exception& error) {
    loaded.coordinator.reset();
    fprintf(stderr, "clap-llama: cache coordinator unavailable; using no-cache fresh mode: %s\n", error.what());
  }
}

json retention_telemetry(const LoadedLlama& loaded, std::size_t active, std::size_t queued = 0) {
  uint32_t retained_total = 0;
  uint32_t retained_sessions = 0;
  uint32_t retained_anchors = 0;
  uint64_t evictions = 0;
  if (loaded.coordinator) {
    const auto retention = loaded.coordinator->retention_telemetry();
    const auto telemetry = loaded.coordinator->telemetry();
    retained_total = retention.active_slots;
    retained_sessions = retention.session_slots;
    retained_anchors = retention.anchor_slots;
    evictions = telemetry.evictions;
  }
  return json{
    {"max_active", loaded.max_active},
    {"queued", queued},
    {"previous_max_active", loaded.previous_max_active > 0
        ? json(loaded.previous_max_active) : json(nullptr)},
    {"last_adjustment_reason", loaded.last_adjustment_reason.empty()
        ? json(nullptr) : json(loaded.last_adjustment_reason)},
    {"last_adjustment_at", loaded.last_adjustment_at.empty()
        ? json(nullptr) : json(loaded.last_adjustment_at)},
    {"retained_growth_reserve_bytes", loaded.retained_growth_reserve_bytes},
    {"global_resident_memory_bytes", loaded.global_resident_memory_bytes > 0
        ? json(loaded.global_resident_memory_bytes) : json(nullptr)},
    {"pressure_state", loaded.pressure_state.empty()
        ? json(nullptr) : json(loaded.pressure_state)},
    {"active_policy", {
      {"mode", loaded.active_policy.mode},
      {"selected_max", loaded.active_policy.selected_max},
      {"backend_ceiling", loaded.active_policy.backend_ceiling},
      {"hardware_ceiling", loaded.active_policy.hardware_ceiling},
      {"model_ceiling", loaded.active_policy.model_ceiling},
      {"memory_ceiling", loaded.active_policy.memory_ceiling},
      {"reason", loaded.active_policy.reason},
      {"inputs", {
        {"startup_available_bytes", loaded.startup_available_bytes > 0
            ? json(loaded.startup_available_bytes) : json(nullptr)},
        {"model_file_bytes", loaded.model_file_bytes > 0
            ? json(loaded.model_file_bytes) : json(nullptr)},
        {"context_capacity", loaded.backend_allocation_cap},
        {"context_ceiling", loaded.active_policy.context_ceiling},
        {"per_active_reserve_cells", loaded.active_policy.per_active_reserve_cells},
        {"per_active_reserve_bytes", loaded.active_policy.per_active_reserve_bytes},
        {"processor_count", std::max(1u, std::thread::hardware_concurrency())},
        {"hybrid_or_recurrent", loaded.hybrid},
      }},
    }},
    {"active", active},
    {"retained_total", retained_total},
    {"retained_sessions", retained_sessions},
    {"retained_anchors", retained_anchors},
    // Byte pressure is intentionally disabled: pinned llama.cpp has no public
    // authoritative KV byte or used/free-cell telemetry.
    {"retained_bytes", 0}, {"session_bytes", 0}, {"anchor_bytes", 0},
    {"budget_bytes", 0}, {"high_watermark_bytes", 0}, {"low_watermark_bytes", 0},
    {"under_pressure", false},
    {"hard_ceiling", loaded.retained_max},
    {"eviction_reason", loaded.last_eviction_reason.empty()
        ? json(nullptr) : json(loaded.last_eviction_reason)},
    {"eviction_count", evictions},
    {"physical_cell_capacity", loaded.backend_allocation_cap},
    {"physical_cells_used", nullptr},
    {"physical_cells_free", nullptr},
  };
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
  clap::llama_cache::Coordinator* coordinator = nullptr;
  clap::llama_cache::Identity cache_identity;

  std::vector<llama_token> prompt_tokens;       // un-ingested remainder
  std::vector<llama_token> full_prompt_tokens;  // for the one ingest retry
  std::size_t ingested = 0;
  int32_t n_pos = 0;
  int prompt_token_count = 0;
  int cached_prompt_tokens = 0;
  std::string cache_reuse_kind;
  std::string cache_reuse_scope;
  std::string cache_namespace;
  int cache_donor_slot = -1;
  uint64_t cache_donor_generation = 0;
  uint64_t cache_target_generation = 0;
  std::vector<uint32_t> cache_evicted_slots;
  uint64_t cache_planned_reuse_tokens = 0;
  uint64_t cache_realized_reuse_tokens = 0;
  uint64_t cache_decision_us = 0;
  std::string cache_fallback;
  std::string prompt_token_hash;
  clap::llama_boundary::StableBoundary stable_boundary;
  struct BoundaryInfo {
    std::size_t token_count;
    std::string kind;
    std::string label;
    bool requested;
    std::string status;
    std::string skip_reason;
  };
  std::vector<std::size_t> anchor_boundaries;
  std::vector<std::size_t> structural_boundaries;
  std::vector<std::size_t> materialized_boundaries;
  std::vector<BoundaryInfo> resolved_boundaries;
  json cache_candidates = json::array();
  bool cache_side_request = false;

  enum class Phase { Prefill, Decode };
  Phase phase = Phase::Prefill;
  llama_token pending_token = 0;  // sampled but not yet decoded
  std::string held;               // tail held for stop/UTF-8 boundaries
  int completion_tokens = 0;
  std::string finish_reason = "stop";
  bool cancelled = false;
  bool retried = false;
  bool done = false;
  int32_t anchor_at = -1;
  bool anchor_planted = false;

  // per-step scratch
  int32_t logits_index = -1;
  int32_t step_tokens = 0;
};

std::string cache_string(const json& cache, const char* key) {
  if (!cache.is_object() || !cache.contains(key) || !cache[key].is_string()) return "";
  return cache[key].get<std::string>();
}

const std::string& telemetry_key() {
  static const std::string key = [] {
    if (const char* installed = std::getenv("CLAP_TELEMETRY_HMAC_KEY"); installed && *installed) {
      return std::string(installed);
    }
    std::random_device random;
    std::ostringstream out;
    for (int index = 0; index < 8; ++index) out << std::hex << random();
    return out.str();
  }();
  return key;
}

template <typename Token>
std::string token_fingerprint(const std::vector<Token>& tokens, std::size_t count) {
  count = std::min(count, tokens.size());
  std::ostringstream encoded;
  encoded << telemetry_key() << "|tokens-v1|" << count << '|';
  for (std::size_t index = 0; index < count; ++index) {
    const uint32_t token = static_cast<uint32_t>(tokens[index]);
    encoded.write(reinterpret_cast<const char*>(&token), sizeof(token));
  }
  const std::string material = encoded.str();
  std::ostringstream result;
  for (int domain = 0; domain < 4; ++domain) {
    result << std::hex << clap::llama_cache::hash(std::to_string(domain) + material);
  }
  return result.str();
}

const char* candidate_state(uint32_t state) {
  switch (state) {
    case CLAP_CACHE_SLOT_SESSION: return "session";
    case CLAP_CACHE_SLOT_PROMPT_BOUNDARY: return "prompt_boundary";
    case CLAP_CACHE_SLOT_ANCHOR: return "anchor";
    default: return "empty";
  }
}

const char* candidate_rejection(uint32_t rejection) {
  switch (rejection) {
    case CLAP_CACHE_REJECTION_NAMESPACE: return "namespace";
    case CLAP_CACHE_REJECTION_MODEL_DOMAIN: return "model_domain";
    case CLAP_CACHE_REJECTION_GENERATION: return "generation";
    case CLAP_CACHE_REJECTION_BUSY_LEASE: return "busy_lease";
    case CLAP_CACHE_REJECTION_MATERIALIZATION: return "materialization";
    case CLAP_CACHE_REJECTION_SESSION: return "session";
    case CLAP_CACHE_REJECTION_NONTRIM: return "nontrim";
    case CLAP_CACHE_REJECTION_CAPABILITY: return "capability";
    case CLAP_CACHE_REJECTION_MIN_PREFIX: return "min_prefix";
    case CLAP_CACHE_REJECTION_CAPACITY: return "capacity";
    case CLAP_CACHE_REJECTION_ABSENT_ANCHOR: return "absent_anchor";
    case CLAP_CACHE_REJECTION_LOWER_RANK: return "lower_rank";
    default: return nullptr;
  }
}

clap::llama_cache::Identity cache_identity(const LoadedLlama& loaded,
                                            const std::string&,
                                            const json& request) {
  const json cache = request.contains("cache") && request["cache"].is_object()
      ? request["cache"] : json::object();
  const std::string requested_namespace = cache_string(cache, "namespace");
  const std::string tenant = requested_namespace.empty()
      ? cache_string(cache, "tenant") : requested_namespace;
  const std::string keyed = telemetry_key() + "|";
  clap::llama_cache::Identity identity;
  identity.name_space = clap::llama_cache::fingerprint(
      keyed + loaded.cache_domain + "|tenant=" + (tenant.empty() ? "local" : tenant));
  identity.tenant = clap::llama_cache::hash(keyed + (tenant.empty() ? "local" : tenant));
  identity.project = clap::llama_cache::hash(keyed + cache_string(cache, "project"));
  identity.harness = clap::llama_cache::hash(keyed + cache_string(cache, "harness"));
  identity.agent = clap::llama_cache::hash(keyed + cache_string(cache, "agent"));
  const std::string session = cache_string(cache, "session");
  identity.session = session.empty() ? 0 : clap::llama_cache::hash(keyed + session);
  identity.side_request = cache.value("side_request", false);
  const std::string priority = cache_string(cache, "priority");
  identity.priority = priority == "background" ? CLAP_CACHE_PRIORITY_BACKGROUND
                                                : CLAP_CACHE_PRIORITY_INTERACTIVE;
  if (!session.empty()) identity.scope = CLAP_CACHE_SCOPE_SESSION;
  else if (!cache_string(cache, "agent").empty()) identity.scope = CLAP_CACHE_SCOPE_AGENT;
  else if (!cache_string(cache, "project").empty()) identity.scope = CLAP_CACHE_SCOPE_PROJECT;
  else if (!cache_string(cache, "harness").empty()) identity.scope = CLAP_CACHE_SCOPE_HARNESS;
  else identity.scope = CLAP_CACHE_SCOPE_TENANT;
  return identity;
}

const char* cache_scope_name(uint32_t scope) {
  switch (scope) {
    case CLAP_CACHE_SCOPE_SESSION: return "session";
    case CLAP_CACHE_SCOPE_AGENT: return "agent";
    case CLAP_CACHE_SCOPE_PROJECT: return "project";
    case CLAP_CACHE_SCOPE_HARNESS: return "harness";
    case CLAP_CACHE_SCOPE_TENANT: return "tenant";
    default: return "none";
  }
}

uint64_t llama_cache_capabilities(const LoadedLlama& loaded) {
  uint64_t capabilities = CLAP_CACHE_CAP_WHOLE_STATE_COPY |
      CLAP_CACHE_CAP_SAFE_BUSY_DONOR | CLAP_CACHE_CAP_RELIABLE_RESIDENT_LENGTH |
      CLAP_CACHE_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS |
      CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT;
  if (loaded.hybrid) {
    capabilities |= CLAP_CACHE_CAP_RECURRENT_OR_HYBRID;
  } else {
    capabilities |= CLAP_CACHE_CAP_PARTIAL_SUFFIX_TRIM |
        CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH | CLAP_CACHE_CAP_ZERO_COPY_BRANCH |
        CLAP_CACHE_CAP_UNIFIED_STORAGE;
  }
  return capabilities;
}

bool prepare_with_coordinator(LoadedLlama& loaded, ActiveRequest& req,
                              std::vector<llama_token>& prompt_tokens,
                              int32_t output_reserve,
                              const std::vector<uint64_t>& stable_boundaries) {
  if (!loaded.coordinator) return false;
  clap::llama_cache::Plan plan;
  std::vector<uint8_t> slot_capabilities;
  slot_capabilities.reserve(loaded.slots.size());
  for (const auto& slot : loaded.slots) {
    uint8_t flags = slot.busy ? 0 : CLAP_CACHE_SLOT_WRITABLE;
    if (!slot.tokens.empty()) {
      flags |= CLAP_CACHE_SLOT_MATERIALIZED | CLAP_CACHE_SLOT_COPY;
      if (!loaded.hybrid) flags |= CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM;
    }
    slot_capabilities.push_back(flags);
  }
  try {
    plan = loaded.coordinator->plan(prompt_tokens, req.cache_identity,
        llama_cache_capabilities(loaded), static_cast<uint64_t>(output_reserve),
        CLAP_CACHE_SLOT_SESSION, slot_capabilities, stable_boundaries);
  } catch (const clap::llama_cache::Error& error) {
    if (error.status == CLAP_CACHE_NO_CAPACITY || error.status == CLAP_CACHE_SLOT_BUSY) {
      throw CacheBackpressure(error.what());
    }
    req.cache_fallback = "coordinator_plan_failed_closed";
    fprintf(stderr, "clap-llama: cache coordinator plan failed closed: %s\n", error.what());
    throw;
  }

  const auto view = plan.view();
  req.cache_candidates = json::array();
  for (const auto& candidate : plan.candidates()) {
    const char* rejection = candidate_rejection(candidate.rejection);
    req.cache_candidates.push_back(json{
      {"slot", candidate.slot}, {"generation", candidate.generation},
      {"state", candidate_state(candidate.state)},
      {"shared_prefix_tokens", candidate.shared_prefix_tokens},
      {"namespace_compatible", candidate.namespace_compatible != 0},
      {"model_compatible", candidate.model_compatible != 0},
      {"session_compatible", candidate.session_compatible != 0},
      {"generation_compatible", candidate.generation_compatible != 0},
      {"busy_eligible", candidate.busy_eligible != 0},
      {"lease_eligible", candidate.lease_eligible != 0},
      {"materialized", candidate.materialized != 0},
      {"trim_eligible", candidate.trim_eligible != 0},
      {"copy_eligible", candidate.copy_eligible != 0},
      {"eligible", candidate.eligible != 0}, {"selected", candidate.selected != 0},
      {"rejection", rejection ? json(rejection) : json(nullptr)},
    });
  }
  const std::size_t target = view.target.slot;
  const std::size_t donor = view.has_donor ? view.donor.slot : SIZE_MAX;
  if (target >= loaded.slots.size() || (donor != SIZE_MAX && donor >= loaded.slots.size())) {
    plan.abort();
    throw std::runtime_error("cache coordinator returned an invalid slot");
  }

  llama_memory_t mem = llama_get_memory(loaded.ctx);
  try {
    for (const auto& victim : plan.evictions()) {
      if (victim.slot >= loaded.slots.size() || victim.slot == target) continue;
      llama_memory_seq_rm(mem, static_cast<llama_seq_id>(victim.slot), -1, -1);
      loaded.slots[victim.slot] = {};
      req.cache_evicted_slots.push_back(victim.slot);
      loaded.last_eviction_reason = "hard_ceiling";
    }

    auto& slot = loaded.slots[target];
    std::size_t resident = 0;
    if (view.operation != CLAP_CACHE_OPERATION_CONTINUE) {
      llama_memory_seq_rm(mem, static_cast<llama_seq_id>(target), -1, -1);
      slot.tokens.clear();
      slot.is_anchor = false;
    }
    if (view.operation == CLAP_CACHE_OPERATION_CONTINUE) {
      resident = std::min<std::size_t>(view.reuse_tokens, prompt_tokens.size() - 1);
      if (resident == 0 || !llama_memory_seq_rm(mem, static_cast<llama_seq_id>(target),
                                                static_cast<llama_pos>(resident), -1)) {
        throw std::runtime_error("coordinator-selected continuation could not be materialized");
      }
    } else if (view.operation == CLAP_CACHE_OPERATION_BRANCH) {
      resident = std::min<std::size_t>(view.reuse_tokens, prompt_tokens.size() - 1);
      if (resident > 0) {
        if (loaded.hybrid && resident == loaded.slots[donor].tokens.size()) {
          llama_memory_seq_cp(mem, static_cast<llama_seq_id>(donor),
                              static_cast<llama_seq_id>(target), -1, -1);
        } else if (!loaded.hybrid) {
          llama_memory_seq_cp(mem, static_cast<llama_seq_id>(donor),
                              static_cast<llama_seq_id>(target), 0,
                              static_cast<llama_pos>(resident));
        } else {
          throw std::runtime_error("coordinator-selected branch could not be materialized");
        }
      }
    } else if (view.operation == CLAP_CACHE_OPERATION_RESTORE &&
               view.reuse_tokens < prompt_tokens.size()) {
      resident = static_cast<std::size_t>(view.reuse_tokens);
      llama_memory_seq_cp(mem, static_cast<llama_seq_id>(donor),
                          static_cast<llama_seq_id>(target), -1, -1);
    }

    const auto decision = plan.commit(resident, CLAP_CACHE_SLOT_SESSION);
    const auto info = loaded.coordinator->slot(static_cast<uint32_t>(target));
    slot.tokens.assign(prompt_tokens.begin(), prompt_tokens.begin() +
        static_cast<std::ptrdiff_t>(resident));
    slot.coordinator_generation = info.generation;
    slot.last_used = ++loaded.use_counter;
    slot.busy = true;
    req.coordinator = loaded.coordinator.get();
    req.slot = &slot;
    req.seq = static_cast<llama_seq_id>(target);
    req.n_pos = static_cast<int32_t>(resident);
    req.cached_prompt_tokens = static_cast<int>(decision.realized_reuse_tokens);
    req.cache_planned_reuse_tokens = decision.planned_reuse_tokens;
    req.cache_realized_reuse_tokens = decision.realized_reuse_tokens;
    req.cache_decision_us = decision.decision_us;
    req.anchor_boundaries.clear();
    for (const auto boundary : plan.anchor_boundaries()) {
      const auto count = static_cast<std::size_t>(boundary);
      req.anchor_boundaries.push_back(count);
      const auto known = std::find_if(req.resolved_boundaries.begin(),
          req.resolved_boundaries.end(), [count](const auto& value) {
            return value.token_count == count;
          });
      if (known == req.resolved_boundaries.end()) {
        req.resolved_boundaries.push_back(
            {count, "automatic_token", "", false, "authorized", ""});
      }
    }
    req.anchor_at = req.anchor_boundaries.empty()
        ? -1 : static_cast<int32_t>(req.anchor_boundaries.front());
    req.stable_boundary = clap::llama_boundary::exact(
        req.full_prompt_tokens, static_cast<std::size_t>(view.anchor_tokens), "prompt",
        [](const auto& tokens, std::size_t count) {
          return token_fingerprint(tokens, count);
        });
    req.cache_donor_slot = decision.has_donor ? static_cast<int>(decision.donor_slot) : -1;
    req.cache_donor_generation = decision.has_donor
        ? loaded.coordinator->slot(decision.donor_slot).generation : 0;
    req.cache_target_generation = info.generation;
    req.cache_reuse_scope = cache_scope_name(decision.scope);
    if (decision.operation == CLAP_CACHE_OPERATION_CONTINUE && resident > 0) {
      req.cache_reuse_kind = "slot";
    } else if (decision.operation == CLAP_CACHE_OPERATION_RESTORE && resident > 0) {
      req.cache_reuse_kind = "anchor";
    } else if (decision.operation == CLAP_CACHE_OPERATION_BRANCH && resident > 0) {
      req.cache_reuse_kind = "branch";
    }
    for (const auto& victim : plan.evictions()) {
      if (victim.slot == target) req.cache_evicted_slots.push_back(victim.slot);
      if (victim.slot == target) loaded.last_eviction_reason = "hard_ceiling";
    }
    prompt_tokens.erase(prompt_tokens.begin(), prompt_tokens.begin() +
        static_cast<std::ptrdiff_t>(resident));
    return true;
  } catch (...) {
    llama_memory_seq_rm(mem, static_cast<llama_seq_id>(target), -1, -1);
    loaded.slots[target] = {};
    throw;
  }
}

void maybe_create_anchor(LoadedLlama& loaded, ActiveRequest& req) {
  if (req.anchor_at < 0 || !req.coordinator) return;
  const std::size_t count = static_cast<std::size_t>(req.anchor_at);
  if (count < 16 || static_cast<std::size_t>(req.cached_prompt_tokens) + req.ingested != count ||
      req.slot == nullptr) return;
  const auto next = std::upper_bound(
      req.anchor_boundaries.begin(), req.anchor_boundaries.end(), count);
  req.anchor_at = next == req.anchor_boundaries.end()
      ? -1 : static_cast<int32_t>(*next);
  std::vector<llama_token> boundary(
      req.full_prompt_tokens.begin(),
      req.full_prompt_tokens.begin() + static_cast<std::ptrdiff_t>(count));
  std::vector<uint8_t> slot_capabilities;
  slot_capabilities.reserve(loaded.slots.size());
  for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
    const auto& slot = loaded.slots[index];
    uint8_t flags = slot.busy ? 0 : CLAP_CACHE_SLOT_WRITABLE;
    if (!slot.tokens.empty()) flags |= CLAP_CACHE_SLOT_MATERIALIZED | CLAP_CACHE_SLOT_COPY;
    slot_capabilities.push_back(flags);
  }
  try {
    auto anchor_identity = req.cache_identity;
    const bool structural = std::find(req.structural_boundaries.begin(),
        req.structural_boundaries.end(), count) != req.structural_boundaries.end();
    anchor_identity.scope = structural ? CLAP_CACHE_SCOPE_HARNESS : CLAP_CACHE_SCOPE_PROJECT;
    auto plan = req.coordinator->plan(boundary, anchor_identity,
        CLAP_CACHE_CAP_WHOLE_STATE_COPY | CLAP_CACHE_CAP_SAFE_BUSY_DONOR |
            CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT,
        0, CLAP_CACHE_SLOT_ANCHOR, slot_capabilities);
    const auto view = plan.view();
    if (view.operation == CLAP_CACHE_OPERATION_NOOP) {
      plan.commit(count, CLAP_CACHE_SLOT_ANCHOR);
      req.materialized_boundaries.push_back(count);
      return;
    }
    if (view.target.slot >= loaded.slots.size()) {
      plan.abort();
      return;
    }
    llama_memory_t mem = llama_get_memory(loaded.ctx);
    for (const auto& victim : plan.evictions()) {
      if (victim.slot >= loaded.slots.size() || victim.slot == view.target.slot) continue;
      llama_memory_seq_rm(mem, static_cast<llama_seq_id>(victim.slot), -1, -1);
      loaded.slots[victim.slot] = {};
      loaded.last_eviction_reason = "hard_ceiling";
    }
    llama_memory_seq_rm(mem, static_cast<llama_seq_id>(view.target.slot), -1, -1);
    llama_memory_seq_cp(mem, req.seq, static_cast<llama_seq_id>(view.target.slot), -1, -1);
    auto& anchor = loaded.slots[view.target.slot];
    anchor.tokens = boundary;
    anchor.is_anchor = true;
    anchor.last_used = ++loaded.use_counter;
    try {
      plan.commit(count, CLAP_CACHE_SLOT_ANCHOR);
      anchor.coordinator_generation = req.coordinator->slot(view.target.slot).generation;
      if (structural) {
        req.coordinator->set_anchor_protected(
            {view.target.slot, 0, anchor.coordinator_generation}, true);
      }
      req.materialized_boundaries.push_back(count);
    } catch (...) {
      llama_memory_seq_rm(mem, static_cast<llama_seq_id>(view.target.slot), -1, -1);
      anchor = {};
      throw;
    }
  } catch (const std::exception& error) {
    fprintf(stderr, "clap-llama: coordinated anchor skipped: %s\n", error.what());
  }
}

void finalize(ActiveRequest& req) {
  if (req.sampler) {
    llama_sampler_free(req.sampler);
    req.sampler = nullptr;
  }
  if (req.slot) {
    req.slot->busy = false;
    if (req.coordinator && req.slot->coordinator_generation != 0) {
      try {
        req.coordinator->set_busy(
            {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation}, false);
      } catch (const std::exception& error) {
        fprintf(stderr, "clap-llama: cache finalize metadata failed: %s\n", error.what());
      }
    }
  }
  req.done = true;
  json cache{
    {"hit", req.cached_prompt_tokens > 0},
    {"reused_tokens", req.cached_prompt_tokens},
    {"reuse_kind", req.cache_reuse_kind.empty() ? json(nullptr) : json(req.cache_reuse_kind)},
    {"reuse_scope", req.cache_reuse_scope.empty() ? json(nullptr) : json(req.cache_reuse_scope)},
    {"namespace", req.cache_namespace.empty() ? json(nullptr) : json(req.cache_namespace)},
    {"donor_slot", req.cache_donor_slot < 0 ? json(nullptr) : json(req.cache_donor_slot)},
    {"donor_generation", req.cache_donor_slot < 0 ? json(nullptr) : json(req.cache_donor_generation)},
    {"target_slot", static_cast<int>(req.seq)},
    {"target_generation", req.cache_target_generation},
    {"miss_reason", req.cached_prompt_tokens > 0 ? json(nullptr) : json("no_shared_prefix")},
    {"candidates", req.cache_candidates},
    {"prompt_token_hash", req.prompt_token_hash},
    {"prompt_token_count", req.prompt_token_count},
    {"evicted_slots", req.cache_evicted_slots},
    {"decision_us", req.cache_decision_us},
    {"planned_reuse_tokens", req.cache_planned_reuse_tokens},
    {"realized_reuse_tokens", req.cache_realized_reuse_tokens},
    {"side_request", req.cache_side_request},
    {"fallback", req.cache_fallback.empty() ? json(nullptr) : json(req.cache_fallback)},
    {"slot", static_cast<int>(req.seq)},
  };
  if (req.stable_boundary.available()) {
    cache["stable_boundary_token_hash"] = req.stable_boundary.token_hash;
    cache["stable_boundary_token_count"] = req.stable_boundary.token_count;
    cache["stable_boundary_kind"] = req.stable_boundary.kind;
  }
  cache["stable_boundaries"] = json::array();
  for (const auto& resolved : req.resolved_boundaries) {
    const auto boundary = resolved.token_count;
    const bool available = (resolved.status == "resolved" || resolved.status == "authorized") &&
        boundary > 0;
    cache["stable_boundaries"].push_back(json{
      {"token_hash", available ? json(token_fingerprint(req.full_prompt_tokens, boundary)) : json(nullptr)},
      {"token_count", available ? json(boundary) : json(nullptr)},
      {"kind", resolved.kind},
      {"label", !resolved.label.empty() ? json(resolved.label) : json(nullptr)},
      {"requested", resolved.requested},
      {"status", resolved.status},
      {"skip_reason", !resolved.skip_reason.empty() ? json(resolved.skip_reason) : json(nullptr)},
      {"materialized", available ? json(std::find(req.materialized_boundaries.begin(),
          req.materialized_boundaries.end(), boundary) != req.materialized_boundaries.end()) : json(nullptr)},
    });
  }
  emit(req.id, json{
    {"done", true},
    {"finish_reason", req.finish_reason},
    {"cancelled", req.cancelled},
    {"usage", json{
      {"prompt_tokens", req.prompt_token_count},
      {"completion_tokens", req.completion_tokens},
    }},
    {"cache", std::move(cache)},
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
    if (req.coordinator && req.slot->coordinator_generation != 0) {
      try {
        req.slot->coordinator_generation = req.coordinator->invalidate(
            {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation});
      } catch (const std::exception& error) {
        fprintf(stderr, "clap-llama: cache failure invalidation failed: %s\n", error.what());
      }
    }
  }
  llama_memory_seq_rm(llama_get_memory(loaded.ctx), req.seq, -1, -1);
  req.done = true;
  emit_error(req.id, message);
}

void flush_held(ActiveRequest& req) {
  req.held.resize(req.held.size() - clap::llama_native::utf8_incomplete_suffix(req.held));
  if (!req.held.empty()) emit(req.id, json{{"token", req.held}});
  req.held.clear();
}

// Emits the visible portion of req.held, holding back partial stop sequences
// and incomplete UTF-8 tails. Returns true when a stop sequence completed.
bool emit_visible(ActiveRequest& req) {
  if (!req.params.stops.empty()) {
    const std::size_t stop_index = clap::llama_native::find_stop(req.held, req.params.stops);
    if (stop_index != std::string::npos) {
      std::string visible = req.held.substr(0, stop_index);
      visible.resize(visible.size() - clap::llama_native::utf8_incomplete_suffix(visible));
      if (!visible.empty()) emit(req.id, json{{"token", visible}});
      req.held.clear();
      return true;
    }
    const std::size_t stop_hold = clap::llama_native::partial_stop_suffix(req.held, req.params.stops);
    const std::size_t utf8_hold = clap::llama_native::utf8_incomplete_suffix(req.held);
    const std::size_t hold = std::max(stop_hold, utf8_hold);
    const std::string visible = req.held.substr(0, req.held.size() - hold);
    if (!visible.empty()) {
      emit(req.id, json{{"token", visible}});
      req.held = req.held.substr(visible.size());
    }
  } else {
    const std::size_t hold = clap::llama_native::utf8_incomplete_suffix(req.held);
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
    const std::size_t chunk_start = req.ingested;
    if (req.slot) {
      req.slot->tokens.insert(
        req.slot->tokens.end(),
        req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(req.ingested),
        req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(req.ingested + chunk));
      if (req.coordinator && req.slot->coordinator_generation != 0) {
        try {
          req.slot->coordinator_generation = req.coordinator->advance(
              {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation},
              req.prompt_tokens.data() + chunk_start, chunk, CLAP_CACHE_SLOT_SESSION, true);
        } catch (const std::exception& error) {
          req.cache_fallback = "coordinator_advance_failed";
          try {
            req.slot->coordinator_generation = req.coordinator->invalidate(
                {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation});
          } catch (...) {
            req.slot->coordinator_generation = 0;
          }
          emit(req.id, json{{"error", "cache coordinator advance failed closed"},
                            {"code", "cache_coordinator_error"}});
          req.done = true;
          fprintf(stderr, "clap-llama: cache metadata advance failed closed: %s\n", error.what());
          return;
        }
      }
    }
    req.n_pos += req.step_tokens;
    req.ingested += chunk;
    if (req.anchor_at >= 0 &&
        static_cast<std::size_t>(req.cached_prompt_tokens) + req.ingested ==
            static_cast<std::size_t>(req.anchor_at)) {
      maybe_create_anchor(loaded, req);
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
  if (req.slot) {
    req.slot->tokens.push_back(req.pending_token);
    if (req.coordinator && req.slot->coordinator_generation != 0) {
      try {
        req.slot->coordinator_generation = req.coordinator->advance(
            {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation},
            &req.pending_token, 1, CLAP_CACHE_SLOT_SESSION, true);
      } catch (const std::exception& error) {
        req.cache_fallback = "coordinator_advance_failed";
        try {
          req.slot->coordinator_generation = req.coordinator->invalidate(
              {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation});
        } catch (...) {
          req.slot->coordinator_generation = 0;
        }
        emit(req.id, json{{"error", "cache coordinator advance failed closed"},
                          {"code", "cache_coordinator_error"}});
        req.done = true;
        fprintf(stderr, "clap-llama: cache metadata advance failed closed: %s\n", error.what());
        return;
      }
    }
  }
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
      if (req.coordinator) {
        req.coordinator->reset();
        req.slot->coordinator_generation = req.coordinator->slot(
            static_cast<uint32_t>(req.seq)).generation;
        req.coordinator->set_busy(
            {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation}, true);
      }
    } else {
      llama_memory_seq_rm(llama_get_memory(loaded.ctx), req.seq, -1, -1);
      if (req.slot) {
        req.slot->tokens.clear();
        if (req.coordinator && req.slot->coordinator_generation != 0) {
          req.slot->coordinator_generation = req.coordinator->invalidate(
              {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation});
          req.coordinator->set_busy(
              {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation}, true);
        }
      }
    }
    req.prompt_tokens = req.full_prompt_tokens;
    req.ingested = 0;
    req.n_pos = 0;
    req.cached_prompt_tokens = 0;
    req.materialized_boundaries.clear();
    req.anchor_at = req.anchor_boundaries.empty()
        ? -1 : static_cast<int32_t>(req.anchor_boundaries.front());
    req.cache_realized_reuse_tokens = 0;
    req.cache_reuse_kind.clear();
    req.cache_fallback = "decode_retry_full_prefill";
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
  const std::size_t absolute_ingested =
      static_cast<std::size_t>(req.cached_prompt_tokens) + req.ingested;
  if (req.anchor_at >= 0 && static_cast<std::size_t>(req.anchor_at) > absolute_ingested) {
    remaining = std::min(
        remaining, static_cast<std::size_t>(req.anchor_at) - absolute_ingested);
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

  std::vector<clap::llama_native::ScheduleRequest> schedule_requests;
  schedule_requests.reserve(active.size());
  for (const auto& req : active) {
    const bool has_work = req->phase == ActiveRequest::Phase::Decode ||
        req->prompt_tokens.size() != req->ingested;
    schedule_requests.push_back({
        req->phase == ActiveRequest::Phase::Decode
            ? clap::llama_native::SchedulePhase::Decode
            : clap::llama_native::SchedulePhase::Prefill,
        !req->done && has_work,
    });
  }
  // Decode streams first, then prefill in admission order.
  for (const std::size_t index : clap::llama_native::decode_first_order(schedule_requests)) {
    if (budget <= 0) break;
    budget -= add_contribution(batch, *active[index], budget);
    contributors.push_back(active[index].get());
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
  req->cache_identity = cache_identity(loaded, id, request);
  req->cache_side_request = req->cache_identity.side_request;
  if (!loaded.coordinator) req->cache_fallback = "coordinator_unavailable";
  if (request.contains("cache") && request["cache"].is_object()) {
    req->cache_namespace = cache_string(request["cache"], "namespace");
  }

  const bool add_special = prompt.find("<bos>") == std::string::npos && prompt.find("<|turn>") == std::string::npos;
  const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, add_special, true);
  if (n_prompt <= 0) throw std::runtime_error("failed to tokenize prompt");
  std::vector<llama_token> prompt_tokens(n_prompt);
  if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), add_special, true) < 0) {
    throw std::runtime_error("failed to tokenize prompt");
  }
  std::vector<uint64_t> stable_boundaries;
  const auto resolve_boundary = [&](std::size_t message_count, const std::string& kind,
                                    const std::string& label, bool requested) {
    if (message_count == 0 || message_count > entries.size()) {
      if (requested) req->resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    const std::vector<ChatEntry> prefix_entries(entries.begin(), entries.begin() + message_count);
    const std::string prefix_prompt = templated_prompt(loaded.model, prefix_entries, false);
    if (prefix_prompt.empty()) {
      if (requested) req->resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    const bool prefix_add_special = prefix_prompt.find("<bos>") == std::string::npos &&
        prefix_prompt.find("<|turn>") == std::string::npos;
    const int prefix_count = -llama_tokenize(vocab, prefix_prompt.c_str(),
        prefix_prompt.size(), nullptr, 0, prefix_add_special, true);
    if (prefix_count <= 0) {
      if (requested) req->resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    std::vector<llama_token> prefix_tokens(static_cast<std::size_t>(prefix_count));
    if (llama_tokenize(vocab, prefix_prompt.c_str(), prefix_prompt.size(),
        prefix_tokens.data(), prefix_tokens.size(), prefix_add_special, true) < 0) {
      if (requested) req->resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      return;
    }
    const auto exact = clap::llama_boundary::exact_template_boundary(
        prefix_tokens, prompt_tokens,
        [vocab](llama_token token) { return llama_vocab_is_eog(vocab, token); });
    if (!exact) {
      if (requested) req->resolved_boundaries.push_back(
          {0, kind, label, true, "skipped", "non_prefix_template_boundary"});
      return;
    }
    const std::size_t boundary = *exact;
    stable_boundaries.push_back(boundary);
    if (kind != "prompt") req->structural_boundaries.push_back(boundary);
    const auto existing = std::find_if(req->resolved_boundaries.begin(),
        req->resolved_boundaries.end(), [boundary](const auto& value) {
          return value.token_count == boundary;
        });
    if (existing == req->resolved_boundaries.end()) {
      req->resolved_boundaries.push_back(
          {boundary, kind, label, requested, "resolved", ""});
    } else if (requested) {
      existing->kind = kind;
      existing->label = label;
      existing->requested = true;
      existing->status = "resolved";
      existing->skip_reason.clear();
    }
  };
  std::size_t leading_systems = 0;
  while (leading_systems < entries.size() && entries[leading_systems].role == "system") {
    ++leading_systems;
  }
  for (std::size_t count = 1; count <= leading_systems; ++count) {
    resolve_boundary(count, "messages", "", false);
  }
  if (request.contains("cache") && request["cache"].is_object() &&
      request["cache"].contains("boundaries") && request["cache"]["boundaries"].is_array()) {
    for (const auto& descriptor : request["cache"]["boundaries"]) {
      const std::string kind = descriptor.value("kind", "");
      const std::string label = descriptor.value("label", "");
      if (kind == "messages") {
        resolve_boundary(descriptor.value("through_message", entries.size()) + 1,
            kind, label, true);
      } else if (kind == "tools") {
        // The llama worker receives no native tool-template structure. Any
        // compatibility instructions are ordinary messages, so there is no
        // independently provable tools-only prefix.
        req->resolved_boundaries.push_back(
            {0, kind, label, true, "skipped", "unsupported_template_boundary"});
      }
    }
  }
  if (prompt_tokens.size() > 16) {
    stable_boundaries.push_back(prompt_tokens.size() - 1);
    req->resolved_boundaries.push_back(
        {prompt_tokens.size() - 1, "prompt", "", false, "resolved", ""});
  }
  std::sort(stable_boundaries.begin(), stable_boundaries.end());
  stable_boundaries.erase(
      std::unique(stable_boundaries.begin(), stable_boundaries.end()), stable_boundaries.end());

  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.ctx));
  const int32_t prompt_count = static_cast<int32_t>(prompt_tokens.size());
  if (prompt_count >= n_ctx) {
    throw RequestError("context_length_exceeded",
      "prompt is too long for the loaded model; prompt_tokens=" + std::to_string(prompt_count) +
      ", max_input_tokens=" + std::to_string(n_ctx - 1) +
      ", effective_context_window=" + std::to_string(n_ctx) + "."
    );
  }
  if (req->params.max_tokens > 0 && loaded.max_output_tokens > 0 &&
      req->params.max_tokens > loaded.max_output_tokens) {
    throw RequestError("max_output_tokens_exceeded",
      "requested max_tokens=" + std::to_string(req->params.max_tokens) +
      " exceeds the loaded model maximum output tokens=" +
      std::to_string(loaded.max_output_tokens) + ".");
  }
  const int32_t available_output = n_ctx - prompt_count;
  if (req->params.max_tokens == 0) {
    req->params.max_tokens = loaded.max_output_tokens > 0
        ? std::min(loaded.max_output_tokens, available_output) : available_output;
  }
  const int32_t output_reserve = req->params.max_tokens;
  if (prompt_count + output_reserve > n_ctx) {
    throw RequestError("context_length_exceeded",
      "prompt plus requested output exceeds the loaded model context; prompt_tokens=" +
      std::to_string(prompt_count) + ", requested_output_tokens=" +
      std::to_string(output_reserve) + ", effective_context_window=" +
      std::to_string(n_ctx) + ".");
  }
  // Per-session context cap: bounds one session's share of the unified KV
  // pool so a single conversation cannot promise itself the full window on a
  // box shared by many sessions. Admin policy, not a physical limit.
  const int32_t session_cap = env_int("CLAP_LLAMA_MAX_SESSION_CTX", 0);
  if (session_cap > 0 && prompt_count + output_reserve > session_cap) {
    throw RequestError("context_length_exceeded",
      "prompt exceeds the per-session context cap; prompt tokens=" + std::to_string(prompt_tokens.size()) +
      ", max_session_ctx=" + std::to_string(session_cap) +
      ", reserved output tokens=" + std::to_string(output_reserve) +
      ". Reduce the prompt/tool history or raise max_session_ctx / CLAP_LLAMA_MAX_SESSION_CTX."
    );
  }

  req->prompt_token_count = static_cast<int>(prompt_tokens.size());
  req->full_prompt_tokens = prompt_tokens;
  req->prompt_token_hash = token_fingerprint(prompt_tokens, prompt_tokens.size());

  if (llama_model_has_encoder(loaded.model)) {
    // Encoder-decoder models run alone (admission guarantees no other active
    // request) and reset all cache state.
    for (auto& s : loaded.slots) {
      s.tokens.clear();
      s.is_anchor = false;
    }
    llama_memory_clear(llama_get_memory(loaded.ctx), true);
    if (loaded.coordinator) loaded.coordinator->reset();
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

  if (prepare_with_coordinator(
          loaded, *req, prompt_tokens, output_reserve, stable_boundaries)) {
    req->prompt_tokens = std::move(prompt_tokens);
    req->sampler = make_sampler(req->params);
    return req;
  }

  // The only policy fallback is coordinator-unavailable no-cache mode. It may
  // choose an idle execution sequence, but it never inspects tokens, reuses a
  // donor, creates an anchor, or performs policy eviction.
  llama_memory_t mem = llama_get_memory(loaded.ctx);
  std::size_t target = SIZE_MAX;
  for (std::size_t index = 0; index < loaded.slots.size(); ++index) {
    if (!loaded.slots[index].busy) {
      target = index;
      if (loaded.slots[index].tokens.empty()) break;
    }
  }
  if (target == SIZE_MAX) throw std::runtime_error("no idle KV slot available");
  llama_memory_seq_rm(mem, static_cast<llama_seq_id>(target), -1, -1);
  loaded.slots[target] = {};
  auto* slot = &loaded.slots[target];
  slot->last_used = ++loaded.use_counter;
  slot->busy = true;
  req->slot = slot;
  req->seq = static_cast<llama_seq_id>(target);
  req->n_pos = 0;
  req->cache_fallback = "coordinator_unavailable_no_cache";
  req->prompt_tokens = std::move(prompt_tokens);
  req->sampler = make_sampler(req->params);
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
            if (!req->done && clap::llama_native::active_cancel_matches(target, req->id)) {
              req->cancelled = true;
            }
          }
          for (auto it = waiting.begin(); it != waiting.end(); ++it) {
            if (clap::llama_native::queued_cancel_matches(target, it->first)) {
              emit(target, json{{"done", true}, {"finish_reason", "cancel"}, {"cancelled", true}});
              waiting.erase(it);
              break;
            }
          }
          return true;
        }
        if (type == "set_max_active") {
          const int requested = message.value("max_active", 0);
          if (requested <= 0) throw std::runtime_error("set_max_active.max_active must be positive");
          const int previous = loaded.max_active;
          loaded.max_active = std::max(1, std::min({requested,
              loaded.active_policy.backend_ceiling, loaded.active_policy.hardware_ceiling,
              loaded.active_policy.model_ceiling, loaded.active_policy.context_ceiling}));
          loaded.active_policy.selected_max = loaded.max_active;
          loaded.previous_max_active = message.value("previous_max_active", previous);
          loaded.active_policy.reason = message.value("limiting_reason", loaded.active_policy.reason);
          loaded.last_adjustment_reason = message.value("last_adjustment_reason", "");
          loaded.last_adjustment_at = message.value("last_adjustment_at", "");
          loaded.retained_growth_reserve_bytes = message.value("retained_growth_reserve_bytes", UINT64_C(0));
          loaded.global_resident_memory_bytes = message.value("global_resident_memory_bytes", UINT64_C(0));
          loaded.pressure_state = message.value("pressure_state", "");
          emit(id, json{{"done", true}, {"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
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
          const int32_t effective = loaded.backend_allocation_cap;
          emit(id, json{{"loaded", true}, {"done", true}, {"token_capabilities", {
            {"model_context_window", loaded.model_context_window > 0 ? json(loaded.model_context_window) : json(nullptr)},
            {"effective_context_window", effective},
            {"max_input_tokens", std::max(0, effective - 1)},
            {"max_output_tokens", loaded.max_output_tokens > 0 ? json(loaded.max_output_tokens) : json(nullptr)},
            {"backend_allocation_cap", loaded.backend_allocation_cap},
            {"user_configured_override", loaded.context_override > 0 ? json(loaded.context_override) : json(nullptr)},
          }}, {"retention", retention_telemetry(loaded, active.size())}});
          return true;
        }
        // Anything else is a chat request; it queues until a slot frees up.
        waiting.emplace_back(id, message);
        emit("", json{{"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
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
        if (!clap::llama_cache::can_admit(active.size(),
                                          static_cast<uint32_t>(loaded.max_active))) break;
        auto [wid, wreq] = std::move(waiting.front());
        waiting.pop_front();
        try {
          auto prepared = prepare_request(loaded, wid, wreq);
          active.push_back(std::move(prepared));
          emit(wid, json{{"started", true},
                         {"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
        } catch (const CacheBackpressure&) {
          waiting.emplace_front(std::move(wid), std::move(wreq));
          break;
        } catch (const RequestError& error) {
          emit_error(wid, error.what(), error.code);
        } catch (const std::exception& error) {
          emit_error(wid, error.what());
        }
      }

      if (!active.empty()) {
        step(loaded, active);
        const std::size_t before_cleanup = active.size();
        active.erase(
          std::remove_if(active.begin(), active.end(), [](const std::unique_ptr<ActiveRequest>& r) { return r->done; }),
          active.end());
        if (active.size() != before_cleanup) {
          emit("", json{{"retention", retention_telemetry(loaded, active.size(), waiting.size())}});
        }
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
