#include "llama.h"
#include "active-concurrency.h"
#include "cache-adapter.h"
#include "clap/llama/cache-executor.h"
#include "clap/llama/environment.h"
#include "clap/llama/model-runtime.h"
#include "clap/llama/prompt.h"
#include "clap/llama/protocol.h"
#include "clap/llama/request-preparer.h"
#include "clap/llama/request-state.h"
#include "clap/llama/sampling.h"
#include "clap/llama/stop-buffer.h"
#include "clap/llama/telemetry.h"
#include "native-characterization.h"
#include "stable-boundary.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <deque>
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

using clap::llama::env_int;
using clap::llama::env_u64;
using clap::llama::emit;
using clap::llama::emit_error;
using clap::llama::make_sampler;
using clap::llama::RequestError;
using clap::llama::StdinReader;
using clap::llama::ActiveRequest;
using clap::llama::CacheBackpressure;

struct LoadedLlama {
  clap::llama::ModelRuntime runtime;
  int32_t max_active = 0;
  clap::llama_active::Decision active_policy{};
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
  std::unique_ptr<clap::llama::CacheExecutor> cache_executor;
  clap::llama_cache::Coordinator* coordinator = nullptr;
};

void unload(LoadedLlama& loaded) {
  if (loaded.cache_executor) loaded.cache_executor->reset();
  loaded.coordinator = nullptr;
  loaded.cache_executor.reset();
  loaded.slots.clear();
  loaded.runtime.reset();
  loaded.max_active = 0;
  loaded.active_policy = {};
  loaded.last_eviction_reason.clear();
}

void load_model(LoadedLlama& loaded, const std::string& model_path) {
  if (loaded.runtime.same_path(model_path)) return;
  unload(loaded);
  loaded.runtime.load(model_path);
  const int32_t n_ctx = loaded.runtime.backend_allocation_cap();
  const int32_t retained_max = loaded.runtime.retained_max();
  loaded.slots.assign(static_cast<std::size_t>(retained_max), {});
  loaded.use_counter = 0;
  loaded.active_policy = clap::llama_active::select({
      env_int("CLAP_MAX_ACTIVE", 0), loaded.runtime.startup_available_bytes(),
      loaded.runtime.model_file_bytes(),
      static_cast<int>(std::max(1u, std::thread::hardware_concurrency())), n_ctx,
      retained_max, loaded.runtime.hybrid(), loaded.runtime.has_encoder()});
  loaded.max_active = loaded.active_policy.selected_max;
  try {
    clap::llama::CacheExecutorConfig config;
    config.slot_count = static_cast<uint32_t>(retained_max);
    config.logical_token_capacity = static_cast<uint64_t>(n_ctx);
    config.max_anchors = static_cast<uint32_t>(retained_max);
    config.hard_max_retained_entries = static_cast<uint32_t>(retained_max);
    config.automatic_checkpoints = env_int("CLAP_CACHE_CHECKPOINTS_ENABLED", 1) != 0;
    config.checkpoint_minimum_tokens = static_cast<uint64_t>(
        env_int("CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS", 2048));
    config.checkpoint_interval_tokens = static_cast<uint64_t>(
        env_int("CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS", 2048));
    config.checkpoint_max = static_cast<uint32_t>(env_int("CLAP_CACHE_CHECKPOINT_MAX", 8));
    config.checkpoint_budget_basis_points = static_cast<uint32_t>(
        env_int("CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS", 2500));
    config.checkpoint_budget_bytes = env_u64("CLAP_CACHE_CHECKPOINT_BUDGET_BYTES", 0);
    loaded.cache_executor = std::make_unique<clap::llama::CacheExecutor>(
        config, std::make_unique<clap::llama::LlamaPhysicalCacheBackend>(
            loaded.runtime.context()));
    loaded.coordinator = &loaded.cache_executor->coordinator();
  } catch (const std::exception& error) {
    loaded.coordinator = nullptr;
    loaded.cache_executor.reset();
    fprintf(stderr, "clap-llama: cache coordinator unavailable; using no-cache fresh mode: %s\n", error.what());
  }
}

json retention_telemetry(const LoadedLlama& loaded, std::size_t active, std::size_t queued = 0) {
  uint32_t retained_total = 0;
  uint32_t retained_sessions = 0;
  uint32_t retained_anchors = 0;
  uint64_t evictions = 0;
  if (loaded.cache_executor) {
    const auto retention = loaded.cache_executor->retention_telemetry();
    const auto telemetry = loaded.cache_executor->telemetry();
    retained_total = retention.active_slots;
    retained_sessions = retention.session_slots;
    retained_anchors = retention.anchor_slots;
    evictions = telemetry.evictions;
  }
  const clap::llama::RetentionTelemetrySnapshot snapshot{
    loaded.max_active,
    queued,
    loaded.previous_max_active,
    loaded.last_adjustment_reason,
    loaded.last_adjustment_at,
    loaded.retained_growth_reserve_bytes,
    loaded.global_resident_memory_bytes,
    loaded.pressure_state,
    {
      loaded.active_policy.mode,
      loaded.active_policy.selected_max,
      loaded.active_policy.backend_ceiling,
      loaded.active_policy.hardware_ceiling,
      loaded.active_policy.model_ceiling,
      loaded.active_policy.memory_ceiling,
      loaded.active_policy.reason,
      loaded.runtime.startup_available_bytes(),
      loaded.runtime.model_file_bytes(),
      loaded.runtime.backend_allocation_cap(),
      loaded.active_policy.context_ceiling,
      loaded.active_policy.per_active_reserve_cells,
      loaded.active_policy.per_active_reserve_bytes,
      std::max(1u, std::thread::hardware_concurrency()),
      loaded.runtime.hybrid(),
    },
    active,
    retained_total,
    retained_sessions,
    retained_anchors,
    loaded.runtime.retained_max(),
    loaded.last_eviction_reason,
    evictions,
    loaded.runtime.backend_allocation_cap(),
  };
  return clap::llama::serialize_retention_telemetry(snapshot);
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

void maybe_create_anchor(LoadedLlama& loaded, ActiveRequest& req) {
  if (req.anchor_at < 0 || !req.cache_lease) return;
  const std::size_t count = static_cast<std::size_t>(req.anchor_at);
  if (count < 16 || static_cast<std::size_t>(req.cached_prompt_tokens) + req.ingested != count ||
      !req.cache_lease) return;
  const auto next = std::upper_bound(
      req.anchor_boundaries.begin(), req.anchor_boundaries.end(), count);
  req.anchor_at = next == req.anchor_boundaries.end()
      ? -1 : static_cast<int32_t>(*next);
  std::vector<llama_token> boundary(
      req.full_prompt_tokens.begin(),
      req.full_prompt_tokens.begin() + static_cast<std::ptrdiff_t>(count));
  try {
    auto anchor_identity = req.cache_identity;
    const bool structural = std::find(req.structural_boundaries.begin(),
        req.structural_boundaries.end(), count) != req.structural_boundaries.end();
    anchor_identity.scope = structural ? CLAP_CACHE_SCOPE_HARNESS : CLAP_CACHE_SCOPE_PROJECT;
    auto result = loaded.cache_executor->create_anchor(
        boundary, anchor_identity, static_cast<uint32_t>(req.seq), structural);
    if (!result.materialized) return;
    if (!result.no_op && result.target_slot < loaded.slots.size()) {
      auto& anchor = loaded.slots[result.target_slot];
      anchor.tokens = boundary;
      anchor.is_anchor = true;
      anchor.last_used = ++loaded.use_counter;
      anchor.coordinator_generation = result.target_generation;
    }
    for (const uint32_t victim : result.eviction_slots) {
      if (victim != result.target_slot && victim < loaded.slots.size()) loaded.slots[victim] = {};
      loaded.last_eviction_reason = "hard_ceiling";
    }
    req.materialized_boundaries.push_back(count);
  } catch (const std::exception& error) {
    fprintf(stderr, "clap-llama: coordinated anchor skipped: %s\n", error.what());
  }
}

void finalize(LoadedLlama& loaded, ActiveRequest& req) {
  req.sampler.reset();
  if (req.cache_lease) {
    loaded.slots[static_cast<std::size_t>(req.seq)].busy = false;
    req.cache_lease.release();
  }
  if (!req.mark_terminal(ActiveRequest::TerminalState::Completed)) return;
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
  req.sampler.reset();
  if (req.cache_lease) {
    auto& slot = loaded.slots[static_cast<std::size_t>(req.seq)];
    slot.busy = false;
    slot.tokens.clear();
    if (req.cache_lease.generation() != 0) {
      try {
        slot.coordinator_generation = req.cache_lease.invalidate_and_clear();
      } catch (const std::exception& error) {
        fprintf(stderr, "clap-llama: cache failure invalidation failed: %s\n", error.what());
      }
    }
    req.cache_lease.release();
  }
  req.mark_terminal(ActiveRequest::TerminalState::Failed);
  emit_error(req.id, message);
}

// Handles one sampled token: EOS, stop sequences, budget checks, streaming
// emission. Finalizes the request when generation ends.
void process_sampled(LoadedLlama& loaded, ActiveRequest& req, const llama_vocab* vocab, llama_token token) {
  if (llama_vocab_is_eog(vocab, token)) {
    const std::string visible = req.stop_buffer.finish();
    if (!visible.empty()) emit(req.id, json{{"token", visible}});
    finalize(loaded, req);
    return;
  }
  req.completion_tokens += 1;
  const std::string piece = token_to_piece(vocab, token);
  if (!piece.empty()) {
    const auto result = req.stop_buffer.append(piece);
    if (!result.visible.empty()) emit(req.id, json{{"token", result.visible}});
    if (result.stop_complete) {
      req.finish_reason = "stop";
      finalize(loaded, req);
      return;
    }
  }
  if (req.completion_tokens >= req.params.max_tokens) {
    req.finish_reason = "length";
    const std::string visible = req.stop_buffer.finish();
    if (!visible.empty()) emit(req.id, json{{"token", visible}});
    finalize(loaded, req);
    return;
  }
  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.runtime.context()));
  if (req.n_pos + 1 >= n_ctx) {
    req.finish_reason = "length";
    const std::string visible = req.stop_buffer.finish();
    if (!visible.empty()) emit(req.id, json{{"token", visible}});
    finalize(loaded, req);
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
    if (req.cache_lease) {
      auto& slot = loaded.slots[static_cast<std::size_t>(req.seq)];
      slot.tokens.insert(
        slot.tokens.end(),
        req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(req.ingested),
        req.prompt_tokens.begin() + static_cast<std::ptrdiff_t>(req.ingested + chunk));
      if (req.cache_lease.generation() != 0) {
        try {
          slot.coordinator_generation = req.cache_lease.advance(
              req.prompt_tokens.data() + chunk_start, chunk, CLAP_CACHE_SLOT_SESSION, true);
        } catch (const std::exception& error) {
          req.cache_fallback = "coordinator_advance_failed";
          slot.coordinator_generation = 0;
          emit(req.id, json{{"error", "cache coordinator advance failed closed"},
                            {"code", "cache_coordinator_error"}});
          req.mark_terminal(ActiveRequest::TerminalState::Failed);
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
      const llama_token token = llama_sampler_sample(req.sampler.get(), loaded.runtime.context(), req.logits_index);
      process_sampled(loaded, req, vocab, token);
    }
    return;
  }
  req.n_pos += 1;
  if (req.cache_lease) {
    auto& slot = loaded.slots[static_cast<std::size_t>(req.seq)];
    slot.tokens.push_back(req.pending_token);
    if (req.cache_lease.generation() != 0) {
      try {
        slot.coordinator_generation = req.cache_lease.advance(
            &req.pending_token, 1, CLAP_CACHE_SLOT_SESSION, true);
      } catch (const std::exception& error) {
        req.cache_fallback = "coordinator_advance_failed";
        slot.coordinator_generation = 0;
        emit(req.id, json{{"error", "cache coordinator advance failed closed"},
                          {"code", "cache_coordinator_error"}});
        req.mark_terminal(ActiveRequest::TerminalState::Failed);
        fprintf(stderr, "clap-llama: cache metadata advance failed closed: %s\n", error.what());
        return;
      }
    }
  }
  const llama_token token = llama_sampler_sample(req.sampler.get(), loaded.runtime.context(), req.logits_index);
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
      if (req.cache_lease) {
        loaded.slots[static_cast<std::size_t>(req.seq)].coordinator_generation =
            req.cache_lease.reset_for_retry();
      }
    } else {
      if (req.cache_lease) {
        auto& slot = loaded.slots[static_cast<std::size_t>(req.seq)];
        slot.tokens.clear();
        if (req.cache_lease.generation() != 0) {
          slot.coordinator_generation = req.cache_lease.invalidate_and_clear(true);
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
  const int result = llama_decode(loaded.runtime.context(), batch);
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
  const llama_vocab* vocab = loaded.runtime.vocab();
  const int32_t n_batch = static_cast<int32_t>(llama_n_batch(loaded.runtime.context()));

  for (auto& req : active) {
    if (!req->done && req->cancelled) {
      req->finish_reason = "cancel";
      finalize(loaded, *req);
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
  if (llama_decode(loaded.runtime.context(), batch) == 0) {
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
  std::vector<clap::llama::RequestSlotState> slots;
  slots.reserve(loaded.slots.size());
  for (const auto& slot : loaded.slots) {
    slots.push_back({slot.tokens, slot.coordinator_generation, slot.busy, slot.is_anchor});
  }
  clap::llama::RequestPreparer preparer(
      loaded.runtime, loaded.cache_executor.get(), std::move(slots), telemetry_key(),
      [](const auto& tokens, std::size_t count) { return token_fingerprint(tokens, count); });
  auto prepared = preparer.prepare(id, request);
  const std::size_t target = static_cast<std::size_t>(prepared.sequence);
  if (loaded.runtime.has_encoder()) {
    for (auto& slot : loaded.slots) slot = {};
  }
  for (const uint32_t victim : prepared.cache_evicted_slots) {
    if (victim != target && victim < loaded.slots.size()) loaded.slots[victim] = {};
    loaded.last_eviction_reason = "hard_ceiling";
  }
  if (target < loaded.slots.size()) {
    auto& slot = loaded.slots[target];
    slot.tokens.assign(prepared.full_prompt_tokens.begin(),
        prepared.full_prompt_tokens.begin() + prepared.initial_position);
    slot.coordinator_generation = prepared.cache_target_generation;
    slot.last_used = ++loaded.use_counter;
    slot.busy = true;
    slot.is_anchor = false;
  }
  auto active = std::make_unique<ActiveRequest>(std::move(prepared));
  active->sampler.reset(make_sampler(active->params));
  return active;
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
          if (!active.empty() && loaded.runtime.loaded() && loaded.runtime.model_path() != model) {
            throw std::runtime_error("cannot switch models while requests are active");
          }
          load_model(loaded, model);
          const int32_t effective = loaded.runtime.backend_allocation_cap();
          emit(id, json{{"loaded", true}, {"done", true}, {"token_capabilities", {
            {"model_context_window", loaded.runtime.model_context_window() > 0 ? json(loaded.runtime.model_context_window()) : json(nullptr)},
            {"effective_context_window", effective},
            {"max_input_tokens", std::max(0, effective - 1)},
            {"max_output_tokens", loaded.runtime.max_output_tokens() > 0 ? json(loaded.runtime.max_output_tokens()) : json(nullptr)},
            {"backend_allocation_cap", loaded.runtime.backend_allocation_cap()},
            {"user_configured_override", loaded.runtime.context_override() > 0 ? json(loaded.runtime.context_override()) : json(nullptr)},
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
        if (!active.empty() && loaded.runtime.loaded() && loaded.runtime.model_path() != req_model) break;
        try {
          load_model(loaded, req_model);
        } catch (const std::exception& error) {
          emit_error(waiting.front().first, error.what());
          waiting.pop_front();
          continue;
        }
        if (loaded.runtime.has_encoder() && !active.empty()) break;
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
        finalize(loaded, *req);
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
