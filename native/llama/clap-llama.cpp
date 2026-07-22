#include "llama.h"
#include "active-concurrency.h"
#include "cache-adapter.h"
#include "clap/llama/environment.h"
#include "clap/llama/model-runtime.h"
#include "clap/llama/prompt.h"
#include "clap/llama/protocol.h"
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
using clap::llama::sampling_from_request;
using clap::llama::SamplingParams;
using clap::llama::StdinReader;

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
  std::unique_ptr<clap::llama_cache::Coordinator> coordinator;
};

struct CacheBackpressure : std::runtime_error {
  using std::runtime_error::runtime_error;
};

void unload(LoadedLlama& loaded) {
  if (loaded.coordinator) loaded.coordinator->reset();
  loaded.coordinator.reset();
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
  std::vector<std::size_t> anchor_boundaries;
  std::vector<std::size_t> structural_boundaries;
  std::vector<std::size_t> materialized_boundaries;
  std::vector<clap::llama::ResolvedPromptBoundary> resolved_boundaries;
  json cache_candidates = json::array();
  bool cache_side_request = false;

  enum class Phase { Prefill, Decode };
  Phase phase = Phase::Prefill;
  llama_token pending_token = 0;  // sampled but not yet decoded
  clap::llama::StopBuffer stop_buffer;
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
      keyed + loaded.runtime.cache_domain() + "|tenant=" + (tenant.empty() ? "local" : tenant));
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
  if (loaded.runtime.hybrid()) {
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
      if (!loaded.runtime.hybrid()) flags |= CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM;
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

  llama_memory_t mem = llama_get_memory(loaded.runtime.context());
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
        if (loaded.runtime.hybrid() && resident == loaded.slots[donor].tokens.size()) {
          llama_memory_seq_cp(mem, static_cast<llama_seq_id>(donor),
                              static_cast<llama_seq_id>(target), -1, -1);
        } else if (!loaded.runtime.hybrid()) {
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
    llama_memory_t mem = llama_get_memory(loaded.runtime.context());
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
  llama_memory_seq_rm(llama_get_memory(loaded.runtime.context()), req.seq, -1, -1);
  req.done = true;
  emit_error(req.id, message);
}

// Handles one sampled token: EOS, stop sequences, budget checks, streaming
// emission. Finalizes the request when generation ends.
void process_sampled(LoadedLlama& loaded, ActiveRequest& req, const llama_vocab* vocab, llama_token token) {
  if (llama_vocab_is_eog(vocab, token)) {
    const std::string visible = req.stop_buffer.finish();
    if (!visible.empty()) emit(req.id, json{{"token", visible}});
    finalize(req);
    return;
  }
  req.completion_tokens += 1;
  const std::string piece = token_to_piece(vocab, token);
  if (!piece.empty()) {
    const auto result = req.stop_buffer.append(piece);
    if (!result.visible.empty()) emit(req.id, json{{"token", result.visible}});
    if (result.stop_complete) {
      req.finish_reason = "stop";
      finalize(req);
      return;
    }
  }
  if (req.completion_tokens >= req.params.max_tokens) {
    req.finish_reason = "length";
    const std::string visible = req.stop_buffer.finish();
    if (!visible.empty()) emit(req.id, json{{"token", visible}});
    finalize(req);
    return;
  }
  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.runtime.context()));
  if (req.n_pos + 1 >= n_ctx) {
    req.finish_reason = "length";
    const std::string visible = req.stop_buffer.finish();
    if (!visible.empty()) emit(req.id, json{{"token", visible}});
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
      const llama_token token = llama_sampler_sample(req.sampler, loaded.runtime.context(), req.logits_index);
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
  const llama_token token = llama_sampler_sample(req.sampler, loaded.runtime.context(), req.logits_index);
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
      llama_memory_clear(llama_get_memory(loaded.runtime.context()), true);
      if (req.coordinator) {
        req.coordinator->reset();
        req.slot->coordinator_generation = req.coordinator->slot(
            static_cast<uint32_t>(req.seq)).generation;
        req.coordinator->set_busy(
            {static_cast<uint32_t>(req.seq), 0, req.slot->coordinator_generation}, true);
      }
    } else {
      llama_memory_seq_rm(llama_get_memory(loaded.runtime.context()), req.seq, -1, -1);
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
  const llama_vocab* vocab = loaded.runtime.vocab();
  clap::llama::PromptRenderer renderer(loaded.runtime);
  auto prepared_prompt = renderer.prepare(clap::llama::prompt_input_from_request(request));
  std::vector<llama_token> prompt_tokens = std::move(prepared_prompt.tokens);
  const std::vector<uint64_t> stable_boundaries = std::move(prepared_prompt.stable_boundaries);

  auto req = std::make_unique<ActiveRequest>();
  req->id = id;
  req->params = sampling_from_request(request);
  req->stop_buffer.reset(req->params.stops);
  req->cache_identity = cache_identity(loaded, id, request);
  req->cache_side_request = req->cache_identity.side_request;
  if (!loaded.coordinator) req->cache_fallback = "coordinator_unavailable";
  if (request.contains("cache") && request["cache"].is_object()) {
    req->cache_namespace = cache_string(request["cache"], "namespace");
  }
  req->structural_boundaries = std::move(prepared_prompt.structural_boundaries);
  req->resolved_boundaries = std::move(prepared_prompt.resolved_boundaries);

  const int32_t n_ctx = static_cast<int32_t>(llama_n_ctx(loaded.runtime.context()));
  const int32_t prompt_count = static_cast<int32_t>(prompt_tokens.size());
  if (prompt_count >= n_ctx) {
    throw RequestError("context_length_exceeded",
      "prompt is too long for the loaded model; prompt_tokens=" + std::to_string(prompt_count) +
      ", max_input_tokens=" + std::to_string(n_ctx - 1) +
      ", effective_context_window=" + std::to_string(n_ctx) + "."
    );
  }
  if (req->params.max_tokens > 0 && loaded.runtime.max_output_tokens() > 0 &&
      req->params.max_tokens > loaded.runtime.max_output_tokens()) {
    throw RequestError("max_output_tokens_exceeded",
      "requested max_tokens=" + std::to_string(req->params.max_tokens) +
      " exceeds the loaded model maximum output tokens=" +
      std::to_string(loaded.runtime.max_output_tokens()) + ".");
  }
  const int32_t available_output = n_ctx - prompt_count;
  if (req->params.max_tokens == 0) {
    req->params.max_tokens = loaded.runtime.max_output_tokens() > 0
        ? std::min(loaded.runtime.max_output_tokens(), available_output) : available_output;
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

  if (loaded.runtime.has_encoder()) {
    // Encoder-decoder models run alone (admission guarantees no other active
    // request) and reset all cache state.
    for (auto& s : loaded.slots) {
      s.tokens.clear();
      s.is_anchor = false;
    }
    llama_memory_clear(llama_get_memory(loaded.runtime.context()), true);
    if (loaded.coordinator) loaded.coordinator->reset();
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_encode(loaded.runtime.context(), batch)) throw std::runtime_error("llama_encode failed");
    llama_token decoder_start = llama_model_decoder_start_token(loaded.runtime.model());
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
  llama_memory_t mem = llama_get_memory(loaded.runtime.context());
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
