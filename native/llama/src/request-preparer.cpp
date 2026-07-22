#include "clap/llama/request-preparer.h"

#include "clap/llama/environment.h"
#include "clap/llama/prompt.h"
#include "clap/llama/protocol.h"

#include <algorithm>
#include <cstdio>

namespace clap::llama {
namespace {

std::string cache_string(const nlohmann::json& cache, const char* key) {
  if (!cache.is_object() || !cache.contains(key) || !cache[key].is_string()) return "";
  return cache[key].get<std::string>();
}

const char* scope_name(uint32_t scope) {
  switch (scope) {
    case CLAP_CACHE_SCOPE_SESSION: return "session";
    case CLAP_CACHE_SCOPE_AGENT: return "agent";
    case CLAP_CACHE_SCOPE_HARNESS: return "harness";
    case CLAP_CACHE_SCOPE_PROJECT: return "project";
    case CLAP_CACHE_SCOPE_TENANT: return "tenant";
    default: return "none";
  }
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

}  // namespace

RequestBudget validate_request_budget(int prompt_count, int context_size,
                                      int requested_max_tokens,
                                      int configured_max_output,
                                      int session_cap) {
  if (prompt_count >= context_size) {
    throw RequestError("context_length_exceeded",
      "prompt is too long for the loaded model; prompt_tokens=" + std::to_string(prompt_count) +
      ", max_input_tokens=" + std::to_string(context_size - 1) +
      ", effective_context_window=" + std::to_string(context_size) + ".");
  }
  if (requested_max_tokens > 0 && configured_max_output > 0 &&
      requested_max_tokens > configured_max_output) {
    throw RequestError("max_output_tokens_exceeded",
      "requested max_tokens=" + std::to_string(requested_max_tokens) +
      " exceeds the loaded model maximum output tokens=" +
      std::to_string(configured_max_output) + ".");
  }
  const int available_output = context_size - prompt_count;
  const int max_tokens = requested_max_tokens == 0
      ? (configured_max_output > 0 ? std::min(configured_max_output, available_output)
                                   : available_output)
      : requested_max_tokens;
  if (prompt_count + max_tokens > context_size) {
    throw RequestError("context_length_exceeded",
      "prompt plus requested output exceeds the loaded model context; prompt_tokens=" +
      std::to_string(prompt_count) + ", requested_output_tokens=" +
      std::to_string(max_tokens) + ", effective_context_window=" +
      std::to_string(context_size) + ".");
  }
  if (session_cap > 0 && prompt_count + max_tokens > session_cap) {
    throw RequestError("context_length_exceeded",
      "prompt exceeds the per-session context cap; prompt tokens=" + std::to_string(prompt_count) +
      ", max_session_ctx=" + std::to_string(session_cap) +
      ", reserved output tokens=" + std::to_string(max_tokens) +
      ". Reduce the prompt/tool history or raise max_session_ctx / CLAP_LLAMA_MAX_SESSION_CTX.");
  }
  return {max_tokens, max_tokens};
}

RequestPreparer::RequestPreparer(ModelRuntime& runtime, CacheExecutor* cache_executor,
                                 std::vector<RequestSlotState> slots,
                                 std::string identity_key, Fingerprint fingerprint)
    : runtime_(runtime), cache_executor_(cache_executor), slots_(std::move(slots)),
      identity_key_(std::move(identity_key)), fingerprint_(std::move(fingerprint)) {}

clap::llama_cache::Identity RequestPreparer::cache_identity(
    const nlohmann::json& request) const {
  const auto cache = request.contains("cache") && request["cache"].is_object()
      ? request["cache"] : nlohmann::json::object();
  const std::string requested_namespace = cache_string(cache, "namespace");
  const std::string tenant = requested_namespace.empty()
      ? cache_string(cache, "tenant") : requested_namespace;
  const std::string keyed = identity_key_ + "|";
  clap::llama_cache::Identity identity;
  identity.name_space = clap::llama_cache::fingerprint(
      keyed + runtime_.cache_domain() + "|tenant=" + (tenant.empty() ? "local" : tenant));
  identity.tenant = clap::llama_cache::hash(keyed + (tenant.empty() ? "local" : tenant));
  identity.project = clap::llama_cache::hash(keyed + cache_string(cache, "project"));
  identity.harness = clap::llama_cache::hash(keyed + cache_string(cache, "harness"));
  identity.agent = clap::llama_cache::hash(keyed + cache_string(cache, "agent"));
  const std::string session = cache_string(cache, "session");
  identity.session = session.empty() ? 0 : clap::llama_cache::hash(keyed + session);
  identity.side_request = cache.value("side_request", false);
  identity.priority = cache_string(cache, "priority") == "background"
      ? CLAP_CACHE_PRIORITY_BACKGROUND : CLAP_CACHE_PRIORITY_INTERACTIVE;
  if (!session.empty()) identity.scope = CLAP_CACHE_SCOPE_SESSION;
  else if (!cache_string(cache, "agent").empty()) identity.scope = CLAP_CACHE_SCOPE_AGENT;
  else if (!cache_string(cache, "project").empty()) identity.scope = CLAP_CACHE_SCOPE_PROJECT;
  else if (!cache_string(cache, "harness").empty()) identity.scope = CLAP_CACHE_SCOPE_HARNESS;
  else identity.scope = CLAP_CACHE_SCOPE_TENANT;
  return identity;
}

uint64_t RequestPreparer::capabilities() const {
  uint64_t value = CLAP_CACHE_CAP_WHOLE_STATE_COPY | CLAP_CACHE_CAP_SAFE_BUSY_DONOR |
      CLAP_CACHE_CAP_RELIABLE_RESIDENT_LENGTH | CLAP_CACHE_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS |
      CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT;
  return runtime_.hybrid() ? value | CLAP_CACHE_CAP_RECURRENT_OR_HYBRID
      : value | CLAP_CACHE_CAP_PARTIAL_SUFFIX_TRIM | CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH |
          CLAP_CACHE_CAP_ZERO_COPY_BRANCH | CLAP_CACHE_CAP_UNIFIED_STORAGE;
}

PreparedRequest RequestPreparer::prepare(const std::string& id, const nlohmann::json& request) {
  PromptRenderer renderer(runtime_);
  auto prompt = renderer.prepare(prompt_input_from_request(request));
  PreparedRequest prepared;
  prepared.id = id;
  prepared.params = sampling_from_request(request);
  prepared.cache_identity = cache_identity(request);
  prepared.cache_side_request = prepared.cache_identity.side_request;
  if (!cache_executor_) prepared.cache_fallback = "coordinator_unavailable";
  if (request.contains("cache") && request["cache"].is_object()) {
    prepared.cache_namespace = cache_string(request["cache"], "namespace");
  }
  prepared.structural_boundaries = std::move(prompt.structural_boundaries);
  prepared.resolved_boundaries = std::move(prompt.resolved_boundaries);
  auto tokens = std::move(prompt.tokens);
  const auto budget = validate_request_budget(static_cast<int>(tokens.size()),
      runtime_.backend_allocation_cap(), prepared.params.max_tokens,
      runtime_.max_output_tokens(), env_int("CLAP_LLAMA_MAX_SESSION_CTX", 0));
  prepared.params.max_tokens = budget.max_tokens;
  prepared.prompt_token_count = static_cast<int>(tokens.size());
  prepared.full_prompt_tokens = tokens;
  prepared.prompt_token_hash = fingerprint_(tokens, tokens.size());

  if (runtime_.has_encoder()) {
    if (cache_executor_) cache_executor_->reset();
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    if (llama_encode(runtime_.context(), batch)) throw std::runtime_error("llama_encode failed");
    llama_token decoder_start = llama_model_decoder_start_token(runtime_.model());
    if (decoder_start == LLAMA_TOKEN_NULL) decoder_start = llama_vocab_bos(runtime_.vocab());
    prepared.prompt_tokens = {decoder_start};
    prepared.full_prompt_tokens = prepared.prompt_tokens;
    prepared.sequence = 0;
    if (cache_executor_) prepared.cache_lease = cache_executor_->acquire(0);
    return prepared;
  }

  if (cache_executor_) {
    auto& executor_slots = cache_executor_->slots();
    for (std::size_t index = 0; index < slots_.size(); ++index) {
      executor_slots[index].tokens = slots_[index].tokens;
      executor_slots[index].busy = slots_[index].busy;
      executor_slots[index].is_anchor = slots_[index].anchor;
      if (slots_[index].generation != 0) executor_slots[index].generation = slots_[index].generation;
    }
    const CacheAdmissionResult decision = [&] {
      try {
        return cache_executor_->admit({tokens, prepared.cache_identity, capabilities(),
            static_cast<uint64_t>(budget.output_reserve), CLAP_CACHE_SLOT_SESSION, {},
            prompt.stable_boundaries, runtime_.hybrid(), {}});
      } catch (const clap::llama_cache::Error& error) {
        if (error.status == CLAP_CACHE_NO_CAPACITY || error.status == CLAP_CACHE_SLOT_BUSY) {
          throw CacheBackpressure(error.what());
        }
        prepared.cache_fallback = "coordinator_plan_failed_closed";
        fprintf(stderr, "clap-llama: cache coordinator plan failed closed: %s\n", error.what());
        throw;
      }
    }();
    prepared.sequence = static_cast<llama_seq_id>(decision.target_slot);
    prepared.initial_position = static_cast<int32_t>(decision.realized_reuse_tokens);
    prepared.cached_prompt_tokens = static_cast<int>(decision.realized_reuse_tokens);
    prepared.cache_planned_reuse_tokens = decision.planned_reuse_tokens;
    prepared.cache_realized_reuse_tokens = decision.realized_reuse_tokens;
    prepared.cache_decision_us = decision.decision_us;
    prepared.cache_evicted_slots = decision.eviction_slots;
    prepared.cache_donor_slot = decision.has_donor ? static_cast<int>(decision.donor_slot) : -1;
    prepared.cache_donor_generation = decision.donor_generation;
    prepared.cache_target_generation = decision.target_generation;
    prepared.cache_reuse_scope = scope_name(decision.scope);
    if (decision.operation == CLAP_CACHE_OPERATION_CONTINUE && decision.realized_reuse_tokens > 0) prepared.cache_reuse_kind = "slot";
    else if (decision.operation == CLAP_CACHE_OPERATION_RESTORE && decision.realized_reuse_tokens > 0) prepared.cache_reuse_kind = "anchor";
    else if (decision.operation == CLAP_CACHE_OPERATION_BRANCH && decision.realized_reuse_tokens > 0) prepared.cache_reuse_kind = "branch";
    prepared.anchor_boundaries.assign(decision.anchor_boundaries.begin(), decision.anchor_boundaries.end());
    for (const std::size_t count : prepared.anchor_boundaries) {
      const auto known = std::find_if(prepared.resolved_boundaries.begin(),
          prepared.resolved_boundaries.end(), [count](const auto& value) {
            return value.token_count == count;
          });
      if (known == prepared.resolved_boundaries.end()) {
        prepared.resolved_boundaries.push_back(
            {count, "automatic_token", "", false, "authorized", ""});
      }
    }
    prepared.initial_anchor_at = prepared.anchor_boundaries.empty()
        ? -1 : static_cast<int32_t>(prepared.anchor_boundaries.front());
    prepared.stable_boundary = clap::llama_boundary::exact(prepared.full_prompt_tokens,
        static_cast<std::size_t>(decision.anchor_tokens), "prompt", fingerprint_);
    for (const auto& candidate : decision.candidates) {
      const char* rejection = candidate_rejection(candidate.rejection);
      prepared.cache_candidates.push_back({
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
        {"rejection", rejection ? nlohmann::json(rejection) : nlohmann::json(nullptr)}});
    }
    tokens.erase(tokens.begin(), tokens.begin() + prepared.initial_position);
    prepared.prompt_tokens = std::move(tokens);
    prepared.cache_lease = cache_executor_->lease_admitted(
        decision.target_slot, decision.target_generation);
    return prepared;
  }

  std::size_t target = SIZE_MAX;
  for (std::size_t index = 0; index < slots_.size(); ++index) {
    if (!slots_[index].busy) {
      target = index;
      if (slots_[index].tokens.empty()) break;
    }
  }
  if (target == SIZE_MAX) throw std::runtime_error("no idle KV slot available");
  prepared.sequence = static_cast<llama_seq_id>(target);
  prepared.cache_fallback = "coordinator_unavailable_no_cache";
  prepared.prompt_tokens = std::move(tokens);
  return prepared;
}

}  // namespace clap::llama
