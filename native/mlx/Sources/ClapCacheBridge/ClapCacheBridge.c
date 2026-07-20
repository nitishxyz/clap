#include "ClapCacheBridge.h"
#include "clap_cache.h"

#include <stdlib.h>
#include <string.h>

struct cc_manager {
  clap_cache_t *cache;
  int32_t last_status;
};

struct cc_plan {
  cc_manager_t *owner;
  clap_cache_plan_t *plan;
};

cc_manager_t *cc_manager_create_with_retention(
    uint32_t initial_slots, uint64_t min_reuse_tokens,
    uint64_t token_capacity, uint32_t max_anchors,
    uint32_t hard_max_retained_entries, uint64_t physical_byte_budget,
    uint64_t high_watermark_bytes, uint64_t low_watermark_bytes) {
  cc_manager_t *manager = calloc(1, sizeof(*manager));
  if (!manager) return NULL;
  clap_cache_config_t config = {0};
  config.version = CLAP_CACHE_ABI_VERSION;
  config.struct_size = sizeof(config);
  config.slot_count = initial_slots;
  config.max_anchors = max_anchors;
  config.min_reuse_tokens = min_reuse_tokens;
  config.logical_token_capacity = token_capacity;
  clap_cache_retention_config_t retention = {0};
  retention.version = CLAP_CACHE_ABI_VERSION;
  retention.struct_size = sizeof(retention);
  retention.hard_max_retained_entries = hard_max_retained_entries;
  retention.physical_byte_budget = physical_byte_budget;
  retention.high_watermark_bytes = high_watermark_bytes;
  retention.low_watermark_bytes = low_watermark_bytes;
  manager->last_status = clap_cache_create_with_retention(
      &config, &retention, &manager->cache);
  if (manager->last_status != CLAP_CACHE_OK) {
    free(manager);
    return NULL;
  }
  return manager;
}

void cc_manager_destroy(cc_manager_t *manager) {
  if (!manager) return;
  clap_cache_destroy(manager->cache);
  free(manager);
}

int32_t cc_manager_last_status(const cc_manager_t *manager) {
  return manager ? manager->last_status : CLAP_CACHE_INVALID_ARGUMENT;
}

cc_plan_t *cc_manager_plan(cc_manager_t *manager, const int32_t *tokens,
                           size_t tokens_len, const uint8_t namespace_fingerprint[32],
                           uint64_t tenant, uint64_t project, uint64_t harness,
                           uint64_t agent, uint64_t session, uint32_t scope,
                           uint32_t priority, uint8_t side_request,
                           uint64_t capabilities, const uint8_t *slot_capabilities,
                           size_t slot_capabilities_len, const uint64_t *stable_boundaries,
                           size_t stable_boundaries_len, uint64_t output_reserve,
                           uint32_t result_state) {
  if (!manager || !namespace_fingerprint) return NULL;
  clap_cache_request_t request = {0};
  request.version = CLAP_CACHE_ABI_VERSION;
  request.struct_size = sizeof(request);
  memcpy(request.namespace_fingerprint, namespace_fingerprint, 32);
  request.tokens = tokens;
  request.tokens_len = tokens_len;
  request.labels.version = CLAP_CACHE_ABI_VERSION;
  request.labels.struct_size = sizeof(request.labels);
  request.labels.tenant = tenant;
  request.labels.project = project;
  request.labels.harness = harness;
  request.labels.agent = agent;
  request.labels.session = session;
  request.labels.scope = scope;
  request.labels.priority = priority;
  request.labels.side_request = side_request;
  request.capabilities = capabilities;
  request.slot_capabilities = slot_capabilities;
  request.slot_capabilities_len = slot_capabilities_len;
  request.stable_boundaries = stable_boundaries;
  request.stable_boundaries_len = stable_boundaries_len;
  request.output_reserve = output_reserve;
  request.result_state = result_state;

  clap_cache_plan_t *inner = NULL;
  manager->last_status = clap_cache_plan(manager->cache, &request, &inner);
  if (manager->last_status != CLAP_CACHE_OK) return NULL;
  cc_plan_t *plan = calloc(1, sizeof(*plan));
  if (!plan) {
    clap_cache_abort(manager->cache, inner);
    clap_cache_plan_destroy(inner);
    manager->last_status = CLAP_CACHE_NO_CAPACITY;
    return NULL;
  }
  plan->owner = manager;
  plan->plan = inner;
  return plan;
}

int32_t cc_plan_view(const cc_plan_t *plan, cc_plan_view_t *out) {
  if (!plan || !out) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_plan_view_t view = {0};
  int32_t status = clap_cache_plan_view(plan->plan, &view);
  if (status != CLAP_CACHE_OK) return status;
  out->epoch = view.epoch;
  out->operation = view.operation;
  out->target_slot = view.target.slot;
  out->target_generation = view.target.generation;
  out->has_donor = view.has_donor;
  out->donor_slot = view.donor.slot;
  out->donor_generation = view.donor.generation;
  out->reuse_tokens = view.reuse_tokens;
  out->anchor_tokens = view.anchor_tokens;
  out->decision_us = view.decision_us;
  out->eviction_count = view.eviction_count;
  return CLAP_CACHE_OK;
}

int32_t cc_plan_eviction(const cc_plan_t *plan, uint32_t index,
                         uint32_t *slot, uint64_t *generation) {
  if (!plan || !slot || !generation) return CLAP_CACHE_INVALID_ARGUMENT;
  size_t count = 0;
  int32_t status = clap_cache_plan_evictions(plan->plan, NULL, 0, &count);
  if (status != CLAP_CACHE_OK && status != CLAP_CACHE_NO_CAPACITY) return status;
  if (index >= count) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t *entries = calloc(count, sizeof(*entries));
  if (!entries) return CLAP_CACHE_NO_CAPACITY;
  status = clap_cache_plan_evictions(plan->plan, entries, count, &count);
  if (status == CLAP_CACHE_OK) {
    *slot = entries[index].slot;
    *generation = entries[index].generation;
  }
  free(entries);
  return status;
}

int32_t cc_plan_candidate_count(const cc_plan_t *plan, uint32_t *count) {
  if (!plan || !count) return CLAP_CACHE_INVALID_ARGUMENT;
  size_t required = 0;
  int32_t status = clap_cache_plan_candidates(plan->plan, NULL, 0, &required);
  if (status != CLAP_CACHE_OK && status != CLAP_CACHE_NO_CAPACITY) return status;
  *count = (uint32_t)required;
  return CLAP_CACHE_OK;
}

int32_t cc_plan_candidate(const cc_plan_t *plan, uint32_t index,
                          cc_candidate_t *out) {
  if (!plan || !out) return CLAP_CACHE_INVALID_ARGUMENT;
  size_t count = 0;
  int32_t status = clap_cache_plan_candidates(plan->plan, NULL, 0, &count);
  if (status != CLAP_CACHE_OK && status != CLAP_CACHE_NO_CAPACITY) return status;
  if (index >= count) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_candidate_evaluation_t *entries = calloc(count, sizeof(*entries));
  if (!entries) return CLAP_CACHE_NO_CAPACITY;
  status = clap_cache_plan_candidates(plan->plan, entries, count, &count);
  if (status == CLAP_CACHE_OK) {
    const clap_cache_candidate_evaluation_t *entry = &entries[index];
    out->slot = entry->slot;
    out->state = entry->state;
    out->generation = entry->generation;
    out->shared_prefix_tokens = entry->shared_prefix_tokens;
    out->namespace_compatible = entry->namespace_compatible;
    out->model_compatible = entry->model_compatible;
    out->session_compatible = entry->session_compatible;
    out->generation_compatible = entry->generation_compatible;
    out->busy_eligible = entry->busy_eligible;
    out->lease_eligible = entry->lease_eligible;
    out->materialized = entry->materialized;
    out->trim_eligible = entry->trim_eligible;
    out->copy_eligible = entry->copy_eligible;
    out->eligible = entry->eligible;
    out->selected = entry->selected;
    out->rejection = entry->rejection;
  }
  free(entries);
  return status;
}

int32_t cc_plan_commit(cc_plan_t *plan, uint64_t resident_tokens,
                       uint32_t state, uint64_t physical_bytes,
                       cc_decision_t *out) {
  if (!plan || !out) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_commit_t commit = {0};
  commit.version = CLAP_CACHE_ABI_VERSION;
  commit.struct_size = sizeof(commit);
  commit.resident_tokens = resident_tokens;
  commit.actual_state = state;
  commit.physical_bytes = physical_bytes;
  clap_cache_decision_t decision = {0};
  int32_t status = clap_cache_commit(plan->owner->cache, plan->plan, &commit, &decision);
  plan->owner->last_status = status;
  if (status != CLAP_CACHE_OK) return status;
  out->hit = decision.hit;
  out->has_donor = decision.has_donor;
  out->operation = decision.operation;
  out->scope = decision.scope;
  out->target_slot = decision.target_slot;
  out->donor_slot = decision.donor_slot;
  out->planned_reuse_tokens = decision.planned_reuse_tokens;
  out->realized_reuse_tokens = decision.realized_reuse_tokens;
  out->decision_us = decision.decision_us;
  out->eviction_count = decision.eviction_count;
  return status;
}

int32_t cc_plan_abort(cc_plan_t *plan) {
  if (!plan) return CLAP_CACHE_INVALID_ARGUMENT;
  int32_t status = clap_cache_abort(plan->owner->cache, plan->plan);
  plan->owner->last_status = status;
  return status;
}

void cc_plan_destroy(cc_plan_t *plan) {
  if (!plan) return;
  clap_cache_plan_destroy(plan->plan);
  free(plan);
}

int32_t cc_manager_slot(cc_manager_t *manager, uint32_t slot, cc_slot_info_t *out) {
  if (!manager || !out) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_info_t info = {0};
  int32_t status = clap_cache_get_slot(manager->cache, slot, &info);
  manager->last_status = status;
  if (status != CLAP_CACHE_OK) return status;
  out->generation = info.generation;
  out->resident_len = info.resident_len;
  out->state = info.state;
  out->busy = info.busy;
  out->read_leases = info.read_leases;
  out->write_leased = info.write_leased;
  out->physical_bytes = info.physical_bytes;
  return status;
}

int32_t cc_manager_advance(cc_manager_t *manager, uint32_t slot,
                           uint64_t generation, const int32_t *tokens,
                           size_t tokens_len, uint32_t state, uint8_t busy,
                           uint64_t physical_bytes, uint64_t *out_generation) {
  if (!manager) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t ref = {slot, 0, generation};
  int32_t status = clap_cache_advance(manager->cache, ref, tokens, tokens_len,
                                      state, busy, physical_bytes, out_generation);
  manager->last_status = status;
  return status;
}

int32_t cc_manager_confirm(cc_manager_t *manager, uint32_t slot,
                           uint64_t generation, const int32_t *tokens,
                           size_t tokens_len, uint32_t state, uint8_t busy,
                           uint64_t physical_bytes, uint64_t *out_generation) {
  if (!manager) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t ref = {slot, 0, generation};
  int32_t status = clap_cache_confirm(manager->cache, ref, tokens, tokens_len,
                                      state, busy, physical_bytes, out_generation);
  manager->last_status = status;
  return status;
}

int32_t cc_manager_set_busy(cc_manager_t *manager, uint32_t slot,
                            uint64_t generation, uint8_t busy) {
  if (!manager) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t ref = {slot, 0, generation};
  int32_t status = clap_cache_set_busy(manager->cache, ref, busy);
  manager->last_status = status;
  return status;
}

int32_t cc_manager_register_slot(cc_manager_t *manager, uint32_t *slot,
                                 uint64_t *generation) {
  if (!manager || !slot || !generation) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t ref = {0};
  int32_t status = clap_cache_register_slot(manager->cache, &ref);
  manager->last_status = status;
  if (status == CLAP_CACHE_OK) {
    *slot = ref.slot;
    *generation = ref.generation;
  }
  return status;
}

int32_t cc_manager_set_anchor_protected(cc_manager_t *manager, uint32_t slot,
                                        uint64_t generation, uint8_t protected_anchor) {
  if (!manager) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t ref = {slot, 0, generation};
  int32_t status = clap_cache_set_anchor_protected(
      manager->cache, ref, protected_anchor);
  manager->last_status = status;
  return status;
}

int32_t cc_manager_invalidate(cc_manager_t *manager, uint32_t slot,
                              uint64_t generation, uint64_t *out_generation) {
  if (!manager) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_slot_ref_t ref = {slot, 0, generation};
  int32_t status = clap_cache_invalidate(manager->cache, ref, out_generation);
  manager->last_status = status;
  return status;
}

int32_t cc_manager_reset(cc_manager_t *manager, uint64_t *out_epoch) {
  if (!manager) return CLAP_CACHE_INVALID_ARGUMENT;
  int32_t status = clap_cache_reset(manager->cache, out_epoch);
  manager->last_status = status;
  return status;
}

int32_t cc_manager_retention_telemetry(cc_manager_t *manager,
                                       cc_retention_telemetry_t *out) {
  if (!manager || !out) return CLAP_CACHE_INVALID_ARGUMENT;
  clap_cache_retention_telemetry_t retention = {0};
  clap_cache_telemetry_t telemetry = {0};
  int32_t status = clap_cache_get_retention_telemetry(manager->cache, &retention);
  if (status == CLAP_CACHE_OK) {
    status = clap_cache_get_telemetry(manager->cache, &telemetry);
  }
  manager->last_status = status;
  if (status != CLAP_CACHE_OK) return status;
  out->total_slots = retention.total_slots;
  out->session_slots = retention.session_slots;
  out->anchor_slots = retention.anchor_slots;
  out->active_slots = retention.active_slots;
  out->total_bytes = retention.total_bytes;
  out->session_bytes = retention.session_bytes;
  out->anchor_bytes = retention.anchor_bytes;
  out->active_bytes = retention.active_bytes;
  out->physical_byte_budget = retention.physical_byte_budget;
  out->high_watermark_bytes = retention.high_watermark_bytes;
  out->low_watermark_bytes = retention.low_watermark_bytes;
  out->under_pressure = retention.under_pressure;
  out->evictions = telemetry.evictions;
  return status;
}

uint64_t cc_hash_string(const char *value) {
  uint64_t result = UINT64_C(1469598103934665603);
  if (value) {
    while (*value) {
      result ^= (uint8_t)*value++;
      result *= UINT64_C(1099511628211);
    }
  }
  return result == 0 ? 1 : result;
}

void cc_fingerprint_string(const char *value, uint8_t out[32]) {
  uint64_t state = cc_hash_string(value);
  for (size_t word = 0; word < 4; ++word) {
    state ^= UINT64_C(0x9e3779b97f4a7c15) + word + (state << 6) + (state >> 2);
    for (size_t byte = 0; byte < 8; ++byte) out[word * 8 + byte] = state >> (byte * 8);
  }
}
