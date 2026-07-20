#ifndef CLAP_CACHE_BRIDGE_H
#define CLAP_CACHE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct cc_manager cc_manager_t;
typedef struct cc_plan cc_plan_t;

enum {
  CC_OK = 0,
  CC_INVALID_ARGUMENT = 1,
  CC_NO_CAPACITY = 2,
  CC_STALE_PLAN = 3,
  CC_PLAN_CONSUMED = 4,
  CC_SLOT_BUSY = 5,
  CC_UNSUPPORTED = 6,
  CC_SLOT_EMPTY = 0,
  CC_SLOT_SESSION = 1,
  CC_SLOT_PROMPT_BOUNDARY = 2,
  CC_SLOT_ANCHOR = 3,
  CC_SCOPE_NONE = 0,
  CC_SCOPE_SESSION = 1,
  CC_SCOPE_AGENT = 2,
  CC_SCOPE_PROJECT = 3,
  CC_SCOPE_HARNESS = 4,
  CC_SCOPE_TENANT = 5,
  CC_PRIORITY_BACKGROUND = 0,
  CC_PRIORITY_INTERACTIVE = 1,
  CC_OPERATION_FRESH = 0,
  CC_OPERATION_CONTINUE = 1,
  CC_OPERATION_BRANCH = 2,
  CC_OPERATION_RESTORE = 3,
  CC_OPERATION_NOOP = 4,
};

enum {
  CC_CAP_PARTIAL_SUFFIX_TRIM = UINT64_C(1) << 0,
  CC_CAP_PARTIAL_PREFIX_BRANCH = UINT64_C(1) << 1,
  CC_CAP_WHOLE_STATE_COPY = UINT64_C(1) << 2,
  CC_CAP_SAFE_BUSY_DONOR = UINT64_C(1) << 3,
  CC_CAP_PROMPT_BOUNDARY_SNAPSHOT = UINT64_C(1) << 5,
  CC_CAP_SLIDING_WINDOW = UINT64_C(1) << 6,
  CC_CAP_RECURRENT_OR_HYBRID = UINT64_C(1) << 7,
  CC_CAP_RELIABLE_RESIDENT_LENGTH = UINT64_C(1) << 9,
  CC_CAP_KV_QUANTIZED = UINT64_C(1) << 10,
  CC_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS = UINT64_C(1) << 11,
};

enum {
  CC_SLOT_MATERIALIZED = UINT8_C(1) << 0,
  CC_SLOT_WRITABLE = UINT8_C(1) << 1,
  CC_SLOT_PARTIAL_SUFFIX_TRIM = UINT8_C(1) << 2,
  CC_SLOT_COPY = UINT8_C(1) << 3,
};

typedef struct cc_plan_view {
  uint64_t epoch;
  uint32_t operation;
  uint32_t target_slot;
  uint64_t target_generation;
  uint8_t has_donor;
  uint32_t donor_slot;
  uint64_t donor_generation;
  uint64_t reuse_tokens;
  uint64_t anchor_tokens;
  uint64_t decision_us;
  uint32_t eviction_count;
} cc_plan_view_t;

typedef struct cc_candidate {
  uint32_t slot;
  uint32_t state;
  uint64_t generation;
  uint64_t shared_prefix_tokens;
  uint8_t namespace_compatible;
  uint8_t model_compatible;
  uint8_t session_compatible;
  uint8_t generation_compatible;
  uint8_t busy_eligible;
  uint8_t lease_eligible;
  uint8_t materialized;
  uint8_t trim_eligible;
  uint8_t copy_eligible;
  uint8_t eligible;
  uint8_t selected;
  uint32_t rejection;
} cc_candidate_t;

typedef struct cc_decision {
  uint8_t hit;
  uint8_t has_donor;
  uint32_t operation;
  uint32_t scope;
  uint32_t target_slot;
  uint32_t donor_slot;
  uint64_t planned_reuse_tokens;
  uint64_t realized_reuse_tokens;
  uint64_t decision_us;
  uint32_t eviction_count;
} cc_decision_t;

typedef struct cc_slot_info {
  uint64_t generation;
  uint64_t resident_len;
  uint32_t state;
  uint8_t busy;
  uint32_t read_leases;
  uint8_t write_leased;
  uint64_t physical_bytes;
} cc_slot_info_t;

typedef struct cc_retention_telemetry {
  uint32_t total_slots;
  uint32_t session_slots;
  uint32_t anchor_slots;
  uint32_t active_slots;
  uint64_t total_bytes;
  uint64_t session_bytes;
  uint64_t anchor_bytes;
  uint64_t active_bytes;
  uint64_t physical_byte_budget;
  uint64_t high_watermark_bytes;
  uint64_t low_watermark_bytes;
  uint8_t under_pressure;
  uint64_t evictions;
} cc_retention_telemetry_t;

cc_manager_t *cc_manager_create_with_retention(
    uint32_t initial_slots, uint64_t min_reuse_tokens,
    uint64_t token_capacity, uint32_t max_anchors,
    uint32_t hard_max_retained_entries, uint64_t physical_byte_budget,
    uint64_t high_watermark_bytes, uint64_t low_watermark_bytes);
void cc_manager_destroy(cc_manager_t *manager);
int32_t cc_manager_last_status(const cc_manager_t *manager);

cc_plan_t *cc_manager_plan(cc_manager_t *manager, const int32_t *tokens,
                           size_t tokens_len, const uint8_t namespace_fingerprint[32],
                           uint64_t tenant, uint64_t project, uint64_t harness,
                           uint64_t agent, uint64_t session, uint32_t scope,
                           uint32_t priority, uint8_t side_request,
                           uint64_t capabilities, const uint8_t *slot_capabilities,
                           size_t slot_capabilities_len, const uint64_t *stable_boundaries,
                           size_t stable_boundaries_len, uint64_t output_reserve,
                           uint32_t result_state);
int32_t cc_plan_view(const cc_plan_t *plan, cc_plan_view_t *out);
int32_t cc_plan_eviction(const cc_plan_t *plan, uint32_t index,
                         uint32_t *slot, uint64_t *generation);
int32_t cc_plan_candidate_count(const cc_plan_t *plan, uint32_t *count);
int32_t cc_plan_candidate(const cc_plan_t *plan, uint32_t index,
                          cc_candidate_t *out);
int32_t cc_plan_commit(cc_plan_t *plan, uint64_t resident_tokens,
                       uint32_t state, uint64_t physical_bytes,
                       cc_decision_t *out);
int32_t cc_plan_abort(cc_plan_t *plan);
void cc_plan_destroy(cc_plan_t *plan);

int32_t cc_manager_slot(cc_manager_t *manager, uint32_t slot, cc_slot_info_t *out);
int32_t cc_manager_advance(cc_manager_t *manager, uint32_t slot,
                           uint64_t generation, const int32_t *tokens,
                           size_t tokens_len, uint32_t state, uint8_t busy,
                           uint64_t physical_bytes, uint64_t *out_generation);
int32_t cc_manager_confirm(cc_manager_t *manager, uint32_t slot,
                           uint64_t generation, const int32_t *tokens,
                           size_t tokens_len, uint32_t state, uint8_t busy,
                           uint64_t physical_bytes, uint64_t *out_generation);
int32_t cc_manager_set_busy(cc_manager_t *manager, uint32_t slot,
                            uint64_t generation, uint8_t busy);
int32_t cc_manager_register_slot(cc_manager_t *manager, uint32_t *slot,
                                 uint64_t *generation);
int32_t cc_manager_set_anchor_protected(cc_manager_t *manager, uint32_t slot,
                                        uint64_t generation, uint8_t protected_anchor);
int32_t cc_manager_invalidate(cc_manager_t *manager, uint32_t slot,
                              uint64_t generation, uint64_t *out_generation);
int32_t cc_manager_reset(cc_manager_t *manager, uint64_t *out_epoch);
int32_t cc_manager_retention_telemetry(cc_manager_t *manager,
                                       cc_retention_telemetry_t *out);
uint64_t cc_hash_string(const char *value);
void cc_fingerprint_string(const char *value, uint8_t out[32]);

#ifdef __cplusplus
}
#endif
#endif
