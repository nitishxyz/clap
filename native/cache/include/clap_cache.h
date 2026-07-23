#ifndef CLAP_CACHE_H
#define CLAP_CACHE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CLAP_CACHE_ABI_VERSION 3u

#define CLAP_CACHE_CAP_PARTIAL_SUFFIX_TRIM (UINT64_C(1) << 0)
#define CLAP_CACHE_CAP_PARTIAL_PREFIX_BRANCH (UINT64_C(1) << 1)
#define CLAP_CACHE_CAP_WHOLE_STATE_COPY (UINT64_C(1) << 2)
#define CLAP_CACHE_CAP_SAFE_BUSY_DONOR (UINT64_C(1) << 3)
#define CLAP_CACHE_CAP_ZERO_COPY_BRANCH (UINT64_C(1) << 4)
#define CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT (UINT64_C(1) << 5)
#define CLAP_CACHE_CAP_SLIDING_WINDOW (UINT64_C(1) << 6)
#define CLAP_CACHE_CAP_RECURRENT_OR_HYBRID (UINT64_C(1) << 7)
#define CLAP_CACHE_CAP_UNIFIED_STORAGE (UINT64_C(1) << 8)
#define CLAP_CACHE_CAP_RELIABLE_RESIDENT_LENGTH (UINT64_C(1) << 9)
#define CLAP_CACHE_CAP_KV_QUANTIZED (UINT64_C(1) << 10)
#define CLAP_CACHE_CAP_RETAIN_LAST_TOKEN_FOR_LOGITS (UINT64_C(1) << 11)

#define CLAP_CACHE_SLOT_MATERIALIZED (UINT8_C(1) << 0)
#define CLAP_CACHE_SLOT_WRITABLE (UINT8_C(1) << 1)
#define CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM (UINT8_C(1) << 2)
#define CLAP_CACHE_SLOT_COPY (UINT8_C(1) << 3)

typedef struct clap_cache clap_cache_t;
typedef struct clap_cache_plan clap_cache_plan_t;

typedef enum clap_cache_status {
  CLAP_CACHE_OK = 0,
  CLAP_CACHE_INVALID_ARGUMENT = 1,
  CLAP_CACHE_NO_CAPACITY = 2,
  CLAP_CACHE_STALE_PLAN = 3,
  CLAP_CACHE_PLAN_CONSUMED = 4,
  CLAP_CACHE_SLOT_BUSY = 5,
  CLAP_CACHE_UNSUPPORTED = 6,
  CLAP_CACHE_PANIC = 255
} clap_cache_status_t;

typedef enum clap_cache_slot_state {
  CLAP_CACHE_SLOT_EMPTY = 0,
  CLAP_CACHE_SLOT_SESSION = 1,
  CLAP_CACHE_SLOT_PROMPT_BOUNDARY = 2,
  CLAP_CACHE_SLOT_ANCHOR = 3
} clap_cache_slot_state_t;

typedef enum clap_cache_scope {
  CLAP_CACHE_SCOPE_NONE = 0,
  CLAP_CACHE_SCOPE_SESSION = 1,
  CLAP_CACHE_SCOPE_AGENT = 2,
  CLAP_CACHE_SCOPE_PROJECT = 3,
  CLAP_CACHE_SCOPE_HARNESS = 4,
  CLAP_CACHE_SCOPE_TENANT = 5
} clap_cache_scope_t;

typedef enum clap_cache_priority {
  CLAP_CACHE_PRIORITY_BACKGROUND = 0,
  CLAP_CACHE_PRIORITY_NORMAL = 1,
  CLAP_CACHE_PRIORITY_INTERACTIVE = 2
} clap_cache_priority_t;

typedef enum clap_cache_operation {
  CLAP_CACHE_OPERATION_FRESH = 0,
  CLAP_CACHE_OPERATION_CONTINUE = 1,
  CLAP_CACHE_OPERATION_BRANCH = 2,
  CLAP_CACHE_OPERATION_RESTORE = 3,
  CLAP_CACHE_OPERATION_NOOP = 4
} clap_cache_operation_t;

typedef struct clap_cache_config {
  uint32_t version;
  uint32_t struct_size;
  uint32_t slot_count;
  uint32_t max_anchors;
  uint64_t min_reuse_tokens;
  uint64_t logical_token_capacity;
  /* mode: 0 safe default, 1 enabled, 2 disabled. Other zero values default. */
  uint32_t automatic_checkpoint_mode;
  uint32_t automatic_checkpoint_max;
  uint64_t automatic_checkpoint_min_tokens;
  uint64_t automatic_checkpoint_interval_tokens;
  uint32_t automatic_checkpoint_memory_basis_points;
  uint32_t reserved;
  uint64_t automatic_checkpoint_memory_cap_bytes;
} clap_cache_config_t;

typedef struct clap_cache_retention_config {
  uint32_t version;
  uint32_t struct_size;
  uint32_t hard_max_retained_entries;
  uint32_t reserved;
  /* Zero disables byte policy; otherwise low <= high <= budget. */
  uint64_t physical_byte_budget;
  uint64_t high_watermark_bytes;
  uint64_t low_watermark_bytes;
} clap_cache_retention_config_t;

typedef struct clap_cache_labels {
  uint32_t version;
  uint32_t struct_size;
  uint64_t tenant;
  uint64_t project;
  uint64_t harness;
  uint64_t agent;
  uint64_t session;
  uint32_t scope;
  uint32_t priority;
  uint8_t side_request;
  uint8_t reserved[7];
} clap_cache_labels_t;

typedef struct clap_cache_request {
  uint32_t version;
  uint32_t struct_size;
  uint8_t namespace_fingerprint[32];
  const int32_t *tokens;
  size_t tokens_len;
  clap_cache_labels_t labels;
  uint64_t capabilities;
  const uint8_t *slot_capabilities;
  size_t slot_capabilities_len;
  const uint64_t *stable_boundaries;
  size_t stable_boundaries_len;
  uint64_t output_reserve;
  uint64_t estimated_bytes_per_token;
  uint32_t result_state;
  uint32_t reserved;
} clap_cache_request_t;

typedef struct clap_cache_slot_ref {
  uint32_t slot;
  uint32_t reserved;
  uint64_t generation;
} clap_cache_slot_ref_t;

typedef struct clap_cache_plan_view {
  uint32_t version;
  uint32_t struct_size;
  uint64_t epoch;
  uint32_t operation;
  uint8_t has_donor;
  uint8_t reserved0[3];
  clap_cache_slot_ref_t target;
  clap_cache_slot_ref_t donor;
  uint64_t reuse_tokens;
  uint64_t anchor_tokens;
  uint32_t eviction_count;
  uint32_t result_state;
  uint64_t decision_us;
} clap_cache_plan_view_t;

typedef enum clap_cache_candidate_rejection {
  CLAP_CACHE_REJECTION_NONE = 0,
  CLAP_CACHE_REJECTION_NAMESPACE = 1,
  CLAP_CACHE_REJECTION_MODEL_DOMAIN = 2,
  CLAP_CACHE_REJECTION_GENERATION = 3,
  CLAP_CACHE_REJECTION_BUSY_LEASE = 4,
  CLAP_CACHE_REJECTION_MATERIALIZATION = 5,
  CLAP_CACHE_REJECTION_SESSION = 6,
  CLAP_CACHE_REJECTION_NONTRIM = 7,
  CLAP_CACHE_REJECTION_CAPABILITY = 8,
  CLAP_CACHE_REJECTION_MIN_PREFIX = 9,
  CLAP_CACHE_REJECTION_CAPACITY = 10,
  CLAP_CACHE_REJECTION_ABSENT_ANCHOR = 11,
  CLAP_CACHE_REJECTION_LOWER_RANK = 12
} clap_cache_candidate_rejection_t;

typedef struct clap_cache_candidate_evaluation {
  uint32_t version;
  uint32_t struct_size;
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
  uint8_t reserved;
  uint32_t rejection;
} clap_cache_candidate_evaluation_t;

typedef struct clap_cache_commit {
  uint32_t version;
  uint32_t struct_size;
  uint64_t resident_tokens;
  uint32_t actual_state;
  uint32_t reserved;
  uint64_t physical_bytes;
  uint64_t prefill_us_saved;
} clap_cache_commit_t;

typedef struct clap_cache_decision {
  uint32_t version;
  uint32_t struct_size;
  uint8_t hit;
  uint8_t has_donor;
  uint8_t reserved0[2];
  uint32_t operation;
  uint32_t scope;
  uint32_t target_slot;
  uint32_t donor_slot;
  uint64_t planned_reuse_tokens;
  uint64_t realized_reuse_tokens;
  uint64_t decision_us;
  uint32_t eviction_count;
  uint32_t reserved1;
} clap_cache_decision_t;

typedef struct clap_cache_telemetry {
  uint32_t version;
  uint32_t struct_size;
  uint64_t plans;
  uint64_t hits;
  uint64_t misses;
  uint64_t commits;
  uint64_t aborts;
  uint64_t stale_commits;
  uint64_t evictions;
  uint64_t resets;
  uint64_t planned_reuse_tokens;
  uint64_t realized_reuse_tokens;
  uint64_t prefill_us_saved;
  uint32_t active_slots;
  uint32_t anchors;
  uint32_t read_leases;
  uint32_t write_leases;
  uint64_t prefix_nodes;
  uint64_t physical_bytes;
} clap_cache_telemetry_t;

typedef struct clap_cache_retention_telemetry {
  uint32_t version;
  uint32_t struct_size;
  uint32_t total_slots;
  uint32_t session_slots;
  uint32_t anchor_slots;
  uint32_t active_slots;
  uint64_t total_bytes;
  uint64_t session_bytes;
  uint64_t anchor_bytes;
  uint32_t automatic_checkpoint_slots;
  uint32_t reserved0;
  uint64_t automatic_checkpoint_bytes;
  uint64_t automatic_checkpoint_byte_budget;
  uint64_t active_bytes;
  uint64_t physical_byte_budget;
  uint64_t high_watermark_bytes;
  uint64_t low_watermark_bytes;
  uint8_t under_pressure;
  uint8_t reserved[7];
} clap_cache_retention_telemetry_t;

typedef struct clap_cache_slot_info {
  uint32_t version;
  uint32_t struct_size;
  uint64_t generation;
  uint64_t resident_len;
  uint32_t state;
  uint8_t busy;
  uint8_t write_leased;
  uint8_t reserved0[2];
  uint32_t read_leases;
  uint32_t scope;
  uint64_t session;
  uint64_t last_used;
  uint64_t reuse_count;
  uint64_t physical_bytes;
} clap_cache_slot_info_t;

clap_cache_status_t clap_cache_create(const clap_cache_config_t *config,
                                      clap_cache_t **out_cache);
clap_cache_status_t clap_cache_create_with_retention(
    const clap_cache_config_t *config,
    const clap_cache_retention_config_t *retention,
    clap_cache_t **out_cache);
void clap_cache_destroy(clap_cache_t *cache);

clap_cache_status_t clap_cache_plan(clap_cache_t *cache,
                                    const clap_cache_request_t *request,
                                    clap_cache_plan_t **out_plan);
clap_cache_status_t clap_cache_plan_view(const clap_cache_plan_t *plan,
                                         clap_cache_plan_view_t *out_view);
clap_cache_status_t clap_cache_plan_evictions(const clap_cache_plan_t *plan,
                                              clap_cache_slot_ref_t *out_slots,
                                              size_t capacity,
                                              size_t *out_count);
clap_cache_status_t clap_cache_plan_anchor_boundaries(
    const clap_cache_plan_t *plan, uint64_t *out_boundaries,
    size_t capacity, size_t *out_count);
clap_cache_status_t clap_cache_plan_candidates(
    const clap_cache_plan_t *plan,
    clap_cache_candidate_evaluation_t *out_candidates,
    size_t capacity,
    size_t *out_count);
clap_cache_status_t clap_cache_commit(clap_cache_t *cache,
                                      clap_cache_plan_t *plan,
                                      const clap_cache_commit_t *commit,
                                      clap_cache_decision_t *out_decision);
clap_cache_status_t clap_cache_abort(clap_cache_t *cache,
                                     clap_cache_plan_t *plan);
/* Unconsumed destruction aborts and invalidates the target. Destroy all plan
 * handles before their owning cache manager. */
void clap_cache_plan_destroy(clap_cache_plan_t *plan);

clap_cache_status_t clap_cache_advance(clap_cache_t *cache,
                                       clap_cache_slot_ref_t slot,
                                       const int32_t *tokens,
                                       size_t tokens_len,
                                       uint32_t state,
                                       uint8_t busy,
                                       uint64_t physical_bytes,
                                       uint64_t *out_generation);
clap_cache_status_t clap_cache_confirm(clap_cache_t *cache,
                                       clap_cache_slot_ref_t slot,
                                       const int32_t *tokens,
                                       size_t tokens_len,
                                       uint32_t state,
                                       uint8_t busy,
                                       uint64_t physical_bytes,
                                       uint64_t *out_generation);
clap_cache_status_t clap_cache_set_busy(clap_cache_t *cache,
                                        clap_cache_slot_ref_t slot,
                                        uint8_t busy);
clap_cache_status_t clap_cache_register_slot(clap_cache_t *cache,
                                              clap_cache_slot_ref_t *out_slot);
clap_cache_status_t clap_cache_set_anchor_protected(
    clap_cache_t *cache, clap_cache_slot_ref_t slot, uint8_t protected_anchor);
clap_cache_status_t clap_cache_invalidate(clap_cache_t *cache,
                                          clap_cache_slot_ref_t slot,
                                          uint64_t *out_generation);
clap_cache_status_t clap_cache_reset(clap_cache_t *cache,
                                     uint64_t *out_epoch);
clap_cache_status_t clap_cache_get_telemetry(const clap_cache_t *cache,
                                             clap_cache_telemetry_t *out);
clap_cache_status_t clap_cache_get_retention_telemetry(
    const clap_cache_t *cache, clap_cache_retention_telemetry_t *out);
clap_cache_status_t clap_cache_get_slot(const clap_cache_t *cache,
                                        uint32_t slot,
                                        clap_cache_slot_info_t *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* CLAP_CACHE_H */
