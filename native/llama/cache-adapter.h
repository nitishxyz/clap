#ifndef CLAP_LLAMA_CACHE_ADAPTER_H
#define CLAP_LLAMA_CACHE_ADAPTER_H

#include "clap_cache.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clap::llama_cache {

class Error : public std::runtime_error {
 public:
  Error(const char* operation, clap_cache_status_t status)
      : std::runtime_error(std::string(operation) + " failed with cache status " +
                           std::to_string(static_cast<int>(status))),
        status(status) {}

  clap_cache_status_t status;
};

inline void check(const char* operation, clap_cache_status_t status) {
  if (status != CLAP_CACHE_OK) throw Error(operation, status);
}

inline bool can_admit(std::size_t active, uint32_t max_active) {
  return max_active > 0 && active < max_active;
}

inline uint64_t hash(const std::string& value) {
  uint64_t result = UINT64_C(1469598103934665603);
  for (const unsigned char byte : value) {
    result ^= byte;
    result *= UINT64_C(1099511628211);
  }
  return result == 0 ? 1 : result;
}

inline std::array<uint8_t, 32> fingerprint(const std::string& value) {
  std::array<uint8_t, 32> out{};
  uint64_t state = hash(value);
  for (std::size_t word = 0; word < 4; ++word) {
    state ^= UINT64_C(0x9e3779b97f4a7c15) + word + (state << 6) + (state >> 2);
    for (std::size_t byte = 0; byte < 8; ++byte) {
      out[word * 8 + byte] = static_cast<uint8_t>(state >> (byte * 8));
    }
  }
  return out;
}

struct Identity {
  std::array<uint8_t, 32> name_space{};
  uint64_t tenant = 0;
  uint64_t project = 0;
  uint64_t harness = 0;
  uint64_t agent = 0;
  uint64_t session = 0;
  uint32_t scope = CLAP_CACHE_SCOPE_NONE;
  uint32_t priority = CLAP_CACHE_PRIORITY_NORMAL;
  bool side_request = false;
};

class Coordinator;

class Plan {
 public:
  Plan() = default;
  Plan(const Plan&) = delete;
  Plan& operator=(const Plan&) = delete;

  Plan(Plan&& other) noexcept
      : owner_(other.owner_), handle_(other.handle_), view_(other.view_),
        evictions_(std::move(other.evictions_)),
        anchor_boundaries_(std::move(other.anchor_boundaries_)),
        candidates_(std::move(other.candidates_)) {
    other.owner_ = nullptr;
    other.handle_ = nullptr;
  }

  Plan& operator=(Plan&& other) noexcept {
    if (this != &other) {
      cleanup();
      owner_ = other.owner_;
      handle_ = other.handle_;
      view_ = other.view_;
      evictions_ = std::move(other.evictions_);
      anchor_boundaries_ = std::move(other.anchor_boundaries_);
      candidates_ = std::move(other.candidates_);
      other.owner_ = nullptr;
      other.handle_ = nullptr;
    }
    return *this;
  }

  ~Plan() { cleanup(); }

  const clap_cache_plan_view_t& view() const { return view_; }
  const std::vector<clap_cache_slot_ref_t>& evictions() const { return evictions_; }
  const std::vector<uint64_t>& anchor_boundaries() const { return anchor_boundaries_; }
  const std::vector<clap_cache_candidate_evaluation_t>& candidates() const {
    return candidates_;
  }

  clap_cache_decision_t commit(uint64_t resident_tokens, uint32_t state,
                               uint64_t physical_bytes = 0,
                               uint64_t prefill_us_saved = 0) {
    clap_cache_commit_t commit{};
    commit.version = CLAP_CACHE_ABI_VERSION;
    commit.struct_size = sizeof(commit);
    commit.resident_tokens = resident_tokens;
    commit.actual_state = state;
    commit.physical_bytes = physical_bytes;
    commit.prefill_us_saved = prefill_us_saved;
    clap_cache_decision_t decision{};
    check("clap_cache_commit", clap_cache_commit(owner_, handle_, &commit, &decision));
    clap_cache_plan_destroy(handle_);
    handle_ = nullptr;
    return decision;
  }

  void abort() {
    if (!handle_) return;
    check("clap_cache_abort", clap_cache_abort(owner_, handle_));
    clap_cache_plan_destroy(handle_);
    handle_ = nullptr;
  }

 private:
  friend class Coordinator;

  Plan(clap_cache_t* owner, clap_cache_plan_t* handle) : owner_(owner), handle_(handle) {
    check("clap_cache_plan_view", clap_cache_plan_view(handle_, &view_));
    std::size_t count = 0;
    clap_cache_status_t status = clap_cache_plan_evictions(handle_, nullptr, 0, &count);
    if (status != CLAP_CACHE_OK && status != CLAP_CACHE_NO_CAPACITY) {
      check("clap_cache_plan_evictions", status);
    }
    evictions_.resize(count);
    if (count > 0) {
      check("clap_cache_plan_evictions",
            clap_cache_plan_evictions(handle_, evictions_.data(), count, &count));
    }
    count = 0;
    status = clap_cache_plan_anchor_boundaries(handle_, nullptr, 0, &count);
    if (status != CLAP_CACHE_OK && status != CLAP_CACHE_NO_CAPACITY) {
      check("clap_cache_plan_anchor_boundaries", status);
    }
    anchor_boundaries_.resize(count);
    if (count > 0) {
      check("clap_cache_plan_anchor_boundaries",
            clap_cache_plan_anchor_boundaries(
                handle_, anchor_boundaries_.data(), count, &count));
    }
    count = 0;
    status = clap_cache_plan_candidates(handle_, nullptr, 0, &count);
    if (status != CLAP_CACHE_OK && status != CLAP_CACHE_NO_CAPACITY) {
      check("clap_cache_plan_candidates", status);
    }
    candidates_.resize(count);
    if (count > 0) {
      check("clap_cache_plan_candidates",
            clap_cache_plan_candidates(handle_, candidates_.data(), count, &count));
    }
  }

  void cleanup() noexcept {
    if (!handle_) return;
    clap_cache_abort(owner_, handle_);
    clap_cache_plan_destroy(handle_);
    handle_ = nullptr;
  }

  clap_cache_t* owner_ = nullptr;
  clap_cache_plan_t* handle_ = nullptr;
  clap_cache_plan_view_t view_{};
  std::vector<clap_cache_slot_ref_t> evictions_;
  std::vector<uint64_t> anchor_boundaries_;
  std::vector<clap_cache_candidate_evaluation_t> candidates_;
};

class Coordinator {
 public:
  Coordinator(uint32_t slots, uint64_t min_reuse_tokens,
              uint64_t logical_token_capacity, uint32_t max_anchors,
              uint32_t hard_max_retained_entries = 0,
              uint64_t physical_byte_budget = 0,
              uint64_t high_watermark_bytes = 0,
              uint64_t low_watermark_bytes = 0,
              bool automatic_checkpoints = true,
              uint64_t checkpoint_minimum_tokens = 2048,
              uint64_t checkpoint_interval_tokens = 2048,
              uint32_t checkpoint_max = 8,
              uint32_t checkpoint_budget_basis_points = 2500,
              uint64_t checkpoint_budget_bytes = 0)
      : slot_count_(slots) {
    clap_cache_config_t config{};
    config.version = CLAP_CACHE_ABI_VERSION;
    config.struct_size = sizeof(config);
    config.slot_count = slots;
    config.max_anchors = max_anchors;
    config.min_reuse_tokens = min_reuse_tokens;
    config.logical_token_capacity = logical_token_capacity;
    config.automatic_checkpoint_mode = automatic_checkpoints ? 1 : 2;
    config.automatic_checkpoint_max = checkpoint_max;
    config.automatic_checkpoint_min_tokens = checkpoint_minimum_tokens;
    config.automatic_checkpoint_interval_tokens = checkpoint_interval_tokens;
    config.automatic_checkpoint_memory_basis_points = checkpoint_budget_basis_points;
    config.automatic_checkpoint_memory_cap_bytes = checkpoint_budget_bytes;
    clap_cache_retention_config_t retention{};
    retention.version = CLAP_CACHE_ABI_VERSION;
    retention.struct_size = sizeof(retention);
    retention.hard_max_retained_entries = hard_max_retained_entries == 0
        ? slots : hard_max_retained_entries;
    retention.physical_byte_budget = physical_byte_budget;
    retention.high_watermark_bytes = high_watermark_bytes;
    retention.low_watermark_bytes = low_watermark_bytes;
    check("clap_cache_create_with_retention",
          clap_cache_create_with_retention(&config, &retention, &handle_));
  }

  Coordinator(const Coordinator&) = delete;
  Coordinator& operator=(const Coordinator&) = delete;
  ~Coordinator() { clap_cache_destroy(handle_); }

  clap_cache_slot_ref_t register_slot() {
    clap_cache_slot_ref_t slot{};
    check("clap_cache_register_slot", clap_cache_register_slot(handle_, &slot));
    slot_count_ += 1;
    return slot;
  }

  Plan plan(const std::vector<int32_t>& tokens, const Identity& identity,
            uint64_t capabilities, uint64_t output_reserve,
            uint32_t result_state = CLAP_CACHE_SLOT_SESSION,
            const std::vector<uint8_t>& supplied_slot_capabilities = {},
            const std::vector<uint64_t>& stable_boundaries = {}) {
    clap_cache_labels_t labels{};
    labels.version = CLAP_CACHE_ABI_VERSION;
    labels.struct_size = sizeof(labels);
    labels.tenant = identity.tenant;
    labels.project = identity.project;
    labels.harness = identity.harness;
    labels.agent = identity.agent;
    labels.session = identity.session;
    labels.scope = identity.scope;
    labels.priority = identity.priority;
    labels.side_request = identity.side_request ? 1 : 0;

    clap_cache_request_t request{};
    request.version = CLAP_CACHE_ABI_VERSION;
    request.struct_size = sizeof(request);
    std::copy(identity.name_space.begin(), identity.name_space.end(),
              request.namespace_fingerprint);
    request.tokens = tokens.data();
    request.tokens_len = tokens.size();
    request.labels = labels;
    request.capabilities = capabilities;
    std::vector<uint8_t> default_slot_capabilities;
    const std::vector<uint8_t>* slot_capabilities = &supplied_slot_capabilities;
    if (slot_capabilities->empty()) {
      default_slot_capabilities.assign(
          slot_count_, CLAP_CACHE_SLOT_MATERIALIZED | CLAP_CACHE_SLOT_WRITABLE |
              CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM | CLAP_CACHE_SLOT_COPY);
      slot_capabilities = &default_slot_capabilities;
    }
    request.slot_capabilities = slot_capabilities->data();
    request.slot_capabilities_len = slot_capabilities->size();
    request.stable_boundaries = stable_boundaries.data();
    request.stable_boundaries_len = stable_boundaries.size();
    request.output_reserve = output_reserve;
    request.result_state = result_state;

    clap_cache_plan_t* plan = nullptr;
    check("clap_cache_plan", clap_cache_plan(handle_, &request, &plan));
    try {
      return Plan(handle_, plan);
    } catch (...) {
      clap_cache_plan_destroy(plan);
      throw;
    }
  }

  clap_cache_slot_info_t slot(uint32_t id) const {
    clap_cache_slot_info_t info{};
    check("clap_cache_get_slot", clap_cache_get_slot(handle_, id, &info));
    return info;
  }

  uint64_t advance(clap_cache_slot_ref_t slot, const int32_t* tokens,
                   std::size_t count, uint32_t state, bool busy,
                   uint64_t physical_bytes = 0) {
    uint64_t generation = 0;
    check("clap_cache_advance",
          clap_cache_advance(handle_, slot, tokens, count, state, busy ? 1 : 0,
                             physical_bytes, &generation));
    return generation;
  }

  void set_busy(clap_cache_slot_ref_t slot, bool busy) {
    check("clap_cache_set_busy", clap_cache_set_busy(handle_, slot, busy ? 1 : 0));
  }

  void set_anchor_protected(clap_cache_slot_ref_t slot, bool protected_anchor) {
    check("clap_cache_set_anchor_protected",
          clap_cache_set_anchor_protected(handle_, slot, protected_anchor ? 1 : 0));
  }

  uint64_t invalidate(clap_cache_slot_ref_t slot) {
    uint64_t generation = 0;
    check("clap_cache_invalidate", clap_cache_invalidate(handle_, slot, &generation));
    return generation;
  }

  uint64_t reset() {
    uint64_t epoch = 0;
    check("clap_cache_reset", clap_cache_reset(handle_, &epoch));
    return epoch;
  }

  clap_cache_telemetry_t telemetry() const {
    clap_cache_telemetry_t telemetry{};
    check("clap_cache_get_telemetry", clap_cache_get_telemetry(handle_, &telemetry));
    return telemetry;
  }

  clap_cache_retention_telemetry_t retention_telemetry() const {
    clap_cache_retention_telemetry_t telemetry{};
    check("clap_cache_get_retention_telemetry",
          clap_cache_get_retention_telemetry(handle_, &telemetry));
    return telemetry;
  }

 private:
  clap_cache_t* handle_ = nullptr;
  uint32_t slot_count_ = 0;
};

}  // namespace clap::llama_cache

#endif
