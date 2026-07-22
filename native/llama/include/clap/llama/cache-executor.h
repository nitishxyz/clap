#pragma once

#include "cache-adapter.h"
#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace clap::llama {

class PhysicalCacheBackend {
 public:
  virtual ~PhysicalCacheBackend() = default;
  virtual bool remove(int32_t sequence, int32_t begin, int32_t end) = 0;
  virtual void copy(int32_t source, int32_t target, int32_t begin, int32_t end) = 0;
  virtual void clear(bool data) = 0;
};

class LlamaPhysicalCacheBackend final : public PhysicalCacheBackend {
 public:
  explicit LlamaPhysicalCacheBackend(llama_context* context) : context_(context) {}

  bool remove(int32_t sequence, int32_t begin, int32_t end) override;
  void copy(int32_t source, int32_t target, int32_t begin, int32_t end) override;
  void clear(bool data) override;

 private:
  llama_context* context_;
};

struct CacheExecutorConfig {
  uint32_t slot_count = 1;
  uint64_t min_reuse_tokens = 16;
  uint64_t logical_token_capacity = 0;
  uint32_t max_anchors = 0;
  uint32_t hard_max_retained_entries = 0;
  bool automatic_checkpoints = true;
  uint64_t checkpoint_minimum_tokens = 2048;
  uint64_t checkpoint_interval_tokens = 2048;
  uint32_t checkpoint_max = 8;
  uint32_t checkpoint_budget_basis_points = 2500;
  uint64_t checkpoint_budget_bytes = 0;
};

struct CacheAdmissionRequest {
  const std::vector<int32_t> tokens;
  const clap::llama_cache::Identity identity;
  const uint64_t capabilities;
  const uint64_t output_reserve;
  const uint32_t result_state;
  const std::vector<uint8_t> slot_capabilities;
  const std::vector<uint64_t> stable_boundaries;
  const bool hybrid = false;
  const std::function<void()> commit_hook = {};
};

struct CacheAdmissionResult {
  const uint32_t operation;
  const uint32_t target_slot;
  const uint64_t target_generation;
  const uint32_t donor_slot;
  const uint64_t donor_generation;
  const uint64_t reuse_tokens;
  const uint64_t planned_reuse_tokens;
  const uint64_t realized_reuse_tokens;
  const uint64_t decision_us;
  const uint32_t scope;
  const uint64_t anchor_tokens;
  const bool has_donor;
  const std::vector<uint32_t> eviction_slots;
  const std::vector<uint64_t> anchor_boundaries;
  const std::vector<clap_cache_candidate_evaluation_t> candidates;
};

struct CacheSlotSnapshot {
  const uint32_t id;
  const uint64_t generation;
  const bool busy;
  const bool anchor;
  const std::size_t resident_tokens;
};

class CacheExecutor;

class CacheLease {
 public:
  CacheLease() = default;
  CacheLease(const CacheLease&) = delete;
  CacheLease& operator=(const CacheLease&) = delete;
  CacheLease(CacheLease&& other) noexcept;
  CacheLease& operator=(CacheLease&& other) noexcept;
  ~CacheLease();

  explicit operator bool() const noexcept { return owner_ != nullptr; }
  uint32_t slot() const noexcept { return slot_; }
  uint64_t generation() const noexcept { return generation_; }
  void release();

 private:
  friend class CacheExecutor;
  CacheLease(CacheExecutor* owner, uint32_t slot, uint64_t generation)
      : owner_(owner), slot_(slot), generation_(generation) {}

  CacheExecutor* owner_ = nullptr;
  uint32_t slot_ = 0;
  uint64_t generation_ = 0;
};

class CacheExecutor {
 public:
  struct Slot {
    uint64_t generation = 0;
    uint64_t last_used = 0;
    bool busy = false;
    bool is_anchor = false;
    std::vector<int32_t> tokens;
  };

  CacheExecutor(CacheExecutorConfig config, std::unique_ptr<PhysicalCacheBackend> backend);
  CacheExecutor(const CacheExecutor&) = delete;
  CacheExecutor& operator=(const CacheExecutor&) = delete;

  CacheAdmissionResult preview(const CacheAdmissionRequest& request);
  CacheAdmissionResult admit(const CacheAdmissionRequest& request);
  CacheLease acquire(uint32_t slot);
  CacheSlotSnapshot slot(uint32_t slot) const;
  std::size_t slot_count() const noexcept { return slots_.size(); }
  uint64_t reset();
  clap::llama_cache::Coordinator& coordinator() { return *coordinator_; }
  const clap::llama_cache::Coordinator& coordinator() const { return *coordinator_; }
  std::vector<Slot>& slots() noexcept { return slots_; }
  const std::vector<Slot>& slots() const noexcept { return slots_; }

  bool remove_sequence(int32_t sequence, int32_t begin, int32_t end);
  void copy_sequence(int32_t source, int32_t target, int32_t begin, int32_t end);
  void clear_physical(bool data);

 private:
  friend class CacheLease;
  void release(uint32_t slot, uint64_t generation);

  std::unique_ptr<PhysicalCacheBackend> backend_;
  std::unique_ptr<clap::llama_cache::Coordinator> coordinator_;
  std::vector<Slot> slots_;
};

}  // namespace clap::llama
