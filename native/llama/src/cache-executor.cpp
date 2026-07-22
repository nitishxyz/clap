#include "clap/llama/cache-executor.h"

#include <stdexcept>
#include <utility>

namespace clap::llama {

bool LlamaPhysicalCacheBackend::remove(int32_t sequence, int32_t begin, int32_t end) {
  if (!context_) throw std::runtime_error("physical cache context is unavailable");
  return llama_memory_seq_rm(llama_get_memory(context_), sequence, begin, end);
}

void LlamaPhysicalCacheBackend::copy(int32_t source, int32_t target,
                                     int32_t begin, int32_t end) {
  if (!context_) throw std::runtime_error("physical cache context is unavailable");
  llama_memory_seq_cp(llama_get_memory(context_), source, target, begin, end);
}

void LlamaPhysicalCacheBackend::clear(bool data) {
  if (!context_) throw std::runtime_error("physical cache context is unavailable");
  llama_memory_clear(llama_get_memory(context_), data);
}

CacheLease::CacheLease(CacheLease&& other) noexcept
    : owner_(other.owner_), slot_(other.slot_), generation_(other.generation_) {
  other.owner_ = nullptr;
}

CacheLease& CacheLease::operator=(CacheLease&& other) noexcept {
  if (this != &other) {
    release();
    owner_ = other.owner_;
    slot_ = other.slot_;
    generation_ = other.generation_;
    other.owner_ = nullptr;
  }
  return *this;
}

CacheLease::~CacheLease() {
  release();
}

void CacheLease::release() {
  if (!owner_) return;
  CacheExecutor* owner = owner_;
  owner_ = nullptr;
  owner->release(slot_, generation_);
}

CacheExecutor::CacheExecutor(CacheExecutorConfig config,
                             std::unique_ptr<PhysicalCacheBackend> backend)
    : backend_(std::move(backend)) {
  if (!backend_) throw std::invalid_argument("physical cache backend is required");
  if (config.slot_count == 0) throw std::invalid_argument("cache slot count must be positive");
  coordinator_ = std::make_unique<clap::llama_cache::Coordinator>(
      1, config.min_reuse_tokens, config.logical_token_capacity, config.max_anchors,
      config.hard_max_retained_entries, 0, 0, 0, config.automatic_checkpoints,
      config.checkpoint_minimum_tokens, config.checkpoint_interval_tokens,
      config.checkpoint_max, config.checkpoint_budget_basis_points,
      config.checkpoint_budget_bytes);
  slots_.resize(config.slot_count);
  slots_[0].generation = coordinator_->slot(0).generation;
  for (uint32_t slot = 1; slot < config.slot_count; ++slot) {
    const auto registered = coordinator_->register_slot();
    if (registered.slot != slot || registered.generation == 0) {
      throw std::runtime_error("cache coordinator returned unstable slot registration");
    }
    slots_[slot].generation = registered.generation;
  }
}

CacheAdmissionResult CacheExecutor::preview(const CacheAdmissionRequest& request) {
  auto plan = coordinator_->plan(request.tokens, request.identity, request.capabilities,
      request.output_reserve, request.result_state, request.slot_capabilities,
      request.stable_boundaries);
  const auto& view = plan.view();
  std::vector<uint32_t> evictions;
  evictions.reserve(plan.evictions().size());
  for (const auto& eviction : plan.evictions()) evictions.push_back(eviction.slot);
  CacheAdmissionResult result{
      view.operation, view.target.slot, view.target.generation,
      view.donor.slot, view.donor.generation, view.reuse_tokens,
      std::move(evictions), plan.anchor_boundaries()};
  plan.abort();
  return result;
}

CacheLease CacheExecutor::acquire(uint32_t slot_id) {
  if (slot_id >= slots_.size()) throw std::out_of_range("cache slot is out of range");
  auto& slot = slots_[slot_id];
  if (slot.busy) throw std::runtime_error("cache slot is already leased");
  clap_cache_slot_ref_t reference{slot_id, 0, slot.generation};
  coordinator_->set_busy(reference, true);
  slot.busy = true;
  return CacheLease(this, slot_id, slot.generation);
}

CacheSlotSnapshot CacheExecutor::slot(uint32_t slot_id) const {
  if (slot_id >= slots_.size()) throw std::out_of_range("cache slot is out of range");
  const auto& slot = slots_[slot_id];
  return {slot_id, slot.generation, slot.busy, slot.anchor, slot.tokens.size()};
}

void CacheExecutor::release(uint32_t slot_id, uint64_t generation) {
  if (slot_id >= slots_.size()) return;
  auto& slot = slots_[slot_id];
  if (!slot.busy || slot.generation != generation) return;
  coordinator_->set_busy({slot_id, 0, generation}, false);
  slot.busy = false;
}

uint64_t CacheExecutor::reset() {
  const uint64_t epoch = coordinator_->reset();
  backend_->clear(true);
  for (uint32_t slot_id = 0; slot_id < slots_.size(); ++slot_id) {
    const auto info = coordinator_->slot(slot_id);
    slots_[slot_id] = {};
    slots_[slot_id].generation = info.generation;
  }
  return epoch;
}

bool CacheExecutor::remove_sequence(int32_t sequence, int32_t begin, int32_t end) {
  return backend_->remove(sequence, begin, end);
}

void CacheExecutor::copy_sequence(int32_t source, int32_t target,
                                  int32_t begin, int32_t end) {
  backend_->copy(source, target, begin, end);
}

void CacheExecutor::clear_physical(bool data) {
  backend_->clear(data);
}

}  // namespace clap::llama
