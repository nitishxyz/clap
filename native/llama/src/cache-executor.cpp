#include "clap/llama/cache-executor.h"

#include <exception>
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
      view.reuse_tokens, 0, view.decision_us, 0, view.anchor_tokens,
      view.has_donor != 0, std::move(evictions), plan.anchor_boundaries(), plan.candidates()};
  plan.abort();
  return result;
}

CacheAnchorResult CacheExecutor::create_anchor(
    const std::vector<int32_t>& tokens, const clap::llama_cache::Identity& identity,
    uint32_t source_slot, bool protect) {
  if (source_slot >= slots_.size()) throw std::out_of_range("cache source slot is out of range");
  std::vector<uint8_t> capabilities;
  capabilities.reserve(slots_.size());
  for (const auto& slot : slots_) {
    uint8_t flags = slot.busy ? 0 : CLAP_CACHE_SLOT_WRITABLE;
    if (!slot.tokens.empty()) flags |= CLAP_CACHE_SLOT_MATERIALIZED | CLAP_CACHE_SLOT_COPY;
    capabilities.push_back(flags);
  }
  auto plan = coordinator_->plan(tokens, identity,
      CLAP_CACHE_CAP_WHOLE_STATE_COPY | CLAP_CACHE_CAP_SAFE_BUSY_DONOR |
          CLAP_CACHE_CAP_PROMPT_BOUNDARY_SNAPSHOT,
      0, CLAP_CACHE_SLOT_ANCHOR, capabilities);
  const auto view = plan.view();
  if (view.operation == CLAP_CACHE_OPERATION_NOOP) {
    plan.commit(tokens.size(), CLAP_CACHE_SLOT_ANCHOR);
    return {true, true, view.target.slot, view.target.generation, {}};
  }
  if (view.target.slot >= slots_.size()) {
    plan.abort();
    return {false, false, view.target.slot, 0, {}};
  }
  const uint32_t target = view.target.slot;
  try {
    backend_->remove(target, -1, -1);
    backend_->copy(source_slot, target, -1, -1);
    plan.commit(tokens.size(), CLAP_CACHE_SLOT_ANCHOR);
    const auto info = coordinator_->slot(target);
    slots_[target].tokens = tokens;
    slots_[target].is_anchor = true;
    slots_[target].generation = info.generation;
    if (protect) coordinator_->set_anchor_protected({target, 0, info.generation}, true);
    std::vector<uint32_t> evictions;
    for (const auto& victim : plan.evictions()) {
      if (victim.slot >= slots_.size()) continue;
      if (victim.slot != target) {
        backend_->remove(victim.slot, -1, -1);
        slots_[victim.slot] = {};
        slots_[victim.slot].generation = coordinator_->slot(victim.slot).generation;
      }
      evictions.push_back(victim.slot);
    }
    return {true, false, target, info.generation, std::move(evictions)};
  } catch (...) {
    try { plan.abort(); } catch (...) {}
    try { backend_->remove(target, -1, -1); } catch (...) {}
    slots_[target] = {};
    try { slots_[target].generation = coordinator_->slot(target).generation; } catch (...) {}
    throw;
  }
}

CacheAdmissionResult CacheExecutor::admit(const CacheAdmissionRequest& request) {
  std::vector<uint8_t> capabilities;
  capabilities.reserve(slots_.size());
  for (const auto& slot : slots_) {
    uint8_t flags = slot.busy ? 0 : CLAP_CACHE_SLOT_WRITABLE;
    if (!slot.tokens.empty()) {
      flags |= CLAP_CACHE_SLOT_MATERIALIZED | CLAP_CACHE_SLOT_COPY;
      if (!request.hybrid) flags |= CLAP_CACHE_SLOT_PARTIAL_SUFFIX_TRIM;
    }
    capabilities.push_back(flags);
  }
  auto plan = coordinator_->plan(request.tokens, request.identity, request.capabilities,
      request.output_reserve, request.result_state,
      request.slot_capabilities.empty() ? capabilities : request.slot_capabilities,
      request.stable_boundaries);
  const auto view = plan.view();
  const std::size_t target = view.target.slot;
  const std::size_t donor = view.has_donor ? view.donor.slot : SIZE_MAX;
  if (target >= slots_.size() || (donor != SIZE_MAX && donor >= slots_.size()) ||
      view.target.generation != slots_[target].generation ||
      (donor != SIZE_MAX && view.donor.generation != slots_[donor].generation)) {
    plan.abort();
    for (uint32_t slot = 0; slot < slots_.size(); ++slot) {
      slots_[slot].generation = coordinator_->slot(slot).generation;
    }
    throw std::runtime_error("cache coordinator returned an invalid slot");
  }
  std::size_t resident = 0;
  bool committed = false;
  try {
    auto& slot = slots_[target];
    if (view.operation != CLAP_CACHE_OPERATION_CONTINUE) {
      backend_->remove(static_cast<int32_t>(target), -1, -1);
      slot.tokens.clear();
      slot.is_anchor = false;
    }
    if (view.operation == CLAP_CACHE_OPERATION_CONTINUE) {
      resident = std::min<std::size_t>(view.reuse_tokens, request.tokens.size() - 1);
      if (resident == 0 || !backend_->remove(static_cast<int32_t>(target),
          static_cast<int32_t>(resident), -1)) {
        throw std::runtime_error("coordinator-selected continuation could not be materialized");
      }
    } else if (view.operation == CLAP_CACHE_OPERATION_BRANCH) {
      resident = std::min<std::size_t>(view.reuse_tokens, request.tokens.size() - 1);
      if (resident > 0) {
        if (request.hybrid && resident == slots_[donor].tokens.size()) {
          backend_->copy(static_cast<int32_t>(donor), static_cast<int32_t>(target), -1, -1);
        } else if (!request.hybrid) {
          backend_->copy(static_cast<int32_t>(donor), static_cast<int32_t>(target),
                         0, static_cast<int32_t>(resident));
        } else {
          throw std::runtime_error("coordinator-selected branch could not be materialized");
        }
      }
    } else if (view.operation == CLAP_CACHE_OPERATION_RESTORE &&
               view.reuse_tokens < request.tokens.size()) {
      resident = static_cast<std::size_t>(view.reuse_tokens);
      backend_->copy(static_cast<int32_t>(donor), static_cast<int32_t>(target), -1, -1);
    }
    if (request.commit_hook) request.commit_hook();
    const auto decision = plan.commit(resident, request.result_state);
    committed = true;
    const auto info = coordinator_->slot(static_cast<uint32_t>(target));
    slot.tokens.assign(request.tokens.begin(), request.tokens.begin() +
        static_cast<std::ptrdiff_t>(resident));
    slot.generation = info.generation;
    slot.busy = true;

    std::vector<uint32_t> evictions;
    for (const auto& victim : plan.evictions()) {
      if (victim.slot >= slots_.size()) continue;
      if (victim.slot != target) {
        backend_->remove(static_cast<int32_t>(victim.slot), -1, -1);
        slots_[victim.slot] = {};
        slots_[victim.slot].generation = coordinator_->slot(victim.slot).generation;
      }
      evictions.push_back(victim.slot);
    }
    return {decision.operation, static_cast<uint32_t>(target), info.generation,
      decision.donor_slot, decision.has_donor ? coordinator_->slot(decision.donor_slot).generation : 0,
      resident, decision.planned_reuse_tokens, decision.realized_reuse_tokens,
      decision.decision_us, decision.scope, view.anchor_tokens, decision.has_donor != 0,
      std::move(evictions), plan.anchor_boundaries(), plan.candidates()};
  } catch (...) {
    const auto failure = std::current_exception();
    if (!committed) {
      try { plan.abort(); } catch (...) {}
    }
    try { backend_->remove(static_cast<int32_t>(target), -1, -1); } catch (...) {}
    for (uint32_t slot = 0; slot < slots_.size(); ++slot) {
      try {
        slots_[slot].generation = coordinator_->slot(slot).generation;
      } catch (...) {
        slots_[slot].generation = 0;
      }
    }
    slots_[target].tokens.clear();
    slots_[target].busy = false;
    slots_[target].is_anchor = false;
    std::rethrow_exception(failure);
  }
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
  return {slot_id, slot.generation, slot.busy, slot.is_anchor, slot.tokens.size()};
}

void CacheExecutor::release(uint32_t slot_id, uint64_t generation) noexcept {
  if (slot_id >= slots_.size()) return;
  auto& slot = slots_[slot_id];
  if (!slot.busy || slot.generation != generation) return;
  try { coordinator_->set_busy({slot_id, 0, generation}, false); } catch (...) { return; }
  slot.busy = false;
}

uint64_t CacheExecutor::advance(uint32_t slot_id, uint64_t generation,
                                const int32_t* tokens, std::size_t count,
                                uint32_t state, bool busy) {
  try {
    const uint64_t next = coordinator_->advance(
        {slot_id, 0, generation}, tokens, count, state, busy);
    slots_[slot_id].generation = next;
    slots_[slot_id].busy = busy;
    slots_[slot_id].tokens.insert(slots_[slot_id].tokens.end(), tokens, tokens + count);
    return next;
  } catch (...) {
    invalidate_and_clear(slot_id, generation);
    throw;
  }
}

uint64_t CacheExecutor::invalidate_and_clear(uint32_t slot_id, uint64_t generation,
                                             bool keep_busy) {
  if (slot_id >= slots_.size()) return 0;
  try { backend_->remove(slot_id, -1, -1); } catch (...) {}
  uint64_t next = coordinator_->invalidate({slot_id, 0, generation});
  slots_[slot_id] = {};
  slots_[slot_id].generation = next;
  if (keep_busy) {
    coordinator_->set_busy({slot_id, 0, next}, true);
    slots_[slot_id].busy = true;
  }
  return next;
}

uint64_t CacheExecutor::reset_for_retry(uint32_t active_slot) {
  const uint64_t epoch = reset();
  const auto info = coordinator_->slot(active_slot);
  slots_[active_slot].generation = info.generation;
  coordinator_->set_busy({active_slot, 0, info.generation}, true);
  slots_[active_slot].busy = true;
  return info.generation;
}

clap_cache_telemetry_t CacheExecutor::telemetry() const { return coordinator_->telemetry(); }
clap_cache_retention_telemetry_t CacheExecutor::retention_telemetry() const {
  return coordinator_->retention_telemetry();
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
