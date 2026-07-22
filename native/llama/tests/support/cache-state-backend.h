#pragma once

#include "clap/llama/cache-executor.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clap::llama::test {

struct CacheMutation {
  std::string operation;
  int32_t source = -1;
  int32_t target = -1;
  int32_t begin = -1;
  int32_t end = -1;
};

class CacheStateBackend final : public PhysicalCacheBackend {
 public:
  explicit CacheStateBackend(std::size_t slot_count) : slots_(slot_count) {}

  bool remove(int32_t sequence, int32_t begin, int32_t end) override {
    CacheMutation mutation{"remove", sequence, sequence, begin, end};
    before(mutation);
    if (fail_next_remove_) {
      fail_next_remove_ = false;
      throw std::runtime_error("test physical remove failed");
    }
    auto& tokens = at(sequence);
    if (begin < 0) {
      tokens.clear();
    } else {
      const std::size_t first = std::min<std::size_t>(begin, tokens.size());
      const std::size_t last = end < 0
          ? tokens.size()
          : std::min<std::size_t>(static_cast<std::size_t>(end), tokens.size());
      if (first < last) tokens.erase(tokens.begin() + first, tokens.begin() + last);
    }
    mutations.push_back(std::move(mutation));
    return true;
  }

  void copy(int32_t source, int32_t target, int32_t begin, int32_t end) override {
    CacheMutation mutation{"copy", source, target, begin, end};
    before(mutation);
    if (fail_next_copy_) {
      fail_next_copy_ = false;
      throw std::runtime_error("test physical copy failed");
    }
    const auto& donor = at(source);
    const std::size_t first = begin < 0 ? 0 : std::min<std::size_t>(begin, donor.size());
    const std::size_t last = end < 0
        ? donor.size()
        : std::min<std::size_t>(static_cast<std::size_t>(end), donor.size());
    at(target).assign(donor.begin() + first, donor.begin() + std::max(first, last));
    mutations.push_back(std::move(mutation));
  }

  void clear(bool data) override {
    CacheMutation mutation{"clear", -1, -1, data ? 1 : 0, -1};
    before(mutation);
    for (auto& slot : slots_) slot.clear();
    mutations.push_back(std::move(mutation));
  }

  void append(uint32_t slot, const std::vector<int32_t>& tokens) {
    auto& resident = slots_.at(slot);
    resident.insert(resident.end(), tokens.begin(), tokens.end());
  }

  const std::vector<int32_t>& slot(uint32_t slot) const { return slots_.at(slot); }
  void fail_next_remove() noexcept { fail_next_remove_ = true; }
  void fail_next_copy() noexcept { fail_next_copy_ = true; }

  std::function<void(const CacheMutation&)> before_mutation;
  std::vector<CacheMutation> mutations;

 private:
  void before(const CacheMutation& mutation) {
    if (before_mutation) before_mutation(mutation);
  }

  std::vector<int32_t>& at(int32_t slot) {
    if (slot < 0) throw std::out_of_range("negative physical cache slot");
    return slots_.at(static_cast<std::size_t>(slot));
  }

  const std::vector<int32_t>& at(int32_t slot) const {
    if (slot < 0) throw std::out_of_range("negative physical cache slot");
    return slots_.at(static_cast<std::size_t>(slot));
  }

  std::vector<std::vector<int32_t>> slots_;
  bool fail_next_remove_ = false;
  bool fail_next_copy_ = false;
};

}  // namespace clap::llama::test
