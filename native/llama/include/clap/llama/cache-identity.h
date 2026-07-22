#pragma once

#include "cache-adapter.h"

#include <cstdint>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace clap::llama {

class ModelRuntime;

struct PhysicalCacheDescriptor {
  std::string backend;
  int64_t context_allocation = 0;
  std::string kv_format;
  bool unified_kv = false;
  int64_t layout_version = 0;
};

struct CacheIdentityDisplay {
  std::string name_space;
  std::string project;
  std::string harness;
  std::string agent;
  std::string session;
};

struct ParsedCacheIdentity {
  clap::llama_cache::Identity authority;
  CacheIdentityDisplay display;
  std::string generation;
  std::string physical_fingerprint;
};

struct CacheIdentityError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

PhysicalCacheDescriptor physical_cache_descriptor(const ModelRuntime& runtime);
ParsedCacheIdentity parse_cache_identity(const nlohmann::json& value,
                                         const PhysicalCacheDescriptor& expected);
ParsedCacheIdentity parse_cache_identity(const nlohmann::json& value,
                                         const ModelRuntime& runtime);

}  // namespace clap::llama
