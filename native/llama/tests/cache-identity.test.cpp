#include "clap/llama/cache-identity.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <fstream>
#include <string>

namespace {

nlohmann::json vector_identity() {
  std::ifstream input(CLAP_CACHE_IDENTITY_VECTOR_PATH);
  assert(input.good());
  nlohmann::json fixture;
  input >> fixture;
  return fixture.at("identity");
}

nlohmann::json vector_expected() {
  std::ifstream input(CLAP_CACHE_IDENTITY_VECTOR_PATH);
  nlohmann::json fixture;
  input >> fixture;
  return fixture.at("expected");
}

const clap::llama::PhysicalCacheDescriptor physical{"llama", 8192, "q8_0", false, 3};

}  // namespace

int main() {
  static_assert(CLAP_CACHE_PRIORITY_BACKGROUND == 0);
  static_assert(CLAP_CACHE_PRIORITY_NORMAL == 1);
  static_assert(CLAP_CACHE_PRIORITY_INTERACTIVE == 2);
  const auto parsed = clap::llama::parse_cache_identity(vector_identity(), physical);
  const auto expected = vector_expected();
  assert(parsed.authority.tenant == std::stoull(expected["tenant_u64_hex"].get<std::string>(), nullptr, 16));
  assert(parsed.authority.project == std::stoull(expected["project_u64_hex"].get<std::string>(), nullptr, 16));
  assert(parsed.authority.harness == std::stoull(expected["harness_u64_hex"].get<std::string>(), nullptr, 16));
  assert(parsed.authority.agent == std::stoull(expected["agent_u64_hex"].get<std::string>(), nullptr, 16));
  assert(parsed.authority.session == std::stoull(expected["session_u64_hex"].get<std::string>(), nullptr, 16));
  assert(parsed.authority.scope == CLAP_CACHE_SCOPE_SESSION);
  assert(parsed.authority.priority == CLAP_CACHE_PRIORITY_NORMAL);
  assert(parsed.authority.side_request);
  assert(parsed.authority.name_space[0] == 0x40 && parsed.authority.name_space[31] == 0x29);
  assert(parsed.display.name_space == "workspace");

  auto omitted_priority = vector_identity();
  omitted_priority.erase("priority");
  assert(clap::llama::parse_cache_identity(omitted_priority, physical).authority.priority ==
      CLAP_CACHE_PRIORITY_NORMAL);

  auto labels = vector_identity();
  labels["display"] = {{"namespace", "untrusted-other"}, {"session", "raw-label"}};
  const auto relabeled = clap::llama::parse_cache_identity(labels, physical);
  assert(relabeled.authority.name_space == parsed.authority.name_space);
  assert(relabeled.authority.tenant == parsed.authority.tenant);
  assert(relabeled.authority.session == parsed.authority.session);

  try {
    clap::llama::parse_cache_identity(nlohmann::json::object(), physical);
    assert(false);
  } catch (const clap::llama::CacheIdentityError&) {}

  auto malformed = vector_identity();
  malformed["tenant_root"] = std::string(64, 'A');
  try {
    clap::llama::parse_cache_identity(malformed, physical);
    assert(false);
  } catch (const clap::llama::CacheIdentityError&) {}

  auto generation = vector_identity();
  generation["generation"] = "";
  try {
    clap::llama::parse_cache_identity(generation, physical);
    assert(false);
  } catch (const clap::llama::CacheIdentityError&) {}

  auto namespace_mismatch = vector_identity();
  namespace_mismatch["namespace_id"] = "1";
  try {
    clap::llama::parse_cache_identity(namespace_mismatch, physical);
    assert(false);
  } catch (const clap::llama::CacheIdentityError&) {}

  auto physical_mismatch = vector_identity();
  physical_mismatch["physical"]["context_allocation"] = 4096;
  try {
    clap::llama::parse_cache_identity(physical_mismatch, physical);
    assert(false);
  } catch (const clap::llama::CacheIdentityError&) {}
}
