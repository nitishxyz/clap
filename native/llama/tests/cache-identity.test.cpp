#include "clap/llama/cache-identity.h"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <string>

namespace {

nlohmann::json vector_identity() {
  return {
    {"version", 1}, {"generation", "sec_123e4567-e89b-42d3-a456-426614174000"},
    {"tenant_root", "6f64c827dbe30bac5d1d2b7e37c70ba23e012c793624af8d64222adac5112f44"},
    {"project_fingerprint", "f1482819163f613fa61f18a23b2aa510319dd4d8db62934fca3a1d7475b47c3b"},
    {"harness_fingerprint", "94bdbacbdf215f56a6e55920c34a287947d6b21bb1194208205e2b83eb302027"},
    {"agent_fingerprint", "d844f6c06cc42e03137243f37db076fcb86b602a409118a62001f67604d793e2"},
    {"session_fingerprint", "f269f6ff4a9cae43ffb8a8a7510e1c6a63669634fb08deca7bfc6dd304959b41"},
    {"scope", "session"},
    {"scope_fingerprint", "f269f6ff4a9cae43ffb8a8a7510e1c6a63669634fb08deca7bfc6dd304959b41"},
    {"namespace_fingerprint", "40d51ffb8e7b5c35e3a4519b9ce017396cc5c8d6c89a2c7652d087742b216329"},
    {"namespace_id", "4671675353754459189"}, {"priority", "background"},
    {"side_request", true},
    {"display", {{"namespace", "workspace"}, {"project", "payments"},
      {"harness", "coding-v4"}, {"agent", "reviewer"}, {"session", "session-99"}}},
    {"physical", {
      {"fingerprint", "5befd882f13033f62c62fd673e60b7758564412e14c6f18cbb799b541847ecb5"},
      {"backend", "llama"}, {"resolved_revision", "sha256:abc123"},
      {"model_artifact_fingerprint", std::string(64, 'd')},
      {"tokenizer_fingerprint", std::string(64, 'e')}, {"context_allocation", 8192},
      {"kv_format", "q8_0"}, {"unified_kv", false}, {"layout_version", 3},
    }},
  };
}

const clap::llama::PhysicalCacheDescriptor physical{"llama", 8192, "q8_0", false, 3};

}  // namespace

int main() {
  const auto parsed = clap::llama::parse_cache_identity(vector_identity(), physical);
  assert(parsed.authority.tenant == UINT64_C(0x6f64c827dbe30bac));
  assert(parsed.authority.project == UINT64_C(0xf1482819163f613f));
  assert(parsed.authority.harness == UINT64_C(0x94bdbacbdf215f56));
  assert(parsed.authority.agent == UINT64_C(0xd844f6c06cc42e03));
  assert(parsed.authority.session == UINT64_C(0xf269f6ff4a9cae43));
  assert(parsed.authority.scope == CLAP_CACHE_SCOPE_SESSION);
  assert(parsed.authority.priority == CLAP_CACHE_PRIORITY_BACKGROUND);
  assert(parsed.authority.side_request);
  assert(parsed.authority.name_space[0] == 0x40 && parsed.authority.name_space[31] == 0x29);
  assert(parsed.display.name_space == "workspace");

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
