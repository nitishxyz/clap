#include "clap/llama/cache-identity.h"

#include "clap/llama/model-runtime.h"

#include <array>
#include <limits>
#include <optional>
#include <set>

namespace clap::llama {
namespace {

using Digest = std::array<uint8_t, 32>;

void exact_keys(const nlohmann::json& value, const std::set<std::string>& allowed,
                const char* name) {
  if (!value.is_object()) throw CacheIdentityError(std::string(name) + " must be an object");
  for (const auto& [key, unused] : value.items()) {
    (void)unused;
    if (!allowed.count(key)) throw CacheIdentityError(std::string(name) + " contains unknown field: " + key);
  }
}

std::string required_string(const nlohmann::json& value, const char* key,
                            std::size_t maximum = std::numeric_limits<std::size_t>::max()) {
  if (!value.contains(key) || !value[key].is_string()) {
    throw CacheIdentityError(std::string("cache_identity.") + key + " must be a string");
  }
  const std::string result = value[key].get<std::string>();
  if (result.empty() || result.size() > maximum) {
    throw CacheIdentityError(std::string("cache_identity.") + key + " has invalid length");
  }
  return result;
}

Digest digest(const nlohmann::json& value, const char* key) {
  const std::string hex = required_string(value, key, 64);
  if (hex.size() != 64) throw CacheIdentityError(std::string("cache_identity.") + key + " must be 64 lowercase hex characters");
  Digest result{};
  for (std::size_t index = 0; index < result.size(); ++index) {
    const char high = hex[index * 2];
    const char low = hex[index * 2 + 1];
    auto nibble = [](char character) -> uint8_t {
      if (character >= '0' && character <= '9') return static_cast<uint8_t>(character - '0');
      if (character >= 'a' && character <= 'f') return static_cast<uint8_t>(character - 'a' + 10);
      throw CacheIdentityError("cache identity fingerprints must use lowercase hexadecimal");
    };
    result[index] = static_cast<uint8_t>((nibble(high) << 4) | nibble(low));
  }
  return result;
}

uint64_t reduce(const Digest& value) {
  uint64_t result = 0;
  for (std::size_t index = 0; index < 8; ++index) result = (result << 8) | value[index];
  return result == 0 ? 1 : result;
}

std::optional<Digest> optional_full_digest(const nlohmann::json& value, const char* key) {
  return value.contains(key) ? std::optional<Digest>(digest(value, key)) : std::nullopt;
}

uint64_t decimal_u64(const std::string& value) {
  if (value.empty() || value[0] == '0') throw CacheIdentityError("cache_identity.namespace_id must be a nonzero decimal u64");
  uint64_t result = 0;
  for (const char character : value) {
    if (character < '0' || character > '9') throw CacheIdentityError("cache_identity.namespace_id must be a nonzero decimal u64");
    const uint64_t digit = static_cast<uint64_t>(character - '0');
    if (result > (UINT64_MAX - digit) / 10) throw CacheIdentityError("cache_identity.namespace_id exceeds u64");
    result = result * 10 + digit;
  }
  if (result == 0) throw CacheIdentityError("cache_identity.namespace_id must be nonzero");
  return result;
}

std::string display(const nlohmann::json& value, const char* key) {
  if (!value.contains(key)) return "";
  if (!value[key].is_string() || value[key].get_ref<const std::string&>().empty() ||
      value[key].get_ref<const std::string&>().size() > 128) {
    throw CacheIdentityError(std::string("cache_identity.display.") + key + " is invalid");
  }
  return value[key].get<std::string>();
}

}  // namespace

PhysicalCacheDescriptor physical_cache_descriptor(const ModelRuntime& runtime) {
  return {"llama", runtime.backend_allocation_cap(), runtime.kv_format(),
          runtime.unified_kv(), 1};
}

ParsedCacheIdentity parse_cache_identity(const nlohmann::json& value,
                                         const ModelRuntime& runtime) {
  return parse_cache_identity(value, physical_cache_descriptor(runtime));
}

ParsedCacheIdentity parse_cache_identity(const nlohmann::json& value,
                                         const PhysicalCacheDescriptor& expected) {
  exact_keys(value, {"version", "generation", "tenant_root", "project_fingerprint",
      "harness_fingerprint", "agent_fingerprint", "session_fingerprint", "scope",
      "scope_fingerprint", "namespace_fingerprint", "namespace_id", "priority",
      "side_request", "display", "physical"}, "cache_identity");
  if (!value.contains("version") || !value["version"].is_number_integer() ||
      value["version"].get<int64_t>() != 1) throw CacheIdentityError("cache_identity.version must be 1");
  ParsedCacheIdentity result;
  result.generation = required_string(value, "generation", 64);
  const Digest tenant = digest(value, "tenant_root");
  const Digest scope_digest = digest(value, "scope_fingerprint");
  const Digest namespace_digest = digest(value, "namespace_fingerprint");
  const auto project = optional_full_digest(value, "project_fingerprint");
  const auto harness = optional_full_digest(value, "harness_fingerprint");
  const auto agent = optional_full_digest(value, "agent_fingerprint");
  const auto session = optional_full_digest(value, "session_fingerprint");
  result.authority.name_space = namespace_digest;
  result.authority.tenant = reduce(tenant);
  result.authority.project = project ? reduce(*project) : 0;
  result.authority.harness = harness ? reduce(*harness) : 0;
  result.authority.agent = agent ? reduce(*agent) : 0;
  result.authority.session = session ? reduce(*session) : 0;
  if (decimal_u64(required_string(value, "namespace_id", 20)) != reduce(namespace_digest)) {
    throw CacheIdentityError("cache_identity.namespace_id does not match namespace_fingerprint");
  }
  const std::string scope = required_string(value, "scope", 16);
  if (scope == "tenant") { result.authority.scope = CLAP_CACHE_SCOPE_TENANT; if (scope_digest != tenant) throw CacheIdentityError("tenant scope fingerprint mismatch"); }
  else if (scope == "project") { result.authority.scope = CLAP_CACHE_SCOPE_PROJECT; if (!project || scope_digest != *project) throw CacheIdentityError("project scope fingerprint mismatch"); }
  else if (scope == "harness") { result.authority.scope = CLAP_CACHE_SCOPE_HARNESS; if (!harness || scope_digest != *harness) throw CacheIdentityError("harness scope fingerprint mismatch"); }
  else if (scope == "agent") { result.authority.scope = CLAP_CACHE_SCOPE_AGENT; if (!agent || scope_digest != *agent) throw CacheIdentityError("agent scope fingerprint mismatch"); }
  else if (scope == "session") { result.authority.scope = CLAP_CACHE_SCOPE_SESSION; if (!session || scope_digest != *session) throw CacheIdentityError("session scope fingerprint mismatch"); }
  else throw CacheIdentityError("cache_identity.scope is invalid");
  const std::string priority = value.contains("priority")
      ? required_string(value, "priority", 16) : "normal";
  if (priority == "interactive") result.authority.priority = CLAP_CACHE_PRIORITY_INTERACTIVE;
  else if (priority == "normal") result.authority.priority = CLAP_CACHE_PRIORITY_NORMAL;
  else if (priority == "background") result.authority.priority = CLAP_CACHE_PRIORITY_BACKGROUND;
  else throw CacheIdentityError("cache_identity.priority is invalid");
  if (!value.contains("side_request") || !value["side_request"].is_boolean()) throw CacheIdentityError("cache_identity.side_request must be boolean");
  result.authority.side_request = value["side_request"].get<bool>();

  if (!value.contains("display")) throw CacheIdentityError("cache_identity.display is required");
  exact_keys(value["display"], {"namespace", "project", "harness", "agent", "session"}, "cache_identity.display");
  result.display = {display(value["display"], "namespace"), display(value["display"], "project"),
      display(value["display"], "harness"), display(value["display"], "agent"),
      display(value["display"], "session")};

  if (!value.contains("physical")) throw CacheIdentityError("cache_identity.physical is required");
  const auto& physical = value["physical"];
  exact_keys(physical, {"fingerprint", "backend", "resolved_revision",
      "model_artifact_fingerprint", "tokenizer_fingerprint", "context_allocation",
      "kv_format", "unified_kv", "layout_version"}, "cache_identity.physical");
  result.physical_fingerprint = required_string(physical, "fingerprint", 64);
  (void)digest(physical, "fingerprint");
  (void)digest(physical, "model_artifact_fingerprint");
  (void)digest(physical, "tokenizer_fingerprint");
  required_string(physical, "resolved_revision", 256);
  const std::string backend = required_string(physical, "backend", 16);
  const std::string kv = required_string(physical, "kv_format", 64);
  if (!physical.contains("context_allocation") || !physical["context_allocation"].is_number_integer() ||
      !physical.contains("layout_version") || !physical["layout_version"].is_number_integer() ||
      !physical.contains("unified_kv") || !physical["unified_kv"].is_boolean()) {
    throw CacheIdentityError("cache_identity physical descriptor types are invalid");
  }
  if (backend != expected.backend || physical["context_allocation"].get<int64_t>() != expected.context_allocation ||
      kv != expected.kv_format || physical["unified_kv"].get<bool>() != expected.unified_kv ||
      physical["layout_version"].get<int64_t>() != expected.layout_version) {
    throw CacheIdentityError("cache_identity physical model domain does not match loaded runtime");
  }
  return result;
}

}  // namespace clap::llama
