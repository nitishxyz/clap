#pragma once

#include <cstddef>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

namespace clap::llama {

inline constexpr std::size_t kMaxStructuredOutputSchemaBytes = 64 * 1024;
inline constexpr std::size_t kMaxStructuredOutputSchemaDepth = 32;
inline constexpr std::size_t kMaxStructuredOutputSchemaProperties = 1024;

struct StructuredOutputConstraint {
  std::string kind;
  std::string strength;
  nlohmann::ordered_json schema;
  std::string grammar;
};

std::optional<StructuredOutputConstraint> parse_structured_output(
    const nlohmann::json& request);

}  // namespace clap::llama
