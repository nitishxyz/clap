#include "clap/llama/structured-output.h"

#include "clap/llama/protocol.h"
#include "json-schema-to-grammar.h"

#include <exception>

namespace clap::llama {
namespace {

[[noreturn]] void invalid(const std::string& message) {
  throw RequestError("invalid_structured_output", message);
}

void validate_schema(const nlohmann::ordered_json& value, std::size_t depth,
                     std::size_t& property_count) {
  if (depth > kMaxStructuredOutputSchemaDepth) {
    invalid("structured output schema exceeds the maximum depth of 32");
  }
  if (value.is_array()) {
    for (const auto& child : value) validate_schema(child, depth + 1, property_count);
    return;
  }
  if (!value.is_object()) return;

  const auto ref = value.find("$ref");
  if (ref != value.end() && ref->is_string() && ref->get_ref<const std::string&>().rfind("#", 0) != 0) {
    invalid("remote structured output schema references are not allowed");
  }
  const auto properties = value.find("properties");
  if (properties != value.end() && properties->is_object()) {
    property_count += properties->size();
    if (property_count > kMaxStructuredOutputSchemaProperties) {
      invalid("structured output schema exceeds the limit of 1024 properties");
    }
  }
  for (const auto& item : value.items()) validate_schema(item.value(), depth + 1, property_count);
}

std::string require_string(const nlohmann::json& object, const char* key) {
  const auto value = object.find(key);
  if (value == object.end() || !value->is_string()) {
    invalid(std::string("structured_output.") + key + " must be a string");
  }
  return value->get<std::string>();
}

}  // namespace

std::optional<StructuredOutputConstraint> parse_structured_output(
    const nlohmann::json& request) {
  const auto field = request.find("structured_output");
  if (field == request.end()) return std::nullopt;
  if (!field->is_object()) invalid("structured_output must be an object");

  const std::string kind = require_string(*field, "kind");
  const std::string strength = require_string(*field, "strength");
  if (kind != "json_object" && kind != "json_schema") {
    invalid("structured_output.kind must be json_object or json_schema");
  }
  if (strength != "best_effort" && strength != "required") {
    invalid("structured_output.strength must be best_effort or required");
  }
  for (const auto& item : field->items()) {
    if (item.key() != "kind" && item.key() != "strength" && item.key() != "schema") {
      invalid("structured_output contains an unsupported field: " + item.key());
    }
  }

  nlohmann::ordered_json schema;
  if (kind == "json_object") {
    if (field->contains("schema")) invalid("structured_output.schema is not allowed for json_object");
    schema = {{"type", "object"}};
  } else {
    const auto input_schema = field->find("schema");
    if (input_schema == field->end() || !input_schema->is_object()) {
      invalid("structured_output.schema must be an object for json_schema");
    }
    schema = *input_schema;
  }

  if (schema.dump().size() > kMaxStructuredOutputSchemaBytes) {
    invalid("structured output schema exceeds the 64 KiB limit");
  }
  std::size_t property_count = 0;
  validate_schema(schema, 1, property_count);

  std::string grammar;
  try {
    grammar = json_schema_to_grammar(schema, true);
  } catch (const std::exception& error) {
    throw RequestError("unsupported_structured_output",
                       std::string("structured output schema conversion failed: ") + error.what());
  }
  if (grammar.empty()) {
    throw RequestError("unsupported_structured_output",
                       "structured output schema conversion produced an empty grammar");
  }
  return StructuredOutputConstraint{kind, strength, std::move(schema), std::move(grammar)};
}

}  // namespace clap::llama
