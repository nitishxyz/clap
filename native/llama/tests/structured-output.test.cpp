#include "clap/llama/protocol.h"
#include "clap/llama/structured-output.h"

#include <cassert>
#include <functional>
#include <string>

using clap::llama::RequestError;
using clap::llama::parse_structured_output;

namespace {

nlohmann::json request(nlohmann::json contract) {
  return {{"structured_output", std::move(contract)}};
}

RequestError failure(const nlohmann::json& value) {
  try {
    (void) parse_structured_output(value);
  } catch (const RequestError& error) {
    return error;
  }
  assert(false && "expected structured output failure");
  return RequestError("unreachable", "unreachable");
}

void expect_compiles(nlohmann::json schema) {
  auto parsed = parse_structured_output(request({
      {"kind", "json_schema"}, {"strength", "required"}, {"schema", std::move(schema)}}));
  assert(parsed);
  assert(parsed->kind == "json_schema");
  assert(parsed->strength == "required");
  assert(!parsed->grammar.empty());
  assert(parsed->grammar.find("root") != std::string::npos);
}

}  // namespace

int main() {
  assert(!parse_structured_output(nlohmann::json::object()));

  auto object = parse_structured_output(request({
      {"kind", "json_object"}, {"strength", "best_effort"}}));
  assert(object);
  assert(object->schema == nlohmann::ordered_json({{"type", "object"}}));
  assert(!object->grammar.empty());

  expect_compiles({
      {"type", "object"},
      {"properties", {{"name", {{"type", "string"}}}, {"age", {{"type", "integer"}}}}},
      {"required", {"name"}},
  });
  expect_compiles({{"type", "array"}, {"items", {{"type", "number"}}}});
  expect_compiles({{"enum", {"red", "green", "blue"}}});
  expect_compiles({
      {"type", "object"},
      {"properties", {{"outer", {{"type", "object"}, {"properties", {{"inner", {{"type", "boolean"}}}}}}}}},
      {"required", {"outer"}},
  });
  expect_compiles({
      {"$defs", {{"item", {{"type", "object"}, {"properties", {{"id", {{"type", "integer"}}}}}}}}},
      {"type", "array"}, {"items", {{"$ref", "#/$defs/item"}}},
  });

  for (const auto& contract : {
      nlohmann::json(nullptr),
      nlohmann::json::object({{"kind", "grammar"}, {"strength", "required"}}),
      nlohmann::json::object({{"kind", "json_object"}, {"strength", "strict"}}),
      nlohmann::json::object({{"kind", "json_schema"}, {"strength", "required"}}),
      nlohmann::json::object({{"kind", "json_object"}, {"strength", "required"}, {"schema", nlohmann::json::object()}}),
      nlohmann::json::object({{"kind", "json_object"}, {"strength", "required"}, {"extra", true}}),
  }) assert(failure(request(contract)).code == "invalid_structured_output");

  std::string oversized(65 * 1024, 'x');
  assert(failure(request({{"kind", "json_schema"}, {"strength", "required"},
      {"schema", {{"description", oversized}}}})).code == "invalid_structured_output");

  nlohmann::json nested = {{"type", "string"}};
  for (int depth = 0; depth < 33; ++depth) nested = nlohmann::json::array({std::move(nested)});
  assert(failure(request({{"kind", "json_schema"}, {"strength", "required"},
      {"schema", {{"allOf", std::move(nested)}}}})).code == "invalid_structured_output");

  nlohmann::json properties = nlohmann::json::object();
  for (int index = 0; index < 1025; ++index) properties["p" + std::to_string(index)] = {{"type", "string"}};
  assert(failure(request({{"kind", "json_schema"}, {"strength", "required"},
      {"schema", {{"type", "object"}, {"properties", std::move(properties)}}}})).code ==
      "invalid_structured_output");

  assert(failure(request({{"kind", "json_schema"}, {"strength", "required"},
      {"schema", {{"$ref", "https://example.com/schema.json"}}}})).code ==
      "invalid_structured_output");

  const auto unsupported = failure(request({{"kind", "json_schema"}, {"strength", "required"},
      {"schema", {{"type", "definitely-not-a-json-schema-type"}}}}));
  assert(unsupported.code == "unsupported_structured_output");
}
