import { beforeEach, describe, expect, test } from "bun:test";
import {
  clearStructuredValidatorCache,
  parseStructuredOutput,
  structuredValidatorCacheSize,
} from "./structured";

describe("structured output post-validation", () => {
  beforeEach(clearStructuredValidatorCache);

  test("parses exact JSON and emits canonical object JSON", () => {
    expect(parseStructuredOutput('{"z":1,"a":{"d":2,"b":3}}', {
      type: "json_object", constraint: "required",
    })).toEqual({
      ok: true,
      value: { z: 1, a: { d: 2, b: 3 } },
      json: '{"a":{"b":3,"d":2},"z":1}',
      repaired: false,
    });
  });

  test("requires json_object outputs to be objects", () => {
    expect(parseStructuredOutput("[1,2]", { type: "json_object" })).toEqual({
      ok: false,
      error: { code: "invalid_json_object", message: "model output must be a JSON object" },
    });
  });

  test("best effort uses deterministic fenced, object, and bracket repair", () => {
    for (const text of [
      '```json\n{"answer": 42}\n```',
      'Answer: {"answer":42} done.',
      '{"answer":42',
    ]) {
      expect(parseStructuredOutput(text, { type: "json_object", constraint: "best_effort" })).toMatchObject({
        ok: true, value: { answer: 42 }, json: '{"answer":42}', repaired: true,
      });
    }
  });

  test("required mode never extracts or repairs JSON", () => {
    for (const text of ['```json\n{"ok":true}\n```', 'prefix {"ok":true}', '{"ok":true']) {
      expect(parseStructuredOutput(text, { type: "json_object", constraint: "required" })).toEqual({
        ok: false,
        error: { code: "invalid_json", message: "model output is not valid JSON" },
      });
    }
  });

  test("validates schemas with local references and returns safe failures", () => {
    const format = {
      type: "json_schema" as const,
      constraint: "required" as const,
      json_schema: {
        name: "Result",
        schema: {
          $defs: { answer: { type: "integer" } },
          type: "object",
          properties: { answer: { $ref: "#/$defs/answer" } },
          required: ["answer"],
          additionalProperties: false,
        },
      },
    };
    expect(parseStructuredOutput('{"answer":42}', format)).toMatchObject({ ok: true });
    expect(parseStructuredOutput('{"answer":"no"}', format)).toEqual({
      ok: false,
      error: {
        code: "schema_validation_failed",
        message: "model output does not match the requested JSON schema",
      },
    });
    expect(parseStructuredOutput("null", {
      ...format, json_schema: { name: "Broken", schema: { type: "not-a-json-schema-type" } },
    })).toEqual({
      ok: false,
      error: { code: "invalid_schema", message: "the requested JSON schema could not be compiled" },
    });
  });

  test("caches validators by canonical schema hash with a bounded LRU", () => {
    const makeFormat = (index: number) => ({
      type: "json_schema" as const,
      json_schema: { name: `Schema${index}`, schema: { type: "object", title: `schema-${index}` } },
    });
    parseStructuredOutput("{}", makeFormat(0));
    parseStructuredOutput("{}", {
      type: "json_schema",
      json_schema: { name: "SameSchema", schema: { title: "schema-0", type: "object" } },
    });
    expect(structuredValidatorCacheSize()).toBe(1);
    for (let index = 1; index < 70; index += 1) parseStructuredOutput("{}", makeFormat(index));
    expect(structuredValidatorCacheSize()).toBe(64);
  });
});
