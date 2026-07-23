import type { ChatCompletionRequest } from "@clap/api";
import Ajv, { type ValidateFunction } from "ajv";
import { parseJsonLike } from "./json";

type ResponseFormat = NonNullable<ChatCompletionRequest["response_format"]>;
type JsonResponseFormat = Exclude<ResponseFormat, { type: "text" }>;

export type StructuredOutputErrorCode =
  | "invalid_json"
  | "invalid_json_object"
  | "schema_validation_failed"
  | "invalid_schema";

export class StructuredOutputError extends Error {
  constructor(readonly code: "structured_output_invalid" | "schema_unsupported", message: string,
              readonly repaired = false, readonly validationMs?: number) {
    super(message);
    this.name = "StructuredOutputError";
  }
}

export type StructuredOutputOutcome = {
  ok: true;
  value: unknown;
  json: string;
  repaired: boolean;
} | {
  ok: false;
  error: {
    code: StructuredOutputErrorCode;
    message: string;
  };
};

const VALIDATOR_CACHE_LIMIT = 64;
const ajv = new Ajv({ allErrors: false, strict: false, allowUnionTypes: true });
const validatorCache = new Map<string, { schema: string; validate: ValidateFunction }>();

export function parseStructuredOutput(text: string, format: JsonResponseFormat): StructuredOutputOutcome {
  const required = format.constraint === "required"
    || (format.type === "json_schema" && format.json_schema.strict === true);
  const trimmed = text.trim();
  let value: unknown;
  let repaired = false;

  try {
    value = JSON.parse(trimmed);
  } catch {
    if (required) return failure("invalid_json", "model output is not valid JSON");
    value = parseJsonLike(trimmed);
    repaired = value !== undefined;
    if (value === undefined) return failure("invalid_json", "model output is not valid JSON");
  }

  if (format.type === "json_object") {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      return failure("invalid_json_object", "model output must be a JSON object");
    }
  } else {
    const validator = validatorFor(format.json_schema.schema);
    if (!validator.ok) return validator;
    if (!validator.validate(value)) {
      return failure("schema_validation_failed", "model output does not match the requested JSON schema");
    }
  }

  return { ok: true, value, json: canonicalJson(value), repaired };
}

export function formatStructuredOutput(text: string, format: JsonResponseFormat): string {
  const outcome = parseStructuredOutput(text, format);
  if (outcome.ok) return outcome.json;
  throw new StructuredOutputError(
    outcome.error.code === "invalid_schema" ? "schema_unsupported" : "structured_output_invalid",
    outcome.error.message);
}

export function structuredValidatorCacheSize(): number {
  return validatorCache.size;
}

export function clearStructuredValidatorCache(): void {
  validatorCache.clear();
}

function validatorFor(schema: Record<string, unknown>):
  | { ok: true; validate: ValidateFunction }
  | Extract<StructuredOutputOutcome, { ok: false }> {
  const serialized = canonicalJson(schema);
  const hash = hashSchema(serialized);
  const cached = validatorCache.get(hash);
  if (cached?.schema === serialized) {
    validatorCache.delete(hash);
    validatorCache.set(hash, cached);
    return { ok: true, validate: cached.validate };
  }

  try {
    const validate = ajv.compile(schema);
    validatorCache.set(hash, { schema: serialized, validate });
    while (validatorCache.size > VALIDATOR_CACHE_LIMIT) {
      const oldest = validatorCache.keys().next().value as string | undefined;
      if (oldest === undefined) break;
      validatorCache.delete(oldest);
    }
    return { ok: true, validate };
  } catch {
    return failure("invalid_schema", "the requested JSON schema could not be compiled");
  }
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(canonicalValue(value));
}

function canonicalValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonicalValue);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .sort(([left], [right]) => left < right ? -1 : left > right ? 1 : 0)
    .map(([key, child]) => [key, canonicalValue(child)]));
}

function hashSchema(value: string): string {
  let first = 0x811c9dc5;
  let second = 0x9e3779b9;
  for (let index = 0; index < value.length; index += 1) {
    const code = value.charCodeAt(index);
    first = Math.imul(first ^ code, 0x01000193);
    second = Math.imul(second ^ code, 0x85ebca6b);
  }
  return `${(first >>> 0).toString(16)}${(second >>> 0).toString(16)}`;
}

function failure(code: StructuredOutputErrorCode, message: string): Extract<StructuredOutputOutcome, { ok: false }> {
  return { ok: false, error: { code, message } };
}
