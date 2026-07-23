import type { ChatCompletionRequest, ChatToolCall } from "@clap/api";
import { parseJsonLike } from "./json";
import { cleanupProtocolText } from "./plain";

export function parseLooseArgs(text: string): unknown {
  const normalized = text
    .trim()
    .replace(/<\|'\|>/g, '"')
    .replace(/"([A-Za-z_$][\w$.-]*):"([^"]*)""(?=\s*[,}\]])/g, '"$1":"$2"')
    .replace(/"([A-Za-z_$][\w$.-]*):"([^"]*)"/g, '"$1":"$2"')
    .replace(/("[^"]*"|\d|true|false|null)\s+([A-Za-z_$][\w$.-]*\s*:)/g, '$1,$2')
    .replace(/([{,]\s*)([A-Za-z_$][\w$.-]*)(\s*:)/g, '$1"$2"$3')
    .replace(/("[^"]*"|\d|true|false|null)\s+"/g, '$1,"')
    .replace(/'([^']*)'/g, (_, value: string) => JSON.stringify(value));
  try {
    return JSON.parse(normalized);
  } catch {
    return { input: cleanupProtocolText(text) };
  }
}

export function parsePythonArgs(text: string): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  const pattern = /([A-Za-z_$][\w$.-]*)\s*=\s*("[^"]*"|'[^']*'|[^,]+)\s*,?/g;
  for (const match of text.matchAll(pattern)) {
    const key = match[1];
    const value = match[2]?.trim();
    if (!key || value === undefined) continue;
    result[key] = parseScalar(value);
  }
  return result;
}

export function toolCall(name: string, args: unknown, index: number): ChatToolCall {
  return {
    id: `call_${index}_${name.replace(/[^A-Za-z0-9_]/g, "_")}`,
    type: "function",
    function: {
      name,
      arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
    },
  };
}

type JsonSchema = {
  type?: string;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
};

export function coerceToolCallArguments(call: ChatToolCall, request: ChatCompletionRequest): ChatToolCall {
  const schema = request.tools?.find((tool) => tool.function.name === call.function.name)?.function.parameters as JsonSchema | undefined;
  if (!schema) return call;
  const args = parseJsonLike(call.function.arguments);
  if (!args || typeof args !== "object" || Array.isArray(args)) return call;
  const coerced = coerceValueToSchema(args, { ...schema, type: schema.type ?? "object" });
  return { ...call, function: { ...call.function, arguments: JSON.stringify(coerced) } };
}

function parseScalar(value: string): unknown {
  try {
    return JSON.parse(value.replace(/^'([^']*)'$/, (_, inner: string) => JSON.stringify(inner)));
  } catch {
    return value;
  }
}

function coerceValueToSchema(value: unknown, schema: JsonSchema | undefined): unknown {
  if (!schema?.type) return value;
  switch (schema.type) {
    case "array": {
      if (Array.isArray(value)) return value.map((item) => coerceValueToSchema(item, schema.items));
      if (typeof value === "string") {
        const parsed = parseJsonLike(value);
        if (Array.isArray(parsed)) return parsed.map((item) => coerceValueToSchema(item, schema.items));
      }
      if (value === undefined || value === null) return value;
      return [coerceValueToSchema(value, schema.items)];
    }
    case "number":
    case "integer":
      if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return Number(value);
      return value;
    case "boolean":
      if (value === "true") return true;
      if (value === "false") return false;
      return value;
    case "string":
      if (typeof value === "number" || typeof value === "boolean") return String(value);
      return value;
    case "object": {
      if (typeof value === "string") {
        const parsed = parseJsonLike(value);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return coerceValueToSchema(parsed, schema);
        return value;
      }
      if (!value || typeof value !== "object" || Array.isArray(value) || !schema.properties) return value;
      const result: Record<string, unknown> = {};
      for (const [key, entry] of Object.entries(value)) result[key] = coerceValueToSchema(entry, schema.properties[key]);
      return result;
    }
    default:
      return value;
  }
}
