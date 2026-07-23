import type { ChatCompletionRequest, ChatToolCall } from "@clap/api";
import type { ParserContext } from "./types";

export function parseJsonToolCalls(text: string, context: ParserContext): ChatToolCall[] {
  if (!context.toolMode) return [];
  return parseToolCalls(text, context.request) ?? [];
}

export function parseToolCalls(text: string, request: ChatCompletionRequest): ChatToolCall[] | undefined {
  const parsed = parseJsonLike(text);
  if (Array.isArray(parsed)) return parsed.map((call, index) => normalizeToolCall(call, index)).filter((call): call is ChatToolCall => Boolean(call));
  if (!parsed || typeof parsed !== "object") return undefined;
  const value = parsed as Record<string, unknown>;
  const rawCalls = Array.isArray(value.tool_calls) ? value.tool_calls : Array.isArray(value.tools) ? value.tools : undefined;
  if (rawCalls) {
    return rawCalls.map((call, index) => {
      const normalized = normalizeToolCall(call, index);
      if (normalized || !call || typeof call !== "object" || Array.isArray(call)) return normalized;
      const record = call as Record<string, unknown>;
      const args = record.arguments ?? record.args ?? record.parameters;
      if (!args || typeof args !== "object" || Array.isArray(args)) return undefined;
      return inferArgsOnlyToolCall(args as Record<string, unknown>, text, request);
    }).filter((call): call is ChatToolCall => Boolean(call));
  }
  const single = normalizeToolCall(value, 0);
  if (single) return [single];
  const inferred = inferArgsOnlyToolCall(value, text, request);
  return inferred ? [inferred] : undefined;
}

export function normalizeToolCall(value: unknown, index: number): ChatToolCall | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  const fn = record.function && typeof record.function === "object" ? record.function as Record<string, unknown> : record;
  const name = typeof fn.name === "string" ? fn.name
    : typeof record.name === "string" ? record.name
    : typeof fn.tool_name === "string" ? fn.tool_name
    : typeof record.tool_name === "string" ? record.tool_name
    : undefined;
  if (!name) return undefined;
  let args = fn.arguments ?? record.arguments ?? fn.args ?? record.args ?? fn.parameters ?? record.parameters ?? {};
  if (fn === record && args && typeof args === "object" && !Array.isArray(args)) {
    const extras = Object.fromEntries(Object.entries(record).filter(([key]) => ![
      "id", "type", "name", "tool_name", "function", "arguments", "args", "parameters",
    ].includes(key)));
    if (Object.keys(extras).length) args = { ...(args as Record<string, unknown>), ...extras };
  }
  const parsedArgs = typeof args === "string" ? parseJsonLike(args) ?? args : args;
  return {
    id: typeof record.id === "string" ? record.id : `call_${index}_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`,
    type: "function",
    function: {
      name,
      arguments: typeof parsedArgs === "string" ? parsedArgs : JSON.stringify(parsedArgs ?? {}),
    },
  };
}

export function parseJsonLike(text: string): unknown {
  const trimmed = text.trim();
  for (const candidate of jsonCandidates(trimmed)) {
    try {
      return JSON.parse(candidate);
    } catch {
      // Try the next repair candidate.
    }
  }
  return undefined;
}

export function extractBalancedObject(text: string): string | undefined {
  const start = text.indexOf("{");
  if (start < 0) return undefined;
  let depth = 0;
  let inString = false;
  let quote = "";
  let escape = false;
  for (let index = start; index < text.length; index += 1) {
    const char = text[index];
    if (escape) {
      escape = false;
      continue;
    }
    if (char === "\\") {
      escape = true;
      continue;
    }
    if (inString) {
      if (char === quote) {
        inString = false;
        quote = "";
      }
      continue;
    }
    if (char === '"' || char === "'") {
      inString = true;
      quote = char;
      continue;
    }
    if (char === "{") depth += 1;
    if (char === "}") {
      depth -= 1;
      if (depth === 0) return text.slice(start, index + 1);
    }
  }
  return undefined;
}

function inferArgsOnlyToolCall(args: Record<string, unknown>, text: string, request: ChatCompletionRequest): ChatToolCall | undefined {
  if (!request.tools?.length || request.response_format?.type === "json_object" || request.response_format?.type === "json_schema") return undefined;
  const keys = Object.keys(args);
  if (!keys.length) return undefined;
  const mentioned = request.tools.filter((tool) => new RegExp(`\\b${escapeRegExp(tool.function.name)}\\b`).test(text));
  if (mentioned.length === 1) return makeToolCall(mentioned[0]!.function.name, args, 0);
  const schemaMatches = request.tools.filter((tool) => toolMatchesArgs(tool.function.parameters, keys));
  if (schemaMatches.length === 1) return makeToolCall(schemaMatches[0]!.function.name, args, 0);
  return undefined;
}

function makeToolCall(name: string, args: unknown, index: number): ChatToolCall {
  return {
    id: `call_${index}_${name.replace(/[^A-Za-z0-9_]/g, "_")}`,
    type: "function",
    function: { name, arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}) },
  };
}

function toolMatchesArgs(parameters: Record<string, unknown> | undefined, keys: string[]): boolean {
  const properties = parameters?.properties;
  if (!properties || typeof properties !== "object" || Array.isArray(properties)) return false;
  const propertyKeys = new Set(Object.keys(properties));
  return keys.every((key) => propertyKeys.has(key));
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function jsonCandidates(text: string): string[] {
  const candidates = [text];
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1]?.trim();
  if (fenced) candidates.push(fenced);
  const firstObject = text.indexOf("{");
  const lastObject = text.lastIndexOf("}");
  if (firstObject >= 0 && lastObject > firstObject) candidates.push(text.slice(firstObject, lastObject + 1));
  if (firstObject >= 0 && text.slice(firstObject).length > 1) candidates.push(text.slice(firstObject));
  for (const candidate of [...candidates]) {
    const repaired = balanceJsonBrackets(candidate);
    if (repaired !== candidate) candidates.push(repaired);
  }
  return candidates;
}

function balanceJsonBrackets(text: string): string {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return text;
  const closerFor: Record<string, string> = { "{": "}", "[": "]" };
  const stack: string[] = [];
  let out = "";
  let inString = false;
  let escape = false;
  for (const char of trimmed) {
    if (escape) {
      escape = false;
      out += char;
      continue;
    }
    if (char === "\\") {
      escape = true;
      out += char;
      continue;
    }
    if (inString) {
      if (char === '"') inString = false;
      out += char;
      continue;
    }
    if (char === '"') {
      inString = true;
      out += char;
      continue;
    }
    if (char === "{" || char === "[") {
      stack.push(char);
      out += char;
      continue;
    }
    if (char === "}" || char === "]") {
      const opener = char === "}" ? "{" : "[";
      if (stack.at(-1) === opener) {
        stack.pop();
        out += char;
        continue;
      }
      const deeper = stack.lastIndexOf(opener);
      if (deeper < 0) continue;
      while (stack.length > deeper + 1) out += closerFor[stack.pop()!]!;
      stack.pop();
      out += char;
      continue;
    }
    out += char;
  }
  if (inString) out += '"';
  while (stack.length) out += closerFor[stack.pop()!]!;
  return out;
}
