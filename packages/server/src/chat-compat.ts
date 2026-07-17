import type { ChatCompletionRequest, ChatMessage, ChatToolCall } from "@clap/api";

export type ParsedAssistantOutput = {
  content: string | null;
  reasoning?: string;
  toolCalls?: ChatToolCall[];
  finishReason: "stop" | "tool_calls";
};

type NormalizedOutput = {
  content: string;
  reasoning?: string;
  toolCalls?: ChatToolCall[];
};

type ParserContext = {
  request: ChatCompletionRequest;
  toolMode: boolean;
};

type ToolParser = (text: string, context: ParserContext) => ChatToolCall[];

type AssistantOutputParser = {
  name: string;
  families: string[];
  toolParsers: ToolParser[];
  parse: (text: string, context: ParserContext) => NormalizedOutput;
};

export type ParserTemplateInfo = {
  familyHints?: string[];
  hasToolCalls?: boolean;
  hasReasoning?: boolean;
  sourceFiles?: string[];
};

export function prepareChatRequest(request: ChatCompletionRequest): ChatCompletionRequest {
  rejectUnsupportedContentParts(request);
  const messages = request.messages.map((message) => ({
    ...message,
    content: stringifyMessageContent(message),
  }));
  const instructions = compatibilityInstructions(request);
  if (instructions) {
    messages.unshift({ role: "system", content: instructions });
  }
  return { ...request, messages, max_tokens: request.max_tokens ?? 4096 };
}

export function parseAssistantOutput(text: string, request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo): ParsedAssistantOutput {
  const stopped = applyStop(text, request.stop);
  const context = { request, toolMode: Boolean(request.tools?.length) };
  const parser = selectParser(request.model, request, templateInfo);
  const normalized = parser.parse(stopped, context);
  const toolCalls = (normalized.toolCalls ?? []).map((call) => coerceToolCallArguments(call, request));
  if (toolCalls.length) {
    return { content: null, reasoning: normalized.reasoning, toolCalls, finishReason: "tool_calls" };
  }

  const formatted = formatStructuredContent(normalized.content, request);
  return { content: formatted, reasoning: normalized.reasoning, finishReason: "stop" };
}

// Parser registry follows the same broad shape as OSS servers such as Ollama,
// vLLM, SGLang, and Unsloth: choose a model/template-aware parser first, then
// extract reasoning before interpreting remaining content as tool calls.
export function selectParser(model: string, _request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo): AssistantOutputParser {
  for (const hint of templateInfo?.familyHints ?? []) {
    const parser = parserRegistry.find((candidate) => candidate.name === hint || candidate.families.includes(hint));
    if (parser) return parser;
  }
  const id = model.toLowerCase();
  return parserRegistry.find((parser) => parser.families.some((family) => id.includes(family))) ?? genericParser;
}

function applyStop(text: string, stop: ChatCompletionRequest["stop"]): string {
  const stops = typeof stop === "string" ? [stop] : stop ?? [];
  const indexes = stops.map((value) => text.indexOf(value)).filter((index) => index >= 0);
  return indexes.length ? text.slice(0, Math.min(...indexes)) : text;
}

export function rejectUnsupportedContentParts(request: ChatCompletionRequest): void {
  for (const message of request.messages) {
    if (!Array.isArray(message.content)) continue;
    const unsupported = message.content.find((part) => part.type !== "text");
    if (unsupported?.type === "image_url") {
      throw new Error("image_url content parts are not supported by the selected local text runtime yet");
    }
  }
}

function stringifyMessageContent(message: ChatMessage): string {
  const content = Array.isArray(message.content)
    ? message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n")
    : message.content ?? "";
  if (message.role !== "tool") return content;
  return `Tool result${message.tool_call_id ? ` (${message.tool_call_id})` : ""}: ${content}`;
}

function compatibilityInstructions(request: ChatCompletionRequest): string {
  const blocks: string[] = [];
  if (request.tools?.length) {
    blocks.push([
      "You may call tools by responding with JSON only.",
      "Use this exact shape when calling tools:",
      '{"tool_calls":[{"name":"tool_name","arguments":{}}]}',
      "Do not include natural language when calling tools.",
      `Available tools: ${JSON.stringify(request.tools.map((tool) => tool.function))}`,
      request.tool_choice && request.tool_choice !== "auto" ? `Tool choice: ${JSON.stringify(request.tool_choice)}` : "",
      request.parallel_tool_calls === false ? "Call at most one tool." : "",
    ].filter(Boolean).join("\n"));
  }
  if (request.response_format?.type === "json_object") {
    blocks.push("Respond with a valid JSON object and no surrounding markdown.");
  }
  if (request.response_format?.type === "json_schema") {
    blocks.push(`Respond with JSON matching this schema and no surrounding markdown: ${JSON.stringify(request.response_format.json_schema)}`);
  }
  return blocks.join("\n\n");
}

const harmonyParser = createParser("harmony", ["harmony", "gpt-oss", "codex"], [
  parseHarmonyToolCalls,
  parseFunctionMessageCalls,
  parseTaggedToolCalls,
  parseJsonToolCalls,
]);

const hermesParser = createParser("hermes", ["hermes", "nous", "functionary", "xlam"], [
  parseTaggedToolCalls,
  parseJsonToolCalls,
]);

const qwenParser = createParser("qwen", ["qwen"], [
  parseQwenToolCalls,
  parseTaggedToolCalls,
  parseJsonToolCalls,
]);

const deepSeekParser = createParser("deepseek", ["deepseek"], [
  parseDeepSeekToolCalls,
  parseTaggedToolCalls,
  parseJsonToolCalls,
]);

const mistralParser = createParser("mistral", ["mistral", "mixtral"], [
  parseMistralToolCalls,
  parseJsonToolCalls,
]);

const llamaParser = createParser("llama", ["llama", "granite"], [
  parsePythonTagToolCalls,
  parseJsonToolCalls,
]);

const gemmaParser = createParser("gemma", ["gemma", "functiongemma"], [
  parseFunctionGemmaCalls,
  parseJsonToolCalls,
]);

const genericParser = createParser("generic", [], [
  parseHarmonyToolCalls,
  parseFunctionMessageCalls,
  parseTaggedToolCalls,
  parseQwenToolCalls,
  parseDeepSeekToolCalls,
  parseMistralToolCalls,
  parsePythonTagToolCalls,
  parseFunctionGemmaCalls,
  parseJsonToolCalls,
]);

const parserRegistry = [harmonyParser, hermesParser, qwenParser, deepSeekParser, mistralParser, llamaParser, gemmaParser];

function createParser(name: string, families: string[], toolParsers: ToolParser[]): AssistantOutputParser {
  return {
    name,
    families,
    toolParsers,
    parse(text, context) {
      const reasoned = extractReasoning(text);
      const toolCalls = runToolParsers(reasoned.content, context, toolParsers);
      const visible = suppressProtocolMarkers(reasoned.content);
      const content = toolCalls.length ? "" : cleanupProtocolText(visible);
      return {
        content,
        reasoning: reasoned.reasoning,
        toolCalls: toolCalls.length ? toolCalls : undefined,
      };
    },
  };
}

function extractReasoning(text: string): { content: string; reasoning?: string } {
  const pieces = extractChannelReasoning(text);
  const withThink = extractThinkReasoning(pieces.content, pieces.reasoning);
  return { content: withThink.content, reasoning: withThink.reasoning };
}

function extractChannelReasoning(text: string): { content: string; reasoning?: string } {
  const channelPattern = /<\|?channel\|?>\s*(thought|analysis|commentary|final)\b\s*/gi;
  const matches = [...text.matchAll(channelPattern)];
  if (!matches.length) return { content: text };

  const reasoning: string[] = [];
  const content: string[] = [];
  for (let i = 0; i < matches.length; i += 1) {
    const match = matches[i]!;
    const channel = match[1]?.toLowerCase();
    const start = (match.index ?? 0) + match[0].length;
    const nextStart = matches[i + 1]?.index ?? text.length;
    const segment = text.slice(start, nextStart);
    const toolStart = findProtocolStart(segment);
    const body = toolStart >= 0 ? segment.slice(0, toolStart) : segment;
    const remainder = toolStart >= 0 ? segment.slice(toolStart) : "";
    if (channel === "final") {
      content.push(body, remainder);
    } else {
      const cleaned = cleanupProtocolText(body);
      if (cleaned) reasoning.push(cleaned);
      if (remainder) content.push(remainder);
    }
  }
  return { content: content.join("\n").trim(), reasoning: reasoning.join("\n\n") || undefined };
}

function findProtocolStart(text: string): number {
  const starts = [
    text.search(/<\|?tool_call\|?>/),
    text.search(/to=functions\.[\w.-]+/),
    text.search(/<tool_call>/),
    text.search(/<\|tool_call_start\|>/),
    text.search(/<｜tool▁calls▁begin｜>/),
    text.search(/<\|python_tag\|>/),
    text.search(/\[TOOL_CALLS\]/),
    text.search(/call::?[\w.-]+\s*\{/),
  ].filter((index) => index >= 0);
  return starts.length ? Math.min(...starts) : -1;
}

function extractThinkReasoning(text: string, existingReasoning?: string): { content: string; reasoning?: string } {
  const reasoning = existingReasoning ? [existingReasoning] : [];
  let content = text.replace(/<think>([\s\S]*?)(?:<\/think>|$)/gi, (_match, thought: string) => {
    const cleaned = cleanupProtocolText(thought);
    if (cleaned) reasoning.push(cleaned);
    return "";
  });
  if (!reasoning.length) {
    const missingOpen = content.match(/^([\s\S]*?)<\/think>/i);
    if (missingOpen?.[1]) {
      reasoning.push(cleanupProtocolText(missingOpen[1]));
      content = content.slice(missingOpen[0].length);
    }
  }
  return { content: content.trim(), reasoning: reasoning.filter(Boolean).join("\n\n") || undefined };
}

function runToolParsers(text: string, context: ParserContext, parsers: ToolParser[]): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  for (const parser of parsers) {
    if (parser === parseJsonToolCalls && !context.toolMode) continue;
    if ((parser === parsePythonTagToolCalls || parser === parseFunctionGemmaCalls) && !context.toolMode && !hasExplicitParserMarker(text)) continue;
    calls.push(...parser(text, context));
    if (calls.length) break;
  }
  return calls.map((call, index) => ({ ...call, id: call.id.startsWith("call_") ? call.id.replace(/^call_\d+_/, `call_${index}_`) : call.id }));
}

function hasExplicitParserMarker(text: string): boolean {
  return /<\|python_tag\|>|call:[\w.-]+\s*\{|<\|?tool_call\|?>|<tool_call>|\[TOOL_CALLS\]/.test(text);
}

function suppressProtocolMarkers(text: string): string {
  return text
    .replace(/<\|?tool_call\|?>[\s\S]*?<\|?\/tool_call\|?>/g, "")
    .replace(/<\|?tool_call\|?>[\s\S]*$/g, "")
    .replace(/to=functions\.[\w.-]+[\s\S]*?<\|message\|>[\s\S]*?<\|call\|>/g, "")
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/g, "")
    .replace(/<\|tool_call_start\|>[\s\S]*?<\|tool_call_end\|>/g, "")
    .replace(/<｜tool▁calls▁begin｜>[\s\S]*?<｜tool▁calls▁end｜>/g, "")
    .replace(/<\|python_tag\|>[\s\S]*$/g, "")
    .replace(/\[TOOL_CALLS\][\s\S]*$/g, "")
    .replace(/call:[\w.-]+\s*\{[\s\S]*?\}/g, "");
}

function cleanupProtocolText(text: string): string {
  return text
    .replace(/<\|(?:message|call|\/tool_call|tool_call)\|>/g, "")
    .replace(/<\|?channel\|?>/g, "")
    .replace(/<\|(?:im_end|eot_id|end_of_text)\|>/g, "")
    .replace(/<\|turn>(?:assistant|user|model|system)?\s*$/g, "")
    .replace(/<turn\|>/g, "")
    .replace(/<\|turn>/g, "")
    .replace(/<\|tool_call_(?:start|end)\|>/g, "")
    .replace(/<\/?tool_call>/g, "")
    .replace(/<\|?tool_call\|?>[\s\S]*$/g, "")
    .replace(/call::?[\w.-]+[\s\S]*$/g, "")
    .replace(/<｜tool▁(?:calls▁begin|calls▁end|call▁begin|call▁end)｜>/g, "")
    .replace(/<\|python_tag\|>/g, "")
    .replace(/\[TOOL_CALLS\]/g, "")
    .replace(/<\/s>/g, "")
    .trim();
}

function parseHarmonyToolCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const pattern = /<\|?tool_call\|?>\s*call::?([\w.-]+)\s*/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (!name) continue;
    const object = extractBalancedObject(text.slice((match.index ?? 0) + match[0].length));
    if (object) calls.push(toolCall(name, parseLooseArgs(object), calls.length));
  }
  return calls;
}

function parseTaggedToolCalls(text: string): ChatToolCall[] {
  return [...text.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)]
    .map((match, index) => normalizeToolCall(parseJsonLike(match[1] ?? ""), index))
    .filter((call): call is ChatToolCall => Boolean(call));
}

function parseQwenToolCalls(text: string): ChatToolCall[] {
  return [...text.matchAll(/<\|tool_call_start\|>\s*([\s\S]*?)\s*<\|tool_call_end\|>/g)]
    .map((match, index) => normalizeToolCall(parseJsonLike(match[1] ?? ""), index))
    .filter((call): call is ChatToolCall => Boolean(call));
}

function parseDeepSeekToolCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const pattern = /<｜tool▁call▁begin｜>\s*function\s+([\w.-]+)\s*```(?:json)?\s*([\s\S]*?)```\s*<｜tool▁call▁end｜>/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (!name) continue;
    calls.push(toolCall(name, parseLooseArgs(match[2] ?? "{}"), calls.length));
  }
  return calls;
}

function parsePythonTagToolCalls(text: string): ChatToolCall[] {
  const tag = text.match(/<\|python_tag\|>\s*([\s\S]*)/);
  if (!tag?.[1]) return [];
  const body = cleanupProtocolText(tag[1]);
  const json = normalizeToolCall(parseJsonLike(body), 0);
  if (json) return [json];
  const call = body.match(/^([\w.-]+)\((.*)\)$/s);
  if (!call?.[1]) return [];
  return [toolCall(call[1], parsePythonArgs(call[2] ?? ""), 0)];
}

function parseMistralToolCalls(text: string): ChatToolCall[] {
  const marker = text.match(/\[TOOL_CALLS\]\s*([\s\S]*)/);
  if (!marker?.[1]) return [];
  const body = cleanupProtocolText(marker[1]);
  const parsed = parseJsonLike(body);
  if (Array.isArray(parsed)) return parsed.map((call, index) => normalizeToolCall(call, index)).filter((call): call is ChatToolCall => Boolean(call));
  const objectCall = normalizeToolCall(parsed, 0);
  if (objectCall) return [objectCall];
  const compact = body.match(/^([\w.-]+)\s*(\{[\s\S]*\})$/);
  if (!compact?.[1]) return [];
  return [toolCall(compact[1], parseLooseArgs(compact[2] ?? "{}"), 0)];
}

function parseFunctionGemmaCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const pattern = /call:([\w.-]+)\s*(\{[\s\S]*?\})/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (!name) continue;
    calls.push(toolCall(name, parseLooseArgs(match[2] ?? "{}"), calls.length));
  }
  if (calls.length) return calls;
  // gemma-4 template format: <|tool_call>call:{"tool_name":"x","arguments":{...}}<tool_call|>
  for (const match of text.matchAll(/call:\s*(?=\{)/g)) {
    const body = extractBalancedObject(text.slice(match.index));
    if (!body) continue;
    const parsed = normalizeToolCall(parseJsonLike(body), calls.length);
    if (parsed) calls.push(parsed);
  }
  return calls;
}

function parseFunctionMessageCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const pattern = /to=functions\.([\w.-]+)[\s\S]*?<\|message\|>\s*([\s\S]*?)\s*<\|call\|>/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (!name) continue;
    calls.push(toolCall(name, parseLooseArgs(match[2] ?? "{}"), calls.length));
  }
  return calls;
}

function parseJsonToolCalls(text: string, context: ParserContext): ChatToolCall[] {
  if (!context.toolMode) return [];
  return parseToolCalls(text, context.request) ?? [];
}

function extractBalancedObject(text: string): string | undefined {
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

function parseLooseArgs(text: string): unknown {
  const normalized = text
    .trim()
    .replace(/<\|'\|>/g, '"')
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

function parsePythonArgs(text: string): Record<string, unknown> {
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

function parseScalar(value: string): unknown {
  try {
    return JSON.parse(value.replace(/^'([^']*)'$/, (_, inner: string) => JSON.stringify(inner)));
  } catch {
    return value;
  }
}

function toolCall(name: string, args: unknown, index: number): ChatToolCall {
  return {
    id: `call_${index}_${name.replace(/[^A-Za-z0-9_]/g, "_")}`,
    type: "function",
    function: {
      name,
      arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
    },
  };
}

function parseToolCalls(text: string, request: ChatCompletionRequest): ChatToolCall[] | undefined {
  const parsed = parseJsonLike(text);
  if (Array.isArray(parsed)) return parsed.map((call, index) => normalizeToolCall(call, index)).filter((call): call is ChatToolCall => Boolean(call));
  if (!parsed || typeof parsed !== "object") return undefined;
  const value = parsed as Record<string, unknown>;
  const rawCalls = Array.isArray(value.tool_calls) ? value.tool_calls : Array.isArray(value.tools) ? value.tools : undefined;
  if (rawCalls) return rawCalls.map((call, index) => normalizeToolCall(call, index)).filter((call): call is ChatToolCall => Boolean(call));
  const single = normalizeToolCall(value, 0);
  if (single) return [single];
  const inferred = inferArgsOnlyToolCall(value, text, request);
  return inferred ? [inferred] : undefined;
}

function inferArgsOnlyToolCall(args: Record<string, unknown>, text: string, request: ChatCompletionRequest): ChatToolCall | undefined {
  if (!request.tools?.length || request.response_format?.type === "json_object" || request.response_format?.type === "json_schema") return undefined;
  const keys = Object.keys(args);
  if (!keys.length) return undefined;
  const mentioned = request.tools.filter((tool) => new RegExp(`\\b${escapeRegExp(tool.function.name)}\\b`).test(text));
  if (mentioned.length === 1) return toolCall(mentioned[0]!.function.name, args, 0);

  const schemaMatches = request.tools.filter((tool) => toolMatchesArgs(tool.function.parameters, keys));
  if (schemaMatches.length === 1) return toolCall(schemaMatches[0]!.function.name, args, 0);
  return undefined;
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

function normalizeToolCall(value: unknown, index: number): ChatToolCall | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  const fn = record.function && typeof record.function === "object" ? record.function as Record<string, unknown> : record;
  const name = typeof fn.name === "string" ? fn.name
    : typeof record.name === "string" ? record.name
    : typeof fn.tool_name === "string" ? fn.tool_name
    : typeof record.tool_name === "string" ? record.tool_name
    : undefined;
  if (!name) return undefined;
  const args = fn.arguments ?? record.arguments ?? fn.parameters ?? record.parameters ?? {};
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

type JsonSchema = {
  type?: string;
  items?: JsonSchema;
  properties?: Record<string, JsonSchema>;
};

// Small local models frequently emit near-miss argument types (a string where
// the schema wants an array, "5" for a number). Harnesses reject those calls
// outright, so coerce obvious mismatches to the declared schema, the way a
// human would read them. Only safe, information-preserving conversions.
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
    case "integer": {
      if (typeof value === "string" && value.trim() !== "" && Number.isFinite(Number(value))) return Number(value);
      return value;
    }
    case "boolean": {
      if (value === "true") return true;
      if (value === "false") return false;
      return value;
    }
    case "string": {
      if (typeof value === "number" || typeof value === "boolean") return String(value);
      return value;
    }
    case "object": {
      if (typeof value === "string") {
        const parsed = parseJsonLike(value);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) return coerceValueToSchema(parsed, schema);
        return value;
      }
      if (!value || typeof value !== "object" || Array.isArray(value)) return value;
      if (!schema.properties) return value;
      const record = value as Record<string, unknown>;
      const result: Record<string, unknown> = {};
      for (const [key, entry] of Object.entries(record)) {
        result[key] = coerceValueToSchema(entry, schema.properties[key]);
      }
      return result;
    }
    default:
      return value;
  }
}

function coerceToolCallArguments(call: ChatToolCall, request: ChatCompletionRequest): ChatToolCall {
  const schema = request.tools?.find((tool) => tool.function.name === call.function.name)?.function.parameters as JsonSchema | undefined;
  if (!schema) return call;
  const args = parseJsonLike(call.function.arguments);
  if (!args || typeof args !== "object" || Array.isArray(args)) return call;
  const coerced = coerceValueToSchema(args, { ...schema, type: schema.type ?? "object" });
  return { ...call, function: { ...call.function, arguments: JSON.stringify(coerced) } };
}

function formatStructuredContent(text: string, request: ChatCompletionRequest): string {
  if (!request.response_format || request.response_format.type === "text") return text;
  const parsed = parseJsonLike(text);
  if (parsed === undefined) return text;
  if (request.response_format.type === "json_object" && (!parsed || typeof parsed !== "object" || Array.isArray(parsed))) return text;
  return JSON.stringify(parsed);
}

function parseJsonLike(text: string): unknown {
  const trimmed = text.trim();
  for (const candidate of jsonCandidates(trimmed)) {
    try {
      return JSON.parse(candidate);
    } catch {
      // try the next candidate
    }
  }
  return undefined;
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

// Local models frequently emit almost-valid JSON with mismatched or missing
// closing brackets (e.g. `{"tool_calls":[{...}}}`). Rebuild the text tracking a
// bracket stack: insert missing closers when a closer matches a deeper opener,
// drop closers that match nothing, and append unclosed brackets at the end.
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
      if (deeper < 0) continue; // closer matches nothing: drop it
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

export type StreamDelta = { type: "content" | "reasoning"; text: string };

export type StreamFilterOptions = {
  toolMode: boolean;
  bufferAll?: boolean;
  stops?: string[];
};

type StreamMarkerAction = "think-open" | "think-close" | "channel" | "suppress" | "strip" | "stop";

type StreamMarker = { text: string; action: StreamMarkerAction; toolModeOnly?: boolean };

const streamMarkers: StreamMarker[] = [
  { text: "<think>", action: "think-open" },
  { text: "</think>", action: "think-close" },
  { text: "<|channel|>", action: "channel" },
  { text: "<channel>", action: "channel" },
  { text: "<|channel>", action: "channel" },
  { text: "<channel|>", action: "channel" },
  { text: "<tool_call>", action: "suppress" },
  { text: "<|tool_call", action: "suppress" },
  { text: "<｜tool▁calls▁begin｜>", action: "suppress" },
  { text: "<|python_tag|>", action: "suppress" },
  { text: "[TOOL_CALLS]", action: "suppress" },
  { text: "to=functions.", action: "suppress", toolModeOnly: true },
  { text: "call:", action: "suppress", toolModeOnly: true },
  { text: "<|eot_id|>", action: "strip" },
  { text: "<|im_end|>", action: "strip" },
  { text: "<|end_of_text|>", action: "strip" },
  { text: "</s>", action: "strip" },
  { text: "{", action: "suppress", toolModeOnly: true },
  { text: "```", action: "suppress", toolModeOnly: true },
];

const channelWords = ["thought", "analysis", "commentary", "final"];

type MarkerMatch = {
  index: number;
  length: number;
  action: StreamMarkerAction;
  resolved: boolean;
  target?: "content" | "reasoning";
};

export class StreamingOutputFilter {
  private pending = "";
  private mode: "content" | "reasoning" = "content";
  private heldWhitespace = "";
  private startedSegment = false;
  private separator = "";
  suppressed = false;
  stopped = false;
  emittedContent = "";
  emittedReasoning = "";

  private readonly markers: StreamMarker[];
  private readonly markerTexts: string[];

  constructor(private readonly options: StreamFilterOptions) {
    this.markers = streamMarkers.filter((marker) => !marker.toolModeOnly || options.toolMode);
    this.markerTexts = [
      ...this.markers.map((marker) => marker.text),
      ...(options.stops ?? []),
    ];
  }

  feed(chunk: string): StreamDelta[] {
    if (this.options.bufferAll || this.suppressed || this.stopped || !chunk) return [];
    this.pending += chunk;
    const deltas: StreamDelta[] = [];
    this.drain(deltas);
    return deltas;
  }

  private drain(deltas: StreamDelta[]): void {
    while (this.pending && !this.suppressed && !this.stopped) {
      const match = this.earliestMarker();
      if (!match) {
        const hold = this.partialSuffixLength();
        const emittable = this.pending.slice(0, this.pending.length - hold);
        if (!emittable) return;
        this.emitText(emittable, deltas);
        this.pending = this.pending.slice(emittable.length);
        return;
      }
      if (match.index > 0) {
        this.emitText(this.pending.slice(0, match.index), deltas);
        this.pending = this.pending.slice(match.index);
        continue;
      }
      if (!match.resolved) return;
      this.pending = this.pending.slice(match.length);
      if (match.action === "think-open") this.setMode("reasoning");
      else if (match.action === "think-close") this.setMode("content");
      else if (match.action === "channel") this.setMode(match.target ?? "content");
      else if (match.action === "stop") {
        this.stopped = true;
        return;
      } else if (match.action === "suppress") {
        this.suppressed = true;
        return;
      }
    }
  }

  private earliestMarker(): MarkerMatch | undefined {
    let best: MarkerMatch | undefined;
    const consider = (candidate: MarkerMatch) => {
      if (!best || candidate.index < best.index) best = candidate;
    };
    for (const marker of this.markers) {
      const index = this.pending.indexOf(marker.text);
      if (index < 0) continue;
      if (marker.action === "channel") {
        const resolved = this.resolveChannel(index, marker.text.length);
        if (resolved) consider(resolved);
        else consider({ index, length: marker.text.length, action: "channel", resolved: false });
        continue;
      }
      if (marker.text === "call:") {
        const resolved = this.resolveFunctionCall(index);
        if (resolved) consider(resolved);
        continue;
      }
      consider({ index, length: marker.text.length, action: marker.action, resolved: true });
    }
    for (const stop of this.options.stops ?? []) {
      if (!stop) continue;
      const index = this.pending.indexOf(stop);
      if (index >= 0) consider({ index, length: stop.length, action: "stop", resolved: true });
    }
    return best;
  }

  private resolveChannel(index: number, markerLength: number): MarkerMatch | undefined {
    const rest = this.pending.slice(index + markerLength);
    const wordMatch = rest.match(/^\s*(thought|analysis|commentary|final)\b\s*/i);
    if (wordMatch && (wordMatch[0].length < rest.length || /\s$/.test(wordMatch[0]))) {
      const target = wordMatch[1]!.toLowerCase() === "final" ? "content" : "reasoning";
      return { index, length: markerLength + wordMatch[0].length, action: "channel", resolved: true, target };
    }
    const trimmed = rest.replace(/^\s*/, "").toLowerCase();
    if (channelWords.some((word) => word.startsWith(trimmed) || trimmed.startsWith(word))) {
      return { index, length: markerLength, action: "channel", resolved: false };
    }
    return { index, length: markerLength, action: "strip", resolved: true };
  }

  private resolveFunctionCall(index: number): MarkerMatch | undefined {
    const rest = this.pending.slice(index);
    const full = rest.match(/^call::?[\w.-]+\s*\{/);
    if (full) return { index, length: 5, action: "suppress", resolved: true };
    if (/^call::?[\w.-]*\s*$/.test(rest)) {
      return { index, length: 5, action: "suppress", resolved: false };
    }
    return undefined;
  }

  private partialSuffixLength(): number {
    const max = Math.min(this.pending.length, this.maxMarkerLength());
    for (let length = max; length > 0; length -= 1) {
      const suffix = this.pending.slice(-length);
      if (this.markerTexts.some((text) => text.startsWith(suffix))) return length;
    }
    return 0;
  }

  private maxMarkerLength(): number {
    return this.markerTexts.reduce((max, text) => Math.max(max, text.length), 0);
  }

  private emitText(text: string, deltas: StreamDelta[]): void {
    let value = text;
    if (!this.startedSegment) {
      value = value.replace(/^\s+/, "");
      if (!value) return;
      this.startedSegment = true;
    }
    value = this.heldWhitespace + value;
    this.heldWhitespace = "";
    const trailing = value.match(/\s+$/)?.[0] ?? "";
    if (trailing) {
      this.heldWhitespace = trailing;
      value = value.slice(0, value.length - trailing.length);
    }
    if (!value) return;
    if (this.separator) {
      value = this.separator + value;
      this.separator = "";
    }
    if (this.mode === "content") this.emittedContent += value;
    else this.emittedReasoning += value;
    deltas.push({ type: this.mode, text: value });
  }

  private setMode(mode: "content" | "reasoning"): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.startedSegment = false;
    this.heldWhitespace = "";
    const emitted = mode === "reasoning" ? this.emittedReasoning : this.emittedContent;
    this.separator = emitted ? (mode === "reasoning" ? "\n\n" : "\n") : "";
  }
}

export function remainingDelta(full: string | null | undefined, emitted: string): string {
  if (!full) return "";
  if (!emitted) return full;
  if (full.startsWith(emitted)) return full.slice(emitted.length);
  const trimmed = emitted.trimEnd();
  if (trimmed && full.startsWith(trimmed)) return full.slice(trimmed.length);
  return "";
}
