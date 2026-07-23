import type { ChatToolCall } from "@clap/api";
import { extractBalancedObject, normalizeToolCall, parseJsonLike, parseJsonToolCalls } from "./json";
import { cleanupProtocolText } from "./plain";
import { parseLooseArgs, parsePythonArgs, toolCall } from "./tool-arguments";
import type { ParserContext, ToolParser } from "./types";
import { parseTaggedToolCalls, parseXmlFunctionCalls } from "./xml";

export const toolParserPrimitives: Record<string, ToolParser> = {
  harmony: parseHarmonyToolCalls,
  "function-message": parseFunctionMessageCalls,
  "xml-function": parseXmlFunctionCalls,
  "tagged-json": parseTaggedToolCalls,
  "qwen-bracket": parseQwenToolCalls,
  deepseek: parseDeepSeekToolCalls,
  mistral: parseMistralToolCalls,
  "python-tag": parsePythonTagToolCalls,
  "gemma-call": parseFunctionGemmaCalls,
  json: parseJsonToolCalls,
};

export function runToolParsers(text: string, context: ParserContext, parsers: ToolParser[]): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  for (const parser of parsers) {
    if (parser === parseJsonToolCalls && !context.toolMode) continue;
    if ((parser === parsePythonTagToolCalls || parser === parseFunctionGemmaCalls) && !context.toolMode && !hasExplicitParserMarker(text)) continue;
    calls.push(...parser(text, context));
    if (calls.length) break;
  }
  return calls.map((call, index) => ({ ...call, id: call.id.startsWith("call_") ? call.id.replace(/^call_\d+_/, `call_${index}_`) : call.id }));
}

export function hasExplicitParserMarker(text: string): boolean {
  return /<\|python_tag\|>|call:[\w.-]+\s*\{|<\|?tool_call\|?>|<tool_call>|<function=[\w.-]+>|\[TOOL_CALLS\]/.test(text);
}

export function parseHarmonyToolCalls(text: string): ChatToolCall[] {
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

export function parseQwenToolCalls(text: string): ChatToolCall[] {
  return [...text.matchAll(/<\|tool_call_start\|>\s*([\s\S]*?)\s*<\|tool_call_end\|>/g)]
    .map((match, index) => normalizeToolCall(parseJsonLike(match[1] ?? ""), index))
    .filter((call): call is ChatToolCall => Boolean(call));
}

export function parseDeepSeekToolCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const pattern = /<｜tool▁call▁begin｜>\s*function\s+([\w.-]+)\s*```(?:json)?\s*([\s\S]*?)```\s*<｜tool▁call▁end｜>/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (name) calls.push(toolCall(name, parseLooseArgs(match[2] ?? "{}"), calls.length));
  }
  return calls;
}

export function parsePythonTagToolCalls(text: string): ChatToolCall[] {
  const tag = text.match(/<\|python_tag\|>\s*([\s\S]*)/);
  if (!tag?.[1]) return [];
  const body = cleanupProtocolText(tag[1]);
  const json = normalizeToolCall(parseJsonLike(body), 0);
  if (json) return [json];
  const call = body.match(/^([\w.-]+)\((.*)\)$/s);
  if (!call?.[1]) return [];
  return [toolCall(call[1], parsePythonArgs(call[2] ?? ""), 0)];
}

export function parseMistralToolCalls(text: string): ChatToolCall[] {
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

export function parseFunctionGemmaCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const listMarker = text.match(/call:tool_calls:\s*\[/);
  if (listMarker?.index !== undefined) {
    const start = text.indexOf("[", listMarker.index);
    const close = text.indexOf("<tool_call|>", start);
    const body = text.slice(start, close >= 0 ? close : undefined);
    const parsed = parseJsonLike(body) ?? parseLooseArgs(body);
    if (Array.isArray(parsed)) return parsed.map((call, index) => normalizeToolCall(call, index)).filter((call): call is ChatToolCall => Boolean(call));
  }
  const pattern = /call:([\w.-]+)\s*(\{[\s\S]*?\})/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (name) calls.push(toolCall(name, parseLooseArgs(match[2] ?? "{}"), calls.length));
  }
  if (calls.length) return calls;
  for (const match of text.matchAll(/call:\s*(?=\{)/g)) {
    const body = extractBalancedObject(text.slice(match.index));
    if (!body) continue;
    const parsed = normalizeToolCall(parseJsonLike(body), calls.length);
    if (parsed) calls.push(parsed);
  }
  return calls;
}

export function parseFunctionMessageCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  const pattern = /to=functions\.([\w.-]+)[\s\S]*?<\|message\|>\s*([\s\S]*?)\s*<\|call\|>/g;
  for (const match of text.matchAll(pattern)) {
    const name = match[1];
    if (name) calls.push(toolCall(name, parseLooseArgs(match[2] ?? "{}"), calls.length));
  }
  return calls;
}
