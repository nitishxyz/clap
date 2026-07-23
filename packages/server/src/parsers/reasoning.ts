import type { ChatToolCall } from "@clap/api";
import { extractBalancedObject, normalizeToolCall, parseJsonLike, parseJsonToolCalls } from "./json";
import { hasExplicitParserMarker, runToolParsers } from "./native";
import { cleanupProtocolText } from "./plain";
import type { ParserContext, ToolParser } from "./types";

export function extractReasoning(text: string): { content: string; reasoning?: string } {
  const pieces = extractChannelReasoning(text);
  const withThink = extractThinkReasoning(pieces.content, pieces.reasoning);
  return { content: withThink.content, reasoning: withThink.reasoning };
}

// Reasoning is removed before tool parsing. Explicit native envelopes can be
// recovered from reasoning, but unmarked tool-looking JSON is ignored whenever
// visible content exists.
export function recoverToolCallsFromReasoning(rawText: string, reasoning: string, context: ParserContext, toolParsers: ToolParser[], content: string): ChatToolCall[] {
  if (hasExplicitParserMarker(rawText)) {
    const calls = runToolParsers(rawText, context, toolParsers.filter((parser) => parser !== parseJsonToolCalls));
    if (calls.length) return calls;
  }
  if (!context.toolMode || content) return [];
  const object = trailingBalancedObject(reasoning);
  if (!object) return [];
  const call = normalizeToolCall(parseJsonLike(object), 0);
  return call ? [call] : [];
}

function extractChannelReasoning(text: string): { content: string; reasoning?: string } {
  const channelPattern = /<\|?channel\|?>\s*(thought|analysis|commentary|final)?\s*/gi;
  const matches = [...text.matchAll(channelPattern)];
  if (!matches.some((match) => match[1])) return { content: text };
  const reasoning: string[] = [];
  const content: string[] = [];
  for (let i = 0; i < matches.length; i += 1) {
    const match = matches[i]!;
    const channel = match[1]?.toLowerCase() ?? "final";
    const start = (match.index ?? 0) + match[0].length;
    if (i === 0 && (match.index ?? 0) > 0) content.push(text.slice(0, match.index));
    const segment = text.slice(start, matches[i + 1]?.index ?? text.length);
    const toolStart = findProtocolStart(segment);
    const body = toolStart >= 0 ? segment.slice(0, toolStart) : segment;
    const remainder = toolStart >= 0 ? segment.slice(toolStart) : "";
    if (channel === "final") content.push(body, remainder);
    else {
      const cleaned = cleanupProtocolText(body);
      if (cleaned) reasoning.push(cleaned);
      if (remainder) content.push(remainder);
    }
  }
  return { content: content.join("\n").trim(), reasoning: reasoning.join("\n\n") || undefined };
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

function trailingBalancedObject(text: string): string | undefined {
  const trimmed = text.trimEnd();
  for (let index = trimmed.lastIndexOf("{"), attempts = 0; index >= 0 && attempts < 24; index = trimmed.lastIndexOf("{", index - 1), attempts += 1) {
    const candidate = extractBalancedObject(trimmed.slice(index));
    if (candidate && index + candidate.length === trimmed.length) return candidate;
  }
  return undefined;
}

function findProtocolStart(text: string): number {
  const starts = [
    text.search(/<\|?tool_call\|?>/), text.search(/to=functions\.[\w.-]+/), text.search(/<tool_call>/),
    text.search(/<function=[\w.-]+>/), text.search(/<\|tool_call_start\|>/), text.search(/<｜tool▁calls▁begin｜>/),
    text.search(/<\|python_tag\|>/), text.search(/\[TOOL_CALLS\]/), text.search(/call::?[\w.-]+\s*\{/),
  ].filter((index) => index >= 0);
  return starts.length ? Math.min(...starts) : -1;
}
