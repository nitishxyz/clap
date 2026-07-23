import type { ChatToolCall } from "@clap/api";
import { normalizeToolCall, parseJsonLike } from "./json";
import { toolCall } from "./tool-arguments";

export function parseTaggedToolCalls(text: string): ChatToolCall[] {
  return [...text.matchAll(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g)]
    .map((match, index) => normalizeToolCall(parseJsonLike(match[1] ?? ""), index))
    .filter((call): call is ChatToolCall => Boolean(call));
}

// Qwen coder-style XML calls may omit wrappers and closing tags when output is
// truncated. Parse complete parameters and never return marker text as content.
export function parseXmlFunctionCalls(text: string): ChatToolCall[] {
  const calls: ChatToolCall[] = [];
  for (const match of text.matchAll(/<function=([\w.-]+)>\s*([\s\S]*?)(?:<\/function>|$)/g)) {
    const name = match[1];
    if (!name) continue;
    const args: Record<string, unknown> = {};
    for (const param of (match[2] ?? "").matchAll(/<parameter=([\w.-]+)>\n?([\s\S]*?)(?:\n?<\/parameter>|$(?![\s\S]))/g)) {
      const key = param[1];
      if (key) args[key] = param[2] ?? "";
    }
    calls.push(toolCall(name, args, calls.length));
  }
  return calls;
}
