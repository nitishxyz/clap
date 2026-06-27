import type { ChatCompletionRequest, ChatMessage, ChatToolCall } from "@clap/api";

export type ParsedAssistantOutput = {
  content: string | null;
  reasoning?: string;
  toolCalls?: ChatToolCall[];
  finishReason: "stop" | "tool_calls";
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
  return { ...request, messages };
}

export function parseAssistantOutput(text: string, request: ChatCompletionRequest): ParsedAssistantOutput {
  const stopped = applyStop(text, request.stop);
  const { content, reasoning } = stripReasoning(stopped);
  const toolCalls = request.tools?.length ? parseToolCalls(content) : undefined;
  if (toolCalls?.length) {
    return { content: null, reasoning, toolCalls, finishReason: "tool_calls" };
  }

  const formatted = formatStructuredContent(content, request);
  return { content: formatted, reasoning, finishReason: "stop" };
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

function stripReasoning(text: string): { content: string; reasoning?: string } {
  const think = text.match(/<think>([\s\S]*?)<\/think>/i);
  if (!think) return { content: text.trim() };
  return {
    reasoning: think[1]?.trim(),
    content: text.replace(think[0], "").trim(),
  };
}

function parseToolCalls(text: string): ChatToolCall[] | undefined {
  const parsed = parseJsonLike(text);
  if (!parsed || typeof parsed !== "object") return undefined;
  const value = parsed as Record<string, unknown>;
  const rawCalls = Array.isArray(value.tool_calls) ? value.tool_calls : Array.isArray(value.tools) ? value.tools : undefined;
  if (rawCalls) return rawCalls.map((call, index) => normalizeToolCall(call, index)).filter((call): call is ChatToolCall => Boolean(call));
  const single = normalizeToolCall(value, 0);
  return single ? [single] : undefined;
}

function normalizeToolCall(value: unknown, index: number): ChatToolCall | undefined {
  if (!value || typeof value !== "object") return undefined;
  const record = value as Record<string, unknown>;
  const fn = record.function && typeof record.function === "object" ? record.function as Record<string, unknown> : record;
  const name = typeof fn.name === "string" ? fn.name : typeof record.name === "string" ? record.name : undefined;
  if (!name) return undefined;
  const args = fn.arguments ?? record.arguments ?? {};
  return {
    id: typeof record.id === "string" ? record.id : `call_${index}_${crypto.randomUUID().replace(/-/g, "").slice(0, 12)}`,
    type: "function",
    function: {
      name,
      arguments: typeof args === "string" ? args : JSON.stringify(args ?? {}),
    },
  };
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
  return candidates;
}
