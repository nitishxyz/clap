import type { ChatCompletionRequest, ChatMessage, ChatToolCall } from "@clap/api";
import { builtinProfiles, genericProfile, loadUserProfiles, resetUserProfileCache, type CustomParserSpec, type ModelProfileDefinition } from "./model-profiles";
import { formatStructuredContent } from "./parsers/json";
import { hasExplicitParserMarker, runToolParsers, toolParserPrimitives } from "./parsers/native";
import { applyProfileMarkers, applyProfileReplacements, cleanupProtocolText, suppressProtocolMarkers } from "./parsers/plain";
import { extractReasoning, recoverToolCallsFromReasoning } from "./parsers/reasoning";
import { selectRegisteredParser } from "./parsers/registry";
import { coerceToolCallArguments, parseLooseArgs, toolCall } from "./parsers/tool-arguments";
import type { AssistantOutputParser, ParsedAssistantOutput, ParserTemplateInfo, ToolParser } from "./parsers/types";
export type { ParsedAssistantOutput, ParserTemplateInfo } from "./parsers/types";

export type PrepareChatOptions = {
  // The runtime renders tool declarations natively through the model's chat
  // template (MLX + template with tool support). Injecting a second JSON
  // tool-call convention on top confuses small models into narrating calls
  // as text, so the compat instruction block is skipped.
  nativeTools?: boolean;
};

export function prepareChatRequest(request: ChatCompletionRequest, options: PrepareChatOptions = {}): ChatCompletionRequest {
  rejectUnsupportedContentParts(request);
  const explicitBoundaries = request.cache?.boundaries;
  let messages = request.messages.map((message) => ({
    ...message,
    content: stringifyMessageContent(message, options),
  }));
  const instructions = compatibilityInstructions(request, options);
  if (instructions) {
    messages.unshift({ role: "system", content: instructions });
  }
  // Explicit message indexes refer to the caller's original zero-based array.
  // Keep those message boundaries representable and account only for the
  // deterministic compatibility system message added above. The worker still
  // owns rendering, tokenization, and exact-prefix validation.
  if (!explicitBoundaries?.length) messages = mergeLeadingSystemMessages(messages);
  const cache = request.cache && instructions ? {
    ...request.cache,
    boundaries: explicitBoundaries?.map((boundary) => boundary.kind === "messages"
      ? { ...boundary, through_message: boundary.through_message + 1 }
      : boundary),
  } : request.cache;
  return { ...request, messages, cache };
}

// Clients such as agent harnesses send multiple system messages, and we may
// prepend compatibility instructions as another. Strict chat templates
// (e.g. Qwen3.6) reject any system message that is not the single first
// message, so collapse the leading run into one.
function mergeLeadingSystemMessages<T extends { role: string; content: string | null }>(messages: T[]): T[] {
  let count = 0;
  while (count < messages.length && messages[count]?.role === "system") count += 1;
  if (count <= 1) return messages;
  const leading = messages.slice(0, count);
  const merged = {
    ...leading[0],
    content: leading.map((message) => message.content ?? "").filter(Boolean).join("\n\n"),
  } as T;
  return [merged, ...messages.slice(count)];
}

export function parseAssistantOutput(text: string, request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo, options?: { truncated?: boolean }): ParsedAssistantOutput {
  const stopped = applyStop(text, request.stop);
  const parser = selectParser(request.model, request, templateInfo);
  // Templates such as Qwen3.6 pre-fill the opening <think> tag inside the
  // generation prompt, so the model's raw output starts mid-reasoning without
  // the tag. Restore it so reasoning extraction sees the full block.
  const implicitThink = templateInfo?.implicitThink || parser.profile?.implicitThink;
  const input = implicitThink && !/<think>/i.test(stopped) ? `<think>${stopped}` : stopped;
  const context = { request, toolMode: Boolean(request.tools?.length) };
  const normalized = parser.parse(input, context);
  const toolCalls = (normalized.toolCalls ?? []).map((call) => coerceToolCallArguments(call, request));
  if (toolCalls.length) {
    return { content: null, reasoning: normalized.reasoning, toolCalls, finishReason: "tool_calls" };
  }
  if (context.toolMode && (hasExplicitParserMarker(input) || /["']?tool_calls["']?\s*:/.test(normalized.content))) {
    // A max-token stop can land after Gemma's opening protocol token but
    // before any function name or arguments. This is not a malformed call to
    // repair and should not become a 500: suppress only a marker-only prefix
    // when the backend explicitly confirmed truncation. Complete malformed
    // envelopes and arbitrary marked prose still fail below.
    if (options?.truncated && /^<\|tool_call>\s*(?:call)?\s*$/u.test(input.trim())) {
      return { content: "", reasoning: normalized.reasoning, finishReason: "stop" };
    }
    throw new Error("model emitted a tool call that Clap could not parse; retry the request");
  }

  const formatted = formatStructuredContent(normalized.content, request);
  return { content: formatted, reasoning: normalized.reasoning, finishReason: "stop" };
}

// Parser registry follows the same broad shape as OSS servers such as Ollama,
// vLLM, SGLang, and Unsloth: choose a model/template-aware parser first, then
// extract reasoning before interpreting remaining content as tool calls.
export function selectParser(model: string, _request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo): AssistantOutputParser {
  return selectRegisteredParser({
    model,
    request: _request,
    traits: templateInfo,
    user: userRegistry(),
    builtin: builtinRegistry,
    generic: genericParser,
    plain: plainParser,
  });
}

// Extra streaming behavior derived from the selected profile: additional
// markers to suppress/strip mid-stream and implicit reasoning mode.
export function profileStreamExtras(model: string, request: ChatCompletionRequest, templateInfo?: ParserTemplateInfo): { extraMarkers: StreamMarker[]; implicitThink: boolean } {
  const parser = selectParser(model, request, templateInfo);
  const markers = parser.profile?.markers;
  const extraMarkers: StreamMarker[] = [
    ...(markers?.suppress ?? []).map((text) => ({ text, action: "suppress" as const })),
    ...(markers?.strip ?? []).map((text) => ({ text, action: "strip" as const })),
  ];
  return { extraMarkers, implicitThink: Boolean(templateInfo?.implicitThink || parser.profile?.implicitThink) };
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

function stringifyMessageContent(message: ChatMessage, options: PrepareChatOptions = {}): string {
  const content = Array.isArray(message.content)
    ? message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n")
    : message.content ?? "";
  // Without native template tool support the transcript must show the
  // assistant's own tool-call turns; harnesses send them with null content,
  // and a dropped turn teaches the model that narration ends the job.
  if (message.role === "assistant" && message.tool_calls?.length && !options.nativeTools) {
    const calls = message.tool_calls.map((call) => ({
      name: call.function.name,
      arguments: parseArgumentsForTranscript(call.function.arguments),
    }));
    const rendered = JSON.stringify({ tool_calls: calls });
    return content ? `${content}\n${rendered}` : rendered;
  }
  if (message.role !== "tool") return content;
  return `Tool result${message.tool_call_id ? ` (${message.tool_call_id})` : ""}: ${content}`;
}

function parseArgumentsForTranscript(raw: string): unknown {
  try {
    return JSON.parse(raw);
  } catch {
    return raw;
  }
}

function compatibilityInstructions(request: ChatCompletionRequest, options: PrepareChatOptions = {}): string {
  const blocks: string[] = [];
  if (request.tools?.length && !options.nativeTools) {
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
  if (request.tools?.length && options.nativeTools) {
    const extras = [
      request.tool_choice && request.tool_choice !== "auto" ? `Tool choice: ${JSON.stringify(request.tool_choice)}` : "",
      request.parallel_tool_calls === false ? "Call at most one tool." : "",
    ].filter(Boolean);
    if (extras.length) blocks.push(extras.join("\n"));
  }
  if (request.response_format?.type === "json_object") {
    blocks.push("Respond with a valid JSON object and no surrounding markdown.");
  }
  if (request.response_format?.type === "json_schema") {
    blocks.push(`Respond with JSON matching this schema and no surrounding markdown: ${JSON.stringify(request.response_format.json_schema)}`);
  }
  return blocks.join("\n\n");
}

function compileProfile(profile: ModelProfileDefinition): AssistantOutputParser {
  const parsers: ToolParser[] = [
    ...(profile.customParsers ?? []).map(makeRegexToolParser),
    ...(profile.parsers ?? []).map((name) => {
      const parser = toolParserPrimitives[name];
      if (!parser) console.error(`[clap] model profile "${profile.name}": unknown parser primitive "${name}"`);
      return parser;
    }).filter((parser): parser is ToolParser => Boolean(parser)),
  ];
  return createParser(profile.name, profile.families ?? [], parsers, profile);
}

// Compile a JSON custom parser spec into a ToolParser. Named groups:
// (?<name>...) captures the tool name (or use spec.name for a fixed tool),
// (?<args>...) captures the argument payload.
function makeRegexToolParser(spec: CustomParserSpec): ToolParser {
  let regex: RegExp;
  try {
    const flags = spec.flags ?? "g";
    regex = new RegExp(spec.pattern, flags.includes("g") ? flags : `${flags}g`);
  } catch (error) {
    console.error(`[clap] invalid custom parser pattern ${JSON.stringify(spec.pattern)}: ${error}`);
    return () => [];
  }
  return (text) => {
    const calls: ChatToolCall[] = [];
    for (const match of text.matchAll(regex)) {
      const name = match.groups?.name ?? spec.name;
      if (!name) continue;
      calls.push(toolCall(name, parseLooseArgs(match.groups?.args ?? "{}"), calls.length));
    }
    return calls;
  };
}

const genericParser = compileProfile(genericProfile);
const plainParser = compileProfile({ ...genericProfile, name: "plain" });
const builtinRegistry = builtinProfiles.map(compileProfile);

let compiledUserRegistry: AssistantOutputParser[] | undefined;

function userRegistry(): AssistantOutputParser[] {
  compiledUserRegistry ??= loadUserProfiles().map(compileProfile);
  return compiledUserRegistry;
}

export function resetCompiledProfiles(): void {
  compiledUserRegistry = undefined;
  resetUserProfileCache();
}
function createParser(name: string, families: string[], toolParsers: ToolParser[], profile?: ModelProfileDefinition): AssistantOutputParser {
  return {
    name,
    families,
    toolParsers,
    profile,
    parse(rawText, context) {
      const text = applyProfileReplacements(rawText, profile);
      const reasoned = extractReasoning(text);
      let toolCalls = runToolParsers(reasoned.content, context, toolParsers);
      const visible = applyProfileMarkers(suppressProtocolMarkers(reasoned.content), profile);
      let content = toolCalls.length ? "" : cleanupProtocolText(visible);
      if (!toolCalls.length && reasoned.reasoning) {
        toolCalls = recoverToolCallsFromReasoning(text, reasoned.reasoning, context, toolParsers, content);
        if (toolCalls.length) content = "";
      }

      return {
        content,
        reasoning: reasoned.reasoning,
        toolCalls: toolCalls.length ? toolCalls : undefined,
      };
    },
  };
}

export type StreamDelta = { type: "content" | "reasoning"; text: string };

export type StreamFilterOptions = {
  toolMode: boolean;
  bufferAll?: boolean;
  stops?: string[];
  startInReasoning?: boolean;
  extraMarkers?: StreamMarker[];
};

type StreamMarkerAction = "think-open" | "think-close" | "channel" | "suppress" | "strip" | "stop";

export type StreamMarker = { text: string; action: StreamMarkerAction; toolModeOnly?: boolean };

const streamMarkers: StreamMarker[] = [
  { text: "<think>", action: "think-open" },
  { text: "</think>", action: "think-close" },
  { text: "<|channel|>", action: "channel" },
  { text: "<channel>", action: "channel" },
  { text: "<|channel>", action: "channel" },
  { text: "<channel|>", action: "channel" },
  { text: "<tool_call>", action: "suppress" },
  { text: "<|tool_call", action: "suppress" },
  { text: "<function=", action: "suppress" },
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
    this.markers = [...streamMarkers, ...(options.extraMarkers ?? [])].filter((marker) => !marker.toolModeOnly || options.toolMode);
    this.markerTexts = [
      ...this.markers.map((marker) => marker.text),
      ...(options.stops ?? []),
    ];
    if (options.startInReasoning) this.mode = "reasoning";
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
