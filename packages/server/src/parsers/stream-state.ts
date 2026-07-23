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

export type StreamingParserState = "content" | "reasoning" | "possible_marker" | "suppressed_tool" | "stopped";

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
  state: StreamingParserState = "content";
  suppressed = false;
  stopped = false;
  emittedContent = "";
  emittedReasoning = "";

  private readonly markers: StreamMarker[];
  private readonly markerTexts: string[];

  constructor(private readonly options: StreamFilterOptions) {
    this.markers = [...streamMarkers, ...(options.extraMarkers ?? [])].filter((marker) => !marker.toolModeOnly || options.toolMode);
    this.markerTexts = [...this.markers.map((marker) => marker.text), ...(options.stops ?? [])];
    if (options.startInReasoning) {
      this.mode = "reasoning";
      this.state = "reasoning";
    }
  }

  feed(chunk: string): StreamDelta[] {
    if (this.options.bufferAll || this.suppressed || this.stopped || !chunk) return [];
    this.pending += chunk;
    const deltas: StreamDelta[] = [];
    this.drain(deltas);
    this.updateState();
    return deltas;
  }

  cancel(): void {
    this.pending = "";
    this.heldWhitespace = "";
    this.separator = "";
    this.stopped = true;
    this.state = "stopped";
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

  private updateState(): void {
    if (this.suppressed) this.state = "suppressed_tool";
    else if (this.stopped) this.state = "stopped";
    else if (this.pending) this.state = "possible_marker";
    else this.state = this.mode;
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
    if (/^call::?[\w.-]*\s*$/.test(rest)) return { index, length: 5, action: "suppress", resolved: false };
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
