import { describe, expect, test } from "bun:test";
import {
  remainingDelta,
  StreamingOutputFilter,
  type StreamDelta,
  type StreamFilterOptions,
  type StreamingParserState,
} from "./stream-state";

type BoundaryFixture = {
  name: string;
  input: string;
  options: StreamFilterOptions;
  content: string;
  reasoning: string;
  state: StreamingParserState;
};

function feedAll(filter: StreamingOutputFilter, chunks: string[]): StreamDelta[] {
  return chunks.flatMap((chunk) => filter.feed(chunk));
}

function joined(deltas: StreamDelta[], type: "content" | "reasoning"): string {
  return deltas.filter((delta) => delta.type === type).map((delta) => delta.text).join("");
}

function runFilter(_input: string, options: StreamFilterOptions, chunks: string[]) {
  const filter = new StreamingOutputFilter(options);
  const deltas = feedAll(filter, chunks);
  const content = joined(deltas, "content");
  const reasoning = joined(deltas, "reasoning");
  expect(content).toBe(filter.emittedContent);
  expect(reasoning).toBe(filter.emittedReasoning);
  return {
    content: filter.emittedContent,
    reasoning: filter.emittedReasoning,
    state: filter.state,
    suppressed: filter.suppressed,
    stopped: filter.stopped,
    emittedContent: filter.emittedContent,
    emittedReasoning: filter.emittedReasoning,
  };
}

function chunkBySize(input: string, size: number): string[] {
  if (size <= 0) return [input];
  const chunks: string[] = [];
  for (let index = 0; index < input.length; index += size) {
    chunks.push(input.slice(index, index + size));
  }
  return chunks.length ? chunks : [""];
}

function chunkByCodePoints(input: string): string[] {
  return [...input];
}

describe("streaming boundary fixtures", () => {
  test("chunking strategies match whole-buffer output", async () => {
    const fixtures = await Bun.file(new URL("./fixtures/streaming-boundaries.json", import.meta.url)).json() as BoundaryFixture[];
    expect(fixtures.length).toBeGreaterThan(0);

    for (const fixture of fixtures) {
      const whole = runFilter(fixture.input, fixture.options, [fixture.input]);
      expect(whole.content, fixture.name).toBe(fixture.content);
      expect(whole.reasoning, fixture.name).toBe(fixture.reasoning);
      expect(whole.state, fixture.name).toBe(fixture.state);
      expect(whole.emittedContent, fixture.name).toBe(fixture.content);
      expect(whole.emittedReasoning, fixture.name).toBe(fixture.reasoning);

      const strategies: Array<{ label: string; chunks: string[] }> = [
        { label: "code-units", chunks: chunkBySize(fixture.input, 1) },
        { label: "code-points", chunks: chunkByCodePoints(fixture.input) },
        { label: "pairs", chunks: chunkBySize(fixture.input, 2) },
        { label: "triples", chunks: chunkBySize(fixture.input, 3) },
        { label: "sevens", chunks: chunkBySize(fixture.input, 7) },
      ];

      for (const strategy of strategies) {
        const result = runFilter(fixture.input, fixture.options, strategy.chunks);
        expect(result.content, `${fixture.name} / ${strategy.label}`).toBe(whole.content);
        expect(result.reasoning, `${fixture.name} / ${strategy.label}`).toBe(whole.reasoning);
        expect(result.state, `${fixture.name} / ${strategy.label}`).toBe(whole.state);
        expect(result.suppressed, `${fixture.name} / ${strategy.label}`).toBe(whole.suppressed);
        expect(result.stopped, `${fixture.name} / ${strategy.label}`).toBe(whole.stopped);
        expect(result.emittedContent, `${fixture.name} / ${strategy.label}`).toBe(whole.emittedContent);
        expect(result.emittedReasoning, `${fixture.name} / ${strategy.label}`).toBe(whole.emittedReasoning);
      }
    }
  });
});

describe("StreamingOutputFilter cancel", () => {
  test("drops pending text and freezes further emission", () => {
    const filter = new StreamingOutputFilter({ toolMode: false });
    expect(filter.feed("Hel")).toEqual([{ type: "content", text: "Hel" }]);
    filter.cancel();
    expect(filter.state).toBe("stopped");
    expect(filter.stopped).toBe(true);
    expect(filter.feed("lo")).toEqual([]);
    expect(filter.emittedContent).toBe("Hel");
  });
});

describe("remainingDelta", () => {
  test("returns the unemitted tail with whitespace tolerance", () => {
    expect(remainingDelta("Hello world", "Hello")).toBe(" world");
    expect(remainingDelta("Hello", "")).toBe("Hello");
    expect(remainingDelta("xyz", "abc")).toBe("");
    expect(remainingDelta("Hello world", "Hello ")).toBe("world");
    expect(remainingDelta(null, "x")).toBe("");
  });
});
