import { describe, expect, test } from "bun:test";
import { parseAssistantOutput, remainingDelta, StreamingOutputFilter, type StreamDelta } from "./chat-compat";

function feedAll(filter: StreamingOutputFilter, chunks: string[]): StreamDelta[] {
  return chunks.flatMap((chunk) => filter.feed(chunk));
}

function joined(deltas: StreamDelta[], type: "content" | "reasoning"): string {
  return deltas.filter((delta) => delta.type === type).map((delta) => delta.text).join("");
}

describe("streaming output filter", () => {
  test("emits plain content incrementally", () => {
    const filter = new StreamingOutputFilter({ toolMode: false });
    const deltas = feedAll(filter, ["Hel", "lo", " wor", "ld!"]);
    expect(deltas.length).toBeGreaterThan(1);
    expect(joined(deltas, "content")).toBe("Hello world!");
  });

  test("splits think blocks into reasoning deltas across chunk boundaries", () => {
    const filter = new StreamingOutputFilter({ toolMode: false });
    const deltas = feedAll(filter, ["<thi", "nk>plan", " it</th", "ink>Answer", " here"]);
    expect(joined(deltas, "reasoning")).toBe("plan it");
    expect(joined(deltas, "content")).toBe("Answer here");
  });

  test("routes harmony channels split across chunks", () => {
    const filter = new StreamingOutputFilter({ toolMode: false });
    const deltas = feedAll(filter, ["<|channel|>ana", "lysis thinking now ", "<|channel|>final Done"]);
    expect(joined(deltas, "reasoning")).toBe("thinking now");
    expect(joined(deltas, "content")).toBe("Done");
  });

  test("stops emitting at stop sequences split across chunks", () => {
    const filter = new StreamingOutputFilter({ toolMode: false, stops: ["END"] });
    const deltas = feedAll(filter, ["one E", "ND two"]);
    expect(joined(deltas, "content")).toBe("one");
    expect(filter.stopped).toBe(true);
    expect(filter.feed("more")).toEqual([]);
  });

  test("suppresses tool-call JSON in tool mode", () => {
    const filter = new StreamingOutputFilter({ toolMode: true });
    const deltas = feedAll(filter, ['{"tool_calls":[{"name":"search"}]}']);
    expect(deltas).toEqual([]);
    expect(filter.suppressed).toBe(true);
  });

  test("emits leading text then suppresses at tool markers", () => {
    const filter = new StreamingOutputFilter({ toolMode: true });
    const deltas = feedAll(filter, ["Using the tool now ", '<tool_call>{"name":"search"}</tool_call>']);
    expect(joined(deltas, "content")).toBe("Using the tool now");
    expect(filter.suppressed).toBe(true);
  });

  test("buffers everything when bufferAll is set", () => {
    const filter = new StreamingOutputFilter({ toolMode: false, bufferAll: true });
    expect(feedAll(filter, ['{"a', '":1}'])).toEqual([]);
    expect(filter.emittedContent).toBe("");
  });

  test("strips end-of-turn markers from content", () => {
    const filter = new StreamingOutputFilter({ toolMode: false });
    const deltas = feedAll(filter, ["Answer.", "<|eot_id|>"]);
    expect(joined(deltas, "content")).toBe("Answer.");
  });
});

describe("remainingDelta", () => {
  test("returns the unemitted tail", () => {
    expect(remainingDelta("Hello world", "Hello")).toBe(" world");
  });

  test("returns full text when nothing was emitted", () => {
    expect(remainingDelta("Hello", "")).toBe("Hello");
  });

  test("returns empty on divergence", () => {
    expect(remainingDelta("xyz", "abc")).toBe("");
  });

  test("tolerates trailing whitespace differences", () => {
    expect(remainingDelta("Hello world", "Hello ")).toBe("world");
  });
});

describe("malformed tool JSON repair", () => {
  const request = {
    model: "test",
    stream: false,
    messages: [{ role: "user" as const, content: "weather" }],
    tools: [{ type: "function" as const, function: { name: "get_weather", parameters: { type: "object", properties: { city: { type: "string" } } } } }],
  };

  test("parses tool calls with a missing closing bracket", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}}', request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("get_weather");
    expect(parsed.toolCalls?.[0]?.function.arguments).toBe(JSON.stringify({ city: "Paris" }));
  });

  test("parses tool calls truncated before closing braces", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"', request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("get_weather");
  });

  test("parses tool calls with an extra closing brace", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}}', request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("get_weather");
  });

  test("valid JSON is not altered by repair candidates", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"get_weather","arguments":{"city":"Paris"}}]}', request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.arguments).toBe(JSON.stringify({ city: "Paris" }));
  });

  test("parses gemma-4 native tool call markers with tool_name keys", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit" };
    const parsed = parseAssistantOutput('<|tool_call>call:{"tool_name":"get_weather","arguments":{"city":"Paris"}}<tool_call|>', gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls).toHaveLength(1);
    expect(parsed.toolCalls?.[0]?.function.name).toBe("get_weather");
    expect(parsed.toolCalls?.[0]?.function.arguments).toBe(JSON.stringify({ city: "Paris" }));
  });
});

describe("schema-guided argument coercion", () => {
  const lsRequest = {
    model: "test",
    stream: false,
    messages: [{ role: "user" as const, content: "list files" }],
    tools: [{
      type: "function" as const,
      function: {
        name: "ls",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string" },
            ignore: { type: "array", items: { type: "string" } },
            depth: { type: "integer" },
            recursive: { type: "boolean" },
          },
        },
      },
    }],
  };

  test("wraps scalar into array when schema expects array", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"ls","arguments":{"path":".","ignore":"node_modules"}}]}', lsRequest);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: ".", ignore: ["node_modules"] });
  });

  test("parses stringified JSON arrays for array params", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"ls","arguments":{"ignore":"[\\"dist\\",\\"build\\"]"}}]}', lsRequest);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ ignore: ["dist", "build"] });
  });

  test("coerces numeric and boolean strings", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"ls","arguments":{"depth":"3","recursive":"true"}}]}', lsRequest);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ depth: 3, recursive: true });
  });

  test("leaves already-valid arguments untouched", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"ls","arguments":{"path":".","ignore":["x"],"depth":2,"recursive":false}}]}', lsRequest);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: ".", ignore: ["x"], depth: 2, recursive: false });
  });

  test("unknown tools pass through unchanged", () => {
    const parsed = parseAssistantOutput('{"tool_calls":[{"name":"other","arguments":{"ignore":"x"}}]}', lsRequest);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ ignore: "x" });
  });
});
