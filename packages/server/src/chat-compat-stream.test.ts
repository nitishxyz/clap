import { describe, expect, test } from "bun:test";
import { parseAssistantOutput, prepareChatRequest, remainingDelta, StreamingOutputFilter, type StreamDelta } from "./chat-compat";

function feedAll(filter: StreamingOutputFilter, chunks: string[]): StreamDelta[] {
  return chunks.flatMap((chunk) => filter.feed(chunk));
}

function joined(deltas: StreamDelta[], type: "content" | "reasoning"): string {
  return deltas.filter((delta) => delta.type === type).map((delta) => delta.text).join("");
}

describe("streaming output filter", () => {
  test("merges leading system messages for strict chat templates", () => {
    const prepared = prepareChatRequest({
      model: "mlx-community/Qwen3.6-27B-4bit",
      stream: false,
      messages: [
        { role: "system", content: "Main system prompt." },
        { role: "system", content: "Extra harness context." },
        { role: "user", content: "hi" },
      ],
    });
    const systems = prepared.messages.filter((message) => message.role === "system");
    expect(systems).toHaveLength(1);
    expect(systems[0]?.content).toContain("Main system prompt.");
    expect(systems[0]?.content).toContain("Extra harness context.");
    expect(prepared.messages[prepared.messages.length - 1]?.role).toBe("user");
  });

  test("emits plain content incrementally", () => {
    const filter = new StreamingOutputFilter({ toolMode: false });
    const deltas = feedAll(filter, ["Hel", "lo", " wor", "ld!"]);
    expect(deltas.length).toBeGreaterThan(1);
    expect(joined(deltas, "content")).toBe("Hello world!");
  });

  test("startInReasoning streams implicit think blocks as reasoning", () => {
    const filter = new StreamingOutputFilter({ toolMode: false, startInReasoning: true });
    const deltas = feedAll(filter, ["First I", " reason.", "</think>", "Then", " I answer."]);
    expect(joined(deltas, "reasoning")).toBe("First I reason.");
    expect(joined(deltas, "content")).toBe("Then I answer.");
  });

  test("implicitThink final parse treats untagged output as reasoning until close tag", () => {
    const request = { model: "qwen3.6", stream: false, messages: [{ role: "user" as const, content: "hi" }] };
    const parsed = parseAssistantOutput("I am thinking.</think>The answer is 4.", request, { implicitThink: true });
    expect(parsed.reasoning).toBe("I am thinking.");
    expect(parsed.content).toBe("The answer is 4.");
    const truncated = parseAssistantOutput("Still thinking when cut off", request, { implicitThink: true });
    expect(truncated.reasoning).toBe("Still thinking when cut off");
    expect(truncated.content).toBe("");
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

  test("parses gemma-4 calls that use the special quote token and loose keys", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit", tools: [{ type: "function" as const, function: { name: "ls", parameters: { type: "object", properties: { path: { type: "string" } } } } }] };
    const parsed = parseAssistantOutput('<|tool_call>call:ls{path:<|"|>.<|"|>}<tool_call|>', gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("ls");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: "." });
  });

  test("recovers gemma-4 tool_calls lists with misplaced arguments", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-12B-it-8bit", tools: [
      { type: "function" as const, function: { name: "glob", parameters: { type: "object", properties: { path: { type: "string" }, pattern: { type: "string" } } } } },
      { type: "function" as const, function: { name: "ls", parameters: { type: "object", properties: { path: { type: "string" }, pattern: { type: "string" } } } } },
    ] };
    const raw = '<|tool_call>call:tool_calls:[{"arguments":{"path":"."},"name":"glob","pattern":"**/*.md"},{"arguments":{"path":"docs"},"name":"ls","pattern":"**/*"}]<tool_call|>';
    const parsed = parseAssistantOutput(raw, gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls).toHaveLength(2);
    expect(parsed.toolCalls?.map((call) => call.function.name)).toEqual(["glob", "ls"]);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: ".", pattern: "**/*.md" });
    expect(JSON.parse(parsed.toolCalls?.[1]?.function.arguments ?? "{}")).toEqual({ path: "docs", pattern: "**/*" });
  });

  test("recovers gemma-4 tool_calls lists with a trailing unmatched brace", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-12B-it-8bit", tools: [{ type: "function" as const, function: { name: "glob", parameters: { type: "object", properties: { path: { type: "string" }, pattern: { type: "string" } } } } }] };
    const raw = '<|tool_call>call:tool_calls:[{"arguments":{"path":"."},"name":"glob","pattern":"docs/**"},{"arguments":{"path":"."},"name":"glob","pattern":"**/*.md"},{"arguments":{"path":"."},"name":"glob","pattern":"**/*.txt"}]}<tool_call|>';
    const parsed = parseAssistantOutput(raw, gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls).toHaveLength(3);
    expect(parsed.toolCalls?.map((call) => JSON.parse(call.function.arguments))).toEqual([
      { path: ".", pattern: "docs/**" },
      { path: ".", pattern: "**/*.md" },
      { path: ".", pattern: "**/*.txt" },
    ]);
  });

  test("recovers a prose-prefixed truncated tool call with an inferred name", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit", tools: [{
      type: "function" as const,
      function: { name: "write", parameters: { type: "object", properties: { content: { type: "string" }, path: { type: "string" }, createDirs: { type: "boolean" } } } },
    }] };
    const raw = 'I will correct this with the write tool.\n\n{"tool_calls":[{"arguments":{"content":"# Comparison","path":"NATIVE_COMPARISON.md","createDirs":true}';
    const parsed = parseAssistantOutput(raw, gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.content).toBeNull();
    expect(parsed.toolCalls?.[0]?.function.name).toBe("write");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({
      content: "# Comparison",
      path: "NATIVE_COMPARISON.md",
      createDirs: true,
    });
  });

  test("repairs gemma-4 tool_calls lists with quoted-key colons", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-12B-it-8bit", tools: [{ type: "function" as const, function: { name: "glob", parameters: { type: "object", properties: { path: { type: "string" }, pattern: { type: "string" } } } } }] };
    const raw = '<|tool_call>call:tool_calls:[{args:{path:"."},name:"glob","pattern:"**/*.md""}]<tool_call|>';
    const parsed = parseAssistantOutput(raw, gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("glob");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: ".", pattern: "**/*.md" });
  });

  test("fails explicitly instead of returning an empty malformed tool response", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-12B-it-8bit" };
    expect(() => parseAssistantOutput("<|tool_call>unrecoverable<tool_call|>", gemmaRequest)).toThrow("could not parse");
  });

  test("fails explicitly instead of leaking an unrecoverable bare tool envelope", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit" };
    expect(() => parseAssistantOutput('I will call it.\n\n{"tool_calls":[{"arguments":', gemmaRequest)).toThrow("could not parse");
  });

  test("treats observed marker-only Gemma envelopes as length truncation, not parser failure", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit" };
    for (const raw of ["<|tool_call>", "<|tool_call>call"]) {
      const parsed = parseAssistantOutput(raw, gemmaRequest, undefined, { truncated: true });
      expect(parsed).toEqual({ content: "", reasoning: undefined, finishReason: "stop" });
    }
  });

  test("does not accept marker-only or malformed envelopes without a confirmed length stop", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit" };
    expect(() => parseAssistantOutput("<|tool_call>", gemmaRequest)).toThrow("could not parse");
    expect(() => parseAssistantOutput("<|tool_call>call", gemmaRequest, undefined, { truncated: false })).toThrow("could not parse");
    expect(() => parseAssistantOutput("<|tool_call>call:arbitrary", gemmaRequest, undefined, { truncated: true })).toThrow("could not parse");
  });

  test("confirmed truncation does not alter complete calls or normal text", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit" };
    const complete = parseAssistantOutput('<|tool_call>call:{"tool_name":"get_weather","arguments":{"city":"Paris"}}<tool_call|>', gemmaRequest, undefined, { truncated: true });
    expect(complete.finishReason).toBe("tool_calls");
    expect(complete.toolCalls?.[0]?.function.name).toBe("get_weather");
    expect(complete.toolCalls?.[0]?.function.arguments).toBe(JSON.stringify({ city: "Paris" }));
    expect(parseAssistantOutput("The weather is mild.", gemmaRequest, undefined, { truncated: true })).toEqual({
      content: "The weather is mild.",
      reasoning: undefined,
      finishReason: "stop",
    });
  });

  test("bare closing channel markers return following tool JSON to content", () => {
    const gemmaRequest = { ...request, model: "mlx-community/gemma-4-e4b-it-4bit", tools: [{ type: "function" as const, function: { name: "ls", parameters: { type: "object", properties: { path: { type: "string" } } } } }] };
    const raw = '<|channel>thought\nI should list the files.<channel|>{"tool_calls":[{"name":"ls","arguments":{"path":"/tmp"}}]}';
    const parsed = parseAssistantOutput(raw, gemmaRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.reasoning).toContain("I should list the files.");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("ls");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: "/tmp" });
  });
});

describe("qwen xml function calls", () => {
  const request = {
    model: "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit",
    stream: false,
    messages: [{ role: "user" as const, content: "write a file" }],
    tools: [{ type: "function" as const, function: { name: "write_file", parameters: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } } } } }],
  };

  test("parses the coder-style xml function format", () => {
    const text = "<tool_call>\n<function=write_file>\n<parameter=path>\nhello.txt\n</parameter>\n<parameter=content>\nhi\n</parameter>\n</function>\n</tool_call>";
    const parsed = parseAssistantOutput(text, request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("write_file");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: "hello.txt", content: "hi" });
  });

  test("parses the xml format without wrapper and with truncated closing tags", () => {
    const text = "<function=write_file>\n<parameter=path>\nhello.txt\n</parameter>\n<parameter=content>\nmulti\nline";
    const parsed = parseAssistantOutput(text, request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: "hello.txt", content: "multi\nline" });
  });

  test("recovers xml tool calls emitted inside an unterminated think block", () => {
    const text = "<think>I should write the file now.\n<tool_call>\n<function=write_file>\n<parameter=path>\nhello.txt\n</parameter>\n</function>\n</tool_call>";
    const parsed = parseAssistantOutput(text, request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("write_file");
  });

  test("recovers a call object constructed in reasoning when no output was produced", () => {
    const text = '<think>Constructing the tool call:\n{\n  "name": "write_file",\n  "arguments": {\n    "path": "hello.txt",\n    "content": "hi"\n  }\n}';
    const parsed = parseAssistantOutput(text, request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("write_file");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ path: "hello.txt", content: "hi" });
  });

  test("does not invent calls from reasoning when visible content exists", () => {
    const text = '<think>Maybe {"name":"write_file","arguments":{}}</think>Here is my answer.';
    const parsed = parseAssistantOutput(text, request);
    expect(parsed.finishReason).toBe("stop");
    expect(parsed.toolCalls).toBeUndefined();
    expect(parsed.content).toBe("Here is my answer.");
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
