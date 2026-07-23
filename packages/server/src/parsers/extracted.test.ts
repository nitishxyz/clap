import { describe, expect, test } from "bun:test";
import type { ChatCompletionRequest } from "@clap/api";
import { parseAssistantOutput } from "./parser";

const tool = (name: string, parameters?: Record<string, unknown>) => ({ type: "function" as const, function: { name, parameters } });
const request = (tools = [tool("search")]): ChatCompletionRequest => ({
  model: "opaque-model",
  messages: [{ role: "user", content: "run it" }],
  stream: false,
  tools,
});

describe("native template protocols", () => {
  test("parses Harmony, Hermes, Qwen, DeepSeek, Mistral, Llama, and Gemma fixtures", async () => {
    const fixtures = await Bun.file(new URL("./fixtures/native-protocols.json", import.meta.url)).json() as Array<{ family: string; input: string; tool: string }>;
    for (const fixture of fixtures) {
      const parsed = parseAssistantOutput(fixture.input, request(), { familyHints: [fixture.family], templateInferred: true });
      expect(parsed.finishReason, fixture.family).toBe("tool_calls");
      expect(parsed.toolCalls?.[0]?.function.name, fixture.family).toBe(fixture.tool);
      expect(parsed.content, fixture.family).toBeNull();
    }
  });
});

describe("JSON and tool arguments", () => {
  test("parses multiple calls, encoded arguments, and schema coercion", async () => {
    const fixture = await Bun.file(new URL("./fixtures/json-tools.json", import.meta.url)).json();
    const parsed = parseAssistantOutput(fixture.input, request([
      tool("search", { type: "object", properties: { tags: { type: "array", items: { type: "string" } }, limit: { type: "integer" } } }),
      tool("open", { type: "object", properties: { path: { type: "string" } } }),
    ]));
    expect(parsed.toolCalls?.map((call) => call.function.name)).toEqual(fixture.expectedTools);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual(fixture.expectedSearchArguments);
  });
});

describe("reasoning, XML, and plain output", () => {
  test("extracts reasoning before parsing and ignores tool-looking JSON with visible content", async () => {
    const fixture = await Bun.file(new URL("./fixtures/reasoning.json", import.meta.url)).json();
    const parsed = parseAssistantOutput(fixture.input, request());
    expect(parsed.reasoning).toBe(fixture.expectedReasoning);
    expect(parsed.content).toBe(fixture.expectedContent);
    expect(parsed.toolCalls).toBeUndefined();
  });

  test("parses partial XML without leaking markers", async () => {
    const fixture = await Bun.file(new URL("./fixtures/xml-partial.json", import.meta.url)).json();
    const parsed = parseAssistantOutput(fixture.input, request(), { familyHints: ["qwen"], templateInferred: true });
    expect(parsed.toolCalls?.[0]?.function.name).toBe(fixture.expectedTool);
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual(fixture.expectedArguments);
    expect(parsed.content).toBeNull();
  });

  test("keeps ordinary braces as prose outside tool mode", () => {
    const parsed = parseAssistantOutput("Use {braces} in prose.", { ...request(), tools: undefined });
    expect(parsed.content).toBe("Use {braces} in prose.");
    expect(parsed.toolCalls).toBeUndefined();
  });
});
