import { describe, expect, test } from "bun:test";
import type { ChatCompletionRequest } from "@clap/api";
import type { ResolvedModel } from "@clap/models";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseAssistantOutput, selectParser } from "./parser";
import { selectRegisteredParser } from "./registry";
import { inferParserFamilies, resolveParserTemplateInfo } from "./traits";
import type { AssistantOutputParser } from "./types";

const textRequest: ChatCompletionRequest = {
  model: "untyped",
  messages: [{ role: "user", content: "hello" }],
  stream: false,
};

function candidate(name: string, families: string[] = []): AssistantOutputParser {
  return {
    name,
    families,
    toolParsers: [],
    parse: (content) => ({ content }),
  };
}

describe("parser registry", () => {
  test("model names alone cannot select built-in families", () => {
    for (const model of ["qwen/Qwen3", "deepseek-r1", "mistral-7b", "functiongemma", "meta-llama/Llama-3"]) {
      expect(selectParser(model, { ...textRequest, model }).name).toBe("plain");
    }
  });

  test("resolved traits record metadata sources and template inference", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-parser-traits-"));
    try {
      await writeFile(join(dir, "config.json"), JSON.stringify({ model_type: "unknown" }));
      await writeFile(join(dir, "tokenizer_config.json"), JSON.stringify({ chat_template: "<|tool_call_start|>{{ tools }}<|tool_call_end|>" }));
      const traits = await resolveParserTemplateInfo({ input: dir, modelPath: dir } as ResolvedModel);
      expect(traits).toMatchObject({
        familyHints: ["qwen"],
        hasToolCalls: true,
        templateInferred: true,
        sourceFiles: ["tokenizer_config.json", "config.json"],
        sources: [
          { file: "tokenizer_config.json", kind: "template" },
          { file: "config.json", kind: "config" },
        ],
      });
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("template markers select a matching parser", () => {
    const families = inferParserFamilies("<|tool_call_start|>", "");
    expect(families).toEqual(["qwen"]);
    expect(selectParser("opaque-model", textRequest, { familyHints: families, templateInferred: true }).name).toBe("qwen");
  });

  test("an exact user profile wins before template traits", () => {
    const selected = selectRegisteredParser({
      model: "acme/special-model",
      request: textRequest,
      traits: { familyHints: ["qwen"] },
      user: [candidate("custom", ["acme/special-model"])],
      builtin: [candidate("qwen", ["qwen"])],
      generic: candidate("generic"),
      plain: candidate("plain"),
    });
    expect(selected.name).toBe("custom");
  });

  test("structured requests use generic and untyped text falls back to plain", () => {
    expect(selectParser("opaque", textRequest).name).toBe("plain");
    expect(selectParser("opaque", {
      ...textRequest,
      tools: [{ type: "function", function: { name: "search" } }],
    }).name).toBe("generic");
  });
});

describe("initial parser fixtures", () => {
  test("plain text", async () => {
    const fixture = await Bun.file(new URL("./fixtures/plain-text.json", import.meta.url)).json();
    const request = { ...textRequest, model: fixture.model };
    expect(selectParser(fixture.model, request).name).toBe(fixture.expectedParser);
    expect(parseAssistantOutput(fixture.input, request)).toMatchObject(fixture.expected);
  });

  test("template-selected qwen tool call", async () => {
    const fixture = await Bun.file(new URL("./fixtures/qwen-template-tool.json", import.meta.url)).json();
    const request: ChatCompletionRequest = {
      ...textRequest,
      model: fixture.model,
      tools: [{ type: "function", function: { name: "search" } }],
    };
    const traits = { familyHints: fixture.familyHints, templateInferred: true };
    expect(selectParser(fixture.model, request, traits).name).toBe(fixture.expectedParser);
    const parsed = parseAssistantOutput(fixture.input, request, traits);
    expect(parsed.finishReason).toBe(fixture.expected.finishReason);
    expect(parsed.toolCalls?.[0]?.function.name).toBe(fixture.expected.toolName);
  });
});
