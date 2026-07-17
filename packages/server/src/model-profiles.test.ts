import { afterEach, describe, expect, test } from "bun:test";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { parseAssistantOutput, profileStreamExtras, resetCompiledProfiles, StreamingOutputFilter } from "./chat-compat";

const tempHome = join(process.env.TMPDIR ?? "/tmp", `clap-profiles-test-${process.pid}`);

function writeProfile(name: string, profile: unknown): void {
  mkdirSync(join(tempHome, "profiles"), { recursive: true });
  writeFileSync(join(tempHome, "profiles", name), JSON.stringify(profile));
  process.env.CLAP_HOME = tempHome;
  resetCompiledProfiles();
}

afterEach(() => {
  delete process.env.CLAP_HOME;
  rmSync(tempHome, { recursive: true, force: true });
  resetCompiledProfiles();
});

const request = {
  model: "acme/new-model-9b",
  stream: false,
  messages: [{ role: "user" as const, content: "run it" }],
  tools: [{ type: "function" as const, function: { name: "run_task", parameters: { type: "object", properties: { id: { type: "string" } } } } }],
};

describe("user model profiles", () => {
  test("custom regex parser from a user profile parses a novel tool format", () => {
    writeProfile("acme.json", {
      name: "acme",
      families: ["acme"],
      customParsers: [{ pattern: "@@invoke\\s+(?<name>[\\w.-]+)\\s+(?<args>\\{[\\s\\S]*?\\})@@" }],
      parsers: ["json"],
    });
    const parsed = parseAssistantOutput('@@invoke run_task {"id":"42"}@@', request);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("run_task");
    expect(JSON.parse(parsed.toolCalls?.[0]?.function.arguments ?? "{}")).toEqual({ id: "42" });
  });

  test("profile markers suppress leaked protocol text and strip end tokens", () => {
    writeProfile("acme.json", {
      name: "acme",
      families: ["acme"],
      parsers: ["json"],
      markers: { suppress: ["<|acme_call|>"], strip: ["<|acme_end|>"] },
    });
    const parsed = parseAssistantOutput("Sure thing<|acme_end|>!<|acme_call|>secret protocol text", { ...request, tools: undefined });
    expect(parsed.content).toBe("Sure thing!");
  });

  test("profile implicitThink treats untagged output as reasoning", () => {
    writeProfile("acme.json", {
      name: "acme",
      families: ["acme"],
      parsers: ["json"],
      implicitThink: true,
    });
    const parsed = parseAssistantOutput("thinking hard</think>The answer.", { ...request, tools: undefined });
    expect(parsed.reasoning).toBe("thinking hard");
    expect(parsed.content).toBe("The answer.");
  });

  test("profileStreamExtras exposes markers and implicit think to the stream filter", () => {
    writeProfile("acme.json", {
      name: "acme",
      families: ["acme"],
      parsers: ["json"],
      implicitThink: true,
      markers: { suppress: ["<|acme_call|>"] },
    });
    const extras = profileStreamExtras(request.model, request);
    expect(extras.implicitThink).toBe(true);
    const filter = new StreamingOutputFilter({ toolMode: false, extraMarkers: extras.extraMarkers });
    const deltas = [...filter.feed("hello <|acme_call|>hidden"), ...filter.feed("")];
    expect(deltas.map((delta) => delta.text).join("")).toBe("hello");
    expect(filter.suppressed).toBe(true);
  });

  test("user profiles win over built-ins for the same family", () => {
    writeProfile("qwen-override.json", {
      name: "qwen",
      families: ["qwen"],
      customParsers: [{ pattern: "OVERRIDE:(?<name>[\\w.-]+)", name: undefined }],
      parsers: [],
    });
    const qwenRequest = { ...request, model: "lmstudio-community/Qwen3.6-35B-A3B-MLX-4bit" };
    const parsed = parseAssistantOutput("OVERRIDE:run_task", qwenRequest);
    expect(parsed.finishReason).toBe("tool_calls");
    expect(parsed.toolCalls?.[0]?.function.name).toBe("run_task");
  });

  test("invalid profile files are ignored without breaking parsing", () => {
    writeProfile("broken.json", { families: ["acme"] });
    const parsed = parseAssistantOutput("plain answer", { ...request, tools: undefined });
    expect(parsed.content).toBe("plain answer");
  });
});
