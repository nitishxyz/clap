import { describe, expect, test } from "bun:test";
import type { ClapModel } from "@clap/api";
import { formatModelList, formatModelListJson } from "./models-output";

function model(entry: Pick<ClapModel, "id" | "object" | "displayName" | "backend" | "format" | "status">): ClapModel {
  return {
    ...entry,
    name: entry.displayName,
    provider: "test",
    source: { type: "local" },
    modalities: { input: ["text"], output: ["text"] },
    capabilities: {
      chat: true,
      completion: false,
      streaming: true,
      temperature: true,
      system_prompt: true,
      attachment: false,
      reasoning: false,
      tool_call: false,
      structured_output: false,
    },
    limit: { context: null, output: null },
  };
}

describe("CLI model list formatting", () => {
  test("prints model ids with backend, format, and status", () => {
    expect(formatModelList([model({
      id: "llama3.2:3b",
      object: "model",
      displayName: "Llama 3.2 3B",
      backend: "llama",
      format: "gguf",
      status: "available",
    })])).toBe([
      "models:",
      "  id           backend  format  status",
      "  llama3.2:3b  llama    gguf    available",
    ].join("\n"));
  });

  test("prints aliases in a separate section when requested", () => {
    expect(formatModelList([], [model({
      id: "qwen2.5:3b",
      object: "model",
      displayName: "Qwen 2.5 3B",
      backend: "mlx",
      format: "mlx",
      status: "not_downloaded",
    })])).toBe([
      "models:",
      "  (none)",
      "aliases:",
      "  id          backend  format  status",
      "  qwen2.5:3b  mlx      mlx     not_downloaded",
    ].join("\n"));
  });

  test("prints rich JSON for --json", () => {
    const body = JSON.parse(formatModelListJson([model({
      id: "llama3.2:3b",
      object: "model",
      displayName: "Llama 3.2 3B",
      backend: "llama",
      format: "gguf",
      status: "available",
    })]));

    expect(body.models[0].capabilities).toMatchObject({ chat: true, attachment: false });
    expect(body.models[0].modalities).toEqual({ input: ["text"], output: ["text"] });
    expect(body.models[0].limit).toEqual({ context: null, output: null });
  });
});
