import { describe, expect, test } from "bun:test";
import type { ModelResolveResponse } from "@clap/api";
import { chooseOptionByInput, findOptionByQuant, formatResolveOptions, supportedOptions } from "./resolver-output";

const response: ModelResolveResponse = {
  model: "acme/model",
  repo: "acme/model",
  options: [
    {
      id: "acme/model:mlx",
      model: "acme/model",
      backend: "mlx",
      format: "mlx",
      repo: "acme/model-mlx",
      sizeBytes: 1024,
      supported: true,
      recommended: true,
      reason: "MLX is preferred on macOS arm64.",
    },
    {
      id: "acme/model:q4",
      model: "acme/model",
      backend: "gguf",
      format: "gguf",
      repo: "acme/model-gguf",
      file: "model-Q4_K_M.gguf",
      sizeBytes: 2048,
      quantization: "Q4_K_M",
      supported: true,
      recommended: false,
      reason: "GGUF Q4_K_M artifact is runnable with llama.cpp.",
    },
    {
      id: "acme/model:source",
      model: "acme/model",
      backend: "mlx",
      format: "safetensors",
      repo: "acme/model",
      sizeBytes: 4096,
      supported: false,
      unsupportedReason: "Raw safetensors repos are not directly runnable yet.",
      recommended: false,
      reason: "Source weights were found.",
    },
  ],
  selected: undefined,
};

describe("resolver output", () => {
  test("formats supported choices separately from unsupported guidance", () => {
    const text = formatResolveOptions(response);
    expect(text).toContain("Supported runnable options:");
    expect(text).toContain("* 1. mlx/mlx 1.00 KiB (recommended)");
    expect(text).toContain("  2. gguf/gguf Q4_K_M 2.00 KiB model-Q4_K_M.gguf (available)");
    expect(text).toContain("Unsupported / guidance:");
    expect(text).toContain("Raw safetensors repos are not directly runnable yet.");
  });

  test("excludes unsupported options from selectable options", () => {
    expect(supportedOptions(response).map((option) => option.format)).toEqual(["mlx", "gguf"]);
  });

  test("selects default/recommended or numbered choices", () => {
    const options = supportedOptions(response);
    expect(chooseOptionByInput(options, "").backend).toBe("mlx");
    expect(chooseOptionByInput(options, "2").file).toBe("model-Q4_K_M.gguf");
    expect(() => chooseOptionByInput(options, "3")).toThrow("expected a choice");
  });

  test("finds explicit quant overrides", () => {
    const options = supportedOptions(response);
    expect(findOptionByQuant(options, "q4_k_m")?.file).toBe("model-Q4_K_M.gguf");
    expect(findOptionByQuant(options, "Q5_K_M")).toBeUndefined();
  });
});
