import { describe, expect, test } from "bun:test";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { isMlxModelDirectory } from "./index";

describe("MLX runtime", () => {
  test("validates that MLX directories include config, tokenizer, and safetensors", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-model-"));
    await writeFile(join(dir, "config.json"), "{}");
    await writeFile(join(dir, "tokenizer.json"), "{}");

    expect(await isMlxModelDirectory(dir)).toBe(false);
  });
});
