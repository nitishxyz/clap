import { describe, expect, test } from "bun:test";
import { readFile, readdir } from "node:fs/promises";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "..", "..", "..", "..");
const owner = "packages/runtime-router/src/process/resident-worker-process.ts";

async function typescriptFiles(directory: string): Promise<string[]> {
  const entries = await readdir(resolve(root, directory), { withFileTypes: true });
  const files = await Promise.all(entries.map((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? typescriptFiles(path) : path.endsWith(".ts") ? [path] : [];
  }));
  return files.flat();
}

describe("native inference process ownership", () => {
  test("keeps native worker spawn and request tracking in the resident process", async () => {
    const files = await typescriptFiles("packages");
    const violations: string[] = [];
    for (const path of files.filter((item) => !item.endsWith(".test.ts"))) {
      const source = await readFile(resolve(root, path), "utf8");
      const nativeWorkerFile = /runtime-(llama|mlx)|runtime-router/.test(path);
      if (nativeWorkerFile && path !== owner && /Bun\.spawn\s*\(/.test(source)) violations.push(`${path}: Bun.spawn`);
      if (path !== owner && /new\s+V1RequestTracker\s*\(/.test(source)) violations.push(`${path}: V1RequestTracker`);
    }
    expect(violations).toEqual([]);
  });

  test("runtime packages expose discovery only, not stdout parsers or mutable logs", async () => {
    for (const path of ["packages/runtime-llama/src/index.ts", "packages/runtime-mlx/src/index.ts"]) {
      const source = await readFile(resolve(root, path), "utf8");
      expect(source).not.toMatch(/completeWith|streamWith|parseWorkerLine|proc\.stdout|TextDecoder|logPath/);
      expect(source).not.toMatch(/Bun\.spawn\s*\(/);
    }
  });
});
