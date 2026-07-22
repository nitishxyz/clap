import { readFile, readdir } from "node:fs/promises";
import { join, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const owner = "packages/runtime-router/src/process/resident-worker-process.ts";
const runtimeBackends = ["packages/runtime-llama/src/index.ts", "packages/runtime-mlx/src/index.ts"];

async function typescriptFiles(directory: string): Promise<string[]> {
  const entries = await readdir(resolve(root, directory), { withFileTypes: true });
  const files = await Promise.all(entries.map((entry) => {
    const path = join(directory, entry.name);
    return entry.isDirectory() ? typescriptFiles(path) : path.endsWith(".ts") ? [path] : [];
  }));
  return files.flat();
}

export async function runtimeProcessOwnershipViolations(): Promise<string[]> {
  const files = await typescriptFiles("packages");
  const violations: string[] = [];
  for (const path of files.filter((item) => !item.endsWith(".test.ts"))) {
    const source = await readFile(resolve(root, path), "utf8");
    const nativeRuntime = /runtime-(llama|mlx)|runtime-router/.test(path);
    if (nativeRuntime && path !== owner && /Bun\.spawn\s*\(/.test(source)) {
      violations.push(`${path}: native inference Bun.spawn outside resident process`);
    }
    if (path !== owner && /new\s+V1RequestTracker\s*\(/.test(source)) {
      violations.push(`${path}: V1RequestTracker outside resident process`);
    }
  }
  for (const path of runtimeBackends) {
    const source = await readFile(resolve(root, path), "utf8");
    if (/completeWith|streamWith|parseWorkerLine|proc\.stdout|TextDecoder/.test(source)) {
      violations.push(`${path}: worker stdout parser or one-shot inference API`);
    }
    if (/logPath|worker\.err\.log/.test(source)) {
      violations.push(`${path}: mutable backend-global worker log path`);
    }
  }
  const requiredTests: Array<[string, RegExp]> = [
    ["packages/runtime-router/src/process/launch-paths.test.ts", /createWorkerLaunchPaths/],
    ["packages/runtime-router/src/process/launch-log-store.test.ts", /finalizes a launch exactly once/],
    ["packages/runtime-router/src/process/resident-worker-process-launch.test.ts", /per-launch logs/],
  ];
  for (const [path, evidence] of requiredTests) {
    const source = await readFile(resolve(root, path), "utf8").catch(() => "");
    if (!evidence.test(source)) violations.push(`${path}: required launch path/metadata test missing`);
  }
  return violations;
}

if (import.meta.main) {
  const violations = await runtimeProcessOwnershipViolations();
  if (violations.length) {
    console.error(violations.join("\n"));
    process.exit(1);
  }
  console.log("ok: unified native inference process ownership");
}
