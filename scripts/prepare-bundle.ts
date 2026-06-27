#!/usr/bin/env bun
import { chmod, copyFile, mkdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { basename, join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const libexec = join(root, "libexec");
const inputs = [
  { name: "clap-llama", source: process.env.CLAP_LLAMA_WORKER ?? join(libexec, "clap-llama") },
  { name: "clap-mlx", source: process.env.CLAP_MLX_WORKER ?? join(libexec, "clap-mlx") },
];

await mkdir(libexec, { recursive: true });

const missing = inputs.filter((input) => !input.source || !existsSync(input.source));
if (missing.length > 0) {
  console.error("Cannot prepare native runtime bundle.");
  for (const input of missing) {
    const env = input.name === "clap-llama" ? "CLAP_LLAMA_WORKER" : "CLAP_MLX_WORKER";
    console.error(`build ${input.name} or set ${env} to an existing binary`);
  }
  console.error("Expected build flow: bun run runtime:llama:vendor && bun run runtime:llama:build && bun run runtime:mlx:build");
  process.exit(1);
}

for (const input of inputs) {
  const target = join(libexec, input.name);
  if (input.source === target) {
    console.log(`already bundled ${target}`);
    continue;
  }
  await copyFile(input.source!, target);
  await chmod(target, 0o755);
  console.log(`bundled ${basename(input.source!)} -> ${target}`);
}

console.log("Native runtime bundle prepared. Run: bun run bundle:check");
