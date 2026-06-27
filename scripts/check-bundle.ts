#!/usr/bin/env bun
import { existsSync, statSync } from "node:fs";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const required = [
  join(root, "libexec", "clap-llama"),
  join(root, "libexec", "clap-mlx"),
];
const requiredFiles = [
  join(root, "libexec", "mlx.metallib"),
];

const missing = required.filter((path) => !isExecutableFile(path));
const missingFiles = requiredFiles.filter((path) => !isNonEmptyFile(path));

if (missing.length > 0 || missingFiles.length > 0) {
  console.error("Native runtime bundle is incomplete.");
  for (const path of missing) {
    console.error(`missing executable: ${path}`);
  }
  for (const path of missingFiles) {
    console.error(`missing resource: ${path}`);
  }
  console.error("Build the native workers before packaging: bun run runtime:llama:vendor && bun run runtime:llama:build && bun run runtime:mlx:build");
  process.exit(1);
}

for (const path of required) {
  console.log(`ok: ${path}`);
}
for (const path of requiredFiles) {
  console.log(`ok: ${path}`);
}

function isExecutableFile(path: string): boolean {
  if (!existsSync(path)) return false;
  const stat = statSync(path);
  return stat.isFile() && stat.size > 0 && (stat.mode & 0o111) !== 0;
}

function isNonEmptyFile(path: string): boolean {
  if (!existsSync(path)) return false;
  const stat = statSync(path);
  return stat.isFile() && stat.size > 0;
}
