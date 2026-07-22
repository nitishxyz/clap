#!/usr/bin/env bun
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const oldEnvironmentName = ["CLAP", "TELEMETRY", "HMAC", "KEY"].join("_");
const productionGlobs = [
  "apps/**/*.{ts,tsx,js,jsx}",
  "packages/*/src/**/*.{ts,tsx,js,jsx}",
  "native/llama/{src,include}/**/*.{cpp,cc,h,hpp}",
  "native/mlx/Sources/**/*.swift",
];
const violations: string[] = [];

for (const pattern of productionGlobs) {
  for await (const path of new Bun.Glob(pattern).scan({ cwd: root })) {
    if (path.endsWith(".test.ts") || path.endsWith(".test.tsx")) continue;
    const text = await Bun.file(resolve(root, path)).text();
    if (text.includes(oldEnvironmentName)) violations.push(`${path}: legacy telemetry HMAC environment name`);
  }
}

const opaqueIdentitySources = [
  "native/llama/src/cache-identity.cpp",
  "native/mlx/Sources/ClapMLXCache/CacheIdentity.swift",
];
const weakHash = /\b(?:hash|fingerprint|fnv|djb2)\s*\(/gi;
for (const path of opaqueIdentitySources) {
  const text = await Bun.file(resolve(root, path)).text();
  for (const match of text.matchAll(weakHash)) {
    const line = text.slice(0, match.index).split("\n").length;
    violations.push(`${path}:${line}: weak hash call in opaque cache identity parser`);
  }
}

const sharedFixture = "packages/worker-protocol/fixtures/v1/cache-identity-vector.json";
const consumers = [
  "packages/worker-protocol/src/validation.test.ts",
  "native/llama/CMakeLists.txt",
  "native/mlx/Tests/ClapMLXCacheTests/TestCacheIdentity.swift",
];
if (!(await Bun.file(resolve(root, sharedFixture)).exists())) violations.push(`${sharedFixture}: missing shared vector`);
for (const path of consumers) {
  const text = await Bun.file(resolve(root, path)).text();
  if (!text.includes("cache-identity-vector.json")) violations.push(`${path}: does not consume shared identity vector`);
}

if (violations.length > 0) {
  console.error("Cache identity security check failed:");
  for (const violation of violations) console.error(`  ${violation}`);
  process.exit(1);
}

console.log("ok: cache identity uses only opaque protocol authority and truthful telemetry key names");
