#!/usr/bin/env bun
import { resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const violations: string[] = [];

const files = await Promise.all([
  "packages/worker-protocol/src/schemas.ts",
  "packages/api/src/schemas.ts",
  "packages/runtime-router/src/resident.ts",
  "packages/runtime-router/src/resident.test.ts",
  "packages/server/src/prometheus.ts",
  "packages/server/src/prometheus.test.ts",
  "native/llama/src/telemetry.cpp",
  "native/llama/tests/telemetry.test.cpp",
  "native/mlx/Sources/clap-mlx/Telemetry/MemoryTelemetry.swift",
  "native/mlx/Sources/clap-mlx/Telemetry/RetentionTelemetry.swift",
].map(async (path) => [path, await Bun.file(resolve(root, path)).text()] as const));
const text = new Map(files);

requireText("packages/worker-protocol/src/schemas.ts", "MemoryValueSchema", "shared memory schema");
requireText("packages/worker-protocol/src/schemas.ts", "bytes: z.null(), source: z.literal(\"unavailable\")", "unavailable bytes are null");
requireText("packages/api/src/schemas.ts", "retainedBytes: z.number().int().nonnegative().nullable()", "nullable public retained bytes");
requireText("packages/api/src/schemas.ts", "estimatedRetainedBytes", "separate public estimate");
requireText("native/llama/src/telemetry.cpp", "{\"retained_bytes\", nullptr}", "GGUF unavailable retained bytes");
requireText("native/llama/src/telemetry.cpp", "{\"estimated_retained_bytes\"", "GGUF separate estimate");
requireText("native/llama/tests/telemetry.test.cpp", "normal[\"retained_bytes\"].is_null()", "GGUF unavailable regression test");
requireText("native/mlx/Sources/clap-mlx/Telemetry/MemoryTelemetry.swift", "\"worker_allocator\"", "MLX allocator basis");
requireText("native/mlx/Sources/clap-mlx/Telemetry/MemoryTelemetry.swift", "active == nil ? \"unavailable\"", "MLX missing allocator observation handling");
requireText("native/mlx/Sources/clap-mlx/Telemetry/RetentionTelemetry.swift", "retained_bytes_source: \"estimated\"", "MLX cache estimates");
requireText("packages/server/src/prometheus.ts", "source === \"unavailable\" ? []", "Prometheus unavailable omission");
requireText("packages/runtime-router/src/resident.ts", "evictOneIdleForCriticalPressure", "critical pressure eviction path");
requireText("packages/runtime-router/src/resident.test.ts", "protects unsafe residents", "critical pressure protection test");
requireText("packages/runtime-router/src/resident.test.ts", "replans stale snapshots", "critical pressure stale replan test");

for (const [path, source] of files.filter(([path]) => path.startsWith("native/") && !path.includes("/tests/"))) {
  for (const pattern of [/\{"retained_bytes",\s*0\}/g, /retained_bytes_source:\s*"measured"/g,
    /active_bytes:\s*snapshot\.activeMemory/g]) {
    if (pattern.test(source)) violations.push(`${path}: forbidden fake-zero or unlabeled allocator pattern ${pattern}`);
  }
}

if (violations.length) {
  console.error("Honest memory telemetry gate failed:");
  for (const violation of violations) console.error(`  ${violation}`);
  process.exit(1);
}
console.log("ok: honest memory telemetry boundaries");

function requireText(path: string, needle: string, label: string): void {
  if (!text.get(path)?.includes(needle)) violations.push(`${path}: missing ${label}`);
}
