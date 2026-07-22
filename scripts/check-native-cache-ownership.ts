#!/usr/bin/env bun
import { relative, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const violations: string[] = [];

const cppPhysical = /\b(?:llama_memory_seq_cp|llama_memory_seq_rm|llama_memory_clear)\s*\(/g;
const cppCoordinator = /(?:\bcoordinator_?|\.coordinator)\s*(?:->|\.)\s*(?:plan|advance|confirm|invalidate|reset|set_busy|set_anchor_protected|register_slot|commit|abort)\s*\(/g;
const cppDirectCoordinator = /\bclap_cache_(?:plan|advance|confirm|invalidate|reset|set_busy|set_anchor_protected|register_slot|commit|abort)\s*\(/g;
const cppGeneration = /\b(?:llama_decode|llama_sampler_sample|llama_token_to_piece)\s*\(/g;
const cppGenerationDefinition = /\b(?:auto|bool|int|void)\s+(?:decode|prefill|step)(?:_[A-Za-z0-9]+)*\s*\(/g;

for await (const path of new Bun.Glob("native/llama/**/*.{cpp,cc,h,hpp}").scan({ cwd: root })) {
  if (isCppTest(path)) continue;
  const text = await Bun.file(resolve(root, path)).text();
  if (path !== "native/llama/src/cache-executor.cpp") {
    reportMatches(path, text, cppPhysical, "physical llama cache mutation");
    reportMatches(path, text, cppCoordinator, "cache coordinator mutation");
  }
  if (path !== "native/llama/src/cache-executor.cpp" && path !== "native/llama/cache-adapter.h") {
    reportMatches(path, text, cppDirectCoordinator, "direct cache coordinator FFI mutation");
  }
  if (path !== "native/llama/src/generation-stepper.cpp") {
    reportMatches(path, text, cppGeneration, "direct llama generation operation");
  }
}

const swiftCachePrimitive = /\.(?:copy|trim|clear)\s*\(/g;
const swiftCoordinator = /\b(?:cacheCoordinator|coordinator)\s*\??\s*\.\s*(?:plan|advance|confirm|invalidate|reset|setBusy|setAnchorProtected|registerSlot|commit|abort)\s*\(/g;
const swiftRegistry = /\b(?:retainedRegistry|registry)\s*\.\s*(?:register|activate|release|validateEvictions|reconcileEvictions)\s*\(/g;
const swiftGenerationPrimitive = /\b(?:TokenIterator|LMInput)\s*\(|\bdetokenizer\s*\.\s*(?:append|next)\s*\(|\bStopSequencePolicy\s*\.\s*scan\s*\(/g;
const swiftGenerationLoop = /\bwhile\s+steps\s*<|\b(?:prefill|decode)StartedNs\b/g;
const swiftMainSchedulerOwnership = /\b(?:pendingChats|schedulerCore)\b|\bLatencyScheduler\s*\.\s*round\s*\(|\bstate\s*\.\s*(?:prepareRequest|step|finalize)\s*\(/g;
const swiftQueueDeclaration = /\bvar\s+(?:active|pendingChats)\s*(?::|=)/g;

for await (const path of new Bun.Glob("native/mlx/Sources/**/*.swift").scan({ cwd: root })) {
  if (path.startsWith("native/mlx/Sources/ClapMLXCache/")) continue;
  const text = await Bun.file(resolve(root, path)).text();
  reportMatches(path, text, swiftCachePrimitive, "retained cache copy/trim/clear");
  reportMatches(path, text, swiftCoordinator, "cache coordinator mutation");
  reportMatches(path, text, swiftRegistry, "retained registry mutation");
}

const swiftMain = "native/mlx/Sources/clap-mlx/main.swift";
const swiftMainText = await Bun.file(resolve(root, swiftMain)).text();
reportMatches(swiftMain, swiftMainText, swiftGenerationPrimitive,
  "direct MLX generation operation");
reportMatches(swiftMain, swiftMainText, swiftGenerationLoop,
  "mutable generation loop");
reportMatches(swiftMain, swiftMainText, swiftMainSchedulerOwnership,
  "direct inference scheduling operation");

for await (const path of new Bun.Glob("native/mlx/Sources/clap-mlx/**/*.swift").scan({ cwd: root })) {
  if (path.startsWith("native/mlx/Sources/clap-mlx/Scheduling/")) continue;
  const text = await Bun.file(resolve(root, path)).text();
  reportMatches(path, text, swiftQueueDeclaration, "inference queue declaration");
}

if (violations.length > 0) {
  console.error("Native ownership check failed:");
  for (const violation of violations) console.error(`  ${violation}`);
  console.error("Route cache and generation mutations through their native owners.");
  process.exit(1);
}

console.log("ok: native cache and generation ownership boundaries");

function isCppTest(path: string): boolean {
  return path.includes("/tests/") || path.endsWith(".test.cpp") || path.endsWith(".test.cc");
}

function reportMatches(path: string, text: string, pattern: RegExp, label: string): void {
  pattern.lastIndex = 0;
  for (const match of text.matchAll(pattern)) {
    const line = text.slice(0, match.index).split("\n").length;
    violations.push(`${relative(root, resolve(root, path))}:${line}: ${label}: ${match[0].trim()}`);
  }
}
