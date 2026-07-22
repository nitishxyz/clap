#!/usr/bin/env bun
import { relative, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const violations: string[] = [];
const notices: string[] = [];

const entrypoints = [
  "native/llama/src/main.cpp",
  "native/mlx/Sources/clap-mlx/main.swift",
];
const legacySizeExemptions = new Map<string, { max: number; reason: string }>([
  ["native/cache/crates/clap-cache-core/src/lib.rs", {
    max: 1808, reason: "pre-existing shared cache coordinator implementation",
  }],
  ["native/cache/crates/clap-cache-ffi/src/lib.rs", {
    max: 1154, reason: "pre-existing C ABI surface and mappings",
  }],
]);

for (const path of entrypoints) {
  const lines = lineCount(await Bun.file(resolve(root, path)).text());
  if (lines >= 100) violations.push(`${path}: entrypoint has ${lines} lines (must be <100)`);
  else notices.push(`${path}: ${lines} lines`);
}

const production: Array<{ path: string; text: string; lines: number }> = [];
for await (const path of new Bun.Glob("native/**/*.{c,cc,cpp,h,hpp,rs,swift}").scan({ cwd: root })) {
  if (!isProduction(path)) continue;
  const text = await Bun.file(resolve(root, path)).text();
  const lines = lineCount(text);
  production.push({ path, text, lines });
  const exemption = legacySizeExemptions.get(path);
  if (exemption) {
    if (lines > exemption.max) {
      violations.push(`${path}: exempt source grew to ${lines} lines (cap ${exemption.max})`);
    } else {
      notices.push(`${path}: ${lines} lines (justified legacy exception: ${exemption.reason})`);
    }
  } else if (lines > 500) {
    violations.push(`${path}: production source has ${lines} lines (limit 500)`);
  } else if (lines > 400) {
    notices.push(`${path}: ${lines} lines (above 400-line structural target)`);
  }
}

const cppSchedulerOwners = new Set([
  "native/llama/src/scheduler.cpp",
  "native/llama/include/clap/llama/scheduler.h",
]);
const cppQueueOwnership = /\b(?:active_|waiting_|decode_first(?:_order)?)\b/g;
const swiftQueueOwnership = /\b(?:public\s+private\(set\)\s+)?var\s+(?:active|pending(?:Chats)?)\s*:\s*\[/g;
const swiftLatencyRound = /\bLatencyScheduler\s*\.\s*round\s*\(/g;
const swiftStdout = /\bFileHandle\s*\.\s*standardOutput\b/g;
const cppStdout = /\b(?:std::cout|stdout)\b/g;

for (const source of production) {
  if (/\.(?:cc|cpp|h|hpp)$/.test(source.path) && !cppSchedulerOwners.has(source.path)) {
    reportMatches(source.path, source.text, cppQueueOwnership,
      "C++ inference queue/decode ordering outside scheduler");
  }
  if (source.path.endsWith(".swift")) {
    const schedulingOwner = source.path.startsWith("native/mlx/Sources/ClapMLXWorkerCore/") ||
      source.path.startsWith("native/mlx/Sources/clap-mlx/Scheduling/");
    if (!schedulingOwner) {
      reportMatches(source.path, source.text, swiftQueueOwnership,
        "Swift inference queue outside scheduling module");
      reportMatches(source.path, source.text, swiftLatencyRound,
        "Swift latency round outside scheduling module");
    }
    const transportOwner = source.path.startsWith("native/mlx/Sources/clap-mlx/Protocol/") ||
      source.path.startsWith("native/mlx/Sources/clap-mlx/Application/");
    if (!transportOwner) {
      reportMatches(source.path, source.text, swiftStdout,
        "Swift stdout write outside protocol/application transport");
    }
  }
  if (/\.(?:c|cc|cpp|h|hpp)$/.test(source.path) &&
      source.path !== "native/llama/src/protocol.cpp" &&
      source.path !== "native/llama/src/worker.cpp") {
    reportMatches(source.path, source.text, cppStdout,
      "C++ stdout use outside protocol/application transport");
  }
}

for (const notice of notices.sort()) console.log(`note: ${notice}`);
if (violations.length > 0) {
  console.error("Native structure check failed:");
  for (const violation of violations) console.error(`  ${violation}`);
  process.exit(1);
}
console.log(`ok: native structure boundaries (${production.length} production sources)`);

function isProduction(path: string): boolean {
  return !path.includes("/tests/") && !path.includes("/Tests/") &&
    !path.includes("/vendor/") && !path.includes("/target/") &&
    !path.includes("/build/") && !/\.test\.(?:c|cc|cpp|h|hpp|rs|swift)$/.test(path) &&
    !path.includes("native-characterization");
}

function lineCount(text: string): number {
  if (text.length === 0) return 0;
  return text.replace(/\r?\n$/, "").split(/\r?\n/).length;
}

function reportMatches(path: string, text: string, pattern: RegExp, label: string): void {
  pattern.lastIndex = 0;
  for (const match of text.matchAll(pattern)) {
    const line = text.slice(0, match.index).split("\n").length;
    violations.push(`${relative(root, resolve(root, path))}:${line}: ${label}: ${match[0].trim()}`);
  }
}
