#!/usr/bin/env bun

const root = new URL("..", import.meta.url).pathname;
const packageJSON = await Bun.file(`${root}/package.json`).json();
const scripts = packageJSON.scripts as Record<string, string>;
const requiredScripts = [
  "native:cache:tier-a",
  "native:cache:tier-b",
  "native:cache:tier-c",
  "native:probe:leak:check",
  "runtime:llama:physical:test",
  "runtime:mlx:physical:test",
];
const failures: string[] = [];
for (const name of requiredScripts) {
  if (!scripts[name]?.trim()) failures.push(`package.json is missing script ${name}`);
}
for (const required of ["runtime:cache:test", "runtime:llama:test", "runtime:mlx:test",
  "native:probe:leak:check"]) {
  if (!scripts["native:cache:tier-a"]?.includes(required)) {
    failures.push(`native:cache:tier-a does not include ${required}`);
  }
}

const assets = await Bun.file(`${root}/config/cache-correctness-assets.json`).json();
if (assets.schemaVersion !== 1 || !assets.assets?.gguf || !assets.assets?.mlx) {
  failures.push("cache correctness asset manifest schema is unavailable");
}
const matrix = await Bun.file(`${root}/config/cache-correctness-matrix.json`).json();
const expectedCases = new Map([
  ["gguf-standard", "unprovisioned"],
  ["mlx-full", "unprovisioned"],
  ["mlx-sliding", "unprovisioned"],
  ["mlx-recurrent", "unprovisioned"],
  ["mlx-hybrid-antares", "pinned"],
]);
if (matrix.schemaVersion !== 1 || !Array.isArray(matrix.cases)) {
  failures.push("Tier C matrix schema is unavailable");
} else {
  for (const [id, provisioning] of expectedCases) {
    const item = matrix.cases.find((candidate: { id?: string }) => candidate.id === id);
    if (!item || item.tier !== "c" || item.provisioning !== provisioning ||
        !Array.isArray(item.scenarios) || item.scenarios.length === 0 ||
        !Number.isSafeInteger(item.timeoutMs) || !Number.isSafeInteger(item.maxResidentBytes)) {
      failures.push(`Tier C case ${id} is missing or invalid`);
    }
  }
}

const cmake = await Bun.file(`${root}/native/llama/CMakeLists.txt`).text();
for (const target of ["clap-llama-cache-checkpoint", "clap-llama-cache-transaction",
  "clap-llama-cache-request-lifecycle", "clap-llama-cache-model-probe"]) {
  if (!cmake.includes(`set_tests_properties(${target} PROPERTIES LABELS \"tier-a;cache`)) {
    failures.push(`${target} is not labeled as Tier A cache coverage`);
  }
}

const workflow = await Bun.file(`${root}/.github/workflows/cache-correctness-matrix.yml`).text();
for (const marker of ["schedule:", "workflow_dispatch:", "self-hosted", "cache-correctness",
  "validate-cache-test-assets.ts", "upload-artifact@v4"]) {
  if (!workflow.includes(marker)) failures.push(`Tier C workflow is missing ${marker}`);
}
if (/pull_request\s*:/.test(workflow)) failures.push("Tier C workflow must not run on public PRs");

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("ok: cache correctness Tier A/B/C gates are wired and reviewable");
