#!/usr/bin/env bun

const root = new URL("..", import.meta.url).pathname;
const failures: string[] = [];
const read = (path: string) => Bun.file(`${root}/${path}`).text();
const requireText = (source: string, marker: string, label: string) => {
  if (!source.includes(marker)) failures.push(`${label} is missing ${marker}`);
};

const protocolTypes = await read("packages/worker-protocol/src/types.ts");
const protocolSchemas = await read("packages/worker-protocol/src/schemas.ts");
const server = await read("packages/server/src/index.ts");
const resident = await read("packages/runtime-router/src/process/resident-worker-process.ts");
const api = await read("packages/api/src/schemas.ts");
const dashboard = await read("apps/web/src/components/dashboard/ModelTables.tsx");
const limiter = await read("packages/server/src/limits.ts");
const llamaScheduler = await read("native/llama/src/scheduler.cpp");
const llamaStepper = await read("native/llama/src/generation-stepper.cpp");
const llamaTests = await read("native/llama/tests/scheduler.test.cpp");
const mlxScheduler = await read("native/mlx/Sources/ClapMLXWorkerCore/WorkerScheduler.swift");
const mlxLatency = await read("native/mlx/Sources/ClapCachePolicy/LatencyScheduler.swift");
const mlxTests = await read("native/mlx/Tests/ClapMLXWorkerCoreTests/WorkerSchedulerTests.swift");

const backendComparisons = (source: string) => source.split("\n")
  .filter((line) => /\bbackend\s*(?:===|!==)/.test(line)).map((line) => line.trim());
const permittedServerMechanics = [
  "request.backend !== \"mlx\"", // resolver/model format
  "resolved.model.backend === \"llama\"", // GGUF path validation
  "model.backend === \"llama\"", // worker config and physical identity
  "backend === \"mlx\"", // worker context configuration
];
for (const comparison of backendComparisons(server)) {
  if (!permittedServerMechanics.some((marker) => comparison.includes(marker))) {
    failures.push(`server has a non-allowlisted backend decision: ${comparison}`);
  }
}
const permittedResidentMechanics = [
  "this.backend === \"mlx\"", // binary and typed error selection
  "backend === \"llama\"", // backend-specific mechanical error guidance
];
for (const comparison of backendComparisons(resident)) {
  if (!permittedResidentMechanics.some((marker) => comparison.includes(marker))) {
    failures.push(`resident router has a non-allowlisted backend decision: ${comparison}`);
  }
}

for (const marker of ["WorkerCapabilities", "CacheCapabilities", "GenerationCapabilities",
  "ModalCapabilities", "EffectiveModelCapabilities"]) {
  requireText(protocolTypes, `type ${marker}`, "strict worker protocol capability contract");
}
if (/worker_capabilities:\s*z\.record|model_capabilities:\s*z\.record/.test(protocolSchemas)) {
  failures.push("ready capabilities must not accept loose records");
}
if (/worker_capabilities:\s*Record|model_capabilities:\s*Record/.test(protocolTypes)) {
  failures.push("ready capability types must not use loose records");
}

const nativeTools = server.match(/nativeTools:[\s\S]*?\},\n\s*\);/)?.[0] ?? "";
if (!nativeTools.includes("effectiveCapabilities?.generation.toolTemplateSupport")
    || !nativeTools.includes("templateInfo?.hasToolCalls")) {
  failures.push("native tool preparation must require effective capability and actual template traits");
}
if (/backend\s*(?:===|!==)/.test(nativeTools)) {
  failures.push("native tool preparation must not compare backend labels");
}

const structuredSelector = server.match(/function structuredBackendMode[\s\S]*?\n}/)?.[0] ?? "";
if (!structuredSelector.includes("effectiveCapabilities?.generation.structuredOutput")) {
  failures.push("structured output mode must come from effective capabilities");
}
if (/backend\s*(?:===|!==)/.test(structuredSelector)) {
  failures.push("structured output selection must not compare backend labels");
}
const residentStructured = resident.match(/private requireStructuredOutputCapability[\s\S]*?\n  }/)?.[0] ?? "";
if (!residentStructured.includes("effectiveCapabilities?.generation.structuredOutput")) {
  failures.push("resident structured output enforcement must use effective capabilities");
}
if (/backend\s*(?:===|!==)/.test(residentStructured)) {
  failures.push("resident structured output enforcement must not compare backend labels");
}

// Backend checks remain valid for mechanics such as binary selection, model
// format validation, configuration, physical identity, and memory estimates.
// Capability-sensitive behavior must stay in these explicitly checked blocks.
for (const [label, source] of [["native tools", nativeTools],
  ["structured output", structuredSelector], ["resident structured output", residentStructured]] as const) {
  if (/\b(?:tool|structured|modalit|cache|schedul)[\s\S]{0,160}backend\s*(?:===|!==)|backend\s*(?:===|!==)[\s\S]{0,160}\b(?:tool|structured|modalit|cache|schedul)/i.test(source)) {
    failures.push(`${label} contains a behavior-level backend capability decision`);
  }
}

for (const marker of ["EffectiveCapabilitiesSchema", "effectiveCapabilities: EffectiveCapabilitiesSchema.optional()"])
  requireText(api, marker, "loaded-model API capability contract");
for (const marker of ["data-model-details=\"capabilities\"", "generation.structuredOutput",
  "generation.toolTemplateSupport", "cache.partialSuffixTrim", "modalities.input"])
  requireText(dashboard, marker, "dashboard effective capability details");

for (const marker of ['"interactive", "interactive", "interactive", "interactive"',
  '"normal", "normal", "background"', 'request.cache?.priority ?? "normal"'])
  requireText(`${limiter}\n${server}`, marker, "server priority scheduling path");
for (const marker of ["CLAP_CACHE_PRIORITY_INTERACTIVE", "CLAP_CACHE_PRIORITY_NORMAL",
  "CLAP_CACHE_PRIORITY_BACKGROUND", "priority_cursor_", "request->priority", "192", "96", "48"])
  requireText(`${llamaScheduler}\n${llamaStepper}`, marker, "llama priority scheduling path");
for (const marker of ["prioritySchedule", ".interactive, .interactive, .interactive, .interactive",
  "normalContendedPrefillQuantum = 96", "interactivePrefillQuantum = 192",
  "backgroundPrefillQuantum = 48", "Every runnable request advances before extras"])
  requireText(`${mlxScheduler}\n${mlxLatency}`, marker, "MLX priority scheduling path");
for (const [source, label, markers] of [
  [llamaTests, "llama starvation tests", ["i0", "n", "b", "never destructively preempts"]],
  [mlxTests, "MLX starvation tests", ["sustained interactive work cannot starve normal or background",
    "priority rounds preserve FIFO within class and never cancel active work"]],
] as const) for (const marker of markers) requireText(source, marker, label);

if (failures.length > 0) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log("ok: effective capabilities have priority over backend labels");
