#!/usr/bin/env bun

const root = new URL("..", import.meta.url).pathname;
const failures: string[] = [];
const read = (path: string) => Bun.file(`${root}/${path}`).text();

const packageJSON = await Bun.file(`${root}/package.json`).json();
const scripts = packageJSON.scripts as Record<string, string>;
if (!scripts["structured-output:gate"]?.includes("check-structured-output-gates.ts")) {
  failures.push("package.json is missing structured-output:gate");
}
if (!scripts["native:test"]?.includes("structured-output:gate")) {
  failures.push("native:test must include structured-output:gate");
}

const server = await read("packages/server/src/index.ts");
const router = await read("packages/runtime-router/src/process/resident-worker-process.ts");
const parserRegistry = await read("packages/server/src/parsers/registry.ts");
const mlxTransport = await read("native/mlx/Sources/clap-mlx/Protocol/JSONLineTransport.swift");
const mlxPreparation = await read("native/mlx/Sources/clap-mlx/Application/RequestPreparation.swift");
const llamaWorker = await read("native/llama/src/worker.cpp");
const llamaStructured = await read("native/llama/src/structured-output.cpp");
const protocolSchemas = await read("packages/worker-protocol/src/schemas.ts");
const serverTests = await read("packages/server/src/index.test.ts");
const routerTests = await read("packages/runtime-router/src/process/resident-worker-process-lifecycle.test.ts");
const parserSelectionTests = await read("packages/server/src/parsers/index.test.ts");
const structuredParserTests = await read("packages/server/src/parsers/structured.test.ts");
const streamStateTests = await read("packages/server/src/parsers/stream-state.test.ts");
const protocolTests = await read("packages/worker-protocol/src/validation.test.ts");
const mlxTests = await read("native/mlx/Tests/ClapMLXWorkerCoreTests/ProtocolStateTests.swift");
const llamaWorkerTests = await read("native/llama/tests/worker.test.cpp");
const llamaTests = await read("native/llama/tests/structured-output.test.cpp");

for (const marker of ["structuredOutputContract(request)", "structuredBackendMode", "parseStructuredOutput",
  "structured_output_capability_required", "repaired_validated", "native_validated"]) {
  if (!server.includes(marker)) failures.push(`server structured dispatch is missing ${marker}`);
}
for (const marker of ["contract.strength === \"required\"", "mode === \"native\"",
  "mode === \"native\" || mode === \"post_validate\""]) {
  if (!router.includes(marker)) failures.push(`router capability enforcement is missing ${marker}`);
}

const selectorBodies = [
  server.match(/function structuredOutputContract[\s\S]*?\n}\n/)?.[0] ?? "",
  server.match(/function structuredBackendMode[\s\S]*?\n}\n/)?.[0] ?? "",
  router.match(/private requireStructuredOutputCapability[\s\S]*?\n  }/)?.[0] ?? "",
].join("\n");
if (/backend\s*===|model\.(?:includes|match)|request\.model|llama|mlx/i.test(selectorBodies)) {
  failures.push("structured capability selection must not sniff backend or model names");
}

// Exact model matching is reserved for explicit user profiles. Once that
// branch has returned, built-in parser selection must use discovered traits
// and request shape only, never model or backend names.
const builtInParserSelection = parserRegistry.split("if (explicitUser) return explicitUser;")[1] ?? "";
if (!builtInParserSelection
  || /\binput\.model\b|\b(?:input|request)\.backend\b|\bbackend\s*===|\bmodel\.(?:includes|match)|\.test\(\s*model\s*\)/i.test(builtInParserSelection)) {
  failures.push("built-in parser selection must not sniff backend or model names");
}
if (!parserSelectionTests.includes("model names alone cannot select built-in families")) {
  failures.push("parser selection regression coverage is missing model-name isolation");
}

if (mlxTransport.includes('"json_object": "native"') || mlxTransport.includes('"json_schema": "native"')) {
  failures.push("MLX must never advertise native structured output");
}
for (const marker of ['"json_object": "post_validate"', '"json_schema": "post_validate"',
  '"post_validation": true']) {
  if (!mlxTransport.includes(marker)) failures.push(`MLX ready capability is missing ${marker}`);
}
if (!mlxPreparation.includes("structured_output_capability_required")) {
  failures.push("MLX required constraints must fail before admission");
}
for (const marker of ['{"json_object", "native"}', '{"json_schema", "native"}',
  '{"post_validation", true}', '{"max_schema_bytes", 64 * 1024}']) {
  if (!llamaWorker.includes(marker)) failures.push(`llama ready capability is missing ${marker}`);
}
for (const marker of ['{"json_object", "native"}', '{"json_schema", "native"}']) {
  if (!llamaWorkerTests.includes(marker)) failures.push(`llama ready capability test is missing ${marker}`);
}
for (const marker of ["kMaxStructuredOutputSchemaBytes", "validate_schema", "json_schema_to_grammar",
  "unsupported_structured_output"]) {
  if (!llamaStructured.includes(marker)) failures.push(`llama conversion gate is missing ${marker}`);
}

for (const marker of ["StructuredOutputContractSchema", "StructuredOutputCapabilitiesSchema",
  "structured_output: StructuredOutputContractSchema.optional()",
  "structured_output: StructuredOutputCapabilitiesSchema.optional()"]) {
  if (!protocolSchemas.includes(marker)) failures.push(`worker protocol schema is missing ${marker}`);
}
for (const marker of ["validates strict structured-output contracts and ready capabilities",
  "decodeWorkerRequest", "decodeWorkerEvent", "ProtocolValidationError"]) {
  if (!protocolTests.includes(marker)) failures.push(`worker protocol tests are missing ${marker}`);
}

const coverage: Array<[string, string, string[]]> = [
  ["server", serverTests, ["never streams partial JSON", "structured_output_capability_required",
    "structured_output_invalid", "CLAP_FAKE_WORKER_STRUCTURED_MODE"]],
  ["router", routerTests, ["stale launch exit cannot clear its replacement",
    "structuredOutputCapabilities", "injects normalized structured-output contracts"]],
  ["structured parser", structuredParserTests, ["parses exact JSON and emits canonical object JSON",
    "best effort uses deterministic fenced, object, and bracket repair", "model output is not valid JSON",
    "schema_validation_failed", "invalid_schema"]],
  ["stream state", streamStateTests, ["chunking strategies match whole-buffer output",
    "buffered structured output emits no marker or JSON bytes before or after cancel"]],
  ["MLX", mlxTests, ["structured output validates typed contracts and schema bounds",
    "https://example.com/schema.json", "64 KiB", "1024 properties", "depth"]],
  ["llama", llamaTests, ["json_object", "$ref", "oversized", "unsupported_structured_output"]],
];
for (const [name, source, markers] of coverage) {
  for (const marker of markers) if (!source.includes(marker)) failures.push(`${name} matrix is missing ${marker}`);
}

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("ok: structured-output capability, validation, privacy, and fault gates are wired");
