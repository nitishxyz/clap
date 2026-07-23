#!/usr/bin/env bun
import { resolve } from "node:path";
import { WorkerEventSchema } from "../packages/worker-protocol/src/schemas";
import identityFixture from "../packages/worker-protocol/fixtures/v1/cache-identity-vector.json";

const root = resolve(import.meta.dir, "..");
const worker = resolve(root, "libexec/clap-mlx");
const process = Bun.spawn([worker], {
  cwd: root,
  env: Bun.env,
  stdin: "pipe",
  stdout: "pipe",
  stderr: "pipe",
});

const lines = [
  "{bad",
  JSON.stringify({ protocol: 2, type: "shutdown", request_id: "bad-version" }),
  JSON.stringify({ protocol: 1, type: "generate", request_id: "target", prompt: "Hello",
    model: "missing" }),
  JSON.stringify({ protocol: 1, type: "generate", request_id: "required", prompt: "Hello",
    model: "missing", cache_identity: identityFixture.identity,
    structured_output: { kind: "json_object", strength: "required" } }),
  JSON.stringify({ protocol: 1, type: "cancel", request_id: "cancel",
    target_request_id: "target" }),
  JSON.stringify({ protocol: 1, type: "shutdown", request_id: "shutdown" }),
];
process.stdin.write(`${lines.join("\n")}\n`);
process.stdin.end();

const output = await new Response(process.stdout).text();
const stderr = await new Response(process.stderr).text();
const status = await process.exited;
if (status !== 0) throw new Error(`clap-mlx exited ${status}: ${stderr}`);
const events = output.trim().split("\n").filter(Boolean).map((line) =>
  WorkerEventSchema.parse(JSON.parse(line)));
const telemetryVector = WorkerEventSchema.parse({
  protocol: 1,
  type: "telemetry",
  telemetry: {
    memory: {
      active_bytes: 1024, active_bytes_source: "measured", active_bytes_basis: "worker_allocator",
      cache_bytes: null, cache_bytes_source: "unavailable", cache_bytes_basis: "not_observed",
      peak_active_bytes: 2048, peak_active_bytes_source: "measured",
      peak_active_bytes_basis: "worker_allocator",
    },
    retention: {
      retained_bytes: 0, retained_bytes_source: "estimated", retained_bytes_basis: "cache_components",
      session_bytes: 0, session_bytes_source: "estimated", session_bytes_basis: "cache_components",
      anchor_bytes: 0, anchor_bytes_source: "estimated", anchor_bytes_basis: "cache_components",
      evicted_bytes: null, evicted_bytes_source: "unavailable", evicted_bytes_basis: "not_observed",
      estimated_retained_bytes: 0, estimated_retained_bytes_source: "estimated",
      estimated_retained_bytes_basis: "cache_components",
    },
  },
});
if (telemetryVector.type !== "telemetry" ||
    telemetryVector.telemetry.memory?.active_bytes_basis !== "worker_allocator" ||
    telemetryVector.telemetry.retention?.retained_bytes_source !== "estimated" ||
    telemetryVector.telemetry.retention?.evicted_bytes !== null) {
  throw new Error("MLX telemetry must preserve honest allocator and cache provenance");
}
if (events[0]?.type !== "ready") throw new Error("v1 ready must be the first event");
if (events[0].worker_capabilities.backend !== "mlx" ||
    events[0].worker_capabilities.scheduling.fused_multi_sequence_batching !== false ||
    events[0].worker_capabilities.scheduling.interleaved !== true ||
    events[0].model_capabilities !== null) {
  throw new Error("v1 ready must advertise strict MLX worker capabilities without model facts");
}
if (!events.some((event) => event.type === "diagnostic")) {
  throw new Error("malformed JSON must produce an unscoped diagnostic");
}

const scoped = events.filter((event): event is Extract<typeof event, { request_id: string }> =>
  "request_id" in event && typeof event.request_id === "string");
for (const id of ["bad-version", "target", "required", "cancel", "shutdown"]) {
  const requestEvents = scoped.filter((event) => event.request_id === id);
  if (requestEvents[0]?.type !== "accepted" || requestEvents[0]?.sequence !== 0) {
    throw new Error(`${id} must begin with accepted sequence 0`);
  }
  requestEvents.forEach((event, index) => {
    if (event.sequence !== index) throw new Error(`${id} sequence is not contiguous`);
  });
  const terminals = requestEvents.filter((event) =>
    event.type === "completed" || event.type === "failed");
  if (terminals.length !== 1) throw new Error(`${id} must have exactly one terminal`);
}
const versionFailure = scoped.find((event) =>
  event.request_id === "bad-version" && event.type === "failed");
if (versionFailure?.type !== "failed" ||
    versionFailure.error.code !== "unsupported_protocol_version") {
  throw new Error("unsupported versions must produce a structured failure");
}
const identityFailure = scoped.find((event) =>
  event.request_id === "target" && event.type === "failed");
if (identityFailure?.type !== "failed" ||
    identityFailure.error.code !== "cache_identity_required") {
  throw new Error("generate without opaque cache identity must fail closed");
}
const requiredFailure = scoped.find((event) =>
  event.request_id === "required" && event.type === "failed");
if (requiredFailure?.type !== "failed" ||
    requiredFailure.error.code !== "structured_output_capability_required") {
  throw new Error("required structured output must fail before model or cache admission");
}
console.log(`ok: clap-mlx v1 conformance (${events.length} events)`);
