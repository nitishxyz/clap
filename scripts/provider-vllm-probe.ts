#!/usr/bin/env bun
import { createHash } from "node:crypto";
import { VllmReplicaPool, type VllmReplicaConfiguration } from "@clap/runtime-vllm";

const model = required("CLAP_VLLM_MODEL");
const deployment = process.env.CLAP_VLLM_DEPLOYMENT ?? "phase7-validation";
const secret = required("CLAP_VLLM_CACHE_SALT_SECRET");
const shareIdentity = process.env.CLAP_VLLM_SHARE_IDENTITY ?? "validation-tenant";
const harnessWords = positiveInteger(process.env.CLAP_VLLM_HARNESS_WORDS ?? "2048");
const replicas = parseReplicas(required("CLAP_VLLM_REPLICAS"), model);
const pool = new VllmReplicaPool({ deployment, cacheSaltSecret: secret, replicas });

await pool.discover();
await pool.scrape();

const harness = Array.from({ length: harnessWords }, (_, index) => `shared-rule-${index % 97}`).join(" ");
const system = {
  role: "system",
  content: `You are a prefix-cache validation model. Follow this synthetic harness and answer briefly.\n${harness}`,
};
const seed = [system, { role: "user", content: "Return marker ALPHA." }];
const requests: Array<Record<string, unknown>> = [];

for (let index = 0; index < 3; index += 1) {
  requests.push(await chat(`repeat-${index}`, seed));
}
const firstContent = String((requests[0]!.response as Record<string, unknown>).content ?? "ALPHA");
const history = [...seed, { role: "assistant", content: firstContent }];
requests.push(await chat("branch-beta", [...history, { role: "user", content: "Return marker BETA." }]));
requests.push(await chat("branch-gamma", [...history, { role: "user", content: "Return marker GAMMA." }]));

const finalTelemetry = await pool.scrape();
const queryDelta = sum(finalTelemetry.map((value) => value.prefixQueryDelta));
const hitDelta = sum(finalTelemetry.map((value) => value.prefixHitDelta));
if (queryDelta <= 0) throw new Error("vLLM did not report prefix-cache query tokens");
if (hitDelta <= 0) throw new Error("vLLM did not report any native prefix-cache hit tokens");

console.log(JSON.stringify({
  status: "ok",
  deployment,
  model,
  harnessWords,
  requests,
  finalTelemetry,
  tokenWeightedHitRatio: hitDelta / queryDelta,
  pool: pool.snapshot(),
}, null, 2));

async function chat(label: string, messages: Array<Record<string, string>>) {
  const canonical = JSON.stringify(messages);
  const prefixFingerprint = createHash("sha256").update(canonical).digest("hex");
  const started = performance.now();
  const dispatch = await pool.chat({ model, messages, temperature: 0, max_tokens: 8, stream: false }, {
    shareIdentity,
    prefixFingerprint,
  });
  const elapsedMs = performance.now() - started;
  const body = await dispatch.response.json() as {
    choices?: Array<{ message?: { content?: string } }>;
    usage?: unknown;
    error?: unknown;
  };
  if (!dispatch.response.ok || body.error) {
    throw new Error(`${label} failed on ${dispatch.decision.replicaId}: ${JSON.stringify(body.error ?? body)}`);
  }
  const content = body.choices?.[0]?.message?.content ?? "";
  const telemetry = await pool.scrape();
  return {
    label,
    elapsedMs,
    decision: dispatch.decision,
    response: {
      content,
      contentSha256: createHash("sha256").update(content).digest("hex"),
      usage: body.usage,
    },
    telemetry,
  };
}

function parseReplicas(value: string, selectedModel: string): VllmReplicaConfiguration[] {
  if (value.trim().startsWith("[")) {
    const parsed = JSON.parse(value) as VllmReplicaConfiguration[];
    return parsed.map((replica) => ({ ...replica, model: replica.model || selectedModel }));
  }
  return value.split(",").map((entry, index) => {
    const separator = entry.indexOf("=");
    const id = separator >= 0 ? entry.slice(0, separator) : `replica-${index}`;
    const baseUrl = separator >= 0 ? entry.slice(separator + 1) : entry;
    return { id, baseUrl, model: selectedModel };
  });
}

function required(name: string): string {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function positiveInteger(value: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed <= 0) throw new Error(`expected a positive integer, received ${value}`);
  return parsed;
}

function sum(values: Array<number | null>): number {
  return values.reduce<number>((total, value) => total + (value ?? 0), 0);
}
