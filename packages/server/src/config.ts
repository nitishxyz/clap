import { clapHome } from "@clap/models";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { z } from "zod";

// clap.toml — org-deployable configuration. Layering (later wins):
//   defaults < /etc/clap/clap.toml < $CLAP_HOME/clap.toml < environment
// Machine state stays JSON under $CLAP_HOME; this file is for humans.

const KvTypeSchema = z.enum(["f16", "q8_0", "q4_0"]);

const LlamaSectionSchema = z.object({
  slots: z.number().int().positive().optional(),
  context: z.number().int().positive().optional(),
  max_session_ctx: z.number().int().positive().optional(),
  batch: z.number().int().positive().optional(),
  ubatch: z.number().int().positive().optional(),
  gpu_layers: z.number().int().nonnegative().optional(),
  kv_type: KvTypeSchema.optional(),
}).partial();

export const ClapConfigSchema = z.object({
  server: z.object({
    host: z.string().min(1).optional(),
    port: z.number().int().positive().optional(),
    idle_timeout_seconds: z.number().int().nonnegative().optional(),
  }).partial().default({}),
  auth: z.object({
    require_api_key: z.boolean().optional(),
  }).partial().default({}),
  limits: z.object({
    max_inflight: z.number().int().positive().optional(),
    queue_depth: z.number().int().positive().optional(),
  }).partial().default({}),
  llama: LlamaSectionSchema.default({}),
  models: z.record(z.string(), LlamaSectionSchema).default({}),
});

export type ClapConfig = z.infer<typeof ClapConfigSchema>;

export function configPaths(): string[] {
  const paths: string[] = [];
  if (process.platform !== "win32") paths.push("/etc/clap/clap.toml");
  paths.push(join(clapHome(), "clap.toml"));
  return paths;
}

function parseTomlFile(path: string): Record<string, unknown> | undefined {
  if (!existsSync(path)) return undefined;
  try {
    return Bun.TOML.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
  } catch (error) {
    console.error(`[clap] ignoring invalid config ${path}: ${error instanceof Error ? error.message : error}`);
    return undefined;
  }
}

function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
  const result = { ...base };
  for (const [key, value] of Object.entries(override)) {
    const existing = result[key];
    if (value && typeof value === "object" && !Array.isArray(value) && existing && typeof existing === "object" && !Array.isArray(existing)) {
      result[key] = deepMerge(existing as Record<string, unknown>, value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }
  return result;
}

export function loadClapConfig(): { config: ClapConfig; sources: Array<{ path: string; loaded: boolean }> } {
  let merged: Record<string, unknown> = {};
  const sources: Array<{ path: string; loaded: boolean }> = [];
  for (const path of configPaths()) {
    const parsed = parseTomlFile(path);
    sources.push({ path, loaded: parsed !== undefined });
    if (parsed) merged = deepMerge(merged, parsed);
  }
  const result = ClapConfigSchema.safeParse(merged);
  if (!result.success) {
    console.error(`[clap] config invalid; using defaults: ${result.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join("; ")}`);
    return { config: ClapConfigSchema.parse({}), sources };
  }
  return { config: result.data, sources };
}

const LLAMA_ENV_MAP: Array<[keyof z.infer<typeof LlamaSectionSchema>, string]> = [
  ["slots", "CLAP_LLAMA_SLOTS"],
  ["context", "CLAP_LLAMA_CONTEXT"],
  ["max_session_ctx", "CLAP_LLAMA_MAX_SESSION_CTX"],
  ["batch", "CLAP_LLAMA_BATCH"],
  ["ubatch", "CLAP_LLAMA_UBATCH"],
  ["gpu_layers", "CLAP_LLAMA_GPU_LAYERS"],
  ["kv_type", "CLAP_LLAMA_KV_TYPE"],
];

// Applies [llama] globals to the process environment. Explicit environment
// variables always win over the config file.
export function applyConfigToEnv(config: ClapConfig): void {
  for (const [key, envName] of LLAMA_ENV_MAP) {
    const value = config.llama[key];
    if (value !== undefined && process.env[envName] === undefined) {
      process.env[envName] = String(value);
    }
  }
}

// Per-model worker environment overrides ([models."owner/name"] sections).
// Returned entries override both process env and [llama] globals for the
// worker process serving that model; applied on next worker (re)start.
export function workerEnvForModel(config: ClapConfig, modelId: string): Record<string, string> | undefined {
  const section = config.models[modelId];
  if (!section) return undefined;
  const env: Record<string, string> = {};
  for (const [key, envName] of LLAMA_ENV_MAP) {
    const value = section[key];
    if (value !== undefined) env[envName] = String(value);
  }
  return Object.keys(env).length ? env : undefined;
}
