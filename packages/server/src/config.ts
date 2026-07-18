import { clapHome } from "@clap/models";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
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
  // Lifecycle policy (per-model sections; ignored in [llama] globals)
  pinned: z.boolean().optional(),
  keep_alive: z.string().regex(/^(always|\d+(ms|s|m|h|d))$/).optional(),
}).partial();

const MlxSectionSchema = z.object({
  slots: z.number().int().positive().optional(),
  parallel: z.number().int().positive().optional(),
  context: z.number().int().positive().optional(),
  max_session_ctx: z.number().int().positive().optional(),
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
  mlx: MlxSectionSchema.default({}),
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

// Serializes our restricted config shape (sections of scalars, plus the
// [models."owner/name"] map) to TOML. Bun parses TOML but does not stringify.
export function stringifyConfigToml(config: Record<string, unknown>): string {
  const lines: string[] = [];
  const scalar = (value: unknown): string => {
    if (typeof value === "string") return JSON.stringify(value);
    return String(value);
  };
  const section = (name: string, body: Record<string, unknown>) => {
    const entries = Object.entries(body).filter(([, value]) => value !== undefined && typeof value !== "object");
    if (!entries.length) return;
    if (lines.length) lines.push("");
    lines.push(`[${name}]`);
    for (const [key, value] of entries) lines.push(`${key} = ${scalar(value)}`);
  };
  for (const [key, value] of Object.entries(config)) {
    if (key === "models" || !value || typeof value !== "object") continue;
    section(key, value as Record<string, unknown>);
  }
  const models = (config.models ?? {}) as Record<string, Record<string, unknown>>;
  for (const [modelId, body] of Object.entries(models)) {
    section(`models.${JSON.stringify(modelId)}`, body);
  }
  return lines.length ? `${lines.join("\n")}\n` : "";
}

// Applies a validated partial update to the user config file (never the
// system file) and returns the new effective config.
export function updateUserConfig(patch: Record<string, unknown>): { config: ClapConfig; path: string } {
  const parsed = ClapConfigSchema.deepPartial().parse(patch);
  const path = join(clapHome(), "clap.toml");
  const current = parseTomlFile(path) ?? {};
  const merged = deepMerge(current, parsed as Record<string, unknown>);
  const validated = ClapConfigSchema.parse(merged);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, stringifyConfigToml(merged));
  void validated;
  const { config } = loadClapConfig();
  return { config, path };
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

const MLX_ENV_MAP: Array<[keyof z.infer<typeof MlxSectionSchema>, string]> = [
  ["slots", "CLAP_MLX_SLOTS"],
  ["parallel", "CLAP_MLX_PARALLEL"],
  ["context", "CLAP_MLX_CONTEXT"],
  ["max_session_ctx", "CLAP_MLX_MAX_SESSION_CTX"],
  ["kv_type", "CLAP_MLX_KV_TYPE"],
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
  for (const [key, envName] of MLX_ENV_MAP) {
    const value = config.mlx[key];
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
  // Per-model sections apply to whichever backend serves the model; mirror
  // the shared keys onto the MLX worker env too.
  for (const [key, envName] of MLX_ENV_MAP) {
    const value = (section as Record<string, unknown>)[key];
    if (value !== undefined) env[envName] = String(value);
  }
  return Object.keys(env).length ? env : undefined;
}
