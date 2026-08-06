import { afterEach, describe, expect, test } from "bun:test";
import { applyConfigToEnv, ClapConfigSchema } from "./config";

const names = [
  "CLAP_CACHE_MAX_ANCHORS_PER_SESSION",
  "CLAP_CACHE_MAX_ANCHOR_BYTES_PER_SESSION",
  "CLAP_CACHE_SESSION_IDLE_TTL_MS",
  "CLAP_CACHE_CHECKPOINTS_ENABLED",
  "CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS",
  "CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS",
  "CLAP_CACHE_CHECKPOINT_MAX",
  "CLAP_CACHE_CHECKPOINT_MAX_PER_SESSION",
  "CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS",
  "CLAP_CACHE_CHECKPOINT_BUDGET_BYTES",
  "CLAP_LLAMA_PREFILL_BUDGET",
  "CLAP_LLAMA_SLOT_MEMORY_BUDGET_BYTES",
  "CLAP_LLAMA_SPLIT_MODE",
  "CLAP_LLAMA_MAIN_GPU",
  "CLAP_LLAMA_TENSOR_SPLIT",
  "CLAP_MLX_PREFILL_QUANTUM",
] as const;

const original = Object.fromEntries(names.map((name) => [name, process.env[name]]));
afterEach(() => {
  for (const name of names) {
    const value = original[name];
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
});

describe("automatic checkpoint config", () => {
  test("has explicit safe defaults", () => {
    const config = ClapConfigSchema.parse({});
    expect(config.cache.max_anchors_per_session).toBe(4);
    expect(config.cache.max_anchor_bytes_per_session).toBe(0);
    expect(config.cache.session_idle_ttl_ms).toBe(900_000);
    expect(config.cache.checkpoints).toEqual({
      enabled: true,
      minimum_tokens: 2_048,
      interval_tokens: 2_048,
      max_checkpoints: 8,
      max_checkpoints_per_session: 2,
      // Half the retained-KV budget: enough for one full-length anchor on
      // large hybrid models, which is what makes long prompts reusable.
      budget_fraction: 0.5,
      budget_bytes: 0,
    });
  });

  test("validates bounds and exports concrete worker values", () => {
    for (const name of names) delete process.env[name];
    const config = ClapConfigSchema.parse({ cache: {
      max_anchors_per_session: 5,
      max_anchor_bytes_per_session: 67_108_864,
      session_idle_ttl_ms: 120_000,
      checkpoints: {
      enabled: false,
      minimum_tokens: 4_096,
      interval_tokens: 1_024,
      max_checkpoints: 6,
      max_checkpoints_per_session: 3,
      budget_fraction: 0.125,
      budget_bytes: 33_554_432,
    } } });
    applyConfigToEnv(config);
    expect(process.env.CLAP_CACHE_MAX_ANCHORS_PER_SESSION).toBe("5");
    expect(process.env.CLAP_CACHE_MAX_ANCHOR_BYTES_PER_SESSION).toBe("67108864");
    expect(process.env.CLAP_CACHE_SESSION_IDLE_TTL_MS).toBe("120000");
    expect(process.env.CLAP_CACHE_CHECKPOINTS_ENABLED).toBe("0");
    expect(process.env.CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS).toBe("4096");
    expect(process.env.CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS).toBe("1024");
    expect(process.env.CLAP_CACHE_CHECKPOINT_MAX).toBe("6");
    expect(process.env.CLAP_CACHE_CHECKPOINT_MAX_PER_SESSION).toBe("3");
    expect(process.env.CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS).toBe("1250");
    expect(process.env.CLAP_CACHE_CHECKPOINT_BUDGET_BYTES).toBe("33554432");
    expect(() => ClapConfigSchema.parse({ cache: { checkpoints: { budget_fraction: 1.1 } } })).toThrow();
    expect(() => ClapConfigSchema.parse({ cache: { checkpoints: { max_checkpoints: 0 } } })).toThrow();
    expect(ClapConfigSchema.parse({ cache: {
      max_anchors_per_session: 0,
      checkpoints: { max_checkpoints_per_session: 0 },
    } }).cache).toMatchObject({
      max_anchors_per_session: 0,
      checkpoints: { max_checkpoints_per_session: 0 },
    });
    expect(() => ClapConfigSchema.parse({ cache: { max_anchors_per_session: 257 } })).toThrow();
  });
});

describe("prefill latency controls", () => {
  test("maps [llama].prefill_budget and [mlx].prefill_quantum to worker env", () => {
    delete process.env.CLAP_LLAMA_PREFILL_BUDGET;
    delete process.env.CLAP_MLX_PREFILL_QUANTUM;
    const config = ClapConfigSchema.parse({
      llama: { prefill_budget: 256 },
      mlx: { prefill_quantum: 192 },
    });
    applyConfigToEnv(config);
    const env: Record<string, string | undefined> = process.env;
    expect(env.CLAP_LLAMA_PREFILL_BUDGET).toBe("256");
    expect(env.CLAP_MLX_PREFILL_QUANTUM).toBe("192");
    expect(() => ClapConfigSchema.parse({ llama: { prefill_budget: -1 } })).toThrow();
    expect(() => ClapConfigSchema.parse({ mlx: { prefill_quantum: 0 } })).toThrow();
  });
});

describe("active slot derivation controls", () => {
  test("maps [llama].slot_memory_budget_bytes to worker env", () => {
    delete process.env.CLAP_LLAMA_SLOT_MEMORY_BUDGET_BYTES;
    const config = ClapConfigSchema.parse({
      llama: { slot_memory_budget_bytes: 17_179_869_184 },
    });
    applyConfigToEnv(config);
    const env: Record<string, string | undefined> = process.env;
    expect(env.CLAP_LLAMA_SLOT_MEMORY_BUDGET_BYTES).toBe("17179869184");
    expect(() => ClapConfigSchema.parse({ llama: { slot_memory_budget_bytes: 0 } })).toThrow();
    expect(() => ClapConfigSchema.parse({ llama: { slot_memory_budget_bytes: 1.5 } })).toThrow();
  });
});

describe("multi-GPU split controls", () => {
  test("maps [llama] split keys to worker env and validates shapes", () => {
    delete process.env.CLAP_LLAMA_SPLIT_MODE;
    delete process.env.CLAP_LLAMA_MAIN_GPU;
    delete process.env.CLAP_LLAMA_TENSOR_SPLIT;
    const config = ClapConfigSchema.parse({
      llama: { split_mode: "row", main_gpu: 1, tensor_split: "3,1" },
    });
    applyConfigToEnv(config);
    const env: Record<string, string | undefined> = process.env;
    expect(env.CLAP_LLAMA_SPLIT_MODE).toBe("row");
    expect(env.CLAP_LLAMA_MAIN_GPU).toBe("1");
    expect(env.CLAP_LLAMA_TENSOR_SPLIT).toBe("3,1");
    expect(() => ClapConfigSchema.parse({ llama: { split_mode: "diagonal" } })).toThrow();
    expect(() => ClapConfigSchema.parse({ llama: { main_gpu: -1 } })).toThrow();
    expect(() => ClapConfigSchema.parse({ llama: { tensor_split: "3," } })).toThrow();
    expect(() => ClapConfigSchema.parse({ llama: { tensor_split: "-1,2" } })).toThrow();
    expect(ClapConfigSchema.parse({ llama: { tensor_split: "0.75,0.25" } })
      .llama.tensor_split).toBe("0.75,0.25");
  });
});
