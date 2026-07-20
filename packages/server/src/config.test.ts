import { afterEach, describe, expect, test } from "bun:test";
import { applyConfigToEnv, ClapConfigSchema } from "./config";

const names = [
  "CLAP_CACHE_CHECKPOINTS_ENABLED",
  "CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS",
  "CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS",
  "CLAP_CACHE_CHECKPOINT_MAX",
  "CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS",
  "CLAP_CACHE_CHECKPOINT_BUDGET_BYTES",
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
    expect(config.cache.checkpoints).toEqual({
      enabled: true,
      minimum_tokens: 2_048,
      interval_tokens: 2_048,
      max_checkpoints: 8,
      budget_fraction: 0.25,
      budget_bytes: 0,
    });
  });

  test("validates bounds and exports concrete worker values", () => {
    for (const name of names) delete process.env[name];
    const config = ClapConfigSchema.parse({ cache: { checkpoints: {
      enabled: false,
      minimum_tokens: 4_096,
      interval_tokens: 1_024,
      max_checkpoints: 6,
      budget_fraction: 0.125,
      budget_bytes: 33_554_432,
    } } });
    applyConfigToEnv(config);
    expect(process.env.CLAP_CACHE_CHECKPOINTS_ENABLED).toBe("0");
    expect(process.env.CLAP_CACHE_CHECKPOINT_MINIMUM_TOKENS).toBe("4096");
    expect(process.env.CLAP_CACHE_CHECKPOINT_INTERVAL_TOKENS).toBe("1024");
    expect(process.env.CLAP_CACHE_CHECKPOINT_MAX).toBe("6");
    expect(process.env.CLAP_CACHE_CHECKPOINT_BUDGET_BASIS_POINTS).toBe("1250");
    expect(process.env.CLAP_CACHE_CHECKPOINT_BUDGET_BYTES).toBe("33554432");
    expect(() => ClapConfigSchema.parse({ cache: { checkpoints: { budget_fraction: 1.1 } } })).toThrow();
    expect(() => ClapConfigSchema.parse({ cache: { checkpoints: { max_checkpoints: 0 } } })).toThrow();
  });
});
