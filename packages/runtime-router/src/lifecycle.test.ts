import { describe, expect, test } from "bun:test";
import { ModelLifecycleManager, parseKeepAliveMs } from "./lifecycle";
import type { ResolvedModel } from "@clap/models";

const resolved: ResolvedModel = {
  id: "acme/tiny",
  input: "acme/tiny",
  backend: "llama",
  format: "gguf",
  modelPath: "/models/tiny.gguf",
  status: "available",
};

describe("runtime model lifecycle", () => {
  test("parses keep-alive durations and always", () => {
    expect(parseKeepAliveMs("30s")).toBe(30_000);
    expect(parseKeepAliveMs("15m")).toBe(900_000);
    expect(parseKeepAliveMs("1h")).toBe(3_600_000);
    expect(parseKeepAliveMs("always")).toBeNull();
    expect(() => parseKeepAliveMs("soon")).toThrow("invalid keep-alive duration");
  });

  test("extends expiry on usage and expires idle models", () => {
    let now = Date.parse("2026-01-01T00:00:00.000Z");
    const manager = new ModelLifecycleManager(() => now);
    const loaded = manager.load(resolved, { keepAlive: "30s" });
    expect(loaded.expiresAt).toBe("2026-01-01T00:00:30.000Z");

    now += 10_000;
    const active = manager.beginUsage(resolved);
    manager.finishUsage(active);
    expect(active.expiresAt).toBe("2026-01-01T00:00:40.000Z");

    now += 31_000;
    expect(manager.list()).toEqual([]);
  });

  test("keeps always-loaded models until manual cleanup", () => {
    let now = Date.parse("2026-01-01T00:00:00.000Z");
    const manager = new ModelLifecycleManager(() => now);
    const loaded = manager.load(resolved, { keepAlive: "always" });
    expect(loaded.expiresAt).toBeNull();
    expect(loaded.always).toBe(true);

    now += 86_400_000;
    expect(manager.list()).toHaveLength(1);
    expect(manager.unload(resolved).unloaded).toBe(true);
    expect(manager.list()).toEqual([]);
  });
});
