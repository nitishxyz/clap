import { afterEach, describe, expect, test } from "bun:test";
import { createHash } from "node:crypto";
import { mkdtemp, realpath, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { runCacheCorrectnessMatrix, scenarioFor, selectCases,
  type MatrixConfiguration } from "./run-cache-correctness-matrix";
import type { AssetConfiguration } from "./validate-cache-test-assets";

const roots: string[] = [];
afterEach(async () => Promise.all(roots.splice(0).map((root) =>
  rm(root, { recursive: true, force: true }))));
const hash = (value: string) => createHash("sha256").update(value).digest("hex");
function assets(checksum?: string): AssetConfiguration {
  return { schemaVersion: 1, assets: {
    gguf: { required: false, pin: checksum ? { source: "fixture", architecture: "standard",
      revision: "pin-1", maxBytes: 100, sha256: checksum } : null },
    mlx: { required: false, pin: null },
  } };
}
const matrix: MatrixConfiguration = { schemaVersion: 1, cases: [
  { id: "standard", tier: "c", backend: "gguf", architecture: "standard", asset: "gguf",
    provisioning: "pinned", scenarios: ["cold", "branch"], timeoutMs: 1234,
    maxResidentBytes: 2048 },
  { id: "sliding", tier: "c", backend: "mlx", architecture: "sliding", asset: "mlx",
    provisioning: "unprovisioned", skipReason: "not_pinned", scenarios: ["cold"],
    timeoutMs: 2345, maxResidentBytes: 4096 },
] };

describe("cache correctness matrix", () => {
  test("filters reviewed cases by tier case backend and scenario", () => {
    expect(selectCases(matrix, { tier: "c", cases: ["standard"] }).map((x) => x.id))
      .toEqual(["standard"]);
    expect(selectCases(matrix, { tier: "c", backends: ["mlx"] }).map((x) => x.id))
      .toEqual(["sliding"]);
    expect(selectCases(matrix, { tier: "c", scenarios: ["branch"] }).map((x) => x.id))
      .toEqual(["standard"]);
    expect(() => selectCases(matrix, { cases: ["missing"] })).toThrow("unknown");
  });

  test("creates backend command and narrows expected scenarios", () => {
    const plan = scenarioFor(matrix.cases[0], { backend: "gguf", path: "/asset/a",
      architecture: "standard", revision: "1", maxBytes: 100,
      expectedProbe: { scenarios: ["cold", "branch"],
        logicalTokenSha256: "a".repeat(64), physicalStateSha256: "b".repeat(64),
        selectedNextToken: 1, top16QuantizedLogitSha256: "c".repeat(64) } }, ["branch"]);
    expect(plan.command.at(-1)).toBe("runtime:llama:physical:test");
    expect(plan.environment.CLAP_CACHE_TEST_SCENARIOS).toBe("branch");
    expect(JSON.parse(plan.environment.CLAP_CACHE_TEST_EXPECTED_PROBE).scenarios)
      .toEqual(["branch"]);
  });

  test("Tier B no assets produces explicit successful skips", async () => {
    const report = await runCacheCorrectnessMatrix({ env: {}, assetConfig: assets(),
      matrixConfig: matrix, filters: { tier: "b" } });
    expect(report.status).toBe("skipped");
    expect(report.cases.map((entry) => entry.status)).toEqual(["skipped", "skipped"]);
  });

  test("Tier C isolates case home applies bounds and records unprovisioned skip", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-matrix-"))); roots.push(root);
    const model = join(root, "model.gguf"); await writeFile(model, "bytes");
    let captured: Record<string, string> = {};
    const report = await runCacheCorrectnessMatrix({ env: {
      CLAP_CACHE_TEST_ASSET_ROOT: root, CLAP_CACHE_TEST_GGUF_MODEL: model,
      SERVICE_TOKEN: "must-not-propagate",
    }, assetConfig: assets(hash("bytes")), matrixConfig: matrix, filters: { tier: "c" },
    execute: async (_, env, timeout) => {
      captured = env; expect(timeout).toBe(1234);
      expect(env.CLAP_HOME).toContain("clap-cache-standard-");
      expect(env.CLAP_CACHE_TEST_MAX_RESIDENT_BYTES).toBe("2048");
    } });
    expect(report.status).toBe("passed");
    expect(report.cases.find((entry) => entry.id === "standard")?.status).toBe("passed");
    expect(report.cases.find((entry) => entry.id === "sliding")).toMatchObject({
      status: "skipped", reason: "not_pinned",
    });
    expect(JSON.stringify(report)).not.toContain(root);
    expect(captured.SERVICE_TOKEN).toBeUndefined();
  });

  test("required pinned asset missing is an unexpected skip failure", async () => {
    const report = await runCacheCorrectnessMatrix({ env: { REQUIRE_ASSETS: "1" },
      assetConfig: assets(), matrixConfig: matrix,
      filters: { tier: "c", cases: ["standard"] } });
    expect(report.status).toBe("failed");
    expect(report.cases[0]).toMatchObject({ status: "failed", reason: "asset_not_installed" });
  });

  test("failure report sanitizes paths", async () => {
    const root = await realpath(await mkdtemp(join(tmpdir(), "clap-cache-matrix-"))); roots.push(root);
    const model = join(root, "model.gguf"); await writeFile(model, "bytes");
    const report = await runCacheCorrectnessMatrix({ env: {
      CLAP_CACHE_TEST_ASSET_ROOT: root, CLAP_TEST_GGUF_MODEL: model,
    }, assetConfig: assets(hash("bytes")), matrixConfig: matrix,
    filters: { tier: "c", cases: ["standard"] },
    execute: async () => { throw new Error(`failed at ${model}`); } });
    expect(report.cases[0].error).toBe("failed at <path>");
  });
});
