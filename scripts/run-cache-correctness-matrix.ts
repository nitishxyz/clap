#!/usr/bin/env bun
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { validateCacheTestAssets, type Backend, type ValidatedAsset } from
  "./validate-cache-test-assets";
import type { AssetConfiguration } from "./validate-cache-test-assets";

export type MatrixScenario = {
  backend: Backend;
  command: string[];
  environment: Record<string, string>;
};
export type MatrixReport = {
  schemaVersion: 1;
  status: "passed" | "skipped" | "failed";
  startedAt: string;
  finishedAt: string;
  backends: Array<{
    backend: Backend;
    architecture?: string;
    revision?: string;
    status: "passed" | "skipped" | "failed";
    durationMs: number;
    error?: string;
  }>;
};

export function scenariosFor(assets: ValidatedAsset[]): MatrixScenario[] {
  return assets.map((asset) => asset.backend === "gguf" ? {
    backend: "gguf",
    command: ["bun", "run", "runtime:llama:physical:test"],
    environment: { CLAP_TEST_GGUF_MODEL: asset.path },
  } : {
    backend: "mlx",
    command: ["bun", "run", "runtime:mlx:physical:test"],
    environment: { CLAP_TEST_MLX_MODEL: asset.path },
  });
}

export async function runCacheCorrectnessMatrix(options: {
  env?: Record<string, string | undefined>;
  config?: AssetConfiguration;
  execute?: (scenario: MatrixScenario, env: Record<string, string>, timeoutMs: number) =>
    Promise<void>;
  now?: () => Date;
} = {}): Promise<MatrixReport> {
  const env = options.env ?? process.env;
  const now = options.now ?? (() => new Date());
  const startedAt = now().toISOString();
  const validation = await validateCacheTestAssets({ env, config: options.config });
  const timeoutMs = positiveInteger(env.CLAP_CACHE_TEST_TIMEOUT_MS, 10 * 60_000);
  const clapHome = await mkdtemp(resolve(tmpdir(), "clap-cache-correctness-"));
  const execute = options.execute ?? executeScenario;
  const backends: MatrixReport["backends"] = validation.skipped.map((backend) => ({
    backend, status: "skipped", durationMs: 0,
  }));
  try {
    for (const scenario of scenariosFor(validation.assets)) {
      const asset = validation.assets.find((candidate) => candidate.backend === scenario.backend)!;
      const started = performance.now();
      try {
        await execute(scenario, {
          ...cleanEnvironment(env), ...scenario.environment, CLAP_HOME: clapHome,
        }, timeoutMs);
        backends.push({ backend: scenario.backend, architecture: asset.architecture,
          revision: asset.revision, status: "passed",
          durationMs: Math.round(performance.now() - started) });
      } catch (error) {
        backends.push({ backend: scenario.backend, architecture: asset.architecture,
          revision: asset.revision, status: "failed",
          durationMs: Math.round(performance.now() - started),
          error: sanitizeError(error) });
      }
    }
  } finally {
    await rm(clapHome, { recursive: true, force: true });
  }
  const status = backends.some((backend) => backend.status === "failed") ? "failed" :
    backends.every((backend) => backend.status === "skipped") ? "skipped" : "passed";
  return { schemaVersion: 1, status, startedAt, finishedAt: now().toISOString(), backends };
}

async function executeScenario(scenario: MatrixScenario, env: Record<string, string>,
                               timeoutMs: number) {
  const process = Bun.spawn(scenario.command, {
    cwd: resolve(import.meta.dir, ".."), env, stdout: "inherit", stderr: "inherit",
  });
  const timer = setTimeout(() => process.kill(), timeoutMs);
  try {
    const exit = await process.exited;
    if (exit !== 0) throw new Error(`${scenario.backend} physical probe exited ${exit}`);
  } finally {
    clearTimeout(timer);
  }
}

function cleanEnvironment(env: Record<string, string | undefined>): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(env)) {
    if (value !== undefined &&
        !/(TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|API_KEY|ACCESS_KEY|PRIVATE_KEY)/i.test(key)) {
      result[key] = value;
    }
  }
  return result;
}

function positiveInteger(value: string | undefined, fallback: number) {
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function sanitizeError(error: unknown) {
  const message = error instanceof Error ? error.message : "physical probe failed";
  return message.replace(/(?:\/[^\s:]+)+/g, "<path>").slice(0, 300);
}

if (import.meta.main) {
  const output = process.env.CLAP_CACHE_TEST_REPORT ??
    resolve(import.meta.dir, "../build/cache-correctness/report.json");
  try {
    const report = await runCacheCorrectnessMatrix();
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({ status: report.status,
      backends: report.backends.map(({ backend, status }) => ({ backend, status })) }));
    if (report.status === "failed") process.exit(1);
  } catch (error) {
    const report: MatrixReport = { schemaVersion: 1, status: "failed",
      startedAt: new Date().toISOString(), finishedAt: new Date().toISOString(),
      backends: [], };
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
    console.error(sanitizeError(error));
    process.exit(1);
  }
}
