#!/usr/bin/env bun
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { loadAssetConfiguration, validateCacheTestAssets, type AssetConfiguration,
  type Backend, type ValidatedAsset } from "./validate-cache-test-assets";

export type MatrixCase = {
  id: string; tier: "b" | "c"; backend: Backend; architecture: string; asset: string;
  provisioning: "pinned" | "unprovisioned"; skipReason?: string; scenarios: string[];
  timeoutMs: number; maxResidentBytes: number;
};
export type MatrixConfiguration = { schemaVersion: 1; cases: MatrixCase[] };
export type MatrixFilters = { tier?: "b" | "c"; cases?: string[]; backends?: Backend[];
  scenarios?: string[] };
export type MatrixScenario = { caseID: string; backend: Backend; scenarios: string[];
  command: string[]; environment: Record<string, string> };
export type MatrixReport = {
  schemaVersion: 1; tier: "b" | "c"; status: "passed" | "skipped" | "failed";
  startedAt: string; finishedAt: string;
  cases: Array<{ id: string; backend: Backend; architecture: string; scenarios: string[];
    status: "passed" | "skipped" | "failed"; durationMs: number; reason?: string;
    revision?: string; timeoutMs: number; maxResidentBytes: number; error?: string }>;
};

export async function loadMatrixConfiguration(path = resolve(import.meta.dir,
  "../config/cache-correctness-matrix.json")): Promise<MatrixConfiguration> {
  const value = JSON.parse(await readFile(path, "utf8"));
  if (value.schemaVersion !== 1 || !Array.isArray(value.cases)) {
    throw new Error("cache correctness matrix has an unsupported schema");
  }
  for (const item of value.cases as MatrixCase[]) validateCase(item);
  return value;
}

export function selectCases(config: MatrixConfiguration, filters: MatrixFilters): MatrixCase[] {
  const selected = config.cases.filter((item) =>
    (!filters.tier || item.tier === filters.tier) &&
    (!filters.cases?.length || filters.cases.includes(item.id)) &&
    (!filters.backends?.length || filters.backends.includes(item.backend)) &&
    (!filters.scenarios?.length || filters.scenarios.some((value) => item.scenarios.includes(value))));
  if (filters.cases?.some((id) => !config.cases.some((item) => item.id === id))) {
    throw new Error("unknown cache correctness case filter");
  }
  return selected;
}

export function scenarioFor(item: MatrixCase, asset: ValidatedAsset,
                            requestedScenarios?: string[]): MatrixScenario {
  const scenarios = requestedScenarios?.length ?
    item.scenarios.filter((value) => requestedScenarios.includes(value)) : item.scenarios;
  const expected = asset.expectedProbe ? { ...asset.expectedProbe, scenarios } : undefined;
  return item.backend === "gguf" ? { caseID: item.id, backend: "gguf", scenarios,
    command: ["bun", "run", "runtime:llama:physical:test"],
    environment: { CLAP_TEST_GGUF_MODEL: asset.path,
      CLAP_CACHE_TEST_SCENARIOS: scenarios.join(","),
      ...(expected ? { CLAP_CACHE_TEST_EXPECTED_PROBE: JSON.stringify(expected) } : {}) } } :
    { caseID: item.id, backend: "mlx", scenarios,
      command: ["bun", "run", "runtime:mlx:physical:test"],
      environment: { CLAP_TEST_MLX_MODEL: asset.path,
        CLAP_CACHE_TEST_SCENARIOS: scenarios.join(","),
        ...(expected ? { CLAP_CACHE_TEST_EXPECTED_PROBE: JSON.stringify(expected) } : {}) } };
}

export async function runCacheCorrectnessMatrix(options: {
  env?: Record<string, string | undefined>; assetConfig?: AssetConfiguration;
  matrixConfig?: MatrixConfiguration; filters?: MatrixFilters;
  execute?: (scenario: MatrixScenario, env: Record<string, string>, timeoutMs: number) =>
    Promise<void>; now?: () => Date;
} = {}): Promise<MatrixReport> {
  const env = options.env ?? process.env;
  const now = options.now ?? (() => new Date());
  const tier = options.filters?.tier ?? "b";
  const startedAt = now().toISOString();
  const assetConfig = options.assetConfig ?? await loadAssetConfiguration();
  const matrix = options.matrixConfig ?? await loadMatrixConfiguration();
  const configuredCases = tier === "b" ? tierBDefaultCases(assetConfig) : matrix.cases;
  const selected = selectCases({ schemaVersion: 1, cases: configuredCases },
    { ...options.filters, tier });
  if (selected.length === 0) throw new Error("cache correctness filters selected no cases");
  const validation = await validateCacheTestAssets({ env: { ...env,
    REQUIRE_ASSETS: "0", CLAP_CACHE_TEST_REQUIRE_ASSETS: "0" }, config: assetConfig });
  const assets = new Map(validation.assets.map((asset) => [asset.backend, asset]));
  const requireAssets = env.CLAP_CACHE_TEST_REQUIRE_ASSETS === "1" || env.REQUIRE_ASSETS === "1";
  const execute = options.execute ?? executeScenario;
  const results: MatrixReport["cases"] = [];

  for (const item of selected) {
    const scenarios = options.filters?.scenarios?.length ? item.scenarios.filter((value) =>
      options.filters!.scenarios!.includes(value)) : item.scenarios;
    const asset = item.provisioning === "pinned" ? assets.get(item.backend) : undefined;
    if (!asset || item.provisioning === "unprovisioned") {
      const unexpected = item.provisioning === "pinned" && !asset;
      results.push({ id: item.id, backend: item.backend, architecture: item.architecture,
        scenarios, status: unexpected && requireAssets ? "failed" : "skipped", durationMs: 0,
        reason: item.skipReason ?? (unexpected ? "asset_not_installed" : "asset_unprovisioned"),
        timeoutMs: item.timeoutMs, maxResidentBytes: item.maxResidentBytes });
      continue;
    }
    if (asset.maxBytes > item.maxResidentBytes) {
      results.push({ id: item.id, backend: item.backend, architecture: item.architecture,
        scenarios, revision: asset.revision, status: "failed", durationMs: 0,
        reason: "asset_exceeds_case_resource_bound", timeoutMs: item.timeoutMs,
        maxResidentBytes: item.maxResidentBytes });
      continue;
    }
    const home = await mkdtemp(resolve(tmpdir(), `clap-cache-${item.id}-`));
    const started = performance.now();
    try {
      await execute(scenarioFor(item, asset, scenarios), {
        ...cleanEnvironment(env), ...scenarioFor(item, asset, scenarios).environment,
        CLAP_HOME: home, CLAP_CACHE_TEST_MAX_RESIDENT_BYTES: String(item.maxResidentBytes),
      }, item.timeoutMs);
      results.push({ id: item.id, backend: item.backend, architecture: item.architecture,
        scenarios, revision: asset.revision, status: "passed",
        durationMs: Math.round(performance.now() - started), timeoutMs: item.timeoutMs,
        maxResidentBytes: item.maxResidentBytes });
    } catch (error) {
      results.push({ id: item.id, backend: item.backend, architecture: item.architecture,
        scenarios, revision: asset.revision, status: "failed",
        durationMs: Math.round(performance.now() - started), timeoutMs: item.timeoutMs,
        maxResidentBytes: item.maxResidentBytes, error: sanitizeError(error) });
    } finally {
      await rm(home, { recursive: true, force: true });
    }
  }
  const status = results.some((item) => item.status === "failed") ? "failed" :
    results.every((item) => item.status === "skipped") ? "skipped" : "passed";
  return { schemaVersion: 1, tier, status, startedAt, finishedAt: now().toISOString(),
    cases: results };
}

function tierBDefaultCases(config: AssetConfiguration): MatrixCase[] {
  return (["gguf", "mlx"] as Backend[]).map((backend) => ({ id: backend, tier: "b",
    backend, architecture: config.assets[backend].pin?.architecture ?? "unprovisioned",
    asset: backend, provisioning: config.assets[backend].pin ? "pinned" : "unprovisioned",
    skipReason: config.assets[backend].skipReason,
    scenarios: config.assets[backend].pin?.expectedProbe?.scenarios ?? ["cold"],
    timeoutMs: 600_000, maxResidentBytes: 8_589_934_592 }));
}

function validateCase(item: MatrixCase) {
  if (!item.id || !["b", "c"].includes(item.tier) || !["gguf", "mlx"].includes(item.backend) ||
      !item.architecture || !["pinned", "unprovisioned"].includes(item.provisioning) ||
      item.scenarios.length === 0 || !Number.isSafeInteger(item.timeoutMs) || item.timeoutMs <= 0 ||
      !Number.isSafeInteger(item.maxResidentBytes) || item.maxResidentBytes <= 0) {
    throw new Error("cache correctness matrix contains an invalid case");
  }
}

async function executeScenario(scenario: MatrixScenario, env: Record<string, string>, timeoutMs: number) {
  const child = Bun.spawn(scenario.command, { cwd: resolve(import.meta.dir, ".."), env,
    stdout: "inherit", stderr: "inherit" });
  const timer = setTimeout(() => child.kill(), timeoutMs);
  try { const exit = await child.exited;
    if (exit !== 0) throw new Error(`${scenario.caseID} physical probe exited ${exit}`);
  } finally { clearTimeout(timer); }
}

function cleanEnvironment(env: Record<string, string | undefined>): Record<string, string> {
  return Object.fromEntries(Object.entries(env).filter(([key, value]) => value !== undefined &&
    !/(TOKEN|SECRET|PASSWORD|CREDENTIAL|AUTH|API_KEY|ACCESS_KEY|PRIVATE_KEY)/i.test(key))) as
    Record<string, string>;
}
function sanitizeError(error: unknown) {
  return (error instanceof Error ? error.message : "physical probe failed")
    .replace(/(?:\/[^\s:]+)+/g, "<path>").slice(0, 300);
}

function parseCLI(args: string[]): MatrixFilters {
  const filters: MatrixFilters = {};
  for (let index = 0; index < args.length; ++index) {
    const [flag, inline] = args[index].split("=", 2); const value = inline ?? args[++index];
    if (flag === "--tier" && (value === "b" || value === "c")) filters.tier = value;
    else if (flag === "--case") (filters.cases ??= []).push(...value.split(","));
    else if (flag === "--backend" && (value === "gguf" || value === "mlx"))
      (filters.backends ??= []).push(value);
    else if (flag === "--scenario") (filters.scenarios ??= []).push(...value.split(","));
    else throw new Error(`invalid cache correctness argument: ${flag}`);
  }
  return filters;
}

if (import.meta.main) {
  const output = process.env.CLAP_CACHE_TEST_REPORT ??
    resolve(import.meta.dir, "../build/cache-correctness/report.json");
  try {
    const report = await runCacheCorrectnessMatrix({ filters: parseCLI(process.argv.slice(2)) });
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({ tier: report.tier, status: report.status,
      cases: report.cases.map(({ id, status, reason }) => ({ id, status, reason })) }));
    if (report.status === "failed") process.exit(1);
  } catch (error) {
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, `${JSON.stringify({ schemaVersion: 1, status: "failed",
      error: sanitizeError(error) }, null, 2)}\n`);
    console.error(sanitizeError(error)); process.exit(1);
  }
}
