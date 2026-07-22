#!/usr/bin/env bun
import { createHash } from "node:crypto";
import { lstat, readFile, realpath } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";

export type Backend = "gguf" | "mlx";
export type AssetPin = {
  source: string;
  architecture: string;
  revision: string;
  maxBytes: number;
  sha256?: string;
  manifestSha256?: string;
  requiredFiles?: Record<string, string>;
  expectedProbe?: {
    scenarios: string[];
    logicalTokenSha256: string;
    physicalStateSha256: string;
    selectedNextToken: number;
    top16QuantizedLogitSha256: string;
  };
};
export type AssetConfiguration = {
  schemaVersion: 1;
  assets: Record<Backend, { required: boolean; provisioning?: "pinned" | "unprovisioned";
    skipReason?: string; pin: AssetPin | null }>;
};
export type ValidatedAsset = {
  backend: Backend;
  path: string;
  architecture: string;
  revision: string;
  expectedProbe?: AssetPin["expectedProbe"];
};
export type ValidationResult = {
  status: "ready" | "skipped";
  assets: ValidatedAsset[];
  skipped: Array<{ backend: Backend; reason: string }>;
};

const aliases: Record<Backend, string[]> = {
  gguf: ["CLAP_CACHE_TEST_GGUF_MODEL", "CLAP_TEST_GGUF_MODEL"],
  mlx: ["CLAP_CACHE_TEST_MLX_MODEL", "CLAP_TEST_MLX_MODEL"],
};
const shaPattern = /^[a-f0-9]{64}$/;

export async function loadAssetConfiguration(path = resolve(import.meta.dir,
  "../config/cache-correctness-assets.json")): Promise<AssetConfiguration> {
  const value = JSON.parse(await readFile(path, "utf8"));
  if (value.schemaVersion !== 1 || !value.assets?.gguf || !value.assets?.mlx) {
    throw new Error("cache correctness asset configuration has an unsupported schema");
  }
  return value;
}

export async function validateCacheTestAssets(options: {
  env?: Record<string, string | undefined>;
  config?: AssetConfiguration;
} = {}): Promise<ValidationResult> {
  const env = options.env ?? process.env;
  const config = options.config ?? await loadAssetConfiguration();
  const requireAssets = env.CLAP_CACHE_TEST_REQUIRE_ASSETS === "1" || env.REQUIRE_ASSETS === "1";
  const fetchAssets = env.CLAP_CACHE_TEST_FETCH === "1" || env.FETCH === "1";
  const configured = (Object.keys(aliases) as Backend[]).map((backend) => ({
    backend,
    path: aliases[backend].map((name) => env[name]).find(Boolean),
  }));
  const missing = configured.filter((entry) => !entry.path).map((entry) => entry.backend);
  const provisioned = missing.filter((backend) => config.assets[backend].pin !== null);
  const policyRequired = missing.filter((backend) => config.assets[backend].required);
  if (missing.length > 0 && fetchAssets) {
    const unpinned = missing.filter((backend) => !config.assets[backend].pin);
    if (unpinned.length > 0) {
      throw new Error(`asset fetch unsupported until reviewed pins exist: ${unpinned.join(",")}`);
    }
    throw new Error("asset fetch was requested but no fetch transport is configured");
  }
  if (policyRequired.length > 0 || (provisioned.length > 0 && requireAssets)) {
    throw new Error(`required cache correctness assets are absent: ${
      policyRequired.length > 0 ? policyRequired.join(",") : provisioned.join(",")}`);
  }

  const supplied = configured.filter((entry): entry is { backend: Backend; path: string } =>
    Boolean(entry.path));
  const skipped = missing.map((backend) => ({ backend,
    reason: config.assets[backend].pin ? "asset_not_installed" :
      config.assets[backend].skipReason ?? "asset_unprovisioned" }));
  if (supplied.length === 0) return { status: "skipped", assets: [], skipped };
  const rootInput = env.CLAP_CACHE_TEST_ASSET_ROOT;
  if (!rootInput || !isAbsolute(rootInput)) {
    throw new Error("CLAP_CACHE_TEST_ASSET_ROOT must be an absolute canonical asset root");
  }
  const root = await realpath(rootInput).catch(() => {
    throw new Error("cache correctness asset root does not exist");
  });
  if (root !== resolve(rootInput)) {
    throw new Error("cache correctness asset root must be canonical and may not be a symlink");
  }

  const assets: ValidatedAsset[] = [];
  for (const entry of supplied) {
    const pin = config.assets[entry.backend].pin;
    if (!pin) throw new Error(`${entry.backend} asset has no reviewed pin`);
    validatePin(entry.backend, pin);
    const canonical = await realpath(entry.path).catch(() => {
      throw new Error(`${entry.backend} asset is unavailable`);
    });
    assertWithinRoot(root, canonical, entry.backend);
    const stat = await lstat(canonical);
    if (entry.backend === "gguf") {
      if (!stat.isFile()) throw new Error("gguf asset must be a regular file");
      if (stat.size > pin.maxBytes) throw new Error("gguf asset exceeds its pinned byte limit");
      if (await sha256File(canonical) !== pin.sha256) throw new Error("gguf asset checksum mismatch");
    } else {
      if (!stat.isDirectory()) throw new Error("mlx asset must be a directory");
      let total = 0;
      const manifest: string[] = [];
      for (const [file, checksum] of Object.entries(pin.requiredFiles!)) {
        if (file.startsWith("/") || file.split("/").includes("..")) {
          throw new Error("mlx pin contains an invalid required file");
        }
        const target = await realpath(resolve(canonical, file)).catch(() => {
          throw new Error(`mlx required file is absent: ${file}`);
        });
        assertWithinRoot(canonical, target, "mlx");
        const fileStat = await lstat(target);
        if (!fileStat.isFile()) throw new Error(`mlx required file is not regular: ${file}`);
        total += fileStat.size;
        const digest = await sha256File(target);
        if (digest !== checksum) throw new Error(`mlx required file checksum mismatch: ${file}`);
        manifest.push(`${file}\0${digest}\n`);
      }
      if (total > pin.maxBytes) throw new Error("mlx asset exceeds its pinned byte limit");
      const digest = createHash("sha256").update(manifest.sort().join("")).digest("hex");
      if (digest !== pin.manifestSha256) throw new Error("mlx asset manifest checksum mismatch");
    }
    assets.push({ backend: entry.backend, path: canonical,
      architecture: pin.architecture, revision: pin.revision,
      expectedProbe: pin.expectedProbe });
  }
  return { status: "ready", assets, skipped };
}

function validatePin(backend: Backend, pin: AssetPin) {
  if (!pin.source?.trim() || !pin.architecture?.trim() || !pin.revision?.trim() ||
      !Number.isSafeInteger(pin.maxBytes) || pin.maxBytes <= 0) {
    throw new Error(`${backend} pin metadata is incomplete`);
  }
  if (backend === "gguf" && !shaPattern.test(pin.sha256 ?? "")) {
    throw new Error("gguf pin requires a real SHA-256 checksum");
  }
  if (backend === "mlx" && (!shaPattern.test(pin.manifestSha256 ?? "") ||
      !pin.requiredFiles || Object.keys(pin.requiredFiles).length === 0 ||
      Object.values(pin.requiredFiles).some((value) => !shaPattern.test(value)))) {
    throw new Error("mlx pin requires a checksummed non-empty manifest");
  }
  if (pin.expectedProbe && (pin.expectedProbe.scenarios.length === 0 ||
      !shaPattern.test(pin.expectedProbe.logicalTokenSha256) ||
      !shaPattern.test(pin.expectedProbe.physicalStateSha256) ||
      !Number.isSafeInteger(pin.expectedProbe.selectedNextToken) ||
      !shaPattern.test(pin.expectedProbe.top16QuantizedLogitSha256))) {
    throw new Error(`${backend} pin contains invalid expected probe metadata`);
  }
}

function assertWithinRoot(root: string, target: string, backend: string) {
  const nested = relative(root, target);
  if (nested === "" || (!nested.startsWith("..") && !isAbsolute(nested))) return;
  throw new Error(`${backend} asset escapes its canonical root`);
}

async function sha256File(path: string) {
  return createHash("sha256").update(await readFile(path)).digest("hex");
}

if (import.meta.main) {
  try {
    const result = await validateCacheTestAssets();
    console.log(JSON.stringify({ status: result.status,
      ready: result.assets.map(({ backend, architecture, revision }) =>
        ({ backend, architecture, revision })), skipped: result.skipped }));
  } catch (error) {
    console.error(error instanceof Error ? error.message : "asset validation failed");
    process.exit(1);
  }
}
