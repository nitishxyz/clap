import type { ClapModel } from "@clap/api";
import { ggufModelDisplayName, isGgufModel } from "@clap/runtime-llama";
import { isMlxModelDirectorySync, mlxModelDisplayName, mlxModelPaths } from "@clap/runtime-mlx";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { mkdir, open, readdir, rename, rm } from "node:fs/promises";
import { basename, dirname, join } from "node:path";
import { hfAuthGuidance, resolveHfToken } from "./hf-auth";

export { assertFileCredentialPermissions, deleteStoredHfToken, hfAuthGuidance, hfAuthStatus, hfTokenEnvVars, isHfAuthError, resolveHfToken, storeHfToken } from "./hf-auth";

export type PullModelRequest = {
  model: string;
  file?: string;
  backend?: BackendOverride;
  force?: boolean;
};

export type BackendOverride = "gguf" | "mlx";

export type AliasBackendTarget = {
  backend: BackendOverride;
  repo: string;
  file?: string;
};

export type ModelAlias = {
  id: string;
  displayName: string;
  mlx: AliasBackendTarget;
  gguf: AliasBackendTarget;
};

export type ResolvedModel = {
  id: string;
  input: string;
  backend: "llama" | "mlx";
  format: "gguf" | "mlx";
  modelPath?: string;
  alias?: ModelAlias;
  target?: AliasBackendTarget;
  status: "available" | "not_downloaded" | "unsupported" | "not_found";
  message?: string;
};

export type PullResult = {
  id: string;
  repo: string;
  file?: string;
  backend: "llama" | "mlx";
  format: "gguf" | "mlx";
  modelPath: string;
  files: string[];
  selected?: ModelResolveOption;
};

export type ModelResolveOption = {
  id: string;
  model: string;
  backend: BackendOverride;
  format: "gguf" | "mlx" | "safetensors";
  repo: string;
  file?: string;
  sizeBytes?: number;
  quantization?: string;
  supported: boolean;
  unsupportedReason?: string;
  recommended: boolean;
  reason: string;
};

export type ModelResolveResult = {
  model: string;
  repo: string;
  options: ModelResolveOption[];
  selected?: ModelResolveOption;
};

export type PullProgress = {
  bytesReceived?: number;
  totalBytes?: number;
  currentFile?: string;
};

export type PullModelOptions = {
  onProgress?: (progress: PullProgress) => void;
  signal?: AbortSignal;
};

export type PullTarget = {
  model: string;
  repo: string;
  file?: string;
  backend?: BackendOverride;
  alias?: ModelAlias;
  cachePath?: string;
  key: string;
};

type HfSibling = {
  rfilename?: string;
  size?: number;
  lfs?: {
    sha256?: string;
    size?: number;
  };
};

type HfModelInfo = {
  siblings?: HfSibling[];
};

type JsonRecord = Record<string, unknown>;

export const modelAliases: ModelAlias[] = [
  {
    id: "qwen2.5:3b",
    displayName: "Qwen2.5 3B Instruct",
    mlx: { backend: "mlx", repo: "mlx-community/Qwen2.5-3B-Instruct-4bit" },
    gguf: { backend: "gguf", repo: "bartowski/Qwen2.5-3B-Instruct-GGUF", file: "Qwen2.5-3B-Instruct-Q4_K_M.gguf" },
  },
  {
    id: "llama3.2:3b",
    displayName: "Llama 3.2 3B Instruct",
    mlx: { backend: "mlx", repo: "mlx-community/Llama-3.2-3B-Instruct-4bit" },
    gguf: { backend: "gguf", repo: "bartowski/Llama-3.2-3B-Instruct-GGUF", file: "Llama-3.2-3B-Instruct-Q4_K_M.gguf" },
  },
];

function envConfiguredModels(): ClapModel[] {
  const ggufModels = (process.env.CLAP_GGUF_MODEL_PATHS ?? "")
    .split(",")
    .map((model) => model.trim())
    .filter((model) => isGgufModel(model) && existsSync(model))
    .map((model) => ({
      id: model,
      object: "model" as const,
      displayName: ggufModelDisplayName(model),
      backend: "llama" as const,
      format: "gguf",
      status: "available" as const,
    } satisfies Partial<ClapModel> & Pick<ClapModel, "id" | "object" | "displayName" | "backend" | "format" | "status">))
    .map((model) => withModelMetadata(model, { path: model.id }));
  const mlxModels = mlxModelPaths()
    .filter((model) => existsSync(model))
    .map((model) => ({
      id: model,
      object: "model" as const,
      displayName: mlxModelDisplayName(model),
      backend: "mlx" as const,
      format: "mlx",
      status: "available" as const,
    } satisfies Partial<ClapModel> & Pick<ClapModel, "id" | "object" | "displayName" | "backend" | "format" | "status">))
    .map((model) => withModelMetadata(model, { path: model.id }));
  return [...ggufModels, ...mlxModels];
}

export function listModels(): ClapModel[] {
  return [...envConfiguredModels(), ...listCachedModels()];
}

// Non-blocking variant for polling endpoints (dashboard, model lists). The
// synchronous walk can stall the whole event loop for seconds when the disk
// is saturated by model weight reads or downloads, freezing every request.
// Directory listings are awaited here and results are memoized briefly with
// stale-while-revalidate so pollers never wait on a refresh already running.
let listModelsMemo: { key: string; at: number; value: ClapModel[] } | undefined;
let listModelsRefresh: { key: string; promise: Promise<ClapModel[]> } | undefined;

function listModelsTtlMs(): number {
  const raw = Number(process.env.CLAP_MODEL_LIST_TTL_MS);
  return Number.isFinite(raw) ? raw : 2000;
}

function listModelsCacheKey(root: string): string {
  return [root, process.env.CLAP_GGUF_MODEL_PATHS ?? "", process.env.CLAP_MLX_MODEL_PATHS ?? ""].join("\0");
}

export function invalidateModelListCache(): void {
  listModelsMemo = undefined;
  readJsonCache.clear();
}

export async function listModelsAsync(): Promise<ClapModel[]> {
  const ttl = listModelsTtlMs();
  const root = hfCacheRoot();
  const key = listModelsCacheKey(root);
  if (listModelsMemo?.key === key && Date.now() - listModelsMemo.at < ttl) return listModelsMemo.value;
  if (listModelsRefresh?.key !== key) {
    const promise = (async () => {
        const repoEntries = existsSync(root)
          ? (await readdir(root, { withFileTypes: true })).filter((entry) => entry.isDirectory())
          : [];
        const cached: ClapModel[] = [];
        for (const entry of repoEntries) {
          const repoPath = join(root, entry.name);
          const files = await readdir(repoPath, { withFileTypes: true });
          cached.push(...cachedModelsForRepo(repoPath, repoFromCacheDirName(entry.name), files));
        }
        const value = [...envConfiguredModels(), ...cached];
        listModelsMemo = { key, at: Date.now(), value };
        return value;
    })();
    listModelsRefresh = { key, promise };
    void promise.then(
      () => { if (listModelsRefresh?.promise === promise) listModelsRefresh = undefined; },
      () => { if (listModelsRefresh?.promise === promise) listModelsRefresh = undefined; },
    );
  }
  // Serve moderately stale data instantly while a refresh is in flight so a
  // slow disk never delays pollers; hard-cap staleness at 10x the TTL.
  if (listModelsMemo?.key === key && Date.now() - listModelsMemo.at < ttl * 10) return listModelsMemo.value;
  return listModelsRefresh.promise;
}

export function listAliases(): ClapModel[] {
  return aliasModels();
}

export function listCachedModels(): ClapModel[] {
  const root = hfCacheRoot();
  if (!existsSync(root)) return [];
  return readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .flatMap((entry) => cachedModelsForRepo(join(root, entry.name), repoFromCacheDirName(entry.name)));
}

export async function pullModel(request: PullModelRequest, options: PullModelOptions = {}): Promise<PullResult> {
  const useResolver = !request.file && request.backend !== "mlx";
  const selected = useResolver ? (await resolveModelOptions(request, options)).selected : undefined;
  const target = selected
    ? buildPullTarget(selected.repo, selected.file, selected.backend)
    : resolvePullTarget(request);
  if (!request.force) {
    const cached = cachedPullResultForTarget(target);
    if (cached) return { ...cached, selected };
  }
  const repo = target.repo;
  const info = await fetchModelInfo(repo, options);
  const siblings = info.siblings ?? [];
  if (target.file) {
    try {
      return { ...await pullGgufFile(repo, target.file, siblings, options), selected };
    } finally {
      invalidateModelListCache();
    }
  }
  if (target.backend === "gguf") {
    throw new Error(`GGUF pull for ${request.model} requires --file`);
  }

  const ggufFiles = siblings.map((sibling) => sibling.rfilename).filter((file): file is string => Boolean(file?.toLowerCase().endsWith(".gguf")));
  try {
    if (ggufFiles.length === 1) return { ...await pullGgufFile(repo, ggufFiles[0]!, siblings, options), selected };
    if (ggufFiles.length > 1) {
      throw new Error(`multiple GGUF files found for ${repo}; pass --file with one of: ${ggufFiles.join(", ")}`);
    }
    return { ...await pullMlxRepo(repo, siblings, options), selected };
  } finally {
    invalidateModelListCache();
  }
}

export async function removeModel(model: string): Promise<string[]> {
  const alias = findAlias(model);
  const targetRepos = new Set<string>();
  if (alias) {
    targetRepos.add(alias.mlx.repo);
    targetRepos.add(alias.gguf.repo);
  }
  const removed: string[] = [];
  for (const cached of listCachedModels()) {
    const repo = cached.repo ?? "";
    const matches = cached.id === model || repo === model || targetRepos.has(repo);
    if (!matches) continue;
    const path = cached.localPath ?? cached.id;
    if (!existsSync(path)) continue;
    await rm(path, { recursive: true, force: true });
    removed.push(path);
  }
  const root = hfCacheRoot();
  for (const path of removed) {
    const dir = dirname(path);
    if (dir !== root && dir.startsWith(root) && existsSync(dir) && readdirSync(dir).every((name) => name.endsWith(".part"))) {
      await rm(dir, { recursive: true, force: true });
    }
  }
  if (removed.length) invalidateModelListCache();
  return removed;
}

export async function resolveModelOptions(request: PullModelRequest, options: PullModelOptions = {}): Promise<ModelResolveResult> {
  const explicit = resolvePullTarget(request);
  const alias = findAlias(request.model);
  const targets = alias && !request.backend && !request.file ? [alias.mlx, alias.gguf] : [{ backend: explicit.backend, repo: explicit.repo, file: explicit.file }];
  const resolvedOptions: ModelResolveOption[] = [];
  for (const target of targets) {
    const info = await fetchModelInfo(target.repo, options);
    resolvedOptions.push(...optionsForRepo(request.model, target.repo, info.siblings ?? [], target.backend, target.file));
  }
  const filtered = applyOverrides(resolvedOptions, request);
  const ranked = rankOptions(filtered);
  const selected = ranked.find((option) => option.recommended && option.supported);
  return { model: request.model, repo: explicit.repo, options: ranked, selected };
}

export function clapHome(): string {
  return process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap");
}

export function hfCacheRoot(): string {
  return join(clapHome(), "models", "huggingface");
}

export function resolveModel(model: string, backend?: BackendOverride): ResolvedModel {
  if (isGgufModel(model) || existsSync(model)) {
    const cached = findCachedModel(model);
    if (cached) return cached;
    if (isGgufModel(model)) return { id: model, input: model, backend: "llama", format: "gguf", status: "available", modelPath: model };
    if (isMlxModelDirectorySync(model)) return { id: model, input: model, backend: "mlx", format: "mlx", status: "available", modelPath: model };
  }

  const cachedRepo = findCachedRepoModel(model, backend);
  if (cachedRepo) return cachedRepo;

  const alias = findAlias(model);
  if (!alias) {
    return {
      id: model,
      input: model,
      backend: backend === "mlx" ? "mlx" : "llama",
      format: backend === "mlx" ? "mlx" : "gguf",
      status: "not_found",
      message: `Model ${model} was not found in the local cache or aliases. Run: clap pull ${model}${backend ? ` --backend ${backend}` : ""}, or pass a local .gguf file / MLX directory.`,
    };
  }

  const target = aliasTarget(alias, backend);
  const cachedPath = cachedPathForTarget(target);
  if (cachedPath) {
    return {
      id: alias.id,
      input: model,
      backend: target.backend === "mlx" ? "mlx" : "llama",
      format: target.backend === "mlx" ? "mlx" : "gguf",
      modelPath: cachedPath,
      alias,
      target,
      status: "available",
    };
  }
  if (!backend) {
    const fallbackTarget = target.backend === "mlx" ? alias.gguf : alias.mlx;
    const fallbackCachedPath = cachedPathForTarget(fallbackTarget);
    if (fallbackCachedPath) {
      return {
        id: alias.id,
        input: model,
        backend: fallbackTarget.backend === "mlx" ? "mlx" : "llama",
        format: fallbackTarget.backend === "mlx" ? "mlx" : "gguf",
        modelPath: fallbackCachedPath,
        alias,
        target: fallbackTarget,
        status: "available",
      };
    }
  }

  const unsupported = target.backend === "mlx" && !supportsMlxTarget();
  return {
    id: alias.id,
    input: model,
    backend: target.backend === "mlx" ? "mlx" : "llama",
    format: target.backend === "mlx" ? "mlx" : "gguf",
    alias,
    target,
    status: unsupported ? "unsupported" : "not_downloaded",
    message: unsupported
      ? `Alias ${alias.id} requires the MLX backend, which is only available on macOS arm64. Try: clap pull ${alias.id} --backend gguf`
      : `Model ${alias.id} is not cached. Run: clap pull ${alias.id}${target.file ? ` --file ${target.file}` : ""}${backend ? ` --backend ${backend}` : ""}`,
  };
}

export function resolvePullTarget(request: PullModelRequest): PullTarget {
  const alias = findAlias(request.model);
  if (!alias) return buildPullTarget(request.model, request.file, request.backend);
  const target = aliasTarget(alias, request.backend);
  return buildPullTarget(target.repo, request.file ?? target.file, target.backend, alias);
}

export function findAlias(id: string): ModelAlias | undefined {
  return modelAliases.find((alias) => alias.id === id);
}

export function aliasTarget(alias: ModelAlias, backend?: BackendOverride): AliasBackendTarget {
  if (backend === "gguf") return alias.gguf;
  if (backend === "mlx") return alias.mlx;
  return supportsMlxTarget() ? alias.mlx : alias.gguf;
}

function aliasModels(): ClapModel[] {
  return modelAliases.map((alias) => {
    const resolved = resolveModel(alias.id);
    return withModelMetadata({
      id: alias.id,
      object: "model" as const,
      displayName: alias.displayName,
      backend: resolved.backend === "llama" ? "llama" as const : "mlx" as const,
      format: resolved.format,
      status: resolved.status === "not_found" ? "not_downloaded" as const : resolved.status,
      alias: alias.id,
      repo: resolved.target?.repo,
      file: resolved.target?.file,
      pull: resolved.target ? { model: alias.id, file: resolved.target.file, backend: resolved.target.backend } : undefined,
      reason: resolved.message,
    }, { repo: resolved.target?.repo, path: resolved.modelPath, alias: true });
  });
}

function findCachedModel(model: string): ResolvedModel | undefined {
  for (const cached of listCachedModels()) {
    if (cached.id !== model) continue;
    const modelPath = cached.localPath ?? cached.id;
    return {
      id: cached.id,
      input: model,
      backend: cached.backend,
      format: cached.format as "gguf" | "mlx",
      modelPath,
      status: "available",
    };
  }
  return undefined;
}

function findCachedRepoModel(model: string, backend?: BackendOverride): ResolvedModel | undefined {
  let target: PullTarget;
  try {
    target = resolvePullTarget({ model, backend });
  } catch {
    return undefined;
  }
  if (target.alias) return undefined;
  const cached = cachedPullResultForTarget(target);
  if (!cached) return undefined;
  return {
    id: model,
    input: model,
    backend: cached.backend,
    format: cached.format,
    modelPath: cached.modelPath,
    target: {
      backend: cached.format,
      repo: cached.repo,
      file: cached.file,
    },
    status: "available",
  };
}

export function cachedPullResultForTarget(target: PullTarget): PullResult | undefined {
  const repoDir = repoCacheDir(target.repo);
  if (target.file) {
    const modelPath = join(repoDir, basename(target.file));
    if (!existsSync(modelPath)) return undefined;
    return {
      id: `${target.repo}:${target.file}`,
      repo: target.repo,
      file: target.file,
      backend: "llama",
      format: "gguf",
      modelPath,
      files: [modelPath],
    };
  }
  if (!existsSync(repoDir)) return undefined;
  const cached = cachedModelsForRepo(repoDir, target.repo);
  const mlx = cached.find((model) => model.backend === "mlx");
  if (target.backend !== "gguf" && mlx) {
    return {
      id: target.repo,
      repo: target.repo,
      backend: "mlx",
      format: "mlx",
      modelPath: repoDir,
      files: cached.map((model) => model.localPath ?? model.id),
    };
  }
  const gguf = cached.filter((model) => model.backend === "llama");
  if (gguf.length === 1) {
    const modelPath = gguf[0]!.localPath ?? gguf[0]!.id;
    return {
      id: target.repo,
      repo: target.repo,
      file: basename(modelPath),
      backend: "llama",
      format: "gguf",
      modelPath,
      files: [modelPath],
    };
  }
  return undefined;
}

function buildPullTarget(model: string, file?: string, backend?: BackendOverride, alias?: ModelAlias): PullTarget {
  const repo = normalizeRepo(model);
  const cachePath = file ? join(repoCacheDir(repo), basename(file)) : backend === "mlx" ? repoCacheDir(repo) : undefined;
  const backendKey = file ? "gguf" : backend ?? "";
  return {
    model,
    repo,
    file,
    backend,
    alias,
    cachePath,
    key: JSON.stringify([alias?.id ?? "", repo, file ?? "", backendKey, cachePath ?? ""]),
  };
}

function cachedPathForTarget(target: AliasBackendTarget): string | undefined {
  const repoDir = repoCacheDir(target.repo);
  if (target.backend === "gguf") {
    const path = join(repoDir, basename(target.file ?? ""));
    return existsSync(path) ? path : undefined;
  }
  return existsSync(repoDir) && cachedModelsForRepo(repoDir, target.repo).some((model) => model.backend === "mlx") ? repoDir : undefined;
}

function supportsMlxTarget(): boolean {
  if (process.env.CLAP_TEST_PLATFORM) return process.env.CLAP_TEST_PLATFORM === "darwin-arm64";
  return process.platform === "darwin" && process.arch === "arm64";
}

function optionsForRepo(model: string, repo: string, siblings: HfSibling[], backend?: BackendOverride, file?: string): ModelResolveOption[] {
  const files = siblings.map((sibling) => sibling.rfilename).filter((name): name is string => Boolean(name));
  const gguf = siblings.filter((sibling) => sibling.rfilename?.toLowerCase().endsWith(".gguf"));
  const hasMlx = files.includes("config.json") && files.some((name) => name === "tokenizer.json" || name === "tokenizer_config.json") && files.some((name) => name.endsWith(".safetensors"));
  const hasSafetensors = files.some((name) => name.endsWith(".safetensors"));
  const result: ModelResolveOption[] = [];
  if (hasMlx) {
    result.push({
      id: `${repo}:mlx`,
      model,
      backend: "mlx",
      format: "mlx",
      repo,
      sizeBytes: selectedFilesSize(siblings, files.filter((name) => name === "config.json" || name === "tokenizer.json" || name === "tokenizer_config.json" || name.endsWith(".safetensors"))),
      supported: supportsMlxTarget(),
      unsupportedReason: supportsMlxTarget() ? undefined : "MLX runner is supported only on macOS arm64; choose a GGUF option instead.",
      recommended: false,
      reason: supportsMlxTarget() ? "MLX is preferred on macOS arm64 for Apple Silicon." : "MLX artifact found but this platform cannot run MLX.",
    });
  }
  for (const sibling of gguf) {
    const rfilename = sibling.rfilename!;
    const quantization = inferQuantization("gguf", rfilename);
    result.push({
      id: `${repo}:${rfilename}`,
      model,
      backend: "gguf",
      format: "gguf",
      repo,
      file: rfilename,
      sizeBytes: sibling.size,
      quantization,
      supported: true,
      recommended: false,
      reason: quantization ? `GGUF ${quantization} artifact is runnable with llama.cpp.` : "GGUF artifact is runnable with llama.cpp.",
    });
  }
  if (!hasMlx && !gguf.length && hasSafetensors) {
    result.push({
      id: `${repo}:safetensors`,
      model,
      backend: "mlx",
      format: "safetensors",
      repo,
      sizeBytes: selectedFilesSize(siblings, files.filter((name) => name.endsWith(".safetensors"))),
      supported: false,
      unsupportedReason: "Raw safetensors repos are not directly runnable yet. Use an MLX-converted repo or a GGUF quantization, or convert the model first.",
      recommended: false,
      reason: "Source weights were found, but Clap does not yet serve raw safetensors directly.",
    });
  }
  return result.filter((option) => (!backend || option.backend === backend) && (!file || option.file === file));
}

function applyOverrides(options: ModelResolveOption[], request: PullModelRequest): ModelResolveOption[] {
  return options.filter((option) => (!request.backend || option.backend === request.backend) && (!request.file || option.file === request.file));
}

function rankOptions(options: ModelResolveOption[]): ModelResolveOption[] {
  const scored = options.map((option) => ({ option, score: optionScore(option) })).sort((a, b) => b.score - a.score || (a.option.sizeBytes ?? 0) - (b.option.sizeBytes ?? 0));
  const best = scored.find((entry) => entry.option.supported)?.option;
  return scored.map(({ option }) => ({ ...option, recommended: option === best }));
}

function optionScore(option: ModelResolveOption): number {
  if (!option.supported) return -100;
  if (option.backend === "mlx") return supportsMlxTarget() ? 100 : -10;
  const quant = option.quantization?.toUpperCase() ?? "";
  if (quant === "Q4_K_M") return 80;
  if (quant === "Q4_K_S") return 75;
  if (quant === "Q3_K_M") return 70;
  if (quant.startsWith("Q5")) return 50;
  if (quant.startsWith("Q8") || /BF16|F16/.test(quant)) return 10;
  return 40;
}

function cachedModelsForRepo(repoPath: string, repo = repoPath, listed?: Array<{ name: string; isFile(): boolean }>): ClapModel[] {
  const files = listed ?? readdirSync(repoPath, { withFileTypes: true });
  const ggufFiles = files.filter((entry) => entry.isFile() && isGgufModel(entry.name));
  const ggufModels = files
    .filter((entry) => entry.isFile() && isGgufModel(entry.name))
    .map((entry) => ({ file: entry.name, path: join(repoPath, entry.name) }))
    .map((model) => ({
      id: ggufFiles.length === 1 ? repo : `${repo}:${model.file}`,
      object: "model" as const,
      displayName: ggufModelDisplayName(model.path),
      backend: "llama" as const,
      format: "gguf",
      status: "available" as const,
      repo,
      file: model.file,
      localPath: model.path,
    } satisfies Partial<ClapModel> & Pick<ClapModel, "id" | "object" | "displayName" | "backend" | "format" | "status">))
    .map((model) => withModelMetadata(model, { repo, path: model.localPath, file: model.file }));
  const hasMlxLayout = files.some((entry) => entry.name === "config.json") && (
    files.some((entry) => entry.name === "tokenizer.json" || entry.name === "tokenizer_config.json") ||
    files.some((entry) => entry.name.endsWith(".safetensors"))
  );
  const mlxModels = hasMlxLayout ? [withModelMetadata({
    id: repo,
    object: "model" as const,
    displayName: mlxModelDisplayName(repoPath),
    backend: "mlx" as const,
    format: "mlx",
    status: "available" as const,
    repo,
    localPath: repoPath,
  }, { repo, path: repoPath })] : [];
  return [...ggufModels, ...mlxModels];
}

function withModelMetadata(
  model: Partial<ClapModel> & Pick<ClapModel, "id" | "object" | "displayName" | "backend" | "format" | "status">,
  options: { repo?: string; path?: string; file?: string; alias?: boolean } = {},
): ClapModel {
  const config = options.path && model.format === "mlx" ? readJson(join(options.path, "config.json")) : undefined;
  const tokenizerConfig = options.path && model.format === "mlx" ? readJson(join(options.path, "tokenizer_config.json")) : undefined;
  const index = options.path && model.format === "mlx" ? readJson(join(options.path, "model.safetensors.index.json")) : undefined;
  const name = displayNameFromRepo(options.repo) ?? model.displayName;
  const baseRepo = stringValue(config?.base_model) ?? stringValue(config?._name_or_path) ?? stringValue(tokenizerConfig?._name_or_path);
  const upstreamModalities = inferUpstreamModalities(config, tokenizerConfig, index);
  const limit = inferLimit(config, tokenizerConfig);
  return {
    ...model,
    name,
    provider: providerFromRepo(options.repo) ?? (options.alias ? "clap" : "local"),
    source: {
      type: options.alias ? "alias" : options.repo ? "huggingface" : "local",
      repo: options.repo,
      baseRepo,
    },
    modalities: servedModalities(),
    capabilities: servedCapabilities(),
    limit,
    upstream: {
      modalities: upstreamModalities,
      capabilities: upstreamCapabilities(upstreamModalities, config),
      limit,
    },
    architecture: firstString(arrayValue(config?.architectures)),
    modelType: stringValue(config?.model_type),
    quantization: inferQuantization(model.format, options.file ?? model.file, config),
  };
}

function servedModalities(): ClapModel["modalities"] {
  return { input: ["text"], output: ["text"] };
}

function servedCapabilities(): ClapModel["capabilities"] {
  return {
    chat: true,
    completion: false,
    streaming: true,
    temperature: true,
    system_prompt: true,
    attachment: false,
    reasoning: false,
    tool_call: true,
    structured_output: true,
  };
}

function upstreamCapabilities(modalities: ClapModel["modalities"], config?: JsonRecord): Partial<ClapModel["capabilities"]> {
  return {
    chat: true,
    streaming: true,
    temperature: true,
    system_prompt: true,
    attachment: modalities.input.some((modality) => modality === "image" || modality === "audio"),
    reasoning: hasKeyOrValue(config, /reason/i),
    tool_call: hasKeyOrValue(config, /tool[_-]?call|function[_-]?call/i),
    structured_output: hasKeyOrValue(config, /structured[_-]?output|json_schema/i),
  };
}

function inferUpstreamModalities(...records: Array<JsonRecord | undefined>): ClapModel["modalities"] {
  const input = new Set<"text" | "image" | "audio">(["text"]);
  const output = new Set<"text" | "image" | "audio">(["text"]);
  for (const record of records) {
    if (hasKeyOrValue(record, /vision|image|mm_projector/i)) input.add("image");
    if (hasKeyOrValue(record, /audio|speech|sound/i)) input.add("audio");
  }
  return { input: [...input], output: [...output] };
}

function inferLimit(config?: JsonRecord, tokenizerConfig?: JsonRecord): ClapModel["limit"] {
  return {
    context: firstPositiveInteger(
      nestedNumber(config, ["text_config", "max_position_embeddings"]),
      nestedNumber(config, ["max_position_embeddings"]),
      nestedNumber(config, ["max_sequence_length"]),
      nestedNumber(config, ["seq_length"]),
      nestedNumber(config, ["n_ctx"]),
      nestedNumber(config, ["context_length"]),
      nestedNumber(tokenizerConfig, ["model_max_length"]),
    ),
    output: firstPositiveInteger(
      nestedNumber(config, ["max_output_tokens"]),
      nestedNumber(config, ["max_new_tokens"]),
      nestedNumber(config, ["generation_config", "max_new_tokens"]),
    ),
  };
}

function inferQuantization(format: string, file?: string, config?: JsonRecord): string | undefined {
  const quantization = config?.quantization ?? config?.quantization_config;
  if (typeof quantization === "string") return quantization;
  if (isRecord(quantization)) {
    const bits = numberValue(quantization.bits) ?? numberValue(quantization.load_in_bits);
    const groupSize = numberValue(quantization.group_size);
    if (bits && groupSize) return `${bits}-bit group ${groupSize}`;
    if (bits) return `${bits}-bit`;
  }
  if (format === "gguf" && file) return file.match(/(?:^|[.-])(Q\d(?:_[A-Z0-9]+)*)(?:[.-]|$)/i)?.[1];
  return undefined;
}

// Metadata JSON files (config.json etc.) change only when a model is
// re-pulled; cache them briefly so polling endpoints do not re-read files
// from a disk that may be saturated by weight loads or downloads.
const readJsonCache = new Map<string, { at: number; value: JsonRecord | undefined }>();
const READ_JSON_TTL_MS = 10_000;

function readJsonTtlMs(): number {
  const raw = Number(process.env.CLAP_MODEL_LIST_TTL_MS);
  return Number.isFinite(raw) ? raw : READ_JSON_TTL_MS;
}

function readJson(path: string): JsonRecord | undefined {
  const cached = readJsonCache.get(path);
  if (cached && Date.now() - cached.at < readJsonTtlMs()) return cached.value;
  const value = readJsonUncached(path);
  if (readJsonCache.size > 512) readJsonCache.clear();
  readJsonCache.set(path, { at: Date.now(), value });
  return value;
}

function readJsonUncached(path: string): JsonRecord | undefined {
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    return undefined;
  }
}

function hasKeyOrValue(value: unknown, pattern: RegExp): boolean {
  if (typeof value === "string") return pattern.test(value);
  if (Array.isArray(value)) return value.some((entry) => hasKeyOrValue(entry, pattern));
  if (!isRecord(value)) return false;
  return Object.entries(value).some(([key, entry]) => pattern.test(key) || hasKeyOrValue(entry, pattern));
}

function nestedNumber(record: JsonRecord | undefined, path: string[]): number | undefined {
  let value: unknown = record;
  for (const segment of path) {
    if (!isRecord(value)) return undefined;
    value = value[segment];
  }
  return numberValue(value);
}

function firstPositiveInteger(...values: Array<number | undefined>): number | null {
  return values.find((value): value is number => typeof value === "number" && Number.isInteger(value) && value > 0) ?? null;
}

function providerFromRepo(repo?: string): string | undefined {
  return repo?.split("/")[0];
}

function displayNameFromRepo(repo?: string): string | undefined {
  return repo?.split("/").at(-1)?.replace(/[-_]+/g, " ").trim();
}

function isRecord(value: unknown): value is JsonRecord {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function arrayValue(value: unknown): unknown[] | undefined {
  return Array.isArray(value) ? value : undefined;
}

function firstString(values?: unknown[]): string | undefined {
  return values?.find((value): value is string => typeof value === "string");
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function numberValue(value: unknown): number | undefined {
  return typeof value === "number" ? value : undefined;
}

function repoFromCacheDirName(name: string): string {
  const separator = name.indexOf("--");
  if (separator <= 0 || separator === name.length - 2) return name;
  return `${name.slice(0, separator)}/${name.slice(separator + 2)}`;
}

async function pullGgufFile(repo: string, file: string, siblings: HfSibling[], options: PullModelOptions): Promise<PullResult> {
  validateRelativeFile(file);
  if (!file.toLowerCase().endsWith(".gguf")) throw new Error(`GGUF pulls require a .gguf file: ${file}`);
  if (siblings.length > 0 && !siblings.some((sibling) => sibling.rfilename === file)) {
    throw new Error(`file not found in ${repo}: ${file}`);
  }
  const repoDir = repoCacheDir(repo);
  const target = join(repoDir, basename(file));
  const totalBytes = siblingSize(siblings, file);
  options.onProgress?.({ currentFile: file, bytesReceived: 0, totalBytes });
  await downloadFile(repo, file, target, {
    bytesReceived: 0,
    totalBytes,
    currentFile: file,
    onProgress: options.onProgress,
    signal: options.signal,
    expectedSha256: siblingSha256(siblings, file),
  });
  return {
    id: `${repo}:${file}`,
    repo,
    file,
    backend: "llama",
    format: "gguf",
    modelPath: target,
    files: [target],
  };
}

async function pullMlxRepo(repo: string, siblings: HfSibling[], options: PullModelOptions): Promise<PullResult> {
  // Include every small JSON/jinja sidecar: chat_template.jinja (HF moved chat
  // templates out of tokenizer_config.json), model.safetensors.index.json for
  // sharded weights, generation_config.json, special_tokens_map.json, etc.
  const files = siblings
    .map((sibling) => sibling.rfilename)
    .filter((file): file is string => Boolean(file))
    .filter((file) => !file.includes("/"))
    .filter((file) => file.endsWith(".json") || file.endsWith(".jinja") || file.endsWith(".safetensors") || file === "merges.txt" || file === "vocab.txt");
  if (!files.includes("config.json")) throw new Error(`MLX repo ${repo} is missing config.json`);
  if (!files.some((file) => file === "tokenizer.json" || file === "tokenizer_config.json" || file.endsWith(".safetensors"))) {
    throw new Error(`MLX repo ${repo} is missing tokenizer or safetensors files`);
  }

  const repoDir = repoCacheDir(repo);
  const downloaded: string[] = [];
  let bytesReceived = 0;
  const totalBytes = selectedFilesSize(siblings, files);
  options.onProgress?.({ bytesReceived, totalBytes, currentFile: files[0] });
  for (const file of files) {
    validateRelativeFile(file);
    const target = join(repoDir, file);
    const fileBytes = await downloadFile(repo, file, target, {
      bytesReceived,
      totalBytes,
      currentFile: file,
      onProgress: options.onProgress,
      signal: options.signal,
      expectedSha256: siblingSha256(siblings, file),
    });
    bytesReceived += fileBytes;
    downloaded.push(target);
  }
  return {
    id: repo,
    repo,
    backend: "mlx",
    format: "mlx",
    modelPath: repoDir,
    files: downloaded,
  };
}

async function fetchModelInfo(repo: string, options: PullModelOptions): Promise<HfModelInfo> {
  const response = await fetch(`${hfEndpoint()}/api/models/${repo}?blobs=true`, await hfFetchInit(options));
  if (response.status === 404) throw new Error(`Hugging Face repo not found: ${repo}`);
  if (response.status === 401 || response.status === 403) throw hfAuthError(response.status, `inspect Hugging Face repo ${repo}`);
  if (!response.ok) throw new Error(`failed to inspect Hugging Face repo ${repo}: ${response.status} ${response.statusText}`);
  return response.json() as Promise<HfModelInfo>;
}

async function downloadFile(repo: string, file: string, target: string, progress: Required<Pick<PullProgress, "bytesReceived" | "currentFile">> & Pick<PullProgress, "totalBytes"> & Pick<PullModelOptions, "onProgress" | "signal"> & { expectedSha256?: string }): Promise<number> {
  throwIfAborted(progress.signal);
  await mkdir(dirname(target), { recursive: true });
  const partial = `${target}.part`;
  let resumeFrom = existsSync(partial) ? Bun.file(partial).size : 0;
  const init = await hfFetchInit(progress);
  if (resumeFrom > 0) {
    init.headers = { ...(init.headers as Record<string, string> | undefined), Range: `bytes=${resumeFrom}-` };
  }
  const response = await fetch(`${hfEndpoint()}/${repo}/resolve/main/${file}`, init);
  if (response.status === 416 && resumeFrom > 0) {
    // Range not satisfiable: the partial file already contains the full payload.
    await response.body?.cancel().catch(() => undefined);
    await verifyChecksum(partial, repo, file, progress.expectedSha256);
    await rename(partial, target);
    progress.onProgress?.({ bytesReceived: progress.bytesReceived + resumeFrom, totalBytes: progress.totalBytes, currentFile: file });
    return resumeFrom;
  }
  if (response.status === 404) throw new Error(`file not found in ${repo}: ${file}`);
  if (response.status === 401 || response.status === 403) throw hfAuthError(response.status, `download ${repo}/${file}`);
  if (!response.ok) throw new Error(`failed to download ${repo}/${file}: ${response.status} ${response.statusText}`);
  const resumed = response.status === 206 && resumeFrom > 0;
  if (!resumed) resumeFrom = 0;  // server ignored the range; restart from scratch
  const contentLength = parseContentLength(response.headers.get("content-length"));
  const totalBytes = progress.totalBytes ?? (contentLength === undefined ? undefined : progress.bytesReceived + resumeFrom + contentLength);
  const reader = response.body?.getReader();
  if (!reader) throw new Error(`failed to download ${repo}/${file}: response body is empty`);
  const hasher = progress.expectedSha256 ? new Bun.CryptoHasher("sha256") : undefined;
  if (hasher && resumed) {
    for await (const chunk of Bun.file(partial).stream()) hasher.update(chunk);
  }
  const handle = await open(partial, resumed ? "a" : "w");
  let fileBytes = resumeFrom;
  if (resumed) {
    progress.onProgress?.({ bytesReceived: progress.bytesReceived + fileBytes, totalBytes, currentFile: file });
  }
  try {
    while (true) {
      throwIfAborted(progress.signal);
      const { done, value } = await reader.read();
      if (done) break;
      await handle.write(value);
      hasher?.update(value);
      fileBytes += value.byteLength;
      progress.onProgress?.({
        bytesReceived: progress.bytesReceived + fileBytes,
        totalBytes,
        currentFile: file,
      });
    }
    throwIfAborted(progress.signal);
    if (hasher && progress.expectedSha256) {
      const actual = hasher.digest("hex");
      if (actual !== progress.expectedSha256) {
        await rm(partial, { force: true }).catch(() => undefined);
        throw new Error(`checksum mismatch for ${repo}/${file}: expected sha256 ${progress.expectedSha256}, got ${actual}. The corrupt partial download was removed; retry the pull.`);
      }
    }
    await rename(partial, target);
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    // Keep the .part file so an interrupted download can resume later.
    throw error;
  } finally {
    await handle.close();
  }
  return fileBytes;
}

async function verifyChecksum(path: string, repo: string, file: string, expectedSha256?: string): Promise<void> {
  if (!expectedSha256) return;
  const hasher = new Bun.CryptoHasher("sha256");
  for await (const chunk of Bun.file(path).stream()) hasher.update(chunk);
  const actual = hasher.digest("hex");
  if (actual !== expectedSha256) {
    await rm(path, { force: true }).catch(() => undefined);
    throw new Error(`checksum mismatch for ${repo}/${file}: expected sha256 ${expectedSha256}, got ${actual}. The corrupt partial download was removed; retry the pull.`);
  }
}

function siblingSize(siblings: HfSibling[], file: string): number | undefined {
  return siblings.find((sibling) => sibling.rfilename === file)?.size;
}

function siblingSha256(siblings: HfSibling[], file: string): string | undefined {
  return siblings.find((sibling) => sibling.rfilename === file)?.lfs?.sha256;
}

function selectedFilesSize(siblings: HfSibling[], files: string[]): number | undefined {
  let total = 0;
  for (const file of files) {
    const size = siblingSize(siblings, file);
    if (size === undefined) return undefined;
    total += size;
  }
  return total;
}

function parseContentLength(value: string | null): number | undefined {
  if (!value) return undefined;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : undefined;
}

function normalizeRepo(model: string): string {
  const repo = model.trim().replace(/^https:\/\/huggingface\.co\//, "").replace(/\/$/, "");
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repo)) {
    throw new Error(`expected explicit Hugging Face repo like owner/model, got: ${model}`);
  }
  return repo;
}

function validateRelativeFile(file: string): void {
  if (file.startsWith("/") || file.includes("..")) throw new Error(`invalid file path: ${file}`);
}

function repoCacheDir(repo: string): string {
  return join(hfCacheRoot(), repo.replace("/", "--"));
}

function hfEndpoint(): string {
  return (process.env.CLAP_HF_ENDPOINT ?? "https://huggingface.co").replace(/\/$/, "");
}

async function hfFetchInit(options: Pick<PullModelOptions, "signal"> = {}): Promise<RequestInit> {
  const { token } = await resolveHfToken();
  return { signal: options.signal, ...(token ? { headers: { Authorization: `Bearer ${token}` } } : {}) };
}

function hfAuthError(status: number, action: string): Error {
  return new Error(`Hugging Face authentication failed (${status}) while trying to ${action}. ${hfAuthGuidance()}`);
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (!signal?.aborted) return;
  throw new DOMException("download cancelled", "AbortError");
}
