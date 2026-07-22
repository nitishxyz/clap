import type { ResolvedModel } from "@clap/models";
import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readdir, realpath, stat } from "node:fs/promises";
import { basename, relative, resolve } from "node:path";

export const PHYSICAL_CACHE_LAYOUT_VERSION = 1;

export type PhysicalModelDomainOptions = {
  contextAllocation: number;
  kvFormat: string;
  unifiedKv: boolean;
  layoutVersion?: number;
};

export type PhysicalModelDomain = {
  backend: "llama" | "mlx";
  modelRevision: string;
  resolvedRevision: string;
  modelArtifactFingerprint: string;
  tokenizer: string;
  tokenizerFingerprint: string;
  contextAllocation: number;
  kvFormat: string;
  unifiedKv: boolean;
  layoutVersion: number;
};

type HashCacheEntry = { signature: string; fingerprint: string };
const fileHashCache = new Map<string, HashCacheEntry>();
const treeHashCache = new Map<string, HashCacheEntry>();
let contentHashComputations = 0;

export async function derivePhysicalModelDomain(
  model: ResolvedModel,
  options: PhysicalModelDomainOptions,
): Promise<PhysicalModelDomain> {
  if (model.status !== "available" || !model.modelPath) throw new Error("Physical model domain requires an available local model");
  assertOptions(options);
  const layoutVersion = options.layoutVersion ?? PHYSICAL_CACHE_LAYOUT_VERSION;
  const path = await realpath(model.modelPath);

  let modelArtifactFingerprint: string;
  let tokenizerFingerprint: string;
  if (model.backend === "llama") {
    const metadata = await stat(path);
    const contentFingerprint = await fingerprintFile(path);
    modelArtifactFingerprint = createHash("sha256")
      .update("gguf-artifact\0")
      .update(path)
      .update("\0")
      .update(String(metadata.size))
      .update("\0")
      .update(contentFingerprint)
      .digest("hex");
    tokenizerFingerprint = modelArtifactFingerprint;
  } else {
    const files = await listFiles(path);
    const artifactFiles = files.filter((file) => !isTokenizerFile(file));
    const tokenizerFiles = files.filter(isTokenizerFile);
    modelArtifactFingerprint = await fingerprintTree(path, artifactFiles, "mlx-artifacts");
    tokenizerFingerprint = await fingerprintTree(path, tokenizerFiles, "mlx-tokenizer");
  }
  const resolvedRevision = model.revision ?? `local:${modelArtifactFingerprint}`;

  return {
    backend: model.backend,
    modelRevision: `${resolvedRevision}@sha256:${modelArtifactFingerprint}`,
    resolvedRevision,
    modelArtifactFingerprint,
    tokenizer: tokenizerFingerprint,
    tokenizerFingerprint,
    contextAllocation: options.contextAllocation,
    kvFormat: options.kvFormat.trim(),
    unifiedKv: options.unifiedKv,
    layoutVersion,
  };
}

export function physicalModelDomainCacheStats(): { contentHashComputations: number; files: number; trees: number } {
  return { contentHashComputations, files: fileHashCache.size, trees: treeHashCache.size };
}

export function clearPhysicalModelDomainCache(): void {
  fileHashCache.clear();
  treeHashCache.clear();
  contentHashComputations = 0;
}

async function fingerprintFile(path: string): Promise<string> {
  const metadata = await stat(path);
  if (!metadata.isFile()) throw new Error(`Model artifact is not a regular file: ${basename(path)}`);
  const signature = statSignature(metadata);
  const cached = fileHashCache.get(path);
  if (cached?.signature === signature) return cached.fingerprint;

  const digest = createHash("sha256");
  await new Promise<void>((resolvePromise, reject) => {
    const stream = createReadStream(path);
    stream.on("data", (chunk) => digest.update(chunk));
    stream.on("error", reject);
    stream.on("end", resolvePromise);
  });
  contentHashComputations += 1;
  const fingerprint = digest.digest("hex");
  fileHashCache.set(path, { signature, fingerprint });
  return fingerprint;
}

async function fingerprintTree(root: string, files: string[], domain: string): Promise<string> {
  const signatures = await Promise.all(files.map(async (file) => {
    const metadata = await stat(resolve(root, file));
    return `${file}\0${statSignature(metadata)}`;
  }));
  const signature = `${domain}\0${signatures.join("\0")}`;
  const key = `${root}\0${domain}`;
  const cached = treeHashCache.get(key);
  if (cached?.signature === signature) return cached.fingerprint;

  const digest = createHash("sha256");
  digest.update(domain);
  for (const file of files) {
    const fileDigest = await fingerprintFile(resolve(root, file));
    const name = Buffer.from(file);
    const length = Buffer.allocUnsafe(4);
    length.writeUInt32BE(name.byteLength);
    digest.update(length).update(name).update(Buffer.from(fileDigest, "hex"));
  }
  const fingerprint = digest.digest("hex");
  treeHashCache.set(key, { signature, fingerprint });
  return fingerprint;
}

async function listFiles(root: string): Promise<string[]> {
  const output: string[] = [];
  const visit = async (directory: string): Promise<void> => {
    const entries = await readdir(directory, { withFileTypes: true });
    entries.sort((left, right) => left.name.localeCompare(right.name));
    for (const entry of entries) {
      const path = resolve(directory, entry.name);
      if (entry.isSymbolicLink()) throw new Error(`MLX model contains a symbolic link: ${entry.name}`);
      if (entry.isDirectory()) await visit(path);
      else if (entry.isFile()) output.push(relative(root, path));
    }
  };
  await visit(root);
  return output.sort();
}

function isTokenizerFile(path: string): boolean {
  const name = basename(path).toLowerCase();
  return name.startsWith("tokenizer")
    || name === "vocab.json"
    || name === "merges.txt"
    || name === "special_tokens_map.json"
    || name === "added_tokens.json";
}

function statSignature(metadata: { size: number; mtimeMs: number; ctimeMs: number; ino: number | bigint }): string {
  return `${metadata.size}:${metadata.mtimeMs}:${metadata.ctimeMs}:${metadata.ino}`;
}

function assertOptions(options: PhysicalModelDomainOptions): void {
  if (!Number.isSafeInteger(options.contextAllocation) || options.contextAllocation < 0) {
    throw new Error("Physical model context allocation must be a nonnegative safe integer");
  }
  if (!options.kvFormat.trim()) throw new Error("Physical model KV format is required");
  const layout = options.layoutVersion ?? PHYSICAL_CACHE_LAYOUT_VERSION;
  if (!Number.isSafeInteger(layout) || layout <= 0) throw new Error("Physical model layout version must be a positive safe integer");
}
