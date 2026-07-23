import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import type { ResolvedModel } from "@clap/models";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  clearPhysicalModelDomainCache,
  derivePhysicalModelDomain,
  effectivePhysicalContextAllocation,
  physicalModelDomainCacheStats,
} from "./model-domain";

let root = "";

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), "clap-model-domain-"));
  clearPhysicalModelDomainCache();
});

afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

describe("physical model domains", () => {
  test("matches runtime auto-context semantics for llama and MLX", () => {
    expect(effectivePhysicalContextAllocation({ backend: "llama", context: 131072 }, 0)).toBe(131072);
    expect(effectivePhysicalContextAllocation({ backend: "llama", context: 131072 }, 8192)).toBe(8192);
    expect(effectivePhysicalContextAllocation({ backend: "llama", context: undefined }, 8192)).toBe(8192);
    expect(effectivePhysicalContextAllocation({ backend: "mlx", context: 131072 }, 0)).toBe(0);
  });

  test("fingerprints a canonical GGUF artifact and reuses its content hash", async () => {
    const path = join(root, "model.gguf");
    await writeFile(path, "small gguf fixture");
    const model = resolved("llama", path);
    const first = await derivePhysicalModelDomain(model, options());
    const second = await derivePhysicalModelDomain(model, options());

    expect(first).toEqual(second);
    expect(first.modelArtifactFingerprint).toMatch(/^[0-9a-f]{64}$/);
    expect(first.tokenizerFingerprint).toBe(first.modelArtifactFingerprint);
    expect(first.resolvedRevision).toBe(`local:${first.modelArtifactFingerprint}`);
    expect(physicalModelDomainCacheStats().contentHashComputations).toBe(1);
  });

  test("invalidates a cached GGUF hash when the artifact changes", async () => {
    const path = join(root, "model.gguf");
    await writeFile(path, "version one");
    const first = await derivePhysicalModelDomain(resolved("llama", path), options());
    await writeFile(path, "version two has another size");
    const second = await derivePhysicalModelDomain(resolved("llama", path), options());

    expect(second.modelArtifactFingerprint).not.toBe(first.modelArtifactFingerprint);
    expect(second.resolvedRevision).not.toBe(first.resolvedRevision);
    expect(physicalModelDomainCacheStats().contentHashComputations).toBe(2);
  });

  test("fingerprints MLX artifacts and tokenizer files independently", async () => {
    const path = join(root, "mlx-model");
    await mkdir(path);
    await writeFile(join(path, "config.json"), "config-v1");
    await writeFile(join(path, "model.safetensors"), "weights-v1");
    await writeFile(join(path, "tokenizer.json"), "tokenizer-v1");
    const model = resolved("mlx", path);
    const first = await derivePhysicalModelDomain(model, options());
    const repeated = await derivePhysicalModelDomain(model, options());
    expect(repeated).toEqual(first);
    expect(physicalModelDomainCacheStats().contentHashComputations).toBe(3);

    await writeFile(join(path, "tokenizer.json"), "tokenizer-v2-longer");
    const tokenizerChanged = await derivePhysicalModelDomain(model, options());
    expect(tokenizerChanged.modelArtifactFingerprint).toBe(first.modelArtifactFingerprint);
    expect(tokenizerChanged.tokenizerFingerprint).not.toBe(first.tokenizerFingerprint);

    await writeFile(join(path, "model.safetensors"), "weights-v2-longer");
    const weightsChanged = await derivePhysicalModelDomain(model, options());
    expect(weightsChanged.modelArtifactFingerprint).not.toBe(tokenizerChanged.modelArtifactFingerprint);
    expect(weightsChanged.tokenizerFingerprint).toBe(tokenizerChanged.tokenizerFingerprint);
  });

  test.each([
    ["context", { contextAllocation: 4096 }],
    ["KV format", { kvFormat: "q4_0" }],
    ["unified KV", { unifiedKv: false }],
    ["layout", { layoutVersion: 2 }],
  ] as const)("represents %s changes in the descriptor", async (_name, change) => {
    const path = join(root, "model.gguf");
    await writeFile(path, "same artifact");
    const baseline = await derivePhysicalModelDomain(resolved("llama", path), options());
    const changed = await derivePhysicalModelDomain(resolved("llama", path), { ...options(), ...change });
    expect(changed).not.toEqual(baseline);
    expect(changed.modelArtifactFingerprint).toBe(baseline.modelArtifactFingerprint);
  });

  test("preserves known resolved revisions and distinguishes revision changes", async () => {
    const path = join(root, "model.gguf");
    await writeFile(path, "same artifact");
    const first = await derivePhysicalModelDomain({ ...resolved("llama", path), revision: "commit:a" }, options());
    const second = await derivePhysicalModelDomain({ ...resolved("llama", path), revision: "commit:b" }, options());

    expect(first.resolvedRevision).toBe("commit:a");
    expect(second.resolvedRevision).toBe("commit:b");
    expect(second.modelRevision).not.toBe(first.modelRevision);
    expect(second.modelArtifactFingerprint).toBe(first.modelArtifactFingerprint);
    expect(physicalModelDomainCacheStats().contentHashComputations).toBe(1);
  });
});

function resolved(backend: "llama" | "mlx", modelPath: string): ResolvedModel {
  return {
    id: modelPath,
    input: modelPath,
    backend,
    format: backend === "llama" ? "gguf" : "mlx",
    modelPath,
    status: "available",
  };
}

function options() {
  return { contextAllocation: 8192, kvFormat: "q8_0", unifiedKv: true, layoutVersion: 1 };
}
