import { describe, expect, test } from "bun:test";
import { deriveCacheIdentity, sanitizeCacheIdentityLabel, type PhysicalCacheDomain } from "./derive";
import { digestToNonzeroU64, encodeLengthPrefixed, hmacSha256 } from "./encoding";
import { apiKeyPrincipal, trustedLocalPrincipal } from "./principal";
import type { InstallationSecret } from "./types";

const secret: InstallationSecret = {
  generation: "sec_123e4567-e89b-42d3-a456-426614174000",
  key: Uint8Array.from({ length: 32 }, (_, index) => index),
};
const physical: PhysicalCacheDomain = {
  backend: "llama.cpp",
  modelRevision: "sha256:abc123",
  tokenizer: "tok-v2",
  contextAllocation: 8192,
  kvFormat: "q8_0",
  layoutVersion: 3,
};

describe("cache identity derivation", () => {
  test("matches a fixed deterministic vector with lowercase hex", () => {
    const identity = deriveCacheIdentity(secret, apiKeyPrincipal("api-key-record-7"), {
      namespace: "workspace",
      project: "payments",
      harness: "coding-v4",
      agent: "reviewer",
      session: "session-99",
      priority: "background",
    }, physical);

    expect(identity).toEqual({
      generation: secret.generation,
      tenantRoot: "6f64c827dbe30bac5d1d2b7e37c70ba23e012c793624af8d64222adac5112f44",
      fingerprints: {
        project: "f1482819163f613fa61f18a23b2aa510319dd4d8db62934fca3a1d7475b47c3b",
        harness: "94bdbacbdf215f56a6e55920c34a287947d6b21bb1194208205e2b83eb302027",
        agent: "d844f6c06cc42e03137243f37db076fcb86b602a409118a62001f67604d793e2",
        session: "f269f6ff4a9cae43ffb8a8a7510e1c6a63669634fb08deca7bfc6dd304959b41",
      },
      scope: {
        kind: "session",
        fingerprint: "f269f6ff4a9cae43ffb8a8a7510e1c6a63669634fb08deca7bfc6dd304959b41",
      },
      priority: "background",
      display: {
        namespace: "workspace",
        project: "payments",
        harness: "coding-v4",
        agent: "reviewer",
        session: "session-99",
      },
      physical: {
        fingerprint: "b18e8feeea0150d5dfb46d58abf61212f65d3ade3f22eaf255702a2d8b0f7874",
        namespace: "37d89e4f01aeb4d83c701c31f207105880818a464228e884754a4bca115d9706",
        namespaceId: 4024140329223369944n,
      },
    });
    expect(allHex(identity).every((value) => /^[0-9a-f]{64}$/.test(value))).toBe(true);
  });

  test("is stable for identical installation, principal, intent, and physical domain", () => {
    const derive = () => deriveCacheIdentity(secret, apiKeyPrincipal("record-1"), { project: "alpha" }, physical);
    expect(derive()).toEqual(derive());
  });

  test("isolates equal caller labels between authenticated tenants", () => {
    const intent = { namespace: "same", project: "same", session: "same" };
    const first = deriveCacheIdentity(secret, apiKeyPrincipal("record-1"), intent, physical);
    const second = deriveCacheIdentity(secret, apiKeyPrincipal("record-2"), intent, physical);

    expect(first.tenantRoot).not.toBe(second.tenantRoot);
    expect(first.fingerprints.project).not.toBe(second.fingerprints.project);
    expect(first.physical.namespace).not.toBe(second.physical.namespace);
  });

  test("trusted local identity is fixed and distinct from every API key record", () => {
    const local = deriveCacheIdentity(secret, trustedLocalPrincipal(), { project: "alpha" }, physical);
    const api = deriveCacheIdentity(secret, apiKeyPrincipal("trusted_local"), { project: "alpha" }, physical);
    const localAgain = deriveCacheIdentity(secret, trustedLocalPrincipal(), { project: "alpha" }, physical);

    expect(local).toEqual(localAgain);
    expect(local.tenantRoot).not.toBe(api.tenantRoot);
  });

  test("domain-separates scopes and derives each independently under the tenant", () => {
    const all = deriveCacheIdentity(secret, apiKeyPrincipal("record-1"), {
      project: "same",
      harness: "same",
      agent: "same",
      session: "same",
    }, physical);
    const projectOnly = deriveCacheIdentity(secret, apiKeyPrincipal("record-1"), { project: "same" }, physical);
    const values = Object.values(all.fingerprints) as string[];

    expect(new Set(values).size).toBe(4);
    expect(all.fingerprints.project).toBe(projectOnly.fingerprints.project);
    expect(all.scope.kind).toBe("session");
    expect(projectOnly.scope.kind).toBe("project");
  });

  test("length prefixes resist delimiter and field-boundary collisions", () => {
    const first = hmacSha256(secret.key, "test", ["a", "b:c"]);
    const second = hmacSha256(secret.key, "test", ["a:b", "c"]);
    const third = hmacSha256(secret.key, "test:a", ["b", "c"]);

    expect(first).not.toEqual(second);
    expect(first).not.toEqual(third);
    expect(encodeLengthPrefixed(["ab", "c"])).not.toEqual(encodeLengthPrefixed(["a", "bc"]));
  });

  test.each([
    ["backend", { backend: "mlx" }],
    ["model revision", { modelRevision: "sha256:def456" }],
    ["tokenizer", { tokenizer: "tok-v3" }],
    ["context allocation", { contextAllocation: 4096 }],
    ["KV format", { kvFormat: "f16" }],
    ["layout version", { layoutVersion: 4 }],
  ] satisfies Array<[string, Partial<PhysicalCacheDomain>]>)
  ("binds the physical namespace to %s", (_name, change) => {
    const baseline = deriveCacheIdentity(secret, apiKeyPrincipal("record-1"), { project: "alpha" }, physical);
    const changed = deriveCacheIdentity(secret, apiKeyPrincipal("record-1"), { project: "alpha" }, {
      ...physical,
      ...change,
    });
    expect(changed.physical.fingerprint).not.toBe(baseline.physical.fingerprint);
    expect(changed.physical.namespace).not.toBe(baseline.physical.namespace);
  });

  test("reduces digest bytes as nonzero unsigned big endian", () => {
    expect(digestToNonzeroU64(Uint8Array.from([0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]))).toBe(
      0x0123456789abcdefn,
    );
    expect(digestToNonzeroU64(new Uint8Array(32))).toBe(1n);
    expect(() => digestToNonzeroU64(new Uint8Array(7))).toThrow("at least 8 bytes");
  });

  test("normalizes bounded display labels and rejects unsafe labels", () => {
    expect(sanitizeCacheIdentityLabel("  cafe\u0301  ")).toBe("café");
    expect(() => sanitizeCacheIdentityLabel(" ")).toThrow("1-128");
    expect(() => sanitizeCacheIdentityLabel("bad\nlabel")).toThrow("control characters");
    expect(() => sanitizeCacheIdentityLabel("é".repeat(65))).toThrow("1-128");
  });
});

function allHex(identity: ReturnType<typeof deriveCacheIdentity>): string[] {
  return [
    identity.tenantRoot,
    ...Object.values(identity.fingerprints).filter((value): value is string => value !== undefined),
    identity.scope.fingerprint,
    identity.physical.fingerprint,
    identity.physical.namespace,
  ];
}
