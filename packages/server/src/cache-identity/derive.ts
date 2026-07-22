import type { InstallationSecret } from "./types";
import type { CachePrincipal } from "./principal";
import { principalIdentity } from "./principal";
import { digestHex, digestToNonzeroU64, hmacSha256 } from "./encoding";

const DOMAIN_PREFIX = "clap.cache-identity.v1";
export const CACHE_IDENTITY_LABEL_MAX_BYTES = 128;

export type CacheIdentityScope = "tenant" | "project" | "harness" | "agent" | "session";
export type CacheIdentityPriority = "interactive" | "background";

export type CacheIdentityIntent = {
  namespace?: string;
  project?: string;
  harness?: string;
  agent?: string;
  session?: string;
  priority?: CacheIdentityPriority;
};

export type PhysicalCacheDomain = {
  backend: string;
  modelRevision: string;
  tokenizer: string;
  contextAllocation: number;
  kvFormat: string;
  unifiedKv?: boolean;
  layoutVersion: number;
};

export type CacheIdentityDisplay = {
  namespace?: string;
  project?: string;
  harness?: string;
  agent?: string;
  session?: string;
};

export type DerivedCacheIdentity = {
  generation: string;
  tenantRoot: string;
  fingerprints: {
    project?: string;
    harness?: string;
    agent?: string;
    session?: string;
  };
  scope: {
    kind: CacheIdentityScope;
    fingerprint: string;
  };
  priority: CacheIdentityPriority;
  display: CacheIdentityDisplay;
  physical: {
    fingerprint: string;
    namespace: string;
    namespaceId: bigint;
  };
};

export function deriveCacheIdentity(
  secret: InstallationSecret,
  principal: CachePrincipal,
  intent: CacheIdentityIntent,
  physical: PhysicalCacheDomain,
): DerivedCacheIdentity {
  assertSecret(secret);
  const display = sanitizeIntent(intent);
  const [principalKind, principalId] = principalIdentity(principal);
  const tenantDigest = hmacSha256(secret.key, `${DOMAIN_PREFIX}/tenant`, [principalKind, principalId]);

  const projectDigest = deriveOptionalScope(secret.key, tenantDigest, "project", display.project);
  const harnessDigest = deriveOptionalScope(secret.key, tenantDigest, "harness", display.harness);
  const agentDigest = deriveOptionalScope(secret.key, tenantDigest, "agent", display.agent);
  const sessionDigest = deriveOptionalScope(secret.key, tenantDigest, "session", display.session);
  const selected = selectScope(tenantDigest, projectDigest, harnessDigest, agentDigest, sessionDigest);
  const physicalDigest = derivePhysicalDomain(secret.key, physical);
  const namespaceDigest = hmacSha256(secret.key, `${DOMAIN_PREFIX}/namespace`, [
    tenantDigest,
    selected.kind,
    selected.digest,
    display.namespace ?? "",
    physicalDigest,
  ]);

  return {
    generation: secret.generation,
    tenantRoot: digestHex(tenantDigest),
    fingerprints: {
      project: hexOptional(projectDigest),
      harness: hexOptional(harnessDigest),
      agent: hexOptional(agentDigest),
      session: hexOptional(sessionDigest),
    },
    scope: {
      kind: selected.kind,
      fingerprint: digestHex(selected.digest),
    },
    priority: intent.priority ?? "interactive",
    display,
    physical: {
      fingerprint: digestHex(physicalDigest),
      namespace: digestHex(namespaceDigest),
      namespaceId: digestToNonzeroU64(namespaceDigest),
    },
  };
}

export function sanitizeCacheIdentityLabel(label: string): string {
  const normalized = label.normalize("NFC").trim();
  if (!normalized || Buffer.byteLength(normalized, "utf8") > CACHE_IDENTITY_LABEL_MAX_BYTES) {
    throw new Error(`Cache identity labels must be 1-${CACHE_IDENTITY_LABEL_MAX_BYTES} UTF-8 bytes`);
  }
  if (/[\u0000-\u001f\u007f]/u.test(normalized)) throw new Error("Cache identity labels cannot contain control characters");
  return normalized;
}

function sanitizeIntent(intent: CacheIdentityIntent): CacheIdentityDisplay {
  return {
    namespace: sanitizeOptional(intent.namespace),
    project: sanitizeOptional(intent.project),
    harness: sanitizeOptional(intent.harness),
    agent: sanitizeOptional(intent.agent),
    session: sanitizeOptional(intent.session),
  };
}

function sanitizeOptional(value: string | undefined): string | undefined {
  return value === undefined ? undefined : sanitizeCacheIdentityLabel(value);
}

function deriveOptionalScope(
  key: Uint8Array,
  tenantDigest: Uint8Array,
  scope: Exclude<CacheIdentityScope, "tenant">,
  label: string | undefined,
): Uint8Array | undefined {
  return label === undefined ? undefined : hmacSha256(key, `${DOMAIN_PREFIX}/scope/${scope}`, [tenantDigest, label]);
}

function derivePhysicalDomain(key: Uint8Array, physical: PhysicalCacheDomain): Uint8Array {
  const backend = sanitizePhysicalField("backend", physical.backend);
  const modelRevision = sanitizePhysicalField("model revision", physical.modelRevision);
  const tokenizer = sanitizePhysicalField("tokenizer", physical.tokenizer);
  const kvFormat = sanitizePhysicalField("KV format", physical.kvFormat);
  if (!Number.isSafeInteger(physical.contextAllocation) || physical.contextAllocation <= 0) {
    throw new Error("Physical cache context allocation must be a positive safe integer");
  }
  if (!Number.isSafeInteger(physical.layoutVersion) || physical.layoutVersion <= 0) {
    throw new Error("Physical cache layout version must be a positive safe integer");
  }
  return hmacSha256(key, `${DOMAIN_PREFIX}/physical`, [
    backend,
    modelRevision,
    tokenizer,
    String(physical.contextAllocation),
    kvFormat,
    physical.unifiedKv === true ? "unified" : "split",
    String(physical.layoutVersion),
  ]);
}

function sanitizePhysicalField(name: string, value: string): string {
  const normalized = value.normalize("NFC").trim();
  if (!normalized || /[\u0000-\u001f\u007f]/u.test(normalized) || Buffer.byteLength(normalized, "utf8") > 1024) {
    throw new Error(`Physical cache ${name} is invalid`);
  }
  return normalized;
}

function selectScope(
  tenant: Uint8Array,
  project?: Uint8Array,
  harness?: Uint8Array,
  agent?: Uint8Array,
  session?: Uint8Array,
): { kind: CacheIdentityScope; digest: Uint8Array } {
  if (session) return { kind: "session", digest: session };
  if (agent) return { kind: "agent", digest: agent };
  if (harness) return { kind: "harness", digest: harness };
  if (project) return { kind: "project", digest: project };
  return { kind: "tenant", digest: tenant };
}

function hexOptional(value: Uint8Array | undefined): string | undefined {
  return value === undefined ? undefined : digestHex(value);
}

function assertSecret(secret: InstallationSecret): void {
  if (!/^sec_[0-9a-f-]{36}$/.test(secret.generation) || secret.key.byteLength !== 32) {
    throw new Error("Cache identity installation secret is invalid");
  }
}
