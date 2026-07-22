import { clapHome } from "@clap/models";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { apiKeyPrincipal, trustedLocalPrincipal, type CachePrincipal } from "./cache-identity";

export type ApiKeyRecord = {
  id: string;
  name: string;
  sha256: string;
  createdAt: string;
  lastUsedAt?: string;
  revoked?: boolean;
};

export type ApiKeyPublic = Omit<ApiKeyRecord, "sha256">;

const KEY_PREFIX = "clap_sk_";

export function keysFilePath(): string {
  return join(clapHome(), "keys.json");
}

function readRecords(path = keysFilePath()): ApiKeyRecord[] {
  if (!existsSync(path)) return [];
  try {
    const parsed = JSON.parse(readFileSync(path, "utf8"));
    return Array.isArray(parsed) ? (parsed as ApiKeyRecord[]) : [];
  } catch {
    return [];
  }
}

function writeRecords(records: ApiKeyRecord[], path = keysFilePath()): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${JSON.stringify(records, null, 2)}\n`, { mode: 0o600 });
}

function hashKey(key: string): string {
  return new Bun.CryptoHasher("sha256").update(key).digest("hex");
}

export function createApiKey(name: string): { record: ApiKeyPublic; key: string } {
  const secret = `${KEY_PREFIX}${crypto.randomUUID().replaceAll("-", "")}`;
  const record: ApiKeyRecord = {
    id: `key_${crypto.randomUUID().slice(0, 8)}`,
    name,
    sha256: hashKey(secret),
    createdAt: new Date().toISOString(),
  };
  const records = readRecords();
  records.push(record);
  writeRecords(records);
  const { sha256: _sha256, ...publicRecord } = record;
  return { record: publicRecord, key: secret };
}

export function listApiKeys(): ApiKeyPublic[] {
  return readRecords().map(({ sha256: _sha256, ...rest }) => rest);
}

export function revokeApiKey(id: string): boolean {
  const records = readRecords();
  const record = records.find((entry) => entry.id === id && !entry.revoked);
  if (!record) return false;
  record.revoked = true;
  writeRecords(records);
  return true;
}

// Verifier with mtime-based reload so the server observes CLI key changes
// without a restart, while avoiding a full file read per request (the stat
// itself is cheap enough for the auth path).
export class ApiKeyVerifier {
  private records: ApiKeyRecord[] = [];
  private mtimeMs = -1;

  constructor(private readonly path = keysFilePath()) {}

  private refresh(): void {
    let mtimeMs = 0;
    try {
      mtimeMs = statSync(this.path).mtimeMs;
    } catch {
      mtimeMs = 0;
    }
    if (mtimeMs === this.mtimeMs) return;
    this.mtimeMs = mtimeMs;
    this.records = readRecords(this.path);
  }

  hasActiveKeys(): boolean {
    this.refresh();
    return this.records.some((record) => !record.revoked);
  }

  verify(token: string): ApiKeyRecord | undefined {
    if (!token.startsWith(KEY_PREFIX)) return undefined;
    this.refresh();
    const digest = hashKey(token);
    const record = this.records.find((entry) => !entry.revoked && entry.sha256 === digest);
    if (record) {
      // Best-effort usage stamp, throttled to avoid a disk write per request.
      const now = Date.now();
      const last = record.lastUsedAt ? Date.parse(record.lastUsedAt) : 0;
      if (now - last > 60_000) {
        record.lastUsedAt = new Date(now).toISOString();
        try {
          writeRecords(this.records, this.path);
          this.mtimeMs = statSync(this.path).mtimeMs;
        } catch {
          // read-only key file is fine
        }
      }
    }
    return record;
  }
}

export function bearerToken(header: string | undefined): string | undefined {
  if (!header) return undefined;
  const match = /^Bearer\s+(.+)$/i.exec(header.trim());
  return match?.[1];
}

const LOOPBACK = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);

export function isLoopbackAddress(address: string | undefined): boolean {
  if (!address) return false;
  return LOOPBACK.has(address);
}

export type RequestIdentity = {
  clientId: string;
  cachePrincipal?: CachePrincipal;
  loopback: boolean;
  credentialPresented: boolean;
  credentialValid: boolean;
};

export type RequestIdentityInput = {
  authorization?: string;
  apiKey?: string;
  address?: string;
  /** Missing transport metadata denotes trusted embedded use. */
  embedded?: boolean;
};

/** Resolves credentials exactly once. A presented credential never falls back to local trust. */
export function resolveRequestIdentity(
  verifier: Pick<ApiKeyVerifier, "verify">,
  input: RequestIdentityInput,
): RequestIdentity {
  const token = bearerToken(input.authorization) ?? input.apiKey;
  const credentialPresented = input.authorization !== undefined || input.apiKey !== undefined;
  const loopback = input.embedded === true || isLoopbackAddress(input.address);
  if (credentialPresented) {
    const record = token === undefined ? undefined : verifier.verify(token);
    if (record) {
      return {
        clientId: record.id,
        cachePrincipal: apiKeyPrincipal(record.id),
        loopback,
        credentialPresented: true,
        credentialValid: true,
      };
    }
    return {
      clientId: loopback ? "local-invalid-credential" : `remote:${input.address ?? "unknown"}`,
      loopback,
      credentialPresented: true,
      credentialValid: false,
    };
  }
  if (loopback) {
    return {
      clientId: "local",
      cachePrincipal: trustedLocalPrincipal(),
      loopback: true,
      credentialPresented: false,
      credentialValid: false,
    };
  }
  return {
    clientId: `remote:${input.address ?? "unknown"}`,
    loopback: false,
    credentialPresented: false,
    credentialValid: false,
  };
}
