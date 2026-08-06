import { createHash, createHmac } from "node:crypto";

export function deriveVllmCacheSalt(input: {
  secret: Uint8Array | string;
  deployment: string;
  model: string;
  shareIdentity: string;
}): string {
  const secretBytes = typeof input.secret === "string" ? Buffer.from(input.secret) : input.secret;
  if (secretBytes.byteLength < 32) throw new Error("vLLM cache salt secret must contain at least 32 bytes");
  for (const [name, value] of Object.entries({
    deployment: input.deployment,
    model: input.model,
    shareIdentity: input.shareIdentity,
  })) {
    if (!value || value.includes("\0")) throw new Error(`${name} must be non-empty and contain no NUL bytes`);
  }
  return createHmac("sha256", secretBytes)
    .update("clap-vllm-cache-salt-v1\0")
    .update(input.deployment).update("\0")
    .update(input.model).update("\0")
    .update(input.shareIdentity)
    .digest("base64url");
}

export function rendezvousScore(secret: Uint8Array | string, key: string, replicaId: string): number {
  const digest = createHmac("sha256", secret)
    .update("clap-vllm-rendezvous-v1\0")
    .update(key).update("\0").update(replicaId).digest();
  return digest.readUIntBE(0, 6) / 0xffff_ffff_ffff;
}

export function physicalFormatGeneration(value: unknown): string {
  return createHash("sha256").update(JSON.stringify(value)).digest("hex");
}
