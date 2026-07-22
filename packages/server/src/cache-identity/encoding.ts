import { createHmac } from "node:crypto";

const textEncoder = new TextEncoder();

export type DomainField = string | Uint8Array;

/** Encodes a domain and fields without delimiter ambiguity. */
export function encodeDomain(domain: string, fields: readonly DomainField[]): Uint8Array {
  return encodeLengthPrefixed([domain, ...fields]);
}

export function encodeLengthPrefixed(fields: readonly DomainField[]): Uint8Array {
  const encoded = fields.map(toBytes);
  const totalLength = encoded.reduce((total, field) => total + 4 + field.byteLength, 0);
  const output = new Uint8Array(totalLength);
  const view = new DataView(output.buffer);
  let offset = 0;
  for (const field of encoded) {
    if (field.byteLength > 0xffff_ffff) throw new Error("Cache identity field is too large");
    view.setUint32(offset, field.byteLength, false);
    offset += 4;
    output.set(field, offset);
    offset += field.byteLength;
  }
  return output;
}

export function hmacSha256(key: Uint8Array, domain: string, fields: readonly DomainField[]): Uint8Array {
  return createHmac("sha256", key).update(encodeDomain(domain, fields)).digest();
}

export function digestHex(digest: Uint8Array): string {
  return Buffer.from(digest).toString("hex");
}

/** Reduces the first 64 digest bits in network order; zero is reserved. */
export function digestToNonzeroU64(digest: Uint8Array): bigint {
  if (digest.byteLength < 8) throw new Error("Cache identity digest must contain at least 8 bytes");
  const value = Buffer.from(digest.buffer, digest.byteOffset, digest.byteLength).readBigUInt64BE(0);
  return value === 0n ? 1n : value;
}

function toBytes(value: DomainField): Uint8Array {
  return typeof value === "string" ? textEncoder.encode(value) : value;
}
