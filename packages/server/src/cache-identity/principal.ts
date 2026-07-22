export const TRUSTED_LOCAL_PRINCIPAL_ID = "trusted_local" as const;
const API_KEY_RECORD_ID_MAX_BYTES = 256;

export type ApiKeyPrincipal = {
  kind: "api_key";
  /** Stable database record ID; never the API key or caller-provided tenant. */
  recordId: string;
};

export type TrustedLocalPrincipal = {
  kind: "trusted_local";
};

export type CachePrincipal = ApiKeyPrincipal | TrustedLocalPrincipal;

export function apiKeyPrincipal(recordId: string): ApiKeyPrincipal {
  const normalized = recordId.trim();
  if (!normalized || Buffer.byteLength(normalized, "utf8") > API_KEY_RECORD_ID_MAX_BYTES || hasControlCharacter(normalized)) {
    throw new Error("API key principal record ID is invalid");
  }
  return { kind: "api_key", recordId: normalized };
}

export function trustedLocalPrincipal(): TrustedLocalPrincipal {
  return { kind: "trusted_local" };
}

export function principalIdentity(principal: CachePrincipal): readonly [string, string] {
  if (principal.kind === "trusted_local") return [principal.kind, TRUSTED_LOCAL_PRINCIPAL_ID];
  const validated = apiKeyPrincipal(principal.recordId);
  return [validated.kind, validated.recordId];
}

function hasControlCharacter(value: string): boolean {
  return /[\u0000-\u001f\u007f]/u.test(value);
}
