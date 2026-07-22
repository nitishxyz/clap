import { describe, expect, test } from "bun:test";
import { resolveRequestIdentity, type ApiKeyRecord } from "./auth";

const record: ApiKeyRecord = {
  id: "key_record_42",
  name: "test",
  sha256: "unused",
  createdAt: "2026-07-23T00:00:00.000Z",
};

describe("authenticated request identity", () => {
  test("valid API key wins even on loopback and is verified once", () => {
    let calls = 0;
    const identity = resolveRequestIdentity({
      verify(token) {
        calls += 1;
        return token === "valid" ? record : undefined;
      },
    }, { authorization: "Bearer valid", address: "127.0.0.1" });

    expect(calls).toBe(1);
    expect(identity).toEqual({
      clientId: record.id,
      cachePrincipal: { kind: "api_key", recordId: record.id },
      loopback: true,
      credentialPresented: true,
      credentialValid: true,
    });
  });

  test("no credential gives loopback and embedded requests trusted-local identity", () => {
    expect(resolveRequestIdentity({ verify: () => undefined }, { address: "::1" })).toMatchObject({
      clientId: "local",
      cachePrincipal: { kind: "trusted_local" },
      loopback: true,
    });
    expect(resolveRequestIdentity({ verify: () => undefined }, { embedded: true })).toMatchObject({
      clientId: "local",
      cachePrincipal: { kind: "trusted_local" },
      loopback: true,
    });
  });

  test("invalid presented credential never falls back to trusted local", () => {
    const identity = resolveRequestIdentity({ verify: () => undefined }, {
      apiKey: "invalid",
      address: "127.0.0.1",
    });

    expect(identity.credentialPresented).toBe(true);
    expect(identity.credentialValid).toBe(false);
    expect(identity.cachePrincipal).toBeUndefined();
    expect(identity.clientId).not.toBe("local");

    const malformedAuthorization = resolveRequestIdentity({ verify: () => record }, {
      authorization: "Basic unexpected",
      address: "127.0.0.1",
    });
    expect(malformedAuthorization.credentialPresented).toBe(true);
    expect(malformedAuthorization.credentialValid).toBe(false);
    expect(malformedAuthorization.cachePrincipal).toBeUndefined();
  });

  test("remote unauthenticated identity has stable address fairness but no cache principal", () => {
    const identity = resolveRequestIdentity({ verify: () => undefined }, { address: "203.0.113.9" });
    expect(identity).toEqual({
      clientId: "remote:203.0.113.9",
      loopback: false,
      credentialPresented: false,
      credentialValid: false,
    });
  });
});
