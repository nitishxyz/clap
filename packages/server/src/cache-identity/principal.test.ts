import { describe, expect, test } from "bun:test";
import {
  apiKeyPrincipal,
  principalIdentity,
  trustedLocalPrincipal,
} from "./principal";

describe("cache identity principals", () => {
  test("uses a stable API key record ID rather than credential material", () => {
    const principal = apiKeyPrincipal("  key-record-42  ");
    expect(principal).toEqual({ kind: "api_key", recordId: "key-record-42" });
    expect(principalIdentity(principal)).toEqual(["api_key", "key-record-42"]);
  });

  test("uses one fixed identity for trusted local callers", () => {
    expect(principalIdentity(trustedLocalPrincipal())).toEqual(["trusted_local", "trusted_local"]);
    expect(trustedLocalPrincipal()).toEqual({ kind: "trusted_local" });
  });

  test.each(["", "   ", "record\n2", "x".repeat(257)])("rejects an invalid API key record ID", (recordId) => {
    expect(() => apiKeyPrincipal(recordId)).toThrow("record ID is invalid");
  });
});
