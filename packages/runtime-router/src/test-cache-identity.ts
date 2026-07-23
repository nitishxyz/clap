import type { ResidentChatOptions } from "./resident";

export const testResidentChatOptions: ResidentChatOptions = {
  cacheIdentity: {
    version: 1,
    generation: "sec_test",
    tenant_root: "a".repeat(64),
    scope: "tenant",
    scope_fingerprint: "a".repeat(64),
    namespace_fingerprint: "b".repeat(64),
    namespace_id: "1",
    priority: "normal",
    side_request: false,
    display: {},
    physical: {
      fingerprint: "c".repeat(64),
      backend: "llama",
      resolved_revision: "local:test",
      model_artifact_fingerprint: "d".repeat(64),
      tokenizer_fingerprint: "d".repeat(64),
      context_allocation: 4096,
      kv_format: "f16",
      unified_kv: true,
      layout_version: 1,
    },
  },
};
