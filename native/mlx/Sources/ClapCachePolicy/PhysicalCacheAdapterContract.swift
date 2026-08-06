import Foundation

public let physicalCacheAdapterContractVersion = 1

public func mlxPhysicalCacheAdapterDescriptor(kvBits: Int?) -> [String: Any] {
  [
    "contract_version": physicalCacheAdapterContractVersion,
    "kind": "sequence",
    "operations": ["inspect", "continue", "restore", "fork", "snapshot", "release"],
    "format": [
      "backend": "mlx",
      "engine": "mlx-lm",
      "cache_format": "mlx-cache-array",
      "cache_format_version": 1,
      "kv_data_type": kvBits.map { "q\($0)" } ?? "f16",
      "block_tokens": NSNull(),
    ],
    "constraints": [
      "restore_granularity": "whole_state",
      "fork_semantics": "whole_state_copy",
      "minimum_trim_tokens": NSNull(),
      "safe_busy_donor": false,
      "prompt_boundary_snapshots": true,
      "recurrent_or_hybrid": false,
      "byte_accounting": "estimated",
      "tiers": ["device"],
      "transfer_format": NSNull(),
    ],
  ]
}
