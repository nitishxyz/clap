import ClapCachePolicy
import Foundation
import Testing

@Suite("MLX physical cache adapter contract")
struct PhysicalCacheAdapterContractTests {
  @Test("advertises whole-state sequence semantics and explicit estimated bytes")
  func descriptor() throws {
    let descriptor = mlxPhysicalCacheAdapterDescriptor(kvBits: nil)
    #expect(descriptor["contract_version"] as? Int == physicalCacheAdapterContractVersion)
    #expect(descriptor["kind"] as? String == "sequence")
    #expect(descriptor["operations"] as? [String] == [
      "inspect", "continue", "restore", "fork", "snapshot", "release",
    ])
    let format = try #require(descriptor["format"] as? [String: Any])
    #expect(format["cache_format"] as? String == "mlx-cache-array")
    #expect(format["kv_data_type"] as? String == "f16")
    #expect(format["block_tokens"] is NSNull)
    let constraints = try #require(descriptor["constraints"] as? [String: Any])
    #expect(constraints["restore_granularity"] as? String == "whole_state")
    #expect(constraints["fork_semantics"] as? String == "whole_state_copy")
    #expect(constraints["minimum_trim_tokens"] is NSNull)
    #expect(constraints["byte_accounting"] as? String == "estimated")
    #expect(constraints["transfer_format"] is NSNull)
  }

  @Test("reports quantized cache format without claiming transfer support")
  func quantizedDescriptor() throws {
    let descriptor = mlxPhysicalCacheAdapterDescriptor(kvBits: 4)
    let format = try #require(descriptor["format"] as? [String: Any])
    #expect(format["kv_data_type"] as? String == "q4")
    let operations = try #require(descriptor["operations"] as? [String])
    #expect(!operations.contains("export"))
    #expect(!operations.contains("import"))
  }

  @Test("advertises copy-on-write only for proven ordinary attention caches")
  func sharedDescriptor() throws {
    let descriptor = mlxPhysicalCacheAdapterDescriptor(kvBits: nil,
      sharedOrdinaryAttention: true)
    let format = try #require(descriptor["format"] as? [String: Any])
    let constraints = try #require(descriptor["constraints"] as? [String: Any])
    #expect(format["cache_format"] as? String == "mlx-cow-array")
    #expect(constraints["fork_semantics"] as? String == "copy_on_write")

    let restricted = mlxPhysicalCacheAdapterDescriptor(kvBits: nil,
      sharedOrdinaryAttention: true, recurrentOrHybrid: true)
    let restrictedFormat = try #require(restricted["format"] as? [String: Any])
    let restrictedConstraints = try #require(restricted["constraints"] as? [String: Any])
    #expect(restrictedFormat["cache_format"] as? String == "mlx-cache-array")
    #expect(restrictedConstraints["fork_semantics"] as? String == "whole_state_copy")
    #expect(restrictedConstraints["recurrent_or_hybrid"] as? Bool == true)
  }
}
