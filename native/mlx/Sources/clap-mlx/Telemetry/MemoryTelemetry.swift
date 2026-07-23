import MLX

func memorySnapshot() -> WorkerMemory {
  // MLX's allocator snapshot is the authoritative worker allocator reading.
  // A zero means there is no positive observation to report, not measured use.
  let snapshot = Memory.snapshot()
  let active = allocatorMemory(snapshot.activeMemory)
  let cache = allocatorMemory(snapshot.cacheMemory)
  let peak = allocatorMemory(snapshot.peakMemory)
  return WorkerMemory(active_bytes: WorkerMemoryBytes(value: active),
    active_bytes_source: active == nil ? "unavailable" : "measured",
    active_bytes_basis: active == nil ? "not_observed" : "worker_allocator",
    cache_bytes: WorkerMemoryBytes(value: cache),
    cache_bytes_source: cache == nil ? "unavailable" : "measured",
    cache_bytes_basis: cache == nil ? "not_observed" : "worker_allocator",
    peak_active_bytes: WorkerMemoryBytes(value: peak),
    peak_active_bytes_source: peak == nil ? "unavailable" : "measured",
    peak_active_bytes_basis: peak == nil ? "not_observed" : "worker_allocator")
}

private func allocatorMemory(_ bytes: Int) -> UInt64? {
  bytes > 0 ? UInt64(bytes) : nil
}
