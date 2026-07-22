import MLX

func memorySnapshot() -> WorkerMemory {
  let snapshot = Memory.snapshot()
  return WorkerMemory(active_bytes: snapshot.activeMemory, cache_bytes: snapshot.cacheMemory,
    peak_active_bytes: snapshot.peakMemory)
}
