import { describe, expect, test } from "bun:test";
import { mapWorkerResultPayload, parseWorkerMemory } from "./result-mapper";

describe("worker result mapper", () => {
  test("maps terminal v1 results without duplicating streamed content", () => {
    expect(mapWorkerResultPayload({ content: "fallback", finish_reason: "stop" }, "req_1", false)).toEqual({
      content: "fallback",
      finish_reason: "stop",
      id: "req_1",
      done: true,
    });
    expect(mapWorkerResultPayload({ content: "duplicate", finish_reason: "stop" }, "req_1", true)).toEqual({
      finish_reason: "stop",
      id: "req_1",
      done: true,
    });
  });

  test("maps memory source and basis companions without guessing legacy provenance", () => {
    expect(parseWorkerMemory({ active_bytes: 1024, cache_bytes: 512, peak_active_bytes: 2048 })).toBeUndefined();
    expect(parseWorkerMemory({
      active_bytes: 1024, active_bytes_source: "measured", active_bytes_basis: "runtime_allocator",
      cache_bytes: 512, cache_bytes_source: "estimated", cache_bytes_basis: "configured_cache",
      peak_active_bytes: 2048, peak_active_bytes_source: "measured", peak_active_bytes_basis: "runtime_allocator",
    })).toEqual({
      activeBytes: 1024, activeBytesSource: "measured", activeBytesBasis: "runtime_allocator",
      cacheBytes: 512, cacheBytesSource: "estimated", cacheBytesBasis: "configured_cache",
      peakActiveBytes: 2048, peakActiveBytesSource: "measured", peakActiveBytesBasis: "runtime_allocator",
    });
    expect(parseWorkerMemory({
      active_bytes: 0, active_bytes_source: "measured", active_bytes_basis: "runtime_allocator",
      cache_bytes: 512, peak_active_bytes: 2048,
    })).toBeUndefined();
    expect(parseWorkerMemory({
      active_bytes: null, active_bytes_source: "unavailable", active_bytes_basis: "not_reported",
      cache_bytes: 512, cache_bytes_source: "estimated", cache_bytes_basis: "configured_cache",
      peak_active_bytes: 2048, peak_active_bytes_source: "measured", peak_active_bytes_basis: "runtime_allocator",
    })).toMatchObject({ activeBytes: null, activeBytesSource: "unavailable", activeBytesBasis: "not_reported" });
    expect(parseWorkerMemory({
      active_bytes: 1024, active_bytes_source: "measured", active_bytes_basis: "worker_allocator",
      cache_bytes: 512, cache_bytes_source: "measured", cache_bytes_basis: "worker_allocator",
      peak_active_bytes: 2048, peak_active_bytes_source: "measured",
      peak_active_bytes_basis: "worker_allocator",
    })).toMatchObject({
      activeBytesSource: "measured", activeBytesBasis: "worker_allocator",
      cacheBytesSource: "measured", cacheBytesBasis: "worker_allocator",
    });
  });
});
