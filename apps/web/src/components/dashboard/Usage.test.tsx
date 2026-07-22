import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { LoadedModels } from "./ModelTables";
import { formatWorkerCpu, UsagePanel, WorkerMetricCell } from "./Usage";

const worker = {
  key: "mlx-key",
  id: "mlx/model",
  backend: "mlx",
  format: "mlx",
  localPath: "/models/mlx",
  state: "warm",
  activeRequests: 0,
  keepAlive: "15m",
  expiresAt: "2026-07-20T12:00:00.000Z",
  loadedAt: "2026-07-20T10:00:00.000Z",
  lastUsedAt: "2026-07-20T11:00:00.000Z",
  pinned: false,
  worker: {
    pid: 42,
    state: "resident",
    crashes: 0,
    memory: { activeBytes: 4 * 2 ** 30, cacheBytes: 0, peakActiveBytes: 5 * 2 ** 30 },
    retention: {
      maxActive: 4, queued: 2, previousMaxActive: 2,
      lastAdjustmentReason: "global_headroom_available", lastAdjustmentAt: "2026-07-20T11:29:00.000Z",
      retainedGrowthReserveBytes: 512 * 2 ** 20, globalResidentMemoryBytes: 15 * 2 ** 30,
      pressureState: "normal",
      activePolicy: { mode: "auto", selectedMax: 4, backendCeiling: 16,
        hardwareCeiling: 8, modelCeiling: 16, memoryCeiling: 4, reason: "memory_ceiling",
        inputs: { model_active_bytes: 4 * 2 ** 30 } },
      active: 1, retainedTotal: 103, retainedSessions: 100, retainedAnchors: 3,
      retainedBytes: 10 * 2 ** 30, sessionBytes: 8 * 2 ** 30, anchorBytes: 2 * 2 ** 30,
      automaticCheckpointCount: 8, automaticCheckpointBytes: 512 * 2 ** 20,
      automaticCheckpointBudgetBytes: 2 * 2 ** 30, automaticCheckpointsEnabled: true,
      automaticCheckpointMinimumTokens: 2048, automaticCheckpointIntervalTokens: 2048,
      automaticCheckpointMax: 8,
      budgetBytes: 96 * 2 ** 30, highWatermarkBytes: 86.4 * 2 ** 30, lowWatermarkBytes: 72 * 2 ** 30,
      underPressure: false, hardCeiling: 256, evictionReason: "byte_pressure", evictionCount: 2,
    },
  },
  usage: { rssBytes: 5 * 2 ** 30, cpuPercent: 0.1 },
};

const data = {
  server: { pid: 1, platform: "darwin", cpuCount: 12, systemMemoryBytes: 32 * 2 ** 30 },
  gpus: [],
  queue: { inflight: 0, queued: 0, maxInflight: 4, queueDepth: 16 },
  loaded: [worker],
};

describe("worker resource presentation", () => {
  test("does not round a measured sub-percent CPU sample to zero", () => {
    expect(formatWorkerCpu(0)).toBe("0%");
    expect(formatWorkerCpu(0.1)).toBe("<1%");
    expect(formatWorkerCpu(1.4)).toBe("1%");
  });

  test("groups each worker label, value, and meter in one metric cell", () => {
    const html = renderToStaticMarkup(<UsagePanel data={data as never} />);
    expect(html).toContain('data-worker-metric="cpu"');
    expect(html).toContain("CPU</span><span");
    expect(html).toContain("&lt;1%");
    expect(html).toContain('data-worker-metric="rss"');
    expect(html).toContain("RSS</span><span");
    expect(html).toContain('data-worker-metric="accelerator"');
    expect(html).toContain('data-worker-metric="reclaimable"');
  });

  test("renders unavailable explicitly", () => {
    const html = renderToStaticMarkup(<WorkerMetricCell label="Accelerator" />);
    expect(html).toContain("Accelerator");
    expect(html).toContain("unavailable");
  });

  test("labels estimated load memory separately from measured RSS", () => {
    const html = renderToStaticMarkup(<LoadedModels models={[{
      ...worker,
      worker: {
        ...worker.worker,
        loadState: "resident",
        residency: {
          estimateBytes: 2_000,
          estimateSource: "model_artifacts",
          observedRssBytes: 1_500,
          observedRssSource: "resident_rss",
          reservationBytes: 2_000,
          lastAdmissionReason: "within_budget_after_eviction",
          lastEvictionReason: "memory_admission",
        },
      },
    } as never]} now={0} actions={{ busy: {}, run: async () => undefined } as never}
      platform="darwin" systemMemoryBytes={10_000} cpuCount={8} />);
    expect(html).toContain("load estimate");
    expect(html).toContain("estimated · model artifacts");
    expect(html).toContain("observed rss");
    expect(html).toContain("measured · RSS");
    expect(html).toContain("within budget after eviction");
    expect(html).toContain("memory admission");
  });

  test("keeps common loaded-model details ordered before backend memory", () => {
    const html = renderToStaticMarkup(
      <LoadedModels
        models={[worker] as never}
        now={new Date("2026-07-20T11:30:00.000Z").getTime()}
        actions={{ busy: {}, run: async () => undefined } as never}
        platform="darwin"
        systemMemoryBytes={32 * 2 ** 30}
        cpuCount={12}
      />,
    );
    const primary = html.indexOf('data-model-details="primary"');
    const capacity = html.indexOf('data-model-details="concurrency"');
    const backend = html.indexOf('data-model-details="backend"');
    expect(primary).toBeGreaterThan(-1);
    expect(capacity).toBeGreaterThan(primary);
    expect(backend).toBeGreaterThan(capacity);
    for (const label of ["memory", "cpu", "model active / max", "model queued", "keep-alive", "expires", "last used", "pid", "backend"]) {
      const position = html.indexOf(`>${label}</span>`, primary);
      expect(position).toBeGreaterThan(primary);
      expect(position).toBeLessThan(backend);
    }
    for (const label of ["worker active / max", "worker queued", "mode", "selected limit", "previous limit", "backend ceiling", "hardware ceiling", "model ceiling", "memory ceiling", "limiting reason", "last adjustment", "adjusted", "retained growth reserve", "current retained", "global resident memory"]) {
      const position = html.indexOf(`>${label}</span>`, capacity);
      expect(position).toBeGreaterThan(capacity);
      expect(position).toBeLessThan(backend);
    }
    expect(html).toContain('data-model-capacity="summary"');
    expect(html).toContain("1/4 active · 2 queued");
    expect(html.indexOf(">active</span>", backend)).toBeGreaterThan(backend);
    expect(html.indexOf(">cache</span>", backend)).toBeGreaterThan(backend);
    expect(html.indexOf(">peak</span>", backend)).toBeGreaterThan(backend);
    const retained = html.indexOf('data-model-details="retained-cache"');
    expect(retained).toBeGreaterThan(backend);
    expect(html).toContain("Retained Cache");
    for (const label of ["startup available", "active reserve", "retained total", "sessions", "anchors", "retained bytes", "automatic checkpoints", "automatic checkpoint bytes", "checkpoint policy", "checkpoint budget", "budget", "high watermark", "low watermark", "hard ceiling", "evictions", "last reason"]) {
      expect(html.indexOf(`>${label}</span>`, retained)).toBeGreaterThan(retained);
    }
    expect(html).toContain("memory ceiling");
  });

  test("renders responsive capacity details and explicit unavailable values", () => {
    const noTelemetry = { ...worker, worker: { ...worker.worker, retention: undefined } };
    const html = renderToStaticMarkup(
      <LoadedModels models={[noTelemetry] as never} now={0}
        actions={{ busy: {}, run: async () => undefined } as never} />,
    );
    expect(html).toContain("capacity unavailable");
    expect(html).toContain('data-model-details="concurrency"');
    expect(html).toContain("grid-cols-1");
    expect(html).toContain("sm:grid-cols-2");
    expect(html).toContain("xl:grid-cols-4");
    for (const label of ["worker active / max", "worker queued", "selected limit", "last adjustment", "retained growth reserve", "global resident memory"]) {
      const position = html.indexOf(`>${label}</span>`);
      expect(position).toBeGreaterThan(-1);
      expect(html.indexOf("unavailable", position)).toBeGreaterThan(position);
    }
  });
});
