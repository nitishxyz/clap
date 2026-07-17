import { describe, expect, test } from "bun:test";
import { MetricsCollector } from "./metrics";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

describe("metrics queue accounting", () => {
  test("time waiting in queue is tracked separately from ttft", async () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", false);
    handle.phase("queued");
    await sleep(120);
    // Worker picks the request up and starts prefill.
    handle.prefill(10, 100);
    await sleep(60);
    handle.firstToken();
    await sleep(30);
    handle.finish({ status: "ok", promptTokens: 100, completionTokens: 20 });

    const record = handle.record;
    expect(record.queuedMs).toBeGreaterThanOrEqual(100);
    expect(record.queuedMs!).toBeLessThan(400);
    // TTFT is measured from dispatch, not from arrival: it must not include
    // the 120ms spent queued.
    expect(record.ttftMs).toBeGreaterThanOrEqual(40);
    expect(record.ttftMs!).toBeLessThan(115);
    expect(record.durationMs!).toBeGreaterThanOrEqual(200);
  });

  test("requests cancelled while queued record queue time", async () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", true);
    handle.phase("queued");
    await sleep(80);
    handle.finish({ status: "cancelled", finishReason: "cancel" });
    expect(handle.record.queuedMs).toBeGreaterThanOrEqual(60);
    expect(handle.record.ttftMs).toBeUndefined();
  });

  test("fast unqueued requests do not report noise queue time", () => {
    const metrics = new MetricsCollector();
    const handle = metrics.start("test-model", "/v1/chat/completions", false);
    handle.prefill(1, 10);
    handle.firstToken();
    handle.finish({ status: "ok", promptTokens: 10, completionTokens: 5 });
    expect(handle.record.queuedMs).toBeUndefined();
  });
});
