import { describe, expect, test } from "bun:test";
import { FairLimiter, QueueFullError, type RequestPriority } from "./limits";

const normal = "normal" as const;

async function settle() { await Bun.sleep(2); }

describe("fair limiter", () => {
  test("admits immediately, tracks priority counts, and queues beyond capacity", async () => {
    const limiter = new FairLimiter(2, 4);
    const r1 = await limiter.acquire("a", "interactive");
    const r2 = await limiter.acquire("a", "background");
    expect(limiter.stats()).toMatchObject({ inflight: 2, queued: 0,
      inflightByPriority: { interactive: 1, normal: 0, background: 1 } });
    const third = limiter.acquire("a", normal);
    await settle();
    expect(limiter.stats()).toMatchObject({ queued: 1,
      waitingByPriority: { interactive: 0, normal: 1, background: 0 } });
    r1();
    const r3 = await third;
    r2(); r3();
    expect(limiter.stats()).toMatchObject({ inflight: 0, queued: 0,
      inflightByPriority: { interactive: 0, normal: 0, background: 0 } });
  });

  test("interactive bypasses an existing normal and background backlog", async () => {
    const limiter = new FairLimiter(1, 8);
    const blocker = await limiter.acquire("running", normal);
    const order: string[] = [];
    const background = limiter.acquire("b", "background").then((release) => { order.push("background"); return release; });
    const queuedNormal = limiter.acquire("n", normal).then((release) => { order.push("normal"); return release; });
    const interactive = limiter.acquire("i", "interactive").then((release) => { order.push("interactive"); return release; });
    blocker(); await settle();
    expect(order).toEqual(["interactive"]);
    (await interactive)(); await settle();
    expect(order).toEqual(["interactive", "normal"]);
    (await queuedNormal)();
    (await background)();
  });

  test("weighted round robin services continuous classes within a 4:2:1 bound", async () => {
    const limiter = new FairLimiter(1, 64);
    const blocker = await limiter.acquire("running", normal);
    const order: RequestPriority[] = [];
    const releases: Array<() => void> = [];
    const pending: Array<Promise<void>> = [];
    for (const priority of ["interactive", "normal", "background"] as const) {
      for (let index = 0; index < 14; index += 1) {
        pending.push(limiter.acquire(`${priority}-${index}`, priority).then((release) => {
          order.push(priority); releases.push(release);
        }));
      }
    }
    blocker();
    for (let index = 0; index < 14; index += 1) {
      await settle();
      expect(order.length).toBe(index + 1);
      releases.shift()?.();
    }
    expect(order.slice(0, 7)).toEqual([
      "interactive", "interactive", "interactive", "interactive", "normal", "normal", "background",
    ]);
    expect(order.slice(7, 14)).toEqual(order.slice(0, 7));
    while (order.length < pending.length) { await settle(); releases.shift()?.(); }
    releases.shift()?.();
    await Promise.all(pending);
  });

  test("weighted order also prevents a background flood from delaying higher classes", async () => {
    const limiter = new FairLimiter(1, 16);
    const blocker = await limiter.acquire("running", normal);
    const order: string[] = [];
    const releases: Array<() => void> = [];
    const queue = (name: string, priority: RequestPriority) => limiter.acquire(name, priority).then((release) => {
      order.push(name); releases.push(release);
    });
    const pending = [queue("b1", "background"), queue("b2", "background"), queue("b3", "background"),
      queue("normal", normal), queue("interactive", "interactive")];
    blocker();
    while (order.length < pending.length) {
      await settle();
      releases.shift()?.();
    }
    await Promise.all(pending);
    expect(order.slice(0, 2)).toEqual(["interactive", "normal"]);
    releases.shift()?.();
  });

  test("least-inflight client fairness and FIFO ties are preserved inside a class", async () => {
    const limiter = new FairLimiter(2, 8);
    const a1 = await limiter.acquire("chatty", normal);
    const a2 = await limiter.acquire("chatty", normal);
    const order: string[] = [];
    const first = limiter.acquire("chatty", normal).then((release) => { order.push("chatty-1"); return release; });
    const second = limiter.acquire("chatty", normal).then((release) => { order.push("chatty-2"); return release; });
    const quiet = limiter.acquire("quiet", normal).then((release) => { order.push("quiet"); return release; });
    a1(); await settle();
    expect(order).toEqual(["quiet"]);
    a2(); await settle();
    expect(order).toEqual(["quiet", "chatty-1"]);
    (await quiet)(); await settle();
    expect(order).toEqual(["quiet", "chatty-1", "chatty-2"]);
    (await first)(); (await second)();
  });

  test("queue full and abort accounting stay exact by priority", async () => {
    const limiter = new FairLimiter(1, 2);
    const release = await limiter.acquire("a", "interactive");
    const controller = new AbortController();
    const aborted = limiter.acquire("b", "background", controller.signal);
    const queued = limiter.acquire("c", normal);
    expect(limiter.acquire("d", "interactive")).rejects.toBeInstanceOf(QueueFullError);
    expect(limiter.stats()).toMatchObject({ queued: 2,
      waitingByPriority: { interactive: 0, normal: 1, background: 1 } });
    controller.abort();
    expect(aborted).rejects.toThrow("aborted while queued");
    await settle();
    expect(limiter.stats()).toMatchObject({ queued: 1,
      waitingByPriority: { interactive: 0, normal: 1, background: 0 } });
    release();
    (await queued)();
    expect(limiter.stats()).toMatchObject({ inflight: 0, queued: 0 });
  });

  test("double release is a no-op", async () => {
    const limiter = new FairLimiter(1, 1);
    const release = await limiter.acquire("a", normal);
    release(); release();
    expect(limiter.stats().inflight).toBe(0);
  });
});
