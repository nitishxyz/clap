import { describe, expect, test } from "bun:test";
import { FairLimiter, QueueFullError } from "./limits";

describe("fair limiter", () => {
  test("admits up to maxInflight immediately and queues beyond", async () => {
    const limiter = new FairLimiter(2, 4);
    const r1 = await limiter.acquire("a");
    const r2 = await limiter.acquire("a");
    expect(limiter.stats()).toMatchObject({ inflight: 2, queued: 0 });

    let admitted = false;
    const third = limiter.acquire("a").then((release) => {
      admitted = true;
      return release;
    });
    await Bun.sleep(5);
    expect(admitted).toBe(false);
    expect(limiter.stats().queued).toBe(1);

    r1();
    const r3 = await third;
    expect(admitted).toBe(true);
    r2();
    r3();
    expect(limiter.stats()).toMatchObject({ inflight: 0, queued: 0 });
  });

  test("rejects with QueueFullError and retry hint when the queue is full", async () => {
    const limiter = new FairLimiter(1, 1);
    const release = await limiter.acquire("a");
    const queued = limiter.acquire("b");
    expect(limiter.acquire("c")).rejects.toBeInstanceOf(QueueFullError);
    try {
      await limiter.acquire("c");
    } catch (error) {
      expect((error as QueueFullError).retryAfterSeconds).toBeGreaterThan(0);
    }
    release();
    (await queued)();
  });

  test("fair pick: a client with fewer inflight wins over a chatty client's FIFO position", async () => {
    const limiter = new FairLimiter(2, 8);
    const a1 = await limiter.acquire("chatty");
    const a2 = await limiter.acquire("chatty");

    const order: string[] = [];
    const chattyNext = limiter.acquire("chatty").then((release) => {
      order.push("chatty");
      return release;
    });
    const quietNext = limiter.acquire("quiet").then((release) => {
      order.push("quiet");
      return release;
    });
    await Bun.sleep(5);
    expect(order).toEqual([]);

    a1();
    await Bun.sleep(5);
    // quiet has 0 inflight vs chatty's 1: quiet is admitted despite queueing later
    expect(order).toEqual(["quiet"]);
    a2();
    await Bun.sleep(5);
    expect(order).toEqual(["quiet", "chatty"]);
    (await chattyNext)();
    (await quietNext)();
  });

  test("abort while queued removes the waiter", async () => {
    const limiter = new FairLimiter(1, 4);
    const release = await limiter.acquire("a");
    const controller = new AbortController();
    const waiting = limiter.acquire("b", controller.signal);
    controller.abort();
    expect(waiting).rejects.toThrow("aborted while queued");
    await Bun.sleep(5);
    expect(limiter.stats().queued).toBe(0);
    release();
    expect(limiter.stats().inflight).toBe(0);
  });

  test("double release is a no-op", async () => {
    const limiter = new FairLimiter(1, 1);
    const release = await limiter.acquire("a");
    release();
    release();
    expect(limiter.stats().inflight).toBe(0);
  });
});
