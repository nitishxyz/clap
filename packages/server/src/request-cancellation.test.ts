import { describe, expect, test } from "bun:test";
import { RequestCancellationRegistry } from "./request-cancellation";

describe("request cancellation registry", () => {
  test("cancels a registered request exactly once and keeps it cancelling", () => {
    const registry = new RequestCancellationRegistry();
    let aborts = 0;
    registry.register("r1", () => { aborts += 1; });

    expect(registry.cancel("r1")).toBe("cancelled");
    expect(aborts).toBe(1);
    // A second click must not re-abort, and must not look like a fresh cancel.
    expect(registry.cancel("r1")).toBe("already_cancelling");
    expect(aborts).toBe(1);
    expect(registry.isCancelling("r1")).toBe(true);
  });

  test("reports unknown ids instead of pretending to cancel", () => {
    const registry = new RequestCancellationRegistry();
    expect(registry.cancel("missing")).toBe("not_found");
  });

  test("cancelling does not unregister: the request owns its own lifecycle", () => {
    const registry = new RequestCancellationRegistry();
    const unregister = registry.register("r1", () => {});
    registry.cancel("r1");
    // Still tracked while the worker winds down, so status stays observable.
    expect(registry.size).toBe(1);
    unregister();
    expect(registry.size).toBe(0);
    expect(registry.cancel("r1")).toBe("not_found");
  });

  test("unregister is idempotent and does not drop a later same-id request", () => {
    const registry = new RequestCancellationRegistry();
    const unregisterFirst = registry.register("r1", () => {});
    unregisterFirst();
    unregisterFirst();

    let secondAborts = 0;
    registry.register("r1", () => { secondAborts += 1; });
    expect(registry.cancel("r1")).toBe("cancelled");
    expect(secondAborts).toBe(1);
  });

  test("cancelAll aborts every in-flight request and returns the newly cancelled ids", () => {
    const registry = new RequestCancellationRegistry();
    const aborted: string[] = [];
    registry.register("a", () => aborted.push("a"));
    registry.register("b", () => aborted.push("b"));
    registry.register("c", () => aborted.push("c"));
    registry.cancel("b");

    // "b" was already cancelling, so it is not reported again.
    expect(registry.cancelAll().sort()).toEqual(["a", "c"]);
    expect(aborted.sort()).toEqual(["a", "b", "c"]);
  });
});
