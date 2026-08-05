/**
 * Registry of in-flight generation requests that an operator can cancel.
 *
 * Client disconnects are the normal cancellation path, but they depend on the
 * transport noticing a dead socket. A long prefill on a large model can hold a
 * worker and a limiter slot for minutes after the caller has gone away, so the
 * dashboard needs an explicit kill switch keyed by the request id it already
 * displays.
 *
 * Cancellation is recorded even when it arrives before the abort takes effect,
 * so repeated clicks stay idempotent and the caller can tell "cancel accepted"
 * apart from "no such request".
 */
export type CancelOutcome = "cancelled" | "already_cancelling" | "not_found";

type Entry = {
  abort: () => void;
  cancelRequestedAt?: number;
};

export class RequestCancellationRegistry {
  private readonly entries = new Map<string, Entry>();

  /** Registers an in-flight request; returns an idempotent unregister function. */
  register(id: string, abort: () => void): () => void {
    this.entries.set(id, { abort });
    return () => {
      this.entries.delete(id);
    };
  }

  cancel(id: string, now: number = Date.now()): CancelOutcome {
    const entry = this.entries.get(id);
    if (!entry) return "not_found";
    if (entry.cancelRequestedAt !== undefined) return "already_cancelling";
    entry.cancelRequestedAt = now;
    // The abort itself must not remove the entry: the request stays in flight
    // until its own completion path unregisters it, and until then further
    // cancel calls should report that cancellation is already under way.
    entry.abort();
    return "cancelled";
  }

  /** Cancels everything in flight; returns the ids newly moved to cancelling. */
  cancelAll(now: number = Date.now()): string[] {
    const cancelled: string[] = [];
    for (const id of [...this.entries.keys()]) {
      if (this.cancel(id, now) === "cancelled") cancelled.push(id);
    }
    return cancelled;
  }

  isCancelling(id: string): boolean {
    return this.entries.get(id)?.cancelRequestedAt !== undefined;
  }

  get size(): number {
    return this.entries.size;
  }
}
