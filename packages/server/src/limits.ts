// Fair inference admission: bounded in-flight dispatch plus a bounded waiting
// queue with per-client fair scheduling. Clients are API key ids when auth is
// in play, else the remote address, else "local". When the queue is full the
// server answers 429 + Retry-After instead of letting latency grow unbounded.

export class QueueFullError extends Error {
  constructor(public readonly retryAfterSeconds: number, message: string) {
    super(message);
  }
}

type Waiter = {
  client: string;
  resolve: (release: () => void) => void;
  reject: (error: Error) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
};

export class FairLimiter {
  private inflight = 0;
  private readonly perClient = new Map<string, number>();
  private waiters: Waiter[] = [];

  constructor(
    private readonly maxInflight: number,
    private readonly queueDepth: number,
  ) {}

  stats(): { inflight: number; queued: number; maxInflight: number; queueDepth: number } {
    return { inflight: this.inflight, queued: this.waiters.length, maxInflight: this.maxInflight, queueDepth: this.queueDepth };
  }

  acquire(client: string, signal?: AbortSignal): Promise<() => void> {
    if (this.inflight < this.maxInflight) {
      return Promise.resolve(this.admit(client));
    }
    if (this.waiters.length >= this.queueDepth) {
      const retryAfter = Math.min(60, 5 * Math.ceil((this.waiters.length + 1) / Math.max(this.maxInflight, 1)));
      return Promise.reject(new QueueFullError(
        retryAfter,
        `server is at capacity (${this.inflight} in flight, ${this.waiters.length} queued); retry after ${retryAfter}s`,
      ));
    }
    return new Promise<() => void>((resolve, reject) => {
      const waiter: Waiter = { client, resolve, reject, signal };
      if (signal) {
        waiter.onAbort = () => {
          const index = this.waiters.indexOf(waiter);
          if (index >= 0) {
            this.waiters.splice(index, 1);
            reject(new Error("request aborted while queued"));
          }
        };
        if (signal.aborted) {
          waiter.onAbort();
          return;
        }
        signal.addEventListener("abort", waiter.onAbort, { once: true });
      }
      this.waiters.push(waiter);
    });
  }

  private admit(client: string): () => void {
    this.inflight += 1;
    this.perClient.set(client, (this.perClient.get(client) ?? 0) + 1);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.inflight -= 1;
      const count = (this.perClient.get(client) ?? 1) - 1;
      if (count <= 0) this.perClient.delete(client);
      else this.perClient.set(client, count);
      this.dispatchNext();
    };
  }

  // Fair pick: among queued clients, admit the one with the fewest requests
  // currently in flight (FIFO within a client). A chatty client cannot starve
  // others: its queued requests only win ties.
  private dispatchNext(): void {
    if (!this.waiters.length || this.inflight >= this.maxInflight) return;
    let bestIndex = 0;
    let bestScore = Number.POSITIVE_INFINITY;
    for (let index = 0; index < this.waiters.length; index += 1) {
      const waiter = this.waiters[index];
      if (!waiter) continue;
      const score = this.perClient.get(waiter.client) ?? 0;
      if (score < bestScore) {
        bestScore = score;
        bestIndex = index;
        if (score === 0) break;
      }
    }
    const [waiter] = this.waiters.splice(bestIndex, 1);
    if (!waiter) return;
    if (waiter.signal && waiter.onAbort) waiter.signal.removeEventListener("abort", waiter.onAbort);
    waiter.resolve(this.admit(waiter.client));
  }
}

export function limiterFromEnv(configMaxInflight?: number, configQueueDepth?: number): FairLimiter {
  const maxInflight = positiveInt(process.env.CLAP_MAX_INFLIGHT) ?? configMaxInflight ?? 16;
  const queueDepth = positiveInt(process.env.CLAP_QUEUE_DEPTH) ?? configQueueDepth ?? 64;
  return new FairLimiter(maxInflight, queueDepth);
}

function positiveInt(raw: string | undefined): number | undefined {
  if (raw === undefined || raw.trim() === "") return undefined;
  const value = Number(raw);
  if (!Number.isInteger(value) || value <= 0) return undefined;
  return value;
}
