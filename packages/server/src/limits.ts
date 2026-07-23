// Fair inference admission: bounded in-flight dispatch plus bounded priority
// queues. Weighted round-robin across priority classes is 4:2:1, while each
// class preserves least-inflight-per-client selection and FIFO ties.

export type RequestPriority = "interactive" | "normal" | "background";
export const REQUEST_PRIORITIES = ["interactive", "normal", "background"] as const;

export class QueueFullError extends Error {
  constructor(public readonly retryAfterSeconds: number, message: string) {
    super(message);
  }
}

type Waiter = {
  client: string;
  priority: RequestPriority;
  resolve: (release: () => void) => void;
  reject: (error: Error) => void;
  signal?: AbortSignal;
  onAbort?: () => void;
};

type PriorityCounts = Record<RequestPriority, number>;
type PriorityOutcomes = Record<RequestPriority, Record<"admitted" | "rejected" | "aborted", number>>;
export type LimiterStats = {
  inflight: number;
  queued: number;
  maxInflight: number;
  queueDepth: number;
  inflightByPriority: PriorityCounts;
  waitingByPriority: PriorityCounts;
  outcomesByPriority: PriorityOutcomes;
};

const schedule: readonly RequestPriority[] = [
  "interactive", "interactive", "interactive", "interactive",
  "normal", "normal", "background",
];
const emptyCounts = (): PriorityCounts => ({ interactive: 0, normal: 0, background: 0 });
const emptyOutcomes = (): PriorityOutcomes => ({
  interactive: { admitted: 0, rejected: 0, aborted: 0 },
  normal: { admitted: 0, rejected: 0, aborted: 0 },
  background: { admitted: 0, rejected: 0, aborted: 0 },
});

export class FairLimiter {
  private inflight = 0;
  private readonly inflightByPriority = emptyCounts();
  private readonly perClient = new Map<string, number>();
  private readonly outcomesByPriority = emptyOutcomes();
  private readonly waiters: Record<RequestPriority, Waiter[]> = {
    interactive: [], normal: [], background: [],
  };
  private scheduleIndex = 0;

  constructor(
    private readonly maxInflight: number,
    private readonly queueDepth: number,
  ) {}

  stats(): LimiterStats {
    const waitingByPriority = Object.fromEntries(REQUEST_PRIORITIES.map((priority) =>
      [priority, this.waiters[priority].length])) as PriorityCounts;
    return { inflight: this.inflight, queued: this.queued(), maxInflight: this.maxInflight,
      queueDepth: this.queueDepth, inflightByPriority: { ...this.inflightByPriority }, waitingByPriority,
      outcomesByPriority: Object.fromEntries(REQUEST_PRIORITIES.map((priority) =>
        [priority, { ...this.outcomesByPriority[priority] }])) as PriorityOutcomes };
  }

  acquire(client: string, priority: RequestPriority = "normal", signal?: AbortSignal): Promise<() => void> {
    if (this.inflight < this.maxInflight && this.queued() === 0) {
      return Promise.resolve(this.admit(client, priority));
    }
    if (this.queued() >= this.queueDepth) {
      this.outcomesByPriority[priority].rejected += 1;
      const retryAfter = Math.min(60, 5 * Math.ceil((this.queued() + 1) / Math.max(this.maxInflight, 1)));
      return Promise.reject(new QueueFullError(
        retryAfter,
        `server is at capacity (${this.inflight} in flight, ${this.queued()} queued); retry after ${retryAfter}s`,
      ));
    }
    return new Promise<() => void>((resolve, reject) => {
      const waiter: Waiter = { client, priority, resolve, reject, signal };
      if (signal) {
        waiter.onAbort = () => {
          const queue = this.waiters[priority];
          const index = queue.indexOf(waiter);
          if (index >= 0) {
            queue.splice(index, 1);
            this.outcomesByPriority[priority].aborted += 1;
            reject(new Error("request aborted while queued"));
          }
        };
        if (signal.aborted) { waiter.onAbort(); return; }
        signal.addEventListener("abort", waiter.onAbort, { once: true });
      }
      this.waiters[priority].push(waiter);
      this.dispatchNext();
    });
  }

  private queued(): number {
    return REQUEST_PRIORITIES.reduce((total, priority) => total + this.waiters[priority].length, 0);
  }

  private admit(client: string, priority: RequestPriority): () => void {
    this.inflight += 1;
    this.outcomesByPriority[priority].admitted += 1;
    this.inflightByPriority[priority] += 1;
    this.perClient.set(client, (this.perClient.get(client) ?? 0) + 1);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      this.inflight -= 1;
      this.inflightByPriority[priority] -= 1;
      const count = (this.perClient.get(client) ?? 1) - 1;
      if (count <= 0) this.perClient.delete(client);
      else this.perClient.set(client, count);
      this.dispatchNext();
    };
  }

  private nextPriority(): RequestPriority | undefined {
    if (this.queued() === 0) return undefined;
    for (let offset = 0; offset < schedule.length; offset += 1) {
      const index = (this.scheduleIndex + offset) % schedule.length;
      const priority = schedule[index]!;
      if (this.waiters[priority].length > 0) {
        this.scheduleIndex = (index + 1) % schedule.length;
        return priority;
      }
    }
    return undefined;
  }

  private dispatchNext(): void {
    while (this.inflight < this.maxInflight) {
      const priority = this.nextPriority();
      if (!priority) return;
      const queue = this.waiters[priority];
      let bestIndex = 0;
      let bestScore = Number.POSITIVE_INFINITY;
      for (let index = 0; index < queue.length; index += 1) {
        const waiter = queue[index]!;
        const score = this.perClient.get(waiter.client) ?? 0;
        if (score < bestScore) {
          bestScore = score;
          bestIndex = index;
          if (score === 0) break;
        }
      }
      const [waiter] = queue.splice(bestIndex, 1);
      if (!waiter) return;
      if (waiter.signal && waiter.onAbort) waiter.signal.removeEventListener("abort", waiter.onAbort);
      waiter.resolve(this.admit(waiter.client, waiter.priority));
    }
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
