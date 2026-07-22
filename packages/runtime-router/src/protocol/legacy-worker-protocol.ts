export type LegacyWorkerLine =
  | { kind: "message"; message: Record<string, unknown> }
  | { kind: "malformed" };

/** Temporary, explicitly configured adapter for structured pre-v1 workers. */
export class LegacyWorkerProtocol {
  decode(line: string): LegacyWorkerLine {
    try {
      const value = JSON.parse(line) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return { kind: "message", message: value as Record<string, unknown> };
      }
    } catch {}
    return { kind: "malformed" };
  }
}
