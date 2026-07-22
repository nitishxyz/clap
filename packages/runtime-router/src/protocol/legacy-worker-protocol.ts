export type LegacyWorkerLine =
  | { kind: "message"; message: Record<string, unknown> }
  | { kind: "text"; text: string };

/** Temporary decoder for pre-v1 workers. It intentionally preserves raw-text fallback. */
export class LegacyWorkerProtocol {
  decode(line: string): LegacyWorkerLine {
    try {
      const value = JSON.parse(line) as unknown;
      if (value && typeof value === "object" && !Array.isArray(value)) {
        return { kind: "message", message: value as Record<string, unknown> };
      }
    } catch {
      // Legacy stdout historically treated non-JSON as generated text.
    }
    return { kind: "text", text: line };
  }
}

export function allowsLegacyStartupFallback(mode: "legacy" | "v1" | "auto", source: "configured" | "bundled" | "missing"): boolean {
  return mode === "auto" && source === "configured";
}
