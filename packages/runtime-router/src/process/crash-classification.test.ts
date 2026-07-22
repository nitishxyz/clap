import { describe, expect, test } from "bun:test";
import { classifyWorkerExit, classifyWorkerExitPhase } from "./crash-classification";

describe("worker crash classification", () => {
  test("classifies exit phase from handshake and pending state", () => {
    expect(classifyWorkerExitPhase(true, 1)).toBe("during protocol handshake");
    expect(classifyWorkerExitPhase(false, 1)).toBe("during request");
    expect(classifyWorkerExitPhase(false, 0)).toBe("while idle");
  });

  test("distinguishes deliberate, clean, and crashing exits", () => {
    expect(classifyWorkerExit(true, 143)).toBe("expected_exit");
    expect(classifyWorkerExit(false, 0)).toBe("clean_exit");
    expect(classifyWorkerExit(false, 9)).toBe("crash");
  });
});
