import { describe, expect, test } from "bun:test";
import { classifyWorkerCrash, classifyWorkerExit, classifyWorkerExitPhase } from "./crash-classification";

describe("worker crash classification", () => {
  test("classifies exit phase from handshake and pending state", () => {
    expect(classifyWorkerExitPhase(true, 1)).toBe("during protocol handshake");
    expect(classifyWorkerExitPhase(false, 1)).toBe("during request");
    expect(classifyWorkerExitPhase(false, 0)).toBe("while idle");
  });

  test("records protocol, handshake, load, prefill, decode, idle, and clean-unexpected causes", () => {
    const classify = (phase: "handshake" | "load" | "prefill" | "decode" | "idle") =>
      classifyWorkerCrash({ protocolFault: false, expectedExit: false, exitCode: 9, phase });
    expect(["handshake", "load", "prefill", "decode", "idle"].map((phase) =>
      classify(phase as Parameters<typeof classify>[0]))).toEqual(["handshake", "load", "prefill", "decode", "idle"]);
    expect(classifyWorkerCrash({ protocolFault: true, expectedExit: false, exitCode: 9, phase: "handshake" }))
      .toBe("protocol_fault");
    expect(classifyWorkerCrash({ protocolFault: false, expectedExit: false, exitCode: 0, phase: "idle" }))
      .toBe("unexpected_exit_0");
  });

  test("distinguishes deliberate, clean, and crashing exits", () => {
    expect(classifyWorkerExit(true, 143)).toBe("expected_exit");
    expect(classifyWorkerExit(false, 0)).toBe("clean_exit");
    expect(classifyWorkerExit(false, 9)).toBe("crash");
  });
});
