import type { WorkerCrashClassification, WorkerRequestPhase } from "./types";

export type WorkerExitPhase = "during protocol handshake" | "during request" | "while idle";
export type WorkerExitClassification = "expected_exit" | "clean_exit" | "crash";

export function classifyWorkerExitPhase(handshakePending: boolean, pendingRequests: number): WorkerExitPhase {
  if (handshakePending) return "during protocol handshake";
  return pendingRequests > 0 ? "during request" : "while idle";
}

export function classifyWorkerCrash(input: {
  protocolFault: boolean;
  expectedExit: boolean;
  exitCode: number;
  phase: WorkerRequestPhase;
}): WorkerCrashClassification {
  if (input.protocolFault) return "protocol_fault";
  if (input.expectedExit) return "expected_exit";
  if (input.exitCode === 0) return "unexpected_exit_0";
  return input.phase;
}

export function classifyWorkerExit(expectedExit: boolean, exitCode: number): WorkerExitClassification {
  if (expectedExit) return "expected_exit";
  return exitCode === 0 ? "clean_exit" : "crash";
}
