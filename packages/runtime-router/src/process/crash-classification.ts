export type WorkerExitPhase = "during protocol handshake" | "during request" | "while idle";
export type WorkerExitClassification = "expected_exit" | "clean_exit" | "crash";

export function classifyWorkerExitPhase(handshakePending: boolean, pendingRequests: number): WorkerExitPhase {
  if (handshakePending) return "during protocol handshake";
  return pendingRequests > 0 ? "during request" : "while idle";
}

export function classifyWorkerExit(expectedExit: boolean, exitCode: number): WorkerExitClassification {
  if (expectedExit) return "expected_exit";
  return exitCode === 0 ? "clean_exit" : "crash";
}
