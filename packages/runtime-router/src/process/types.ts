export const WORKER_LAUNCH_METADATA_VERSION = 1 as const;

export interface WorkerLaunchIdentity {
  backend: string;
  modelId: string;
  revision?: string | null;
  modelPath: string;
}

export interface WorkerLaunchPaths {
  clapHome: string;
  backend: string;
  modelHash: string;
  launchId: string;
  directory: string;
  stderrPath: string;
  metadataPath: string;
  modelPathFingerprint: string;
}

export interface WorkerLaunchMetadata {
  version: typeof WORKER_LAUNCH_METADATA_VERSION;
  launchId: string;
  modelId: string;
  modelPathFingerprint: string;
  backend: string;
  pid: number;
  command: string[];
  protocolVersion: string;
  startedAt: string;
  endedAt?: string;
  exitStatus?: number | null;
  crashClassification?: string | null;
}

export interface LaunchRetentionLimits {
  maxLaunchesPerModel: number;
  maxBytesPerBackend: number;
}
