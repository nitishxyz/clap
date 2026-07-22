import { readdir, readFile, rename, stat, unlink, writeFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";
import { randomUUID } from "node:crypto";
import {
  WORKER_LAUNCH_METADATA_VERSION,
  type LaunchRetentionLimits,
  type WorkerLaunchMetadata,
  type WorkerLaunchPaths,
  type WorkerLaunchContext,
  type WorkerLaunchIdentity,
} from "./types";
import { createWorkerLaunchPaths } from "./launch-paths";

export const DEFAULT_MAX_LAUNCHES_PER_MODEL = 20;
export const DEFAULT_MAX_BYTES_PER_BACKEND = 256 * 1024 * 1024;
export const RETENTION_COUNT_ENV = "CLAP_WORKER_LOG_MAX_LAUNCHES_PER_MODEL";
export const RETENTION_BYTES_ENV = "CLAP_WORKER_LOG_MAX_BYTES_PER_BACKEND";

const activeMetadataPaths = new Set<string>();
const retentionQueues = new Map<string, Promise<void>>();

function positiveInteger(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 0 ? parsed : fallback;
}

export function launchRetentionLimits(): LaunchRetentionLimits {
  return {
    maxLaunchesPerModel: positiveInteger(
      process.env[RETENTION_COUNT_ENV],
      DEFAULT_MAX_LAUNCHES_PER_MODEL,
    ),
    maxBytesPerBackend: positiveInteger(
      process.env[RETENTION_BYTES_ENV],
      DEFAULT_MAX_BYTES_PER_BACKEND,
    ),
  };
}

async function ignoreMissing(operation: () => Promise<unknown>): Promise<void> {
  try {
    await operation();
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
}

export async function writeLaunchMetadataAtomic(
  metadataPath: string,
  metadata: WorkerLaunchMetadata,
): Promise<void> {
  await mkdir(dirname(metadataPath), { recursive: true });
  const temporaryPath = `${metadataPath}.${process.pid}.${randomUUID()}.tmp`;
  try {
    await writeFile(temporaryPath, `${JSON.stringify(metadata, null, 2)}\n`, { mode: 0o600 });
    await rename(temporaryPath, metadataPath);
  } catch (error) {
    await ignoreMissing(() => unlink(temporaryPath));
    throw error;
  }
}

interface FinalizedLaunch {
  metadataPath: string;
  stderrPath: string;
  endedAt: number;
  bytes: number;
}

function isFinalizedMetadata(value: unknown): value is WorkerLaunchMetadata & { endedAt: string } {
  if (typeof value !== "object" || value === null) return false;
  const item = value as Record<string, unknown>;
  return item.version === WORKER_LAUNCH_METADATA_VERSION
    && typeof item.launchId === "string"
    && typeof item.modelId === "string"
    && typeof item.modelPathFingerprint === "string"
    && typeof item.backend === "string"
    && typeof item.pid === "number"
    && Array.isArray(item.command)
    && item.command.every((part) => typeof part === "string")
    && typeof item.protocolVersion === "string"
    && typeof item.startedAt === "string"
    && typeof item.endedAt === "string";
}

async function finalizedLaunch(metadataPath: string): Promise<FinalizedLaunch | undefined> {
  if (activeMetadataPaths.has(metadataPath)) return undefined;
  try {
    const parsed: unknown = JSON.parse(await readFile(metadataPath, "utf8"));
    if (!isFinalizedMetadata(parsed)) return;
    const endedAt = Date.parse(parsed.endedAt);
    if (!Number.isFinite(endedAt)) return;
    const stderrPath = metadataPath.replace(/\.json$/, ".stderr.log");
    const [metadataInfo, stderrInfo] = await Promise.all([
      stat(metadataPath),
      stat(stderrPath).catch((error: NodeJS.ErrnoException) => error.code === "ENOENT" ? undefined : Promise.reject(error)),
    ]);
    return { metadataPath, stderrPath, endedAt, bytes: metadataInfo.size + (stderrInfo?.size ?? 0) };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT" || error instanceof SyntaxError) return;
    throw error;
  }
}

async function scanModel(modelDirectory: string): Promise<{ launches: FinalizedLaunch[]; bytes: number }> {
  let entries;
  try {
    entries = await readdir(modelDirectory, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return { launches: [], bytes: 0 };
    throw error;
  }
  const files = entries.filter((entry) => entry.isFile());
  const [launches, sizes] = await Promise.all([
    Promise.all(files
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .map((entry) => finalizedLaunch(join(modelDirectory, entry.name)))),
    Promise.all(files.map((entry) => stat(join(modelDirectory, entry.name))
      .then((info) => info.size)
      .catch((error: NodeJS.ErrnoException) => error.code === "ENOENT" ? 0 : Promise.reject(error)))),
  ]);
  return {
    launches: launches.filter((launch): launch is FinalizedLaunch => launch !== undefined),
    bytes: sizes.reduce((total, size) => total + size, 0),
  };
}

async function removeLaunch(launch: FinalizedLaunch): Promise<void> {
  await ignoreMissing(() => unlink(launch.stderrPath));
  await ignoreMissing(() => unlink(launch.metadataPath));
}

async function pruneBackendNow(backendDirectory: string, limits: LaunchRetentionLimits): Promise<void> {
  let modelEntries;
  try {
    modelEntries = await readdir(backendDirectory, { withFileTypes: true });
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
    throw error;
  }
  const retained: FinalizedLaunch[] = [];
  let bytes = 0;
  for (const modelEntry of modelEntries.filter((entry) => entry.isDirectory())) {
    const scan = await scanModel(join(backendDirectory, modelEntry.name));
    bytes += scan.bytes;
    const launches = scan.launches
      .sort((left, right) => left.endedAt - right.endedAt || left.metadataPath.localeCompare(right.metadataPath));
    const excess = Math.max(0, launches.length - limits.maxLaunchesPerModel);
    for (const launch of launches.slice(0, excess)) {
      await removeLaunch(launch);
      bytes -= launch.bytes;
    }
    retained.push(...launches.slice(excess));
  }
  retained.sort((left, right) => left.endedAt - right.endedAt || left.metadataPath.localeCompare(right.metadataPath));
  for (const launch of retained) {
    if (bytes <= limits.maxBytesPerBackend) break;
    await removeLaunch(launch);
    bytes -= launch.bytes;
  }
}

export async function pruneLaunchLogs(
  backendDirectory: string,
  limits = launchRetentionLimits(),
): Promise<void> {
  const previous = retentionQueues.get(backendDirectory) ?? Promise.resolve();
  const current = previous.catch(() => {}).then(() => pruneBackendNow(backendDirectory, limits));
  retentionQueues.set(backendDirectory, current);
  try {
    await current;
  } finally {
    if (retentionQueues.get(backendDirectory) === current) retentionQueues.delete(backendDirectory);
  }
}

export class WorkerLaunchLogStore {
  private readonly writes = new Map<string, Promise<void>>();
  private readonly finalizations = new WeakMap<WorkerLaunchContext, Promise<void>>();

  registerActive(paths: WorkerLaunchPaths): () => void {
    activeMetadataPaths.add(paths.metadataPath);
    let active = true;
    return () => {
      if (!active) return;
      active = false;
      activeMetadataPaths.delete(paths.metadataPath);
    };
  }

  async writeMetadata(paths: WorkerLaunchPaths, metadata: WorkerLaunchMetadata): Promise<void> {
    const previous = this.writes.get(paths.metadataPath) ?? Promise.resolve();
    const current = previous.then(() => writeLaunchMetadataAtomic(paths.metadataPath, metadata));
    this.writes.set(paths.metadataPath, current);
    try {
      await current;
    } finally {
      if (this.writes.get(paths.metadataPath) === current) this.writes.delete(paths.metadataPath);
    }
  }

  async prune(paths: WorkerLaunchPaths, limits = launchRetentionLimits()): Promise<void> {
    await pruneLaunchLogs(dirname(paths.directory), limits);
  }

  async prepareLaunch(identity: WorkerLaunchIdentity, command: string[]): Promise<WorkerLaunchContext> {
    const paths = await createWorkerLaunchPaths(identity);
    const context: WorkerLaunchContext = {
      paths,
      phase: "handshake",
      protocolFault: false,
      releaseActive: this.registerActive(paths),
      metadata: {
        version: WORKER_LAUNCH_METADATA_VERSION,
        launchId: paths.launchId,
        modelId: identity.modelId,
        modelPathFingerprint: paths.modelPathFingerprint,
        backend: identity.backend,
        pid: 0,
        command,
        protocolVersion: "1",
        startedAt: new Date().toISOString(),
      },
    };
    try {
      await this.writeMetadata(paths, context.metadata);
      return context;
    } catch (error) {
      context.releaseActive();
      throw error;
    }
  }

  async markSpawned(context: WorkerLaunchContext, pid: number): Promise<void> {
    context.metadata = { ...context.metadata, pid };
    await this.writeMetadata(context.paths, context.metadata);
  }

  async markReady(context: WorkerLaunchContext): Promise<void> {
    context.metadata = { ...context.metadata, readyAt: new Date().toISOString() };
    await this.writeMetadata(context.paths, context.metadata);
  }

  async finalize(context: WorkerLaunchContext, exitStatus: number | null, classification: string): Promise<void> {
    const existing = this.finalizations.get(context);
    if (existing) return existing;
    const finalization = (async () => {
      context.metadata = {
        ...context.metadata,
        endedAt: new Date().toISOString(),
        exitStatus,
        crashClassification: classification,
      };
      try {
        await this.writeMetadata(context.paths, context.metadata);
      } finally {
        context.releaseActive();
      }
      await this.prune(context.paths);
    })();
    this.finalizations.set(context, finalization);
    return finalization;
  }
}

export { WorkerLaunchLogStore as LaunchLogStore };
