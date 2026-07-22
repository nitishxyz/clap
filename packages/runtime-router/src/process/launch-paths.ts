import { createHash, randomUUID } from "node:crypto";
import { homedir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { realpath } from "node:fs/promises";
import type { WorkerLaunchIdentity, WorkerLaunchPaths } from "./types";

const PATH_FINGERPRINT_DOMAIN = "clap.worker.model-path.v1";
const MODEL_HASH_DOMAIN = "clap.worker.model-identity.v1";

function digest(domain: string, fields: readonly string[]): string {
  const hash = createHash("sha256");
  hash.update(`${domain}\0`);
  for (const field of fields) hash.update(`${Buffer.byteLength(field)}:${field}`);
  return hash.digest("hex");
}

function safeSegment(value: string, name: string): string {
  if (!value || value === "." || value === ".." || value.includes("/") || value.includes("\\")) {
    throw new Error(`Invalid worker launch ${name}: ${JSON.stringify(value)}`);
  }
  return value;
}

export function resolveClapHome(): string {
  return resolve(process.env.CLAP_HOME || join(homedir(), ".clap"));
}

export async function canonicalModelPath(modelPath: string): Promise<string> {
  const absolute = resolve(modelPath);
  try {
    return await realpath(absolute);
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    return join(await realpath(dirname(absolute)), basename(absolute));
  }
}

export async function fingerprintModelPath(modelPath: string): Promise<string> {
  return digest(PATH_FINGERPRINT_DOMAIN, [await canonicalModelPath(modelPath)]);
}

export function hashModelIdentity(input: Omit<WorkerLaunchIdentity, "modelPath"> & {
  modelPathFingerprint: string;
}): string {
  return digest(MODEL_HASH_DOMAIN, [
    input.backend,
    input.modelId,
    input.revision ?? "",
    input.modelPathFingerprint,
  ]);
}

export async function createWorkerLaunchPaths(
  input: WorkerLaunchIdentity,
  launchId = randomUUID(),
): Promise<WorkerLaunchPaths> {
  const clapHome = resolveClapHome();
  const backend = safeSegment(input.backend, "backend");
  const id = safeSegment(launchId, "ID");
  const modelPathFingerprint = await fingerprintModelPath(input.modelPath);
  const modelHash = hashModelIdentity({ ...input, modelPathFingerprint });
  const directory = join(clapHome, "logs", "workers", backend, modelHash);
  return {
    clapHome,
    backend,
    modelHash,
    launchId: id,
    directory,
    stderrPath: join(directory, `${id}.stderr.log`),
    metadataPath: join(directory, `${id}.json`),
    modelPathFingerprint,
  };
}
