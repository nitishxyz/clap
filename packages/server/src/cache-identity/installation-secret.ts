import { constants } from "node:fs";
import {
  chmod,
  lstat,
  mkdir,
  open,
  rename,
  rm,
  stat,
} from "node:fs/promises";
import { dirname, join } from "node:path";
import { randomBytes, randomUUID } from "node:crypto";
import {
  INSTALLATION_SECRET_BYTE_LENGTH,
  INSTALLATION_SECRET_SCHEMA_VERSION,
  type InstallationSecret,
  type InstallationSecretDocumentV1,
  type InstallationSecretRotationOptions,
} from "./types";

const AUTH_DIRECTORY_MODE = 0o700;
const SECRET_FILE_MODE = 0o600;
const SECRET_FILE_NAME = "cache-identity-secret.json";
const GENERATION_PATTERN = /^sec_[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const BASE64URL_PATTERN = /^[A-Za-z0-9_-]{43}$/;
const CREATE_READ_RETRIES = 20;

export async function loadOrCreateInstallationSecret(): Promise<InstallationSecret> {
  const path = installationSecretPath();
  try {
    await prepareAuthDirectory(dirname(path));
    try {
      return await readInstallationSecret(path);
    } catch (error) {
      if (!hasCode(error, "ENOENT")) throw error;
    }

    const document = createDocument();
    try {
      await createSecretFile(path, document);
      await syncDirectory(dirname(path));
      return toInstallationSecret(document);
    } catch (error) {
      if (!hasCode(error, "EEXIST")) throw error;
      return await readWinner(path);
    }
  } catch {
    throw installationSecretError("load or create");
  }
}

export async function rotateInstallationSecret(
  options: InstallationSecretRotationOptions = {},
): Promise<InstallationSecret> {
  const path = installationSecretPath();
  const directory = dirname(path);
  let temporaryPath: string | undefined;
  try {
    await prepareAuthDirectory(directory);
    await readInstallationSecret(path);

    const document = createDocument();
    temporaryPath = join(directory, `.${SECRET_FILE_NAME}.${process.pid}.${randomUUID()}.tmp`);
    await createSecretFile(temporaryPath, document);
    await (options.rename ?? rename)(temporaryPath, path);
    temporaryPath = undefined;
    await syncDirectory(directory);
    return toInstallationSecret(document);
  } catch {
    throw installationSecretError("rotate");
  } finally {
    if (temporaryPath) await rm(temporaryPath, { force: true }).catch(() => undefined);
  }
}

export function installationSecretPath(): string {
  return join(clapHome(), "auth", SECRET_FILE_NAME);
}

function clapHome(): string {
  return process.env.CLAP_HOME ?? join(process.env.HOME ?? ".", ".clap");
}

async function prepareAuthDirectory(directory: string): Promise<void> {
  await mkdir(directory, { recursive: true, mode: AUTH_DIRECTORY_MODE });
  const metadata = await lstat(directory);
  assertOwnedRegularKind(metadata, "directory");
  await chmod(directory, AUTH_DIRECTORY_MODE);
  const repaired = await stat(directory);
  assertOwnedRegularKind(repaired, "directory");
  if ((repaired.mode & 0o777) !== AUTH_DIRECTORY_MODE) throw new Error("directory permissions rejected");
}

async function readInstallationSecret(path: string): Promise<InstallationSecret> {
  const before = await lstat(path);
  assertOwnedRegularKind(before, "file");

  const noFollow = "O_NOFOLLOW" in constants ? constants.O_NOFOLLOW : 0;
  const handle = await open(path, constants.O_RDONLY | noFollow);
  try {
    const opened = await handle.stat();
    assertOwnedRegularKind(opened, "file");
    if (before.dev !== opened.dev || before.ino !== opened.ino) throw new Error("file changed while opening");
    await handle.chmod(SECRET_FILE_MODE);
    const repaired = await handle.stat();
    if ((repaired.mode & 0o777) !== SECRET_FILE_MODE) throw new Error("file permissions rejected");
    const contents = await handle.readFile("utf8");
    return toInstallationSecret(parseDocument(contents));
  } finally {
    await handle.close();
  }
}

async function createSecretFile(path: string, document: InstallationSecretDocumentV1): Promise<void> {
  const handle = await open(path, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL, SECRET_FILE_MODE);
  try {
    await handle.writeFile(`${JSON.stringify(document)}\n`, "utf8");
    await handle.chmod(SECRET_FILE_MODE);
    await handle.sync();
  } finally {
    await handle.close();
  }
}

async function readWinner(path: string): Promise<InstallationSecret> {
  for (let attempt = 0; attempt < CREATE_READ_RETRIES; attempt += 1) {
    try {
      return await readInstallationSecret(path);
    } catch (error) {
      if (!hasCode(error, "ENOENT") && !(error instanceof SyntaxError)) throw error;
      await new Promise((resolve) => setTimeout(resolve, attempt + 1));
    }
  }
  return readInstallationSecret(path);
}

function createDocument(): InstallationSecretDocumentV1 {
  return {
    version: INSTALLATION_SECRET_SCHEMA_VERSION,
    generation: `sec_${randomUUID()}`,
    createdAt: new Date().toISOString(),
    secret: randomBytes(INSTALLATION_SECRET_BYTE_LENGTH).toString("base64url"),
  };
}

function parseDocument(contents: string): InstallationSecretDocumentV1 {
  let value: unknown;
  try {
    value = JSON.parse(contents);
  } catch {
    throw new SyntaxError("installation secret document is malformed");
  }
  if (!isRecord(value)) throw new SyntaxError("installation secret document is malformed");
  const fields = Object.keys(value).sort();
  if (fields.join(",") !== "createdAt,generation,secret,version") {
    throw new SyntaxError("installation secret document has unexpected fields");
  }
  if (value.version !== INSTALLATION_SECRET_SCHEMA_VERSION) throw new SyntaxError("installation secret version is unsupported");
  if (typeof value.generation !== "string" || !GENERATION_PATTERN.test(value.generation)) {
    throw new SyntaxError("installation secret generation is malformed");
  }
  if (typeof value.createdAt !== "string" || !isCanonicalTimestamp(value.createdAt)) {
    throw new SyntaxError("installation secret timestamp is malformed");
  }
  if (typeof value.secret !== "string" || !BASE64URL_PATTERN.test(value.secret)) {
    throw new SyntaxError("installation secret key is malformed");
  }
  const key = Buffer.from(value.secret, "base64url");
  if (key.byteLength !== INSTALLATION_SECRET_BYTE_LENGTH || key.toString("base64url") !== value.secret) {
    throw new SyntaxError("installation secret key has an invalid length");
  }
  return {
    version: INSTALLATION_SECRET_SCHEMA_VERSION,
    generation: value.generation,
    createdAt: value.createdAt,
    secret: value.secret,
  };
}

function toInstallationSecret(document: InstallationSecretDocumentV1): InstallationSecret {
  return {
    generation: document.generation,
    key: Buffer.from(document.secret, "base64url"),
  };
}

function isCanonicalTimestamp(value: string): boolean {
  const timestamp = Date.parse(value);
  return Number.isFinite(timestamp) && new Date(timestamp).toISOString() === value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertOwnedRegularKind(metadata: { isFile(): boolean; isDirectory(): boolean; uid: number }, kind: "file" | "directory"): void {
  if (kind === "file" ? !metadata.isFile() : !metadata.isDirectory()) throw new Error(`${kind} type rejected`);
  const uid = typeof process.getuid === "function" ? process.getuid() : undefined;
  if (uid !== undefined && metadata.uid !== uid) throw new Error(`${kind} owner rejected`);
}

async function syncDirectory(directory: string): Promise<void> {
  let handle;
  try {
    handle = await open(directory, constants.O_RDONLY);
    await handle.sync();
  } catch (error) {
    if (!hasAnyCode(error, ["EINVAL", "ENOTSUP", "EBADF", "EISDIR"])) throw error;
  } finally {
    await handle?.close();
  }
}

function installationSecretError(operation: string): Error {
  return new Error(`Unable to ${operation} the cache identity installation secret securely`);
}

function hasCode(error: unknown, code: string): boolean {
  return isRecord(error) && error.code === code;
}

function hasAnyCode(error: unknown, codes: string[]): boolean {
  return isRecord(error) && typeof error.code === "string" && codes.includes(error.code);
}
