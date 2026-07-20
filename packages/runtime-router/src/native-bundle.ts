import { chmod, mkdir, rename, rm } from "node:fs/promises";
import { dirname } from "node:path";

export const NATIVE_BUNDLE_SCHEMA = 1;

export type NativeArtifactManifest = {
  schema: number;
  artifacts: Array<{
    name: string;
    size: number;
    sha256: string;
    executable: boolean;
  }>;
};

export async function sha256(input: Blob | ArrayBuffer | Uint8Array): Promise<string> {
  const hasher = new Bun.CryptoHasher("sha256");
  if (input instanceof Blob) hasher.update(await input.arrayBuffer());
  else hasher.update(input);
  return hasher.digest("hex");
}

export async function createNativeBundleManifest(
  artifacts: Array<{ name: string; data: Blob | ArrayBuffer | Uint8Array; executable: boolean }>,
): Promise<NativeArtifactManifest> {
  const entries = await Promise.all(artifacts.map(async ({ name, data, executable }) => {
    const bytes = data instanceof Blob ? data.size : data.byteLength;
    return { name, size: bytes, sha256: await sha256(data), executable };
  }));
  return { schema: NATIVE_BUNDLE_SCHEMA, artifacts: entries.sort((a, b) => a.name.localeCompare(b.name)) };
}

export async function nativeBundleBuildId(manifest: NativeArtifactManifest): Promise<string> {
  return (await sha256(new TextEncoder().encode(JSON.stringify(manifest)))).slice(0, 12);
}

export async function ensureNativeArtifact(
  source: Blob,
  target: string,
  expected: { size: number; sha256: string; executable: boolean },
): Promise<"unchanged" | "replaced"> {
  const current = Bun.file(target);
  if (current.size === expected.size && await sha256(current) === expected.sha256) {
    if (expected.executable) await chmod(target, 0o755);
    return "unchanged";
  }
  const sourceDigest = await sha256(source);
  if (source.size !== expected.size || sourceDigest !== expected.sha256) {
    throw new Error(`embedded native artifact digest mismatch for ${target}`);
  }
  await mkdir(dirname(target), { recursive: true });
  const temp = `${target}.tmp-${process.pid}-${crypto.randomUUID()}`;
  try {
    await Bun.write(temp, source);
    if (expected.executable) await chmod(temp, 0o755);
    await rename(temp, target);
  } catch (error) {
    await rm(temp, { force: true });
    throw error;
  }
  return "replaced";
}
