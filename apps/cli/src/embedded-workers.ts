import { existsSync } from "node:fs";
import { chmod, mkdir, readdir, rename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, join } from "node:path";
// Embedded native workers: when compiled with `bun build --compile` these
// imports become assets inside the single-file binary (virtual /$bunfs paths).
// In dev mode they resolve to the real files in <repo>/libexec.
import llamaWorkerAsset from "../../../libexec/clap-llama" with { type: "file" };
import metallibAsset from "../../../libexec/mlx.metallib" with { type: "file" };
import mlxWorkerAsset from "../../../libexec/clap-mlx" with { type: "file" };

const embeddedAssets = [
  { name: "clap-llama", path: llamaWorkerAsset as string, executable: true, env: "CLAP_LLAMA_WORKER" },
  { name: "clap-mlx", path: mlxWorkerAsset as string, executable: true, env: "CLAP_MLX_WORKER" },
  { name: "mlx.metallib", path: metallibAsset as string, executable: false, env: undefined },
];

function isEmbeddedPath(path: string): boolean {
  return path.startsWith("/$bunfs/") || path.includes("/~BUN/") || path.startsWith("B:\\");
}

function clapHome(): string {
  return process.env.CLAP_HOME ?? join(process.env.HOME ?? homedir(), ".clap");
}

// Extracts the embedded native workers next to each other under
// ~/.clap/libexec/<build-id>/ so the single binary is fully self-contained.
// No-op in dev mode (workers already exist on disk) and when the user
// configured explicit worker paths.
export async function ensureEmbeddedWorkers(): Promise<void> {
  const embedded = embeddedAssets.filter((asset) => isEmbeddedPath(asset.path));
  if (embedded.length === 0) return;

  const buildId = process.env.CLAP_EMBED_BUILD ?? "dev";
  const targetDir = join(clapHome(), "libexec", buildId);
  const extracted: Record<string, string> = {};

  for (const asset of embedded) {
    const file = Bun.file(asset.path);
    if (file.size === 0) continue; // placeholder for a worker not built on this platform
    const target = join(targetDir, asset.name);
    extracted[asset.name] = target;
    if (existsSync(target) && Bun.file(target).size === file.size) continue;
    await mkdir(targetDir, { recursive: true });
    const temp = `${target}.tmp-${process.pid}`;
    await Bun.write(temp, file);
    if (asset.executable) await chmod(temp, 0o755);
    await rename(temp, target);
  }

  for (const asset of embedded) {
    const target = extracted[asset.name];
    if (asset.env && target && !process.env[asset.env]) {
      process.env[asset.env] = target;
    }
  }

  void pruneStaleBuilds(join(clapHome(), "libexec"), buildId);
}

async function pruneStaleBuilds(libexecDir: string, keep: string): Promise<void> {
  try {
    const entries = await readdir(libexecDir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isDirectory() || entry.name === keep) continue;
      await rm(join(libexecDir, entry.name), { recursive: true, force: true });
    }
  } catch {
    // best effort; stale build dirs are harmless
  }
}

export function embeddedWorkerInfo(): { name: string; embedded: boolean }[] {
  return embeddedAssets.map((asset) => ({ name: basename(asset.name), embedded: isEmbeddedPath(asset.path) }));
}
