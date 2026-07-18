import { clapVersion } from "@clap/api";
import { existsSync } from "node:fs";
import { chmod, mkdir, rename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const releaseRepo = "nitishxyz/clap";

function clapHome(): string {
  return process.env.CLAP_HOME ?? join(process.env.HOME ?? homedir(), ".clap");
}

export function hasNvidiaGpu(): boolean {
  if (process.platform !== "linux") return false;
  if (existsSync("/dev/nvidia0") || existsSync("/proc/driver/nvidia")) return true;
  return Bun.which("nvidia-smi") !== null;
}

// On Linux boxes with an NVIDIA GPU the bundled clap-llama worker is CPU-only
// (release binaries are compiled on runners without CUDA). Swap in the
// GPU-enabled worker automatically: use the cached copy when present,
// otherwise download it from the matching GitHub release. Any failure falls
// back to the bundled CPU worker so inference still works.
export async function ensureCudaWorker(): Promise<void> {
  if (process.env.CLAP_LLAMA_WORKER) return; // explicit user configuration wins
  if (process.env.CLAP_CUDA === "0") return;
  if (process.platform !== "linux" || process.arch !== "x64") return;
  if (!hasNvidiaGpu()) return;

  const targetDir = join(clapHome(), "libexec", `cuda-${clapVersion}`);
  const worker = join(targetDir, "clap-llama");
  if (isUsable(worker)) {
    process.env.CLAP_LLAMA_WORKER = worker;
    return;
  }

  const tag = `v${clapVersion}`;
  const archive = `clap-llama-cuda-${tag}-linux-x64.tar.gz`;
  const url = `https://github.com/${releaseRepo}/releases/download/${tag}/${archive}`;
  const tmpDir = join(clapHome(), "libexec", `.cuda-download-${process.pid}`);

  try {
    console.error(`[clap] NVIDIA GPU detected; downloading CUDA worker (${tag})...`);
    await mkdir(tmpDir, { recursive: true });

    const archivePath = join(tmpDir, archive);
    await download(url, archivePath);
    await verifySha256(archivePath, `${url}.sha256`);

    const untar = Bun.spawn(["tar", "-xzf", archivePath, "-C", tmpDir], { stdout: "ignore", stderr: "ignore" });
    if (await untar.exited !== 0) throw new Error("failed to extract CUDA worker archive");

    const extracted = join(tmpDir, "clap-llama");
    if (!isUsable(extracted)) throw new Error("archive did not contain a clap-llama binary");
    await chmod(extracted, 0o755);

    await mkdir(targetDir, { recursive: true });
    await rename(extracted, worker);
    process.env.CLAP_LLAMA_WORKER = worker;
    console.error(`[clap] CUDA worker installed at ${worker}`);
  } catch (error) {
    console.error(`[clap] CUDA worker unavailable (${error instanceof Error ? error.message : error}); using bundled CPU worker`);
  } finally {
    await rm(tmpDir, { recursive: true, force: true });
  }
}

function isUsable(path: string): boolean {
  return existsSync(path) && Bun.file(path).size > 0;
}

async function download(url: string, destination: string): Promise<void> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`download failed (${response.status}): ${url}`);
  const total = Number(response.headers.get("content-length") ?? 0);
  if (!response.body || total <= 0 || !process.stderr.isTTY) {
    await Bun.write(destination, response);
    return;
  }

  const writer = Bun.file(destination).writer();
  let received = 0;
  let lastRender = 0;
  const startedAt = Date.now();
  for await (const chunk of response.body) {
    writer.write(chunk);
    received += chunk.byteLength;
    const now = Date.now();
    if (now - lastRender >= 100 || received === total) {
      lastRender = now;
      renderProgress(received, total, now - startedAt);
    }
  }
  await writer.end();
  process.stderr.write("\n");
}

function renderProgress(received: number, total: number, elapsedMs: number, width = 24): void {
  const ratio = Math.min(1, received / total);
  const filled = Math.round(ratio * width);
  const bar = `${"=".repeat(filled)}${"-".repeat(width - filled)}`;
  const percent = Math.floor(ratio * 100).toString().padStart(3, " ");
  const mib = (bytes: number) => (bytes / 1024 / 1024).toFixed(0);
  let stats = "";
  if (elapsedMs >= 500) {
    const rate = received / (elapsedMs / 1000);
    const etaSeconds = Math.max(0, Math.round((total - received) / rate));
    const eta = etaSeconds < 60 ? `${etaSeconds}s` : `${Math.floor(etaSeconds / 60)}m${(etaSeconds % 60).toString().padStart(2, "0")}s`;
    stats = ` ${mib(rate)} MiB/s eta ${eta}`;
  }
  process.stderr.write(`\r[${bar}] ${percent}% ${mib(received)}/${mib(total)} MiB${stats}   `);
}

async function verifySha256(path: string, checksumUrl: string): Promise<void> {
  const response = await fetch(checksumUrl);
  if (!response.ok) return; // checksum asset missing; skip verification
  const expected = (await response.text()).trim().split(/\s+/)[0];
  if (!expected) return;
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(await Bun.file(path).arrayBuffer());
  const actual = hasher.digest("hex");
  if (actual !== expected) throw new Error(`sha256 mismatch for ${path}`);
}
