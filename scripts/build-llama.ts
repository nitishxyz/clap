#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { chmod, mkdir } from "node:fs/promises";
import { availableParallelism } from "node:os";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const llamaDir = join(root, "vendor", "llama.cpp");
const sourceDir = join(root, "native", "llama");
const buildDir = join(root, "build", "llama");
const libexec = join(root, "libexec");
const wrapperOutput = join(libexec, "clap-llama");

if (!existsSync(llamaDir)) {
  console.error("llama.cpp is not vendored. Run: bun run runtime:llama:vendor");
  process.exit(1);
}

const configure = [
  "cmake",
  "-S", sourceDir,
  "-B", buildDir,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DGGML_NATIVE=ON",
];

if (process.platform === "darwin") {
  configure.push("-DGGML_METAL=ON");
  if (process.arch === "arm64") configure.push("-DCMAKE_OSX_ARCHITECTURES=arm64");
}

// Linux: enable CUDA automatically when the toolkit is present (RunPod and
// similar GPU boxes) unless explicitly disabled with CLAP_CUDA=0.
const cudaHome = process.env.CUDA_HOME ?? process.env.CUDA_PATH ?? "/usr/local/cuda";
const wantCuda = process.platform === "linux"
  && process.env.CLAP_CUDA !== "0"
  && (existsSync(join(cudaHome, "bin", "nvcc")) || Bun.which("nvcc") !== null);
if (wantCuda) {
  configure.push("-DGGML_CUDA=ON");
  const cudaArchs = process.env.CLAP_CUDA_ARCHS;
  if (cudaArchs) configure.push(`-DCMAKE_CUDA_ARCHITECTURES=${cudaArchs}`);
  console.log(`CUDA toolkit detected (${cudaHome}); building GPU-enabled worker`);
} else if (process.platform === "linux") {
  console.log("no CUDA toolkit detected; building CPU-only worker (set CUDA_HOME if nvcc is elsewhere)");
}

await run(configure);
const buildJobs = process.env.CLAP_BUILD_JOBS ?? String(availableParallelism());
await run(["cmake", "--build", buildDir, "--config", "Release", "--target", "clap-llama", "-j", buildJobs]);

await mkdir(libexec, { recursive: true });
await run([
  "cmake", "--install", buildDir, "--config", "Release",
  "--component", "runtime", "--prefix", libexec,
]);
await chmod(wrapperOutput, 0o755);

console.log(`built ${wrapperOutput}`);

async function run(command: string[], cwd = root) {
  const proc = Bun.spawn(command, { cwd, stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) throw new Error(`${command.join(" ")} exited ${code}`);
}
