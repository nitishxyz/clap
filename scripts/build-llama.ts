#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { chmod, copyFile, mkdir } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const llamaDir = join(root, "vendor", "llama.cpp");
const buildDir = join(llamaDir, "build");
const libexec = join(root, "libexec");
const wrapperSource = join(root, "native", "llama", "clap-llama.cpp");
const wrapperOutput = join(libexec, "clap-llama");
const llamaCliOutput = join(libexec, "llama-cli");

if (!existsSync(llamaDir)) {
  console.error("llama.cpp is not vendored. Run: bun run runtime:llama:vendor");
  process.exit(1);
}

const configure = [
  "cmake",
  "-S", llamaDir,
  "-B", buildDir,
  "-DCMAKE_BUILD_TYPE=Release",
  "-DBUILD_SHARED_LIBS=OFF",
  "-DLLAMA_BUILD_TESTS=OFF",
  "-DLLAMA_BUILD_EXAMPLES=ON",
  "-DLLAMA_CURL=OFF",
  "-DGGML_NATIVE=ON",
];

if (process.platform === "darwin") {
  configure.push("-DGGML_METAL=ON");
  if (process.arch === "arm64") configure.push("-DCMAKE_OSX_ARCHITECTURES=arm64");
}

await run(configure);
await run(["cmake", "--build", buildDir, "--config", "Release", "--target", "llama-cli", "-j"]);

const llamaCli = findFirst([
  join(buildDir, "bin", "llama-cli"),
  join(buildDir, "examples", "main", "llama-cli"),
  join(buildDir, "bin", "Release", "llama-cli"),
]);
if (!llamaCli) {
  console.error(`llama-cli was not produced under ${buildDir}`);
  process.exit(1);
}

await mkdir(libexec, { recursive: true });
await copyFile(llamaCli, llamaCliOutput);
await chmod(llamaCliOutput, 0o755);

const cxx = process.env.CXX ?? "c++";
await run([cxx, "-std=c++17", "-O2", wrapperSource, "-o", wrapperOutput]);
await chmod(wrapperOutput, 0o755);

console.log(`built ${wrapperOutput}`);
console.log(`bundled ${llamaCliOutput}`);

function findFirst(paths: string[]): string | undefined {
  return paths.find((path) => existsSync(path));
}

async function run(command: string[]) {
  const proc = Bun.spawn(command, { stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) throw new Error(`${command.join(" ")} exited ${code}`);
}
