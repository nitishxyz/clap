#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { chmod, mkdir } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const llamaDir = join(root, "vendor", "llama.cpp");
const buildDir = join(llamaDir, "build");
const libexec = join(root, "libexec");
const wrapperSource = join(root, "native", "llama", "clap-llama.cpp");
const wrapperOutput = join(libexec, "clap-llama");

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
  "-DLLAMA_BUILD_EXAMPLES=OFF",
  "-DLLAMA_CURL=OFF",
  "-DGGML_NATIVE=ON",
];

if (process.platform === "darwin") {
  configure.push("-DGGML_METAL=ON");
  if (process.arch === "arm64") configure.push("-DCMAKE_OSX_ARCHITECTURES=arm64");
}

await run(configure);
await run(["cmake", "--build", buildDir, "--config", "Release", "--target", "llama", "-j"]);

await mkdir(libexec, { recursive: true });

const cxx = process.env.CXX ?? "c++";
const compile = [
  cxx,
  "-std=c++17",
  "-O2",
  wrapperSource,
  "-I", join(llamaDir, "include"),
  "-I", join(llamaDir, "ggml", "include"),
  "-I", join(llamaDir, "ggml", "src"),
  "-L", join(buildDir, "src"),
  "-L", join(buildDir, "ggml", "src"),
  "-L", join(buildDir, "ggml", "src", "ggml-cpu"),
  "-lllama",
  "-lggml",
  "-lggml-base",
  "-lggml-cpu",
];

if (process.platform === "darwin") {
  compile.push(
    "-L", join(buildDir, "ggml", "src", "ggml-metal"),
    "-L", join(buildDir, "ggml", "src", "ggml-blas"),
    "-lggml-metal",
    "-lggml-blas",
    "-framework", "Foundation",
    "-framework", "Metal",
    "-framework", "MetalKit",
    "-framework", "Accelerate",
  );
}

await run([...compile, "-o", wrapperOutput]);
await chmod(wrapperOutput, 0o755);

console.log(`built ${wrapperOutput}`);

async function run(command: string[]) {
  const proc = Bun.spawn(command, { stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) throw new Error(`${command.join(" ")} exited ${code}`);
}
