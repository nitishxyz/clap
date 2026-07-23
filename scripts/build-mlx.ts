#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { chmod, copyFile, mkdir, readdir, rm } from "node:fs/promises";
import { dirname, join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const packageDir = join(root, "native", "mlx");
const cacheDir = join(root, "native", "cache");
const libexec = join(root, "libexec");
const buildConfig = process.env.CLAP_SWIFT_CONFIGURATION ?? "release";
const binary = join(packageDir, ".build", "arm64-apple-macosx", buildConfig, "clap-mlx");

if (process.platform !== "darwin" || process.arch !== "arm64") {
  console.error("clap-mlx can only be built on macOS arm64 with Xcode/Swift installed.");
  process.exit(1);
}

await run(["cargo", "build", "--release", "-p", "clap-cache-ffi"], cacheDir);
// SwiftPM does not track changes to the Rust static archive passed through
// unsafe linker flags. Remove the final product so every Rust cache rebuild
// necessarily relinks clap-mlx instead of shipping stale coordinator code.
await rm(binary, { force: true });
await run(["swift", "build", "--package-path", packageDir, "-c", buildConfig]);

if (!existsSync(binary)) {
  console.error(`swift build completed but ${binary} does not exist`);
  process.exit(1);
}

await mkdir(libexec, { recursive: true });
await copyFile(binary, join(libexec, "clap-mlx"));
await chmod(join(libexec, "clap-mlx"), 0o755);
await copyOrBuildMetalLibrary(binary, libexec);
console.log(`built ${join(libexec, "clap-mlx")}`);

async function run(command: string[], cwd = root) {
  const proc = Bun.spawn(command, { cwd, stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) throw new Error(`${command.join(" ")} exited ${code}`);
}

async function copyOrBuildMetalLibrary(binary: string, libexec: string) {
  const binaryDir = dirname(binary);
  const candidates = [
    join(binaryDir, "mlx.metallib"),
    join(binaryDir, "Resources", "mlx.metallib"),
    join(binaryDir, "Resources", "default.metallib"),
  ];
  const generated = join(packageDir, ".build", "mlx-metallib", "mlx.metallib");
  const source = candidates.find((candidate) => existsSync(candidate)) ?? await buildMetalLibrary(generated);
  if (!source) {
    throw new Error("built clap-mlx but did not find or build mlx.metallib");
  }
  await copyFile(source, join(libexec, "mlx.metallib"));
  console.log(`copied ${join(libexec, "mlx.metallib")}`);
}

async function buildMetalLibrary(output: string): Promise<string | undefined> {
  const sourceDir = join(packageDir, ".build", "checkouts", "mlx-swift", "Source", "Cmlx", "mlx-generated", "metal");
  if (!existsSync(sourceDir)) return undefined;

  const outputDir = dirname(output);
  await rm(outputDir, { recursive: true, force: true });
  await mkdir(outputDir, { recursive: true });

  const metalSources = await listMetalSources(sourceDir);
  if (metalSources.length === 0) return undefined;

  const airFiles: string[] = [];
  for (const source of metalSources) {
    const air = join(outputDir, `${source.replaceAll("/", "_").replace(/\.metal$/, "")}.air`);
    await run([
      "xcrun", "-sdk", "macosx", "metal",
      "-x", "metal",
      "-Wall",
      "-Wextra",
      "-fno-fast-math",
      "-Wno-c++17-extensions",
      "-Wno-c++20-extensions",
      "-c", join(sourceDir, source),
      `-I${sourceDir}`,
      "-o", air,
    ]);
    airFiles.push(air);
  }

  await run(["xcrun", "-sdk", "macosx", "metallib", ...airFiles, "-o", output]);
  return output;
}

async function listMetalSources(dir: string, prefix = ""): Promise<string[]> {
  const entries = await readdir(join(dir, prefix), { withFileTypes: true });
  const files = await Promise.all(entries.map(async (entry) => {
    const relative = join(prefix, entry.name);
    if (entry.isDirectory()) return listMetalSources(dir, relative);
    return entry.isFile() && entry.name.endsWith(".metal") ? [relative] : [];
  }));
  return files.flat().sort();
}
