#!/usr/bin/env bun
import { copyFile, mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";

const root = resolve(import.meta.dir, "..");
const packagePath = join(root, "native/mlx");
const metallib = join(root, "libexec/mlx.metallib");
if (!(await Bun.file(metallib).exists())) {
  console.error("MLX physical tests require libexec/mlx.metallib; run bun run runtime:mlx:build");
  process.exit(1);
}

await run(["cargo", "build", "--release", "--locked", "-p", "clap-cache-ffi",
  "--manifest-path", "native/cache/Cargo.toml"]);
await run(["swift", "test", "--package-path", packagePath, "list"]);

const bundles = new Bun.Glob("**/ClapMLXPackageTests.xctest/Contents/MacOS");
let installed = 0;
for await (const relative of bundles.scan({ cwd: join(packagePath, ".build"), onlyFiles: false })) {
  const macos = join(packagePath, ".build", relative);
  const resources = join(dirname(macos), "Resources");
  await mkdir(resources, { recursive: true });
  for (const directory of [macos, resources]) {
    await copyFile(metallib, join(directory, "mlx.metallib"));
    await copyFile(metallib, join(directory, "default.metallib"));
  }
  installed += 1;
}
if (installed === 0) {
  console.error("SwiftPM did not produce the MLX test bundle");
  process.exit(1);
}
await run(["swift", "test", "--package-path", packagePath,
  "--filter", "ClapMLXPhysicalIntegrationTests.ModelProbeTests.realModel"]);

async function run(command: string[]) {
  const child = Bun.spawn(command, { cwd: root, env: process.env,
    stdout: "inherit", stderr: "inherit" });
  const exit = await child.exited;
  if (exit !== 0) process.exit(exit);
}
