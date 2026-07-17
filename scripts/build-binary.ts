#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { chmod, copyFile, mkdir, rm } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const libexec = join(root, "libexec");
const dist = join(root, "dist");
const binary = join(dist, "clap");
const entrypoint = join(root, "apps", "cli", "src", "index.ts");

const workers = [
  { name: "clap-llama", required: true },
  { name: "clap-mlx", required: process.platform === "darwin" && process.arch === "arm64" },
  { name: "mlx.metallib", required: process.platform === "darwin" && process.arch === "arm64" },
];

const missing = workers.filter((worker) => worker.required && !existsSync(join(libexec, worker.name)));
if (missing.length > 0) {
  console.error(`missing native workers in ${libexec}: ${missing.map((worker) => worker.name).join(", ")}`);
  console.error("build them first: bun run native:build");
  process.exit(1);
}

await rm(dist, { recursive: true, force: true });
await mkdir(join(dist, "libexec"), { recursive: true });

const build = Bun.spawn(["bun", "build", "--compile", "--minify", entrypoint, "--outfile", binary], {
  cwd: root,
  stdout: "inherit",
  stderr: "inherit",
});
if (await build.exited !== 0) process.exit(1);
await chmod(binary, 0o755);

for (const worker of workers) {
  const source = join(libexec, worker.name);
  if (!existsSync(source)) continue;
  const target = join(dist, "libexec", worker.name);
  await copyFile(source, target);
  if (!worker.name.endsWith(".metallib")) await chmod(target, 0o755);
  console.log(`bundled libexec/${worker.name}`);
}

const size = Bun.file(binary).size;
console.log(`built ${binary} (${(size / 1024 / 1024).toFixed(1)} MiB)`);
console.log("install: cp -R dist/clap dist/libexec /usr/local/bin (keep clap and libexec/ side by side)");
