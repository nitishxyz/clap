#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { chmod, mkdir, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { createNativeBundleManifest, nativeBundleBuildId } from "../packages/runtime-router/src/native-bundle";

const root = new URL("..", import.meta.url).pathname;
const libexec = join(root, "libexec");
const dist = join(root, "dist");
const entrypoint = join(root, "apps", "cli", "src", "index.ts");

// Optional cross-compile target, e.g. --target bun-linux-x64. The embedded
// workers come from libexec/, so cross builds require workers compiled for the
// target platform to be placed there first (see scripts/build-linux.sh).
const targetFlagIndex = process.argv.indexOf("--target");
const target = targetFlagIndex >= 0 ? process.argv[targetFlagIndex + 1] : undefined;
const targetIsDarwinArm = target ? target.includes("darwin-arm64") || target.includes("darwin-aarch64") : process.platform === "darwin" && process.arch === "arm64";
const binary = join(dist, target ? `clap-${target.replace(/^bun-/, "")}` : "clap");

const workers = [
  { name: "clap-llama", required: true, executable: true },
  { name: "clap-mlx", required: targetIsDarwinArm, executable: true },
  { name: "mlx.metallib", required: targetIsDarwinArm, executable: false },
];

const missing = workers.filter((worker) => worker.required && !existsSync(join(libexec, worker.name)));
if (missing.length > 0) {
  console.error(`missing native workers in ${libexec}: ${missing.map((worker) => worker.name).join(", ")}`);
  console.error("build them first: bun run native:build");
  process.exit(1);
}

// The CLI statically imports every worker asset; provide empty placeholders for
// optional workers so cross-platform builds still compile (extraction skips
// zero-byte files and the runtime reports the backend as unavailable).
for (const worker of workers) {
  const source = join(libexec, worker.name);
  if (!existsSync(source)) await writeFile(source, "");
}

// Content-addressed build id so extracted workers in ~/.clap/libexec/<id>/ are
// refreshed exactly when the embedded workers change.
const manifest = await createNativeBundleManifest(workers.map((worker) => ({
  name: worker.name,
  data: Bun.file(join(libexec, worker.name)),
  executable: worker.executable,
})));
const embedBuild = await nativeBundleBuildId(manifest);
const embeddedManifest = JSON.stringify(manifest);

if (!target) await rm(dist, { recursive: true, force: true });
await mkdir(dist, { recursive: true });

const build = Bun.spawn(
  [
    "bun", "build", "--compile", "--minify", entrypoint,
    ...(target ? ["--target", target] : []),
    "--define", `process.env.CLAP_EMBED_BUILD="${embedBuild}"`,
    "--define", `process.env.CLAP_EMBED_MANIFEST=${JSON.stringify(embeddedManifest)}`,
    "--outfile", binary,
  ],
  { cwd: root, stdout: "inherit", stderr: "inherit" },
);
if (await build.exited !== 0) process.exit(1);
await chmod(binary, 0o755);

const size = Bun.file(binary).size;
console.log(`built ${binary} (${(size / 1024 / 1024).toFixed(1)} MiB, embed ${embedBuild})`);
console.log("single self-contained binary: native workers are embedded and extracted to ~/.clap/libexec on first run");
console.log("install: cp dist/clap /usr/local/bin/clap");
