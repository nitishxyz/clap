#!/usr/bin/env bun
// The CLI statically imports libexec worker binaries as embedded assets.
// Fresh checkouts (CI, new clones) have no built workers yet, which breaks
// module resolution for every test that loads the CLI. Create zero-byte
// placeholders for missing workers; extraction skips empty files at runtime.
import { existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const libexec = join(root, "libexec");
await mkdir(libexec, { recursive: true });

for (const name of ["clap-llama", "clap-mlx", "mlx.metallib"]) {
  const path = join(libexec, name);
  if (!existsSync(path)) {
    await writeFile(path, "");
    console.log(`created placeholder ${path}`);
  }
}
