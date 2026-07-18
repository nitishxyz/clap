#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const vendor = join(root, "vendor");
const dir = join(vendor, "llama.cpp");
const repo = process.env.CLAP_LLAMA_CPP_REPO ?? "https://github.com/ggerganov/llama.cpp.git";
// Pinned so vendored sources (and cached build objects in CI) stay stable
// across releases; bump deliberately and test.
const ref = process.env.CLAP_LLAMA_CPP_REF ?? "0ed235ea2c17a19fc8238668653946721ed136fd";

await mkdir(vendor, { recursive: true });

if (!existsSync(dir)) {
  await run(["git", "init", dir]);
  await run(["git", "-C", dir, "remote", "add", "origin", repo]);
  await run(["git", "-C", dir, "fetch", "--depth", "1", "origin", ref]);
  await run(["git", "-C", dir, "checkout", "FETCH_HEAD"]);
} else {
  const head = Bun.spawnSync(["git", "-C", dir, "rev-parse", "HEAD"]).stdout.toString().trim();
  if (head !== ref) {
    await run(["git", "-C", dir, "fetch", "--depth", "1", "origin", ref]);
    await run(["git", "-C", dir, "checkout", "FETCH_HEAD"]);
  } else {
    console.log(`llama.cpp already at pinned ref ${ref}`);
  }
}
await run(["git", "-C", dir, "submodule", "update", "--init", "--recursive", "--depth", "1"]);

console.log(`llama.cpp vendored at ${dir}`);

async function run(command: string[]) {
  const proc = Bun.spawn(command, { stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) throw new Error(`${command.join(" ")} exited ${code}`);
}
