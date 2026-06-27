#!/usr/bin/env bun
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { join } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const vendor = join(root, "vendor");
const dir = join(vendor, "llama.cpp");
const repo = process.env.CLAP_LLAMA_CPP_REPO ?? "https://github.com/ggerganov/llama.cpp.git";
const ref = process.env.CLAP_LLAMA_CPP_REF ?? "master";

await mkdir(vendor, { recursive: true });

if (!existsSync(dir)) {
  await run(["git", "clone", "--depth", "1", "--branch", ref, repo, dir]);
} else {
  await run(["git", "-C", dir, "fetch", "--depth", "1", "origin", ref]);
  await run(["git", "-C", dir, "checkout", "FETCH_HEAD"]);
}
await run(["git", "-C", dir, "submodule", "update", "--init", "--recursive", "--depth", "1"]);

console.log(`llama.cpp vendored at ${dir}`);

async function run(command: string[]) {
  const proc = Bun.spawn(command, { stdout: "inherit", stderr: "inherit" });
  const code = await proc.exited;
  if (code !== 0) throw new Error(`${command.join(" ")} exited ${code}`);
}
