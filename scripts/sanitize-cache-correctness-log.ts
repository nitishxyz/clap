#!/usr/bin/env bun
import { readFile, writeFile } from "node:fs/promises";

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: sanitize-cache-correctness-log.ts <input> <output>");
  process.exit(1);
}
let content = await readFile(input, "utf8");
for (const value of [process.env.CLAP_CACHE_TEST_ASSET_ROOT,
  process.env.CLAP_CACHE_TEST_GGUF_MODEL, process.env.CLAP_CACHE_TEST_MLX_MODEL]) {
  if (value) content = content.replaceAll(value, "<asset-path>");
}
content = content
  .replace(/(?:\/Users|\/Volumes|\/private|\/tmp)\/[^\s:"']+/g, "<path>")
  .replace(/((?:token|secret|password|credential|api[_-]?key)\s*[=:]\s*)[^\s]+/gi, "$1<redacted>");
await writeFile(output, content);
