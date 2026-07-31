#!/usr/bin/env bun
import { readFile, writeFile } from "node:fs/promises";
import { sanitizeBenchmarkText } from "./inference-benchmark-lib";

const [input, output] = process.argv.slice(2);
if (!input || !output) {
  console.error("usage: sanitize-inference-benchmark-log.ts <input> <output>");
  process.exit(1);
}
const content = await readFile(input, "utf8");
await writeFile(output, sanitizeBenchmarkText(content, [
  process.env.CLAP_API_KEY, process.env.CLAP_HF_TOKEN, process.env.HF_TOKEN,
  process.env.HUGGINGFACE_HUB_TOKEN, process.env.HUGGINGFACE_TOKEN,
]));
