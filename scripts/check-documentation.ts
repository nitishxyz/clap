#!/usr/bin/env bun
import { stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";

const root = new URL("..", import.meta.url).pathname;
const tracked = Bun.spawnSync(["git", "ls-files", "-z", "--", "*.md"], { cwd: root });
if (tracked.exitCode !== 0) {
  process.stderr.write(tracked.stderr);
  process.exit(tracked.exitCode);
}

const markdownFiles = tracked.stdout.toString().split("\0").filter(Boolean);
const packageJson = await Bun.file(resolve(root, "package.json")).json() as {
  scripts?: Record<string, string>;
};
const scripts = new Set(Object.keys(packageJson.scripts ?? {}));
const failures: string[] = [];

for (const relativeFile of markdownFiles) {
  const absoluteFile = resolve(root, relativeFile);
  const contents = await Bun.file(absoluteFile).text();

  for (const match of contents.matchAll(/!?(?:\[[^\]]*\])\(([^)]+)\)/g)) {
    const rawTarget = match[1]?.trim() ?? "";
    const target = rawTarget.startsWith("<") && rawTarget.endsWith(">")
      ? rawTarget.slice(1, -1)
      : rawTarget.split(/\s+["']/u, 1)[0] ?? "";
    if (!target || /^(?:https?:|mailto:|#)/u.test(target)) continue;

    const path = decodeURIComponent(target.split(/[?#]/u, 1)[0] ?? "");
    if (!path) continue;
    try {
      await stat(resolve(dirname(absoluteFile), path));
    } catch {
      failures.push(`${relativeFile}: broken local link ${JSON.stringify(target)}`);
    }
  }

  for (const match of contents.matchAll(/\bbun run\s+([a-zA-Z0-9_.:-]+)/g)) {
    const script = match[1];
    if (script && !scripts.has(script)) {
      failures.push(`${relativeFile}: unknown package script ${JSON.stringify(script)}`);
    }
  }
}

if (failures.length > 0) {
  console.error("Documentation validation failed:");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exit(1);
}

console.log(`validated ${markdownFiles.length} tracked Markdown files`);
