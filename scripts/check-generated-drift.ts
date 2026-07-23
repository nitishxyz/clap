#!/usr/bin/env bun
const root = new URL("..", import.meta.url).pathname;
const generatedFiles = [
  "packages/api/openapi.json",
  "packages/server/src/web-assets.generated.ts",
];
const diff = Bun.spawnSync(["git", "diff", "--exit-code", "--", ...generatedFiles], {
  cwd: root,
  stdout: "inherit",
  stderr: "inherit",
});

if (diff.exitCode !== 0) {
  console.error("Generated artifacts are stale. Commit the output from generate:openapi and build:web.");
  process.exit(diff.exitCode);
}

console.log("generated artifacts match their committed versions");
