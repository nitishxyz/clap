#!/usr/bin/env bun

const root = new URL("..", import.meta.url).pathname;
const sourceRoots = [
  "apps",
  "packages",
  "native/llama/include",
  "native/llama/src",
  "native/mlx/Sources",
];
const binaryRoot = "libexec";
const markers = [
  ["cache", "model", "probe"].join("-"),
  ["cache", "model", "probe"].join("_"),
  ["cache", "model", "probe"].join("."),
  ["model", "probe", "observation"].join("_"),
  ["modelProbe", "Observation"].join(""),
  ["top16", "quantized", "logit", "sha256"].join("_"),
  ["top16Quantized", "LogitSHA256"].join(""),
  ["CLAP", "TEST", "GGUF", "MODEL"].join("_"),
  ["CLAP", "TEST", "MLX", "MODEL"].join("_"),
];
const sourceGlob = new Bun.Glob("**/*.{c,cc,cpp,h,hpp,json,swift,ts,tsx}");
const failures: string[] = [];

for (const sourceRoot of sourceRoots) {
  for await (const relative of sourceGlob.scan({ cwd: `${root}/${sourceRoot}`, onlyFiles: true })) {
    const path = `${sourceRoot}/${relative}`;
    const content = await Bun.file(`${root}/${path}`).text();
    for (const marker of markers) {
      if (content.toLowerCase().includes(marker.toLowerCase())) {
        failures.push(`${path}: leaked test probe marker ${marker}`);
      }
    }
  }
}

const binaryGlob = new Bun.Glob("**/*");
for await (const relative of binaryGlob.scan({ cwd: `${root}/${binaryRoot}`, onlyFiles: true })) {
  const path = `${binaryRoot}/${relative}`;
  const content = new TextDecoder("latin1").decode(await Bun.file(`${root}/${path}`).arrayBuffer());
  for (const marker of markers) {
    if (content.toLowerCase().includes(marker.toLowerCase())) {
      failures.push(`${path}: production binary contains test probe marker ${marker}`);
    }
  }
}

if (failures.length > 0) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("ok: test-only model probes are absent from production surfaces");
