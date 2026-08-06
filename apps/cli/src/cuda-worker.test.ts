import { afterEach, describe, expect, test } from "bun:test";
import { hasExplicitWorker } from "./cuda-worker";

const originalWorker = process.env.CLAP_LLAMA_WORKER;
const originalBundledWorker = process.env.CLAP_BUNDLED_LLAMA_WORKER;

afterEach(() => {
  if (originalWorker === undefined) delete process.env.CLAP_LLAMA_WORKER;
  else process.env.CLAP_LLAMA_WORKER = originalWorker;
  if (originalBundledWorker === undefined) delete process.env.CLAP_BUNDLED_LLAMA_WORKER;
  else process.env.CLAP_BUNDLED_LLAMA_WORKER = originalBundledWorker;
});

describe("CUDA worker override classification", () => {
  test("allows the provisioner to replace the internally extracted CPU worker", () => {
    process.env.CLAP_LLAMA_WORKER = "/tmp/clap/libexec/build/clap-llama";
    process.env.CLAP_BUNDLED_LLAMA_WORKER = process.env.CLAP_LLAMA_WORKER;
    expect(hasExplicitWorker()).toBe(false);
  });

  test("preserves an operator-configured worker", () => {
    process.env.CLAP_LLAMA_WORKER = "/opt/clap/custom-worker";
    process.env.CLAP_BUNDLED_LLAMA_WORKER = "/tmp/clap/libexec/build/clap-llama";
    expect(hasExplicitWorker()).toBe(true);
  });
});
