import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { chmod, mkdtemp, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { completeWithMlx, isMlxModelDirectory, MlxWorkerError } from "./index";

const envKeys = ["CLAP_MLX_WORKER", "CLAP_MLX_MODEL_PATHS", "CLAP_HOME"] as const;
const previousEnv = new Map<string, string | undefined>();

beforeEach(() => {
  for (const key of envKeys) previousEnv.set(key, process.env[key]);
});

afterEach(() => {
  for (const key of envKeys) {
    const value = previousEnv.get(key);
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
  previousEnv.clear();
});

describe("MLX runtime", () => {
  test("validates that MLX directories include config, tokenizer, and safetensors", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-model-"));
    await writeFile(join(dir, "config.json"), "{}");
    await writeFile(join(dir, "tokenizer.json"), "{}");

    expect(await isMlxModelDirectory(dir)).toBe(false);
  });

  test("surfaces worker JSON errors with exit code and stderr log path", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-worker-error-"));
    const model = await writeMlxModel(dir);
    const worker = join(dir, "worker.sh");
    await writeFile(worker, "#!/bin/sh\necho '{\"error\":\"unsupported model_type gemma4\"}'\nexit 7\n");
    await chmod(worker, 0o755);
    await writeFile(join(dir, "mlx.metallib"), "fake");
    process.env.CLAP_MLX_WORKER = worker;
    process.env.CLAP_HOME = dir;

    await expect(completeWithMlx({
      request: { model, messages: [{ role: "user", content: "hello" }], stream: false },
      stream: false,
    })).rejects.toThrow(/unsupported model_type gemma4.*exitCode=7.*stderrLog=/);
  });

  test("surfaces non-JSON stdout diagnostics from worker crashes", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-worker-stdout-"));
    const model = await writeMlxModel(dir);
    const worker = join(dir, "worker.sh");
    await writeFile(worker, "#!/bin/sh\necho 'MLX error: Failed to load the default metallib. library not found'\nexit 255\n");
    await chmod(worker, 0o755);
    await writeFile(join(dir, "mlx.metallib"), "fake");
    process.env.CLAP_MLX_WORKER = worker;
    process.env.CLAP_HOME = dir;

    await expect(completeWithMlx({
      request: { model, messages: [{ role: "user", content: "hello" }], stream: false },
      stream: false,
    })).rejects.toThrow(/clap-mlx exited with code 255.*Failed to load the default metallib/);
  });

  test("ignores stdout diagnostics when the worker exits successfully", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-worker-success-diagnostic-"));
    const model = await writeMlxModel(dir);
    const worker = join(dir, "worker.sh");
    await writeFile(worker, "#!/bin/sh\necho '{\"content\":\"hi\"}'\necho '{\"done\":true}'\necho 'No chat template was included'\nexit 0\n");
    await chmod(worker, 0o755);
    await writeFile(join(dir, "mlx.metallib"), "fake");
    process.env.CLAP_MLX_WORKER = worker;
    process.env.CLAP_HOME = dir;

    await expect(completeWithMlx({
      request: { model, messages: [{ role: "user", content: "hello" }], stream: false },
      stream: false,
    })).resolves.toBe("hi");
  });

  test("reports missing metallib before launching the worker", async () => {
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-no-metal-"));
    const model = await writeMlxModel(dir);
    const worker = join(dir, "worker.sh");
    await writeFile(worker, "#!/bin/sh\necho should-not-run\n");
    await chmod(worker, 0o755);
    process.env.CLAP_MLX_WORKER = worker;
    process.env.CLAP_HOME = dir;

    await expect(completeWithMlx({
      request: { model, messages: [{ role: "user", content: "hello" }], stream: false },
      stream: false,
    })).rejects.toThrow(/Metal shader library not found/);
  });
});

async function writeMlxModel(dir: string): Promise<string> {
  const model = join(dir, "model");
  await mkdir(model);
  await writeFile(join(model, "config.json"), JSON.stringify({ model_type: "llama" }));
  await writeFile(join(model, "tokenizer.json"), "{}");
  await writeFile(join(model, "model.safetensors"), "fake");
  return model;
}
