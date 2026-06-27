import { describe, expect, test } from "bun:test";
import { HealthResponseSchema } from "@clap/api";
import { assertFileCredentialPermissions, deleteStoredHfToken, hfAuthStatus, resolveHfToken, resolveModel, storeHfToken } from "@clap/models";
import { existsSync } from "node:fs";
import { chmod, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createServer, idleTimeoutFromEnv } from "./index";

describe("clap server", () => {
  test("serves health", async () => {
    const response = await createServer().request("/clap/v1/health");
    expect(response.status).toBe(200);
    const body = HealthResponseSchema.parse(await response.json());
    expect(body.status).toBe("ok");
  });

  test("parses server idle timeout env with a long local inference default", () => {
    const previous = process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS;
    try {
      delete process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS;
      expect(idleTimeoutFromEnv()).toBe(240);
      process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS = "900";
      expect(idleTimeoutFromEnv()).toBe(255);
      process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS = "0";
      expect(idleTimeoutFromEnv()).toBe(0);
      process.env.CLAP_SERVER_IDLE_TIMEOUT_SECONDS = "nope";
      expect(idleTimeoutFromEnv()).toBe(240);
    } finally {
      restoreEnv("CLAP_SERVER_IDLE_TIMEOUT_SECONDS", previous);
    }
  });

  test("reports missing bundled native workers as not installed", async () => {
    const previousLlamaWorker = process.env.CLAP_LLAMA_WORKER;
    const previousMlxWorker = process.env.CLAP_MLX_WORKER;
    try {
      process.env.CLAP_LLAMA_WORKER = "/__missing_clap_llama_worker__";
      process.env.CLAP_MLX_WORKER = "/__missing_clap_mlx_worker__";
      const response = await createServer().request("/clap/v1/backends");
      expect(response.status).toBe(200);
      const body = await response.json();
      const llama = body.backends.find((backend: { id: string }) => backend.id === "llama");
      const mlx = body.backends.find((backend: { id: string }) => backend.id === "mlx");
      expect(llama.status).toBe("not_installed");
      expect(llama.reason).toContain("clap-llama worker not found");
      expect(["not_installed", "unsupported"]).toContain(mlx.status);
      expect(mlx.reason).not.toContain("MOCK");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousLlamaWorker);
      restoreEnv("CLAP_MLX_WORKER", previousMlxWorker);
    }
  });

  test("rejects unknown models instead of serving placeholder completions", async () => {
    const response = await createServer().request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "unknown-local",
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    expect(response.status).toBe(404);
    const body = await response.json();
    expect(body.error.code).toBe("not_found");
    expect(body.error.message).toContain("not found in the local cache or aliases");
  });

  test("requires a model for chat completions", async () => {
    const response = await createServer().request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    expect(response.status).toBe(400);
    const body = await response.json();
    expect(body.error.code).toBe("invalid_json");
  });

  test("fails clearly when a local GGUF model needs an unbuilt llama worker", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-gguf-test-"));
    const model = join(dir, "model.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = join(dir, "missing-clap-llama");
      process.env.CLAP_HOME = dir;
      await writeFile(model, "gguf bytes");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "hello gguf" }],
        }),
      });

      expect(response.status).toBe(503);
      const body = await response.json();
      expect(body.error.code).toBe("worker_not_found");
      expect(body.error.message).toContain("clap-llama worker not found");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("fails clearly when a local MLX directory needs an unbuilt MLX worker", async () => {
    const previousWorker = process.env.CLAP_MLX_WORKER;
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-mlx-test-"));
    const model = join(dir, "mlx-model");
    try {
      process.env.CLAP_MLX_WORKER = join(dir, "missing-clap-mlx");
      process.env.CLAP_HOME = dir;
      await mkdir(model);
      await writeFile(join(model, "config.json"), "{}");
      await writeFile(join(model, "tokenizer.json"), "{}");
      await writeFile(join(model, "model.safetensors"), "fake");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "hello mlx" }],
        }),
      });

      expect(response.status).toBe(503);
      const body = await response.json();
      expect(body.error.message).toContain("clap-mlx worker");
    } finally {
      restoreEnv("CLAP_MLX_WORKER", previousWorker);
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("returns a helpful error when a GGUF model path is missing", async () => {
    const response = await createServer().request("/v1/chat/completions", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "/tmp/missing-clap-model.gguf",
        messages: [{ role: "user", content: "hello" }],
      }),
    });

    expect(response.status).toBe(404);
    const body = await response.json();
    expect(body.error.message).toContain("GGUF model not found");
  });

  test("returns OpenAI-compatible tool calls from generated JSON", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-tool-call-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = JSON.stringify({ tool_calls: [{ name: "get_weather", arguments: { city: "Paris" } }] });

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "weather?" }],
          tools: [{ type: "function", function: { name: "get_weather", parameters: { type: "object" } } }],
        }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices[0].finish_reason).toBe("tool_calls");
      expect(body.choices[0].message.content).toBeNull();
      expect(body.choices[0].message.tool_calls[0]).toMatchObject({
        type: "function",
        function: { name: "get_weather", arguments: JSON.stringify({ city: "Paris" }) },
      });
      expect(body.usage.total_tokens).toBeGreaterThan(0);
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("streams tool call deltas for parsed tool calls", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-tool-stream-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = JSON.stringify({ tool_calls: [{ name: "search", arguments: { q: "clap" } }] });

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          stream: true,
          messages: [{ role: "user", content: "search" }],
          tools: [{ type: "function", function: { name: "search" } }],
        }),
      });

      expect(response.status).toBe(200);
      const events = await sseData(response);
      const toolDelta = events.find((event) => event.choices?.[0]?.delta?.tool_calls);
      expect(toolDelta.choices[0].delta.tool_calls[0]).toMatchObject({
        type: "function",
        function: { name: "search", arguments: JSON.stringify({ q: "clap" }) },
      });
      expect(events.at(-1).choices[0].finish_reason).toBe("tool_calls");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("supports json_object and json_schema response formats best effort", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-json-format-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "```json\n{\"ok\":true}\n```";

      const app = createServer();
      const jsonObject = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "json" }], response_format: { type: "json_object" } }),
      });
      expect((await jsonObject.json()).choices[0].message.content).toBe(JSON.stringify({ ok: true }));

      const jsonSchema = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "json" }], response_format: { type: "json_schema", json_schema: { name: "Result", schema: { type: "object" } } } }),
      });
      expect((await jsonSchema.json()).choices[0].message.content).toBe(JSON.stringify({ ok: true }));
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("accepts Vercel AI style content parts and rejects image parts for text runtimes", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-content-parts-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "plain response";
      const app = createServer();

      const textParts = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: [{ type: "text", text: "hello" }] }], seed: 1, top_p: 0.8, presence_penalty: 0, frequency_penalty: 0, stop: ["###"] }),
      });
      expect(textParts.status).toBe(200);

      const imageParts = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: "https://example.com/image.png" } }] }] }),
      });
      expect(imageParts.status).toBe(400);
      expect((await imageParts.json()).error.code).toBe("unsupported_content_part");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("serves Responses API string input and instructions", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-responses-string-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "response text";
      const response = await createServer().request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, instructions: "Be terse", input: "hello", metadata: { trace: "yes" } }),
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.object).toBe("response");
      expect(body.status).toBe("completed");
      expect(body.output_text).toBe("response text");
      expect(body.output[0]).toMatchObject({ type: "message", role: "assistant" });
      expect(body.usage.total_tokens).toBeGreaterThan(0);
      expect(body.metadata.trace).toBe("yes");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("serves Responses API message input, structured JSON, and tool calls", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-responses-tools-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = JSON.stringify({ tool_calls: [{ name: "lookup", arguments: { q: "x" } }] });
      const tool = await createServer().request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          input: [{ role: "user", content: [{ type: "text", text: "lookup" }] }],
          tools: [{ type: "function", function: { name: "lookup" } }],
        }),
      });
      expect(tool.status).toBe(200);
      const toolBody = await tool.json();
      expect(toolBody.output[0]).toMatchObject({ type: "function_call", name: "lookup", arguments: JSON.stringify({ q: "x" }) });
      expect(toolBody.output_text).toBe("");

      process.env.CLAP_FAKE_WORKER_OUTPUT = "```json\n{\"ok\":true}\n```";
      const json = await createServer().request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: "json", text: { format: { type: "json_object" } } }),
      });
      expect((await json.json()).output_text).toBe(JSON.stringify({ ok: true }));
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("streams Responses API text and tool-call events", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-responses-stream-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "stream text";
      const text = await createServer().request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: "hello", stream: true }),
      });
      const textEvents = await responseSseEvents(text);
      expect(textEvents.map((event) => event.event)).toContain("response.output_text.delta");
      expect(textEvents.at(-1)?.event).toBe("response.completed");

      process.env.CLAP_FAKE_WORKER_OUTPUT = JSON.stringify({ tool_calls: [{ name: "lookup", arguments: { q: "x" } }] });
      const tool = await createServer().request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: "lookup", stream: true, tools: [{ type: "function", function: { name: "lookup" } }] }),
      });
      const toolEvents = await responseSseEvents(tool);
      expect(toolEvents.map((event) => event.event)).toContain("response.function_call_arguments.delta");
      expect(toolEvents.at(-1)?.event).toBe("response.completed");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("rejects unsupported Responses API images and stateful continuation, and has no legacy completions route", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-responses-errors-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      const app = createServer();
      const image = await app.request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: [{ role: "user", content: [{ type: "image_url", image_url: { url: "https://example.com/image.png" } }] }] }),
      });
      expect(image.status).toBe(400);
      expect((await image.json()).error.code).toBe("unsupported_content_part");

      const previous = await app.request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: "continue", previous_response_id: "resp_old" }),
      });
      expect(previous.status).toBe(400);
      expect((await previous.json()).error.code).toBe("unsupported_stateful_continuation");

      const completions = await app.request("/v1/completions", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      expect(completions.status).toBe(404);
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("serves Ollama tags, show, chat, generate, and unsupported endpoints", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-ollama-api-test-"));
    const repoDir = join(dir, "models", "huggingface", "acme--ollama-gguf");
    const modelPath = join(repoDir, "model.Q4_K_M.gguf");
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "ollama response";
      await mkdir(repoDir, { recursive: true });
      await writeFile(modelPath, "gguf bytes");
      const app = createServer();

      const tags = await app.request("/api/tags");
      expect(tags.status).toBe(200);
      const tagsBody = await tags.json();
      expect(tagsBody.models.map((entry: { name: string }) => entry.name)).toContain("acme/ollama-gguf");

      const show = await app.request("/api/show", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/ollama-gguf" }),
      });
      expect(show.status).toBe(200);
      expect((await show.json()).details.quantization_level).toBe("Q4_K_M");

      const chat = await app.request("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/ollama-gguf", stream: false, messages: [{ role: "user", content: "hello" }] }),
      });
      expect(chat.status).toBe(200);
      const chatBody = await chat.json();
      expect(chatBody.message.content).toBe("ollama response");
      expect(chatBody.done).toBe(true);

      const generate = await app.request("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/ollama-gguf", stream: false, prompt: "hello" }),
      });
      expect(generate.status).toBe(200);
      expect((await generate.json()).response).toBe("ollama response");

      const unsupported = await app.request("/api/embeddings", { method: "POST", headers: { "content-type": "application/json" }, body: "{}" });
      expect(unsupported.status).toBe(501);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("maps Ollama chat tools and rejects Ollama image inputs", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-ollama-tools-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = JSON.stringify({ tool_calls: [{ name: "lookup", arguments: { q: "x" } }] });
      const app = createServer();

      const chat = await app.request("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          stream: false,
          messages: [{ role: "user", content: "lookup" }],
          tools: [{ type: "function", function: { name: "lookup" } }],
        }),
      });
      expect(chat.status).toBe(200);
      expect((await chat.json()).message.tool_calls[0].function.name).toBe("lookup");

      const image = await app.request("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "describe", images: ["abc"] }] }),
      });
      expect(image.status).toBe(400);
      expect((await image.json()).error).toContain("image input is not supported");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("pulls models through the Ollama pull endpoint", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-ollama-pull-test-"));
    const hf = mockHuggingFaceServer({
      "acme/ollama-pull": { "model.Q4_K_M.gguf": "gguf bytes" },
    });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const response = await createServer().request("/api/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: "acme/ollama-pull", stream: false }),
      });
      expect(response.status).toBe(200);
      expect((await response.json()).status).toBe("success");
      expect(existsSync(join(dir, "models", "huggingface", "acme--ollama-pull", "model.Q4_K_M.gguf"))).toBe(true);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("uses stored Hugging Face token when env tokens are absent", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const previousBackend = process.env.CLAP_HF_AUTH_BACKEND;
    const previousToken = process.env.CLAP_HF_TOKEN;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-stored-auth-test-"));
    const seenAuth: string[] = [];
    const hf = mockHuggingFaceServer({
      "acme/stored-gguf": {
        "stored.Q4_K_M.gguf": "stored bytes",
      },
    }, { seenAuth });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      process.env.CLAP_HF_AUTH_BACKEND = "file";
      delete process.env.CLAP_HF_TOKEN;
      await storeHfToken("stored-token");

      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/stored-gguf", file: "stored.Q4_K_M.gguf" }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      await waitForDownload(body.download.id, "completed");
      expect(seenAuth).toEqual(["Bearer stored-token", "Bearer stored-token"]);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      restoreEnv("CLAP_HF_AUTH_BACKEND", previousBackend);
      restoreEnv("CLAP_HF_TOKEN", previousToken);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("prefers Hugging Face env token over stored credential", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousBackend = process.env.CLAP_HF_AUTH_BACKEND;
    const previousToken = process.env.CLAP_HF_TOKEN;
    const dir = await mkdtemp(join(tmpdir(), "clap-auth-priority-test-"));
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_AUTH_BACKEND = "file";
      process.env.CLAP_HF_TOKEN = "env-token";
      await storeHfToken("stored-token");

      const resolved = await resolveHfToken();
      expect(resolved.token).toBe("env-token");
      expect(resolved.source).toBe("env");
      expect(resolved.envVar).toBe("CLAP_HF_TOKEN");
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_AUTH_BACKEND", previousBackend);
      restoreEnv("CLAP_HF_TOKEN", previousToken);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("file-backed Hugging Face auth uses restrictive permissions, redacts status, and logs out", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousBackend = process.env.CLAP_HF_AUTH_BACKEND;
    const dir = await mkdtemp(join(tmpdir(), "clap-auth-file-test-"));
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_AUTH_BACKEND = "file";
      const stored = await storeHfToken("hf_abcdefghijklmnopqrstuvwxyz");
      expect(stored.source).toBe("file");
      expect(stored.tokenPreview).toBe("hf_abc...wxyz");

      const permissions = assertFileCredentialPermissions();
      expect(permissions.fileMode).toBe(0o600);
      expect(permissions.dirMode).toBe(0o700);

      const status = await hfAuthStatus();
      expect(status.source).toBe("file");
      expect(status.tokenPreview).toBe("hf_abc...wxyz");
      expect(JSON.stringify(status)).not.toContain("abcdefghijklmnopqrstuvwxyz");

      await deleteStoredHfToken();
      const afterLogout = await hfAuthStatus();
      expect(afterLogout.authenticated).toBe(false);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_AUTH_BACKEND", previousBackend);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("sends Hugging Face authorization token to info and download requests", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const previousToken = process.env.CLAP_HF_TOKEN;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-auth-test-"));
    const seenAuth: string[] = [];
    const hf = mockHuggingFaceServer({
      "acme/private-gguf": {
        "private.Q4_K_M.gguf": "private bytes",
      },
    }, { seenAuth });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      process.env.CLAP_HF_TOKEN = "test-token";

      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/private-gguf", file: "private.Q4_K_M.gguf" }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      await waitForDownload(body.download.id, "completed");
      expect(seenAuth).toEqual(["Bearer test-token", "Bearer test-token"]);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      restoreEnv("CLAP_HF_TOKEN", previousToken);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("returns a clear Hugging Face auth error for private repos", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-auth-error-test-"));
    const hf = mockHuggingFaceServer({
      "acme/private-gguf": {
        "private.Q4_K_M.gguf": "private bytes",
      },
    }, { requireAuth: true });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;

      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/private-gguf", file: "private.Q4_K_M.gguf" }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      const download = await waitForDownload(body.download.id, "failed");
      expect(download.error).toContain("Hugging Face authentication failed (401)");
      expect(download.error).toContain("CLAP_HF_TOKEN");
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("pulls a GGUF file into the Hugging Face cache", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-gguf-test-"));
    const hf = mockHuggingFaceServer({
      "acme/tiny-gguf": {
        "tiny.Q4_K_M.gguf": "mock gguf bytes",
      },
    });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;

      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/tiny-gguf", file: "tiny.Q4_K_M.gguf" }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      const download = await waitForDownload(body.download.id, "completed");
      expect(download.status).toBe("completed");
      expect(download.modelPath).toEndWith("tiny.Q4_K_M.gguf");

      const models = await createServer().request("/clap/v1/models");
      const modelsBody = await models.json();
      const model = modelsBody.models.find((entry: { id: string }) => entry.id === "acme/tiny-gguf");
      expect(model.backend).toBe("llama");
      expect(model.localPath).toBe(download.modelPath);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("pulls an MLX repo directory into the Hugging Face cache", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-mlx-test-"));
    const hf = mockHuggingFaceServer({
      "acme/tiny-mlx": {
        "config.json": "{}",
        "tokenizer.json": "{}",
        "model.safetensors": "mock weights",
      },
    });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;

      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/tiny-mlx" }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      const download = await waitForDownload(body.download.id, "completed");
      expect(download.status).toBe("completed");
      expect(download.modelPath).toEndWith("acme--tiny-mlx");

      const downloads = await createServer().request("/clap/v1/downloads");
      const downloadsBody = await downloads.json();
      expect(downloadsBody.downloads.some((download: { id: string; status: string }) => download.id === body.download.id && download.status === "completed")).toBe(true);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("reuses an existing running pull for the same resolved target", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-dedupe-test-"));
    const requestCounts: Record<string, number> = {};
    const hf = mockHuggingFaceServer({
      "acme/dedupe-gguf": {
        "dedupe.Q4_K_M.gguf": "first chunk second chunk",
      },
    }, { chunkDelayMs: 150, requestCounts });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const app = createServer();

      const first = await app.request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/dedupe-gguf", file: "dedupe.Q4_K_M.gguf" }),
      });
      const second = await app.request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/dedupe-gguf", file: "dedupe.Q4_K_M.gguf" }),
      });

      expect(first.status).toBe(200);
      expect(second.status).toBe(200);
      const firstBody = await first.json();
      const secondBody = await second.json();
      expect(secondBody.download.id).toBe(firstBody.download.id);
      await waitForDownload(firstBody.download.id, "completed");
      expect(requestCounts["GET /acme/dedupe-gguf/resolve/main/dedupe.Q4_K_M.gguf"]).toBe(1);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("skips cached pulls unless force is set", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-cache-skip-test-"));
    const requestCounts: Record<string, number> = {};
    const hf = mockHuggingFaceServer({
      "acme/cached-gguf": {
        "cached.Q4_K_M.gguf": "cached bytes",
      },
    }, { requestCounts });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const app = createServer();
      const body = JSON.stringify({ model: "acme/cached-gguf", file: "cached.Q4_K_M.gguf" });

      const first = await app.request("/clap/v1/models/pull", { method: "POST", headers: { "content-type": "application/json" }, body });
      const firstBody = await first.json();
      await waitForDownload(firstBody.download.id, "completed");
      expect(requestCounts["GET /acme/cached-gguf/resolve/main/cached.Q4_K_M.gguf"]).toBe(1);

      const cached = await app.request("/clap/v1/models/pull", { method: "POST", headers: { "content-type": "application/json" }, body });
      const cachedBody = await cached.json();
      expect(cachedBody.download.status).toBe("completed");
      expect(cachedBody.download.modelPath).toEndWith("cached.Q4_K_M.gguf");
      expect(requestCounts["GET /api/models/acme/cached-gguf"]).toBe(1);
      expect(requestCounts["GET /acme/cached-gguf/resolve/main/cached.Q4_K_M.gguf"]).toBe(1);

      const forced = await app.request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/cached-gguf", file: "cached.Q4_K_M.gguf", force: true }),
      });
      const forcedBody = await forced.json();
      await waitForDownload(forcedBody.download.id, "completed");
      expect(requestCounts["GET /acme/cached-gguf/resolve/main/cached.Q4_K_M.gguf"]).toBe(2);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("cancels a running pull and removes partial files", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-cancel-test-"));
    const hf = mockHuggingFaceServer({
      "acme/cancel-gguf": {
        "cancel.Q4_K_M.gguf": "first chunk second chunk",
      },
    }, { chunkDelayMs: 500 });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const app = createServer();

      const response = await app.request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/cancel-gguf", file: "cancel.Q4_K_M.gguf" }),
      });
      const body = await response.json();
      await waitForDownloadProgress(body.download.id);

      const cancel = await app.request(`/clap/v1/downloads/${body.download.id}/cancel`, { method: "POST" });
      expect(cancel.status).toBe(200);
      const cancelBody = await cancel.json();
      expect(cancelBody.download.status).toBe("cancelled");
      const download = await waitForDownload(body.download.id, "cancelled");
      expect(download.status).toBe("cancelled");
      expect(existsSync(join(dir, "models", "huggingface", "acme--cancel-gguf", "cancel.Q4_K_M.gguf"))).toBe(false);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("updates download progress before a chunked pull completes", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-pull-progress-test-"));
    const hf = mockHuggingFaceServer({
      "acme/chunked-gguf": {
        "chunked.Q4_K_M.gguf": "first chunk second chunk",
      },
    }, { chunkDelayMs: 150 });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;

      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/chunked-gguf", file: "chunked.Q4_K_M.gguf" }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      const running = await waitForDownloadProgress(body.download.id);
      expect(running.status).toBe("running");
      expect(running.bytesReceived).toBeGreaterThan(0);
      expect(running.totalBytes).toBe("first chunk second chunk".length);
      expect(running.currentFile).toBe("chunked.Q4_K_M.gguf");

      const completed = await waitForDownload(body.download.id, "completed");
      expect(completed.bytesReceived).toBe("first chunk second chunk".length);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("keeps not-downloaded aliases out of model inventory", async () => {
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-alias-list-test-"));
    try {
      process.env.CLAP_HOME = dir;
      const response = await createServer().request("/clap/v1/models");
      expect(response.status).toBe(200);
      const body = await response.json();
      const alias = body.models.find((model: { id: string }) => model.id === "qwen2.5:3b");
      expect(alias).toBeUndefined();

      const aliases = await createServer().request("/clap/v1/aliases");
      expect(aliases.status).toBe(200);
      const aliasesBody = await aliases.json();
      const listedAlias = aliasesBody.models.find((model: { id: string }) => model.id === "qwen2.5:3b");
      expect(listedAlias.status).toBe("not_downloaded");
      expect(listedAlias.pull.model).toBe("qwen2.5:3b");
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("keeps nonexistent configured model paths out of model inventory", async () => {
    const previousGgufPaths = process.env.CLAP_GGUF_MODEL_PATHS;
    const previousMlxPaths = process.env.CLAP_MLX_MODEL_PATHS;
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-local-list-test-"));
    const gguf = join(dir, "local.gguf");
    const mlx = join(dir, "local-mlx");
    try {
      process.env.CLAP_HOME = join(dir, "home");
      await writeFile(gguf, "gguf bytes");
      await mkdir(mlx);
      process.env.CLAP_GGUF_MODEL_PATHS = `${join(dir, "missing.gguf")},${gguf}`;
      process.env.CLAP_MLX_MODEL_PATHS = `${join(dir, "missing-mlx")},${mlx}`;

      const response = await createServer().request("/clap/v1/models");
      expect(response.status).toBe(200);
      const body = await response.json();
      const ids = body.models.map((model: { id: string }) => model.id);
      expect(ids).toContain(gguf);
      expect(ids).toContain(mlx);
      expect(ids).not.toContain(join(dir, "missing.gguf"));
      expect(ids).not.toContain(join(dir, "missing-mlx"));
    } finally {
      restoreEnv("CLAP_GGUF_MODEL_PATHS", previousGgufPaths);
      restoreEnv("CLAP_MLX_MODEL_PATHS", previousMlxPaths);
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("resolves an explicit cached Hugging Face MLX repo id", async () => {
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-explicit-mlx-resolve-test-"));
    const repoDir = join(dir, "models", "huggingface", "acme--explicit-mlx");
    try {
      process.env.CLAP_HOME = dir;
      await mkdir(repoDir, { recursive: true });
      await writeFile(join(repoDir, "config.json"), "{}");
      await writeFile(join(repoDir, "tokenizer.json"), "{}");
      await writeFile(join(repoDir, "model.safetensors"), "weights");

      const resolved = resolveModel("acme/explicit-mlx", "mlx");

      expect(resolved.status).toBe("available");
      expect(resolved.backend).toBe("mlx");
      expect(resolved.format).toBe("mlx");
      expect(resolved.modelPath).toBe(repoDir);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("lists cached Hugging Face MLX repo ids instead of cache paths", async () => {
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-cached-mlx-list-test-"));
    const repoDir = join(dir, "models", "huggingface", "mlx-community--gemma-4-e4b-it-4bit");
    try {
      process.env.CLAP_HOME = dir;
      await mkdir(repoDir, { recursive: true });
      await writeFile(join(repoDir, "config.json"), JSON.stringify({
        architectures: ["Gemma4ForConditionalGeneration"],
        model_type: "gemma4",
        text_config: { max_position_embeddings: 131072 },
        vision_config: { image_size: 896 },
        audio_config: { sampling_rate: 16000 },
        quantization: { bits: 4, group_size: 64 },
      }));
      await writeFile(join(repoDir, "tokenizer_config.json"), JSON.stringify({ model_max_length: 131072 }));
      await writeFile(join(repoDir, "model.safetensors"), "weights");

      const clapResponse = await createServer().request("/clap/v1/models");
      expect(clapResponse.status).toBe(200);
      const clapBody = await clapResponse.json();
      const model = clapBody.models.find((entry: { id: string }) => entry.id === "mlx-community/gemma-4-e4b-it-4bit");
      expect(model).toBeDefined();
      expect(model.localPath).toBe(repoDir);
      expect(model.name).toBe("gemma 4 e4b it 4bit");
      expect(model.provider).toBe("mlx-community");
      expect(model.source).toMatchObject({ type: "huggingface", repo: "mlx-community/gemma-4-e4b-it-4bit" });
      expect(model.modalities).toEqual({ input: ["text"], output: ["text"] });
      expect(model.capabilities).toMatchObject({ chat: true, streaming: true, attachment: false });
      expect(model.limit).toEqual({ context: 131072, output: null });
      expect(model.upstream.modalities.input).toEqual(["text", "image", "audio"]);
      expect(model.upstream.capabilities.attachment).toBe(true);
      expect(model.upstream.limit.context).toBe(131072);
      expect(model.architecture).toBe("Gemma4ForConditionalGeneration");
      expect(model.modelType).toBe("gemma4");
      expect(model.quantization).toBe("4-bit group 64");
      expect(clapBody.models.map((entry: { id: string }) => entry.id)).not.toContain(repoDir);

      const openAiResponse = await createServer().request("/v1/models");
      expect(openAiResponse.status).toBe(200);
      const openAiBody = await openAiResponse.json();
      expect(openAiBody.data.map((entry: { id: string }) => entry.id)).toContain("mlx-community/gemma-4-e4b-it-4bit");
      expect(openAiBody.data.map((entry: { id: string }) => entry.id)).not.toContain(repoDir);
      const openAiModel = openAiBody.data.find((entry: { id: string }) => entry.id === "mlx-community/gemma-4-e4b-it-4bit");
      expect(Object.keys(openAiModel).sort()).toEqual(["created", "id", "object", "owned_by"]);

      const richOpenAiResponse = await createServer().request("/v1/models?metadata=1");
      expect(richOpenAiResponse.status).toBe(200);
      const richOpenAiBody = await richOpenAiResponse.json();
      const richModel = richOpenAiBody.data.find((entry: { id: string }) => entry.id === "mlx-community/gemma-4-e4b-it-4bit");
      expect(richOpenAiBody.object).toBe("list");
      expect(richModel).toMatchObject({
        id: "mlx-community/gemma-4-e4b-it-4bit",
        object: "model",
        created: 0,
        owned_by: "clap",
        backend: "mlx",
        format: "mlx",
        status: "available",
        localPath: repoDir,
        provider: "mlx-community",
        source: { type: "huggingface", repo: "mlx-community/gemma-4-e4b-it-4bit" },
        modalities: { input: ["text"], output: ["text"] },
        limit: { context: 131072, output: null },
      });
      expect(richModel.capabilities).toMatchObject({ chat: true, streaming: true, attachment: false });
      expect(richModel.upstream.modalities.input).toEqual(["text", "image", "audio"]);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("resolves an explicit cached Hugging Face GGUF repo id when unambiguous", async () => {
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-explicit-gguf-resolve-test-"));
    const repoDir = join(dir, "models", "huggingface", "acme--explicit-gguf");
    const modelPath = join(repoDir, "model.Q4_K_M.gguf");
    try {
      process.env.CLAP_HOME = dir;
      await mkdir(repoDir, { recursive: true });
      await writeFile(modelPath, "gguf bytes");

      const resolved = resolveModel("acme/explicit-gguf", "gguf");

      expect(resolved.status).toBe("available");
      expect(resolved.backend).toBe("llama");
      expect(resolved.format).toBe("gguf");
      expect(resolved.modelPath).toBe(modelPath);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("lists cached Hugging Face GGUF repo ids and disambiguates multiple files", async () => {
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-cached-gguf-list-test-"));
    const singleRepo = join(dir, "models", "huggingface", "acme--single-gguf");
    const singlePath = join(singleRepo, "model.Q4_K_M.gguf");
    const multiRepo = join(dir, "models", "huggingface", "acme--multi-gguf");
    const firstPath = join(multiRepo, "first.Q4_K_M.gguf");
    const secondPath = join(multiRepo, "second.Q4_K_M.gguf");
    try {
      process.env.CLAP_HOME = dir;
      await mkdir(singleRepo, { recursive: true });
      await writeFile(singlePath, "gguf bytes");
      await mkdir(multiRepo, { recursive: true });
      await writeFile(firstPath, "gguf bytes");
      await writeFile(secondPath, "gguf bytes");

      const response = await createServer().request("/clap/v1/models");
      expect(response.status).toBe(200);
      const body = await response.json();
      const ids = body.models.map((entry: { id: string }) => entry.id);
      expect(ids).toContain("acme/single-gguf");
      expect(ids).toContain("acme/multi-gguf:first.Q4_K_M.gguf");
      expect(ids).toContain("acme/multi-gguf:second.Q4_K_M.gguf");
      expect(ids).not.toContain(singlePath);
      expect(ids).not.toContain(firstPath);
      const single = body.models.find((entry: { id: string }) => entry.id === "acme/single-gguf");
      expect(single.localPath).toBe(singlePath);
      expect(single.modalities).toEqual({ input: ["text"], output: ["text"] });
      expect(single.capabilities).toMatchObject({ chat: true, streaming: true, attachment: false });
      expect(single.limit).toEqual({ context: null, output: null });
      expect(single.quantization).toBe("Q4_K_M");

      const openAiResponse = await createServer().request("/v1/models");
      expect(openAiResponse.status).toBe(200);
      const openAiBody = await openAiResponse.json();
      const openAiIds = openAiBody.data.map((entry: { id: string }) => entry.id);
      expect(openAiIds).toContain("acme/single-gguf");
      expect(openAiIds).toContain("acme/multi-gguf:first.Q4_K_M.gguf");
      expect(openAiIds).not.toContain(singlePath);

      const richOpenAiResponse = await createServer().request("/v1/models?metadata=true");
      expect(richOpenAiResponse.status).toBe(200);
      const richOpenAiBody = await richOpenAiResponse.json();
      const richSingle = richOpenAiBody.data.find((entry: { id: string }) => entry.id === "acme/single-gguf");
      expect(richSingle).toMatchObject({
        id: "acme/single-gguf",
        object: "model",
        created: 0,
        owned_by: "clap",
        backend: "llama",
        format: "gguf",
        status: "available",
        localPath: singlePath,
        quantization: "Q4_K_M",
        capabilities: { chat: true, streaming: true, attachment: false },
      });
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("loads, lists, expires, and unloads runtime model entries", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-lifecycle-load-test-"));
    const repoDir = join(dir, "models", "huggingface", "acme--lifecycle-gguf");
    const modelPath = join(repoDir, "model.Q4_K_M.gguf");
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      await mkdir(repoDir, { recursive: true });
      await writeFile(modelPath, "gguf bytes");
      const app = createServer();

      const load = await app.request("/clap/v1/models/load", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/lifecycle-gguf", backend: "gguf", keepAlive: "30s" }),
      });
      expect(load.status).toBe(200);
      const loadBody = await load.json();
      expect(loadBody.model).toMatchObject({
        id: "acme/lifecycle-gguf",
        backend: "llama",
        format: "gguf",
        localPath: modelPath,
        state: "warm",
        activeRequests: 0,
        keepAlive: "30s",
      });
      expect(loadBody.model.expiresAt).toBeString();
      expect(loadBody.model.worker.state).toBe("resident");
      expect(loadBody.model.worker.pid).toBeNumber();

      const active = await app.request("/clap/v1/runtime/models");
      const activeBody = await active.json();
      expect(activeBody.models.map((entry: { id: string }) => entry.id)).toContain("acme/lifecycle-gguf");
      expect(activeBody.models[0].worker.pid).toBe(loadBody.model.worker.pid);

      const unload = await app.request("/clap/v1/models/unload", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/lifecycle-gguf", backend: "gguf" }),
      });
      const unloadBody = await unload.json();
      expect(unloadBody.unloaded).toBe(true);

      const empty = await app.request("/clap/v1/runtime/models");
      expect((await empty.json()).models).toEqual([]);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("chat requests reuse the same resident worker pid and extend keep-alive", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-lifecycle-chat-test-"));
    const modelPath = join(dir, "local.Q4_K_M.gguf");
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "resident ok";
      await writeFile(modelPath, "gguf bytes");
      const app = createServer();

      const response = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: modelPath,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      expect(response.status).toBe(200);
      expect((await response.json()).choices[0].message.content).toBe("resident ok");

      const active = await app.request("/clap/v1/runtime/models");
      const activeBody = await active.json();
      expect(activeBody.models[0]).toMatchObject({
        id: modelPath,
        backend: "llama",
        localPath: modelPath,
        state: "warm",
        activeRequests: 0,
        worker: { state: "resident" },
      });
      const firstPid = activeBody.models[0].worker.pid;
      const firstExpiry = activeBody.models[0].expiresAt;

      await Bun.sleep(5);
      const second = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "again" }] }),
      });
      expect(second.status).toBe(200);
      const after = await app.request("/clap/v1/runtime/models");
      const afterBody = await after.json();
      expect(afterBody.models[0].worker.pid).toBe(firstPid);
      expect(Date.parse(afterBody.models[0].expiresAt)).toBeGreaterThanOrEqual(Date.parse(firstExpiry));
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("routes cached Hugging Face repo ids to local worker paths", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousWorker = process.env.CLAP_MLX_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-cached-mlx-route-test-"));
    const repoDir = join(dir, "models", "huggingface", "acme--route-mlx");
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_MLX_WORKER = join(dir, "missing-clap-mlx");
      await mkdir(repoDir, { recursive: true });
      await writeFile(join(repoDir, "config.json"), "{}");
      await writeFile(join(repoDir, "tokenizer.json"), "{}");
      await writeFile(join(repoDir, "model.safetensors"), "weights");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "acme/route-mlx",
          backend: "mlx",
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      expect(response.status).toBe(503);
      const body = await response.json();
      expect(body.error.message).toContain("clap-mlx worker not found");
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_MLX_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("pulls an alias using the GGUF backend override", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-alias-gguf-test-"));
    const hf = mockHuggingFaceServer({
      "bartowski/Llama-3.2-3B-Instruct-GGUF": {
        "Llama-3.2-3B-Instruct-Q4_K_M.gguf": "mock alias gguf",
      },
    });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;

      const pull = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "llama3.2:3b", backend: "gguf" }),
      });
      expect(pull.status).toBe(200);
      const body = await pull.json();
      const download = await waitForDownload(body.download.id, "completed");
      expect(download.modelPath).toEndWith("Llama-3.2-3B-Instruct-Q4_K_M.gguf");
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("returns a clear pull instruction when an alias is not cached", async () => {
    const previousHome = process.env.CLAP_HOME;
    const dir = await mkdtemp(join(tmpdir(), "clap-alias-missing-test-"));
    try {
      process.env.CLAP_HOME = dir;
      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "qwen2.5:3b",
          backend: "gguf",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      expect(response.status).toBe(404);
      const body = await response.json();
      expect(body.error.message).toContain("clap pull qwen2.5:3b");
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("returns a platform fallback suggestion for unsupported MLX aliases", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousPlatform = process.env.CLAP_TEST_PLATFORM;
    const dir = await mkdtemp(join(tmpdir(), "clap-alias-platform-test-"));
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_TEST_PLATFORM = "linux-x64";
      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "qwen2.5:3b",
          backend: "mlx",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      expect(response.status).toBe(400);
      const body = await response.json();
      expect(body.error.message).toContain("--backend gguf");
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_TEST_PLATFORM", previousPlatform);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("serves OpenAPI JSON", async () => {
    const response = await createServer().request("/openapi.json");
    expect(response.status).toBe(200);
    const body = await response.json();
    expect(body.openapi).toBe("3.0.0");
  });
});

function restoreEnv(name: string, value: string | undefined) {
  if (value === undefined) {
    delete process.env[name];
    return;
  }
  process.env[name] = value;
}

async function waitForDownload(id: string, status: "completed" | "failed" | "cancelled") {
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    const response = await createServer().request("/clap/v1/downloads");
    const body = await response.json();
    const download = body.downloads.find((entry: { id: string }) => entry.id === id);
    if (download?.status === status) return download;
    await Bun.sleep(10);
  }
  throw new Error(`download ${id} did not become ${status}`);
}

async function waitForDownloadProgress(id: string) {
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    const response = await createServer().request("/clap/v1/downloads");
    const body = await response.json();
    const download = body.downloads.find((entry: { id: string }) => entry.id === id);
    if (download?.status === "running" && download.bytesReceived > 0) return download;
    await Bun.sleep(10);
  }
  throw new Error(`download ${id} did not report in-progress bytes`);
}

async function fakeWorker(dir: string): Promise<string> {
  const path = join(dir, "fake-clap-worker");
  await writeFile(path, `#!/usr/bin/env bun
const decoder = new TextDecoder();
let buffer = "";
for await (const chunk of Bun.stdin.stream()) {
  buffer += decoder.decode(chunk, { stream: true });
  let newline;
  while ((newline = buffer.indexOf("\\n")) >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (!line) continue;
    const request = JSON.parse(line);
    if (request.type === "shutdown") {
      console.log(JSON.stringify({ id: request.id, done: true }));
      process.exit(0);
    }
    if (request.type === "load") {
      console.log(JSON.stringify({ id: request.id, loaded: true, done: true }));
      continue;
    }
    if (request.type === "unload") {
      console.log(JSON.stringify({ id: request.id, unloaded: true, done: true }));
      continue;
    }
    console.log(JSON.stringify({ id: request.id, content: process.env.CLAP_FAKE_WORKER_OUTPUT ?? "ok" }));
    console.log(JSON.stringify({ id: request.id, done: true }));
  }
}
`);
  await chmod(path, 0o755);
  return path;
}

async function sseData(response: Response): Promise<any[]> {
  const text = await response.text();
  return text
    .split("\n")
    .filter((line) => line.startsWith("data: "))
    .map((line) => line.slice(6))
    .filter((line) => line !== "[DONE]")
    .map((line) => JSON.parse(line));
}

async function responseSseEvents(response: Response): Promise<Array<{ event: string; data: unknown }>> {
  const text = await response.text();
  return text.trim().split("\n\n").filter(Boolean).map((block) => {
    const event = block.split("\n").find((line) => line.startsWith("event: "))?.slice(7) ?? "message";
    const data = JSON.parse(block.split("\n").find((line) => line.startsWith("data: "))?.slice(6) ?? "{}");
    return { event, data };
  });
}

function mockHuggingFaceServer(repos: Record<string, Record<string, string>>, options: { requireAuth?: boolean; seenAuth?: string[]; chunkDelayMs?: number; requestCounts?: Record<string, number> } = {}) {
  return Bun.serve({
    port: 0,
    fetch(request) {
      const url = new URL(request.url);
      const countKey = `${request.method} ${url.pathname}`;
      if (options.requestCounts) options.requestCounts[countKey] = (options.requestCounts[countKey] ?? 0) + 1;
      options.seenAuth?.push(request.headers.get("authorization") ?? "");
      if (options.requireAuth && request.headers.get("authorization") !== "Bearer test-token") {
        return Response.json({ error: "authentication required" }, { status: 401 });
      }
      const modelMatch = url.pathname.match(/^\/api\/models\/([^/]+\/[^/]+)$/);
      if (modelMatch) {
        const repo = modelMatch[1]!;
        const files = repos[repo];
        if (!files) return Response.json({ error: "not found" }, { status: 404 });
        return Response.json({
          siblings: Object.entries(files).map(([rfilename, content]) => ({ rfilename, size: content.length })),
        });
      }

      const resolveMatch = url.pathname.match(/^\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/);
      if (resolveMatch) {
        const repo = resolveMatch[1]!;
        const file = decodeURIComponent(resolveMatch[2]!);
        const content = repos[repo]?.[file];
        if (content === undefined) return new Response("not found", { status: 404 });
        if (options.chunkDelayMs) return new Response(chunkedBody(content, options.chunkDelayMs));
        return new Response(content);
      }

      return new Response("not found", { status: 404 });
    },
  });
}

function chunkedBody(content: string, delayMs: number) {
  const encoder = new TextEncoder();
  const midpoint = Math.max(1, Math.floor(content.length / 2));
  let cancelled = false;
  return new ReadableStream({
    start(controller) {
      controller.enqueue(encoder.encode(content.slice(0, midpoint)));
      setTimeout(() => {
        if (cancelled) return;
        controller.enqueue(encoder.encode(content.slice(midpoint)));
        controller.close();
      }, delayMs);
    },
    cancel() {
      cancelled = true;
    },
  });
}
