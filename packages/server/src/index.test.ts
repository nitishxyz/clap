import { describe, expect, test } from "bun:test";
import { HealthResponseSchema } from "@clap/api";
import { assertFileCredentialPermissions, deleteStoredHfToken, hfAuthStatus, pullModel, removeModel, resolveHfToken, resolveModel, storeHfToken } from "@clap/models";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { parseAssistantOutput, selectParser } from "./chat-compat";
import { createServer, idleTimeoutFromEnv, inferParserFamilies } from "./index";

// Tests mutate fake model cache directories directly; disable the model list
// memo so every request observes the current directory state.
process.env.CLAP_MODEL_LIST_TTL_MS = "0";

const mlxSupported = process.platform === "darwin" && process.arch === "arm64";

describe("clap server", () => {
  test("serves health", async () => {
    const response = await createServer().request("/clap/v1/health");
    expect(response.status).toBe(200);
    const body = HealthResponseSchema.parse(await response.json());
    expect(body.status).toBe("ok");
  });

  test("serves dashboard metrics json", async () => {
    const response = await createServer().request("/clap/v1/dashboard");
    expect(response.status).toBe(200);
    const body = await response.json() as Record<string, unknown>;
    expect(body.server).toMatchObject({ platform: process.platform });
    expect(body.totals).toMatchObject({ requests: 0, ok: 0, errors: 0, cacheHits: 0 });
    expect(Array.isArray(body.active)).toBe(true);
    expect(Array.isArray(body.requests)).toBe(true);
    expect(Array.isArray(body.loaded)).toBe(true);
    expect(Array.isArray(body.models)).toBe(true);
    expect(Array.isArray(body.gpus)).toBe(true);
  });

  test("dashboard SSE stream emits a payload and stops on abort", async () => {
    const app = createServer();
    const controller = new AbortController();
    const response = await app.request("/clap/v1/dashboard/stream?interval=500", { signal: controller.signal });
    expect(response.status).toBe(200);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    const reader = response.body!.getReader();
    const { value } = await reader.read();
    const text = new TextDecoder().decode(value);
    expect(text).toContain("event: dashboard");
    expect(text).toContain('"totals"');
    controller.abort();
    await reader.cancel().catch(() => undefined);
  });

  test("serves the embedded dashboard ui", async () => {
    const app = createServer();
    const response = await app.request("/");
    expect([200, 503]).toContain(response.status);
    if (response.status === 200) {
      expect(response.headers.get("content-type")).toContain("text/html");
      expect(await response.text()).toContain("<div id=\"root\">");
    }
    const redirect = await app.request("/dashboard");
    expect([301, 302]).toContain(redirect.status);
  });

  test("warm-on-boot loads pinned models from config", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const home = await mkdtemp(join(tmpdir(), "clap-warm-boot-test-"));
    const dir = await mkdtemp(join(tmpdir(), "clap-warm-boot-model-"));
    const modelPath = join(dir, "pinned.Q4_K_M.gguf");
    try {
      process.env.CLAP_HOME = home;
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      await writeFile(modelPath, "gguf bytes");
      await writeFile(join(home, "clap.toml"), [
        `[models.${JSON.stringify(modelPath)}]`,
        "pinned = true",
      ].join("\n"));

      const app = createServer();
      // warm-on-boot runs on a deferred timer
      await Bun.sleep(300);
      const loaded = await app.request("/clap/v1/runtime/models");
      const body = await loaded.json() as { models: Array<{ id: string; pinned: boolean; keepAlive: string; worker: { state: string } }> };
      const entry = body.models.find((model) => model.id === modelPath);
      expect(entry).toBeDefined();
      expect(entry?.pinned).toBe(true);
      expect(entry?.keepAlive).toBe("always");
      expect(entry?.worker.state).toBe("resident");
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(home, { recursive: true, force: true });
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("PATCH /clap/v1/config writes valid TOML, round-trips, and live-applies auth", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousRequire = process.env.CLAP_REQUIRE_API_KEY;
    const home = await mkdtemp(join(tmpdir(), "clap-config-write-test-"));
    try {
      process.env.CLAP_HOME = home;
      delete process.env.CLAP_REQUIRE_API_KEY;
      await writeFile(join(home, "clap.toml"), '[llama]\nslots = 4\n');
      const app = createServer();

      const patched = await app.request("/clap/v1/config", {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          auth: { require_api_key: true },
          llama: { kv_type: "q8_0" },
          models: { "owner/big-GGUF": { max_session_ctx: 32768 } },
        }),
      });
      expect(patched.status).toBe(200);
      const body = await patched.json() as { config: { llama: { slots?: number; kv_type?: string } } };
      // merged: existing slots preserved, new kv_type added
      expect(body.config.llama.slots).toBe(4);
      expect(String(body.config.llama.kv_type)).toBe("q8_0");

      // The written file is valid TOML and round-trips through the parser
      const written = await readFile(join(home, "clap.toml"), "utf8");
      const reparsed = Bun.TOML.parse(written) as { models: Record<string, { max_session_ctx: number }> };
      expect(reparsed.models["owner/big-GGUF"].max_session_ctx).toBe(32768);

      // auth.require_api_key applied live: next request denied without a key
      const denied = await app.request("/clap/v1/models");
      expect(denied.status).toBe(401);

      // invalid patch rejected with 400 and file untouched
      const invalid = await app.request("/clap/v1/config", {
        method: "PATCH",
        headers: { "content-type": "application/json", "x-api-key": "unused" },
        body: JSON.stringify({ llama: { kv_type: "q2_bogus" } }),
      });
      expect([400, 401]).toContain(invalid.status);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_REQUIRE_API_KEY", previousRequire);
      await rm(home, { recursive: true, force: true });
    }
  });

  test("clap.toml config: llama env mapping, per-model overrides, auth requirement, config endpoint", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousSlots = process.env.CLAP_LLAMA_SLOTS;
    const previousKvType = process.env.CLAP_LLAMA_KV_TYPE;
    const previousRequire = process.env.CLAP_REQUIRE_API_KEY;
    const home = await mkdtemp(join(tmpdir(), "clap-config-test-"));
    try {
      process.env.CLAP_HOME = home;
      delete process.env.CLAP_LLAMA_KV_TYPE;
      delete process.env.CLAP_REQUIRE_API_KEY;
      process.env.CLAP_LLAMA_SLOTS = "7";
      await writeFile(join(home, "clap.toml"), [
        "[server]",
        "port = 12999",
        "[auth]",
        "require_api_key = true",
        "[llama]",
        'kv_type = "q8_0"',
        "slots = 32",
        '[models."owner/big-GGUF"]',
        "context = 65536",
        'kv_type = "q4_0"',
      ].join("\n"));

      const { loadClapConfig, configPaths } = await import("./config");
      expect(configPaths().at(-1)).toBe(join(home, "clap.toml"));
      const { config } = loadClapConfig();
      expect(config.server.port).toBe(12999);
      expect(String(config.llama.kv_type)).toBe("q8_0");

      const app = createServer();
      // config file applied kv_type (env unset), but env wins for slots
      expect(process.env.CLAP_LLAMA_KV_TYPE ?? "").toBe("q8_0");
      expect(process.env.CLAP_LLAMA_SLOTS ?? "").toBe("7");

      // require_api_key = true denies unauthenticated requests even loopback
      const denied = await app.request("/clap/v1/models");
      expect(denied.status).toBe(401);
      const health = await app.request("/clap/v1/health");
      expect(health.status).toBe(200);

      // env CLAP_REQUIRE_API_KEY=0-style override: unset config effect
      process.env.CLAP_REQUIRE_API_KEY = "0";
      const allowed = await app.request("/clap/v1/models");
      expect(allowed.status).toBe(200);

      const configResponse = await app.request("/clap/v1/config");
      expect(configResponse.status).toBe(200);
      const body = await configResponse.json() as { config: { models: Record<string, { context?: number }> }; sources: Array<{ loaded: boolean }> };
      expect(body.config.models["owner/big-GGUF"].context).toBe(65536);
      expect(body.sources.some((s) => s.loaded)).toBe(true);

      const { workerEnvForModel } = await import("./config");
      const env = workerEnvForModel(config, "owner/big-GGUF");
      expect(env).toEqual({ CLAP_LLAMA_CONTEXT: "65536", CLAP_LLAMA_KV_TYPE: "q4_0" });
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_LLAMA_SLOTS", previousSlots);
      restoreEnv("CLAP_LLAMA_KV_TYPE", previousKvType);
      restoreEnv("CLAP_REQUIRE_API_KEY", previousRequire);
      await rm(home, { recursive: true, force: true });
    }
  });

  test("api keys: create, list, enforce with CLAP_REQUIRE_API_KEY, revoke", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousRequire = process.env.CLAP_REQUIRE_API_KEY;
    const home = await mkdtemp(join(tmpdir(), "clap-keys-test-"));
    try {
      process.env.CLAP_HOME = home;
      const app = createServer();

      const created = await app.request("/clap/v1/keys", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name: "ci" }),
      });
      expect(created.status).toBe(201);
      const key = await created.json() as { id: string; key: string };
      expect(key.key).toStartWith("clap_sk_");

      const listed = await app.request("/clap/v1/keys");
      const listBody = await listed.json() as { keys: Array<{ id: string; name: string }> };
      expect(listBody.keys).toHaveLength(1);
      expect(JSON.stringify(listBody)).not.toContain("sha256");

      process.env.CLAP_REQUIRE_API_KEY = "1";
      const denied = await app.request("/clap/v1/models");
      expect(denied.status).toBe(401);
      expect(((await denied.json()) as { error: { code: string } }).error.code).toBe("invalid_api_key");

      const health = await app.request("/clap/v1/health");
      expect(health.status).toBe(200);

      const allowed = await app.request("/clap/v1/models", {
        headers: { authorization: `Bearer ${key.key}` },
      });
      expect(allowed.status).toBe(200);

      const revoked = await app.request(`/clap/v1/keys/${key.id}`, {
        method: "DELETE",
        headers: { authorization: `Bearer ${key.key}` },
      });
      expect(revoked.status).toBe(200);

      const deniedAfterRevoke = await app.request("/clap/v1/models", {
        headers: { authorization: `Bearer ${key.key}` },
      });
      expect(deniedAfterRevoke.status).toBe(401);
    } finally {
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_REQUIRE_API_KEY", previousRequire);
      await rm(home, { recursive: true, force: true });
    }
  });

  test("prometheus /metrics exposes counters, queue gauges, and histograms", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const dir = await mkdtemp(join(tmpdir(), "clap-prom-test-"));
    const modelPath = join(dir, "prom.Q4_K_M.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      await writeFile(modelPath, "gguf bytes");
      const app = createServer();

      const chat = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "hey" }] }),
      });
      expect(chat.status).toBe(200);

      const response = await app.request("/metrics");
      expect(response.status).toBe(200);
      expect(response.headers.get("content-type")).toContain("text/plain");
      const body = await response.text();
      expect(body).toContain('clap_requests_total{status="ok"} 1');
      expect(body).toContain("clap_queue_inflight 0");
      expect(body).toContain("clap_queue_inflight_limit ");
      expect(body).toContain("clap_request_duration_ms_count 1");
      expect(body).toContain('clap_request_duration_ms_bucket{le="+Inf"} 1');
      expect(body).toContain('clap_tokens_total{kind="completion"}');
      expect(body).toContain("# TYPE clap_request_ttft_ms histogram");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("saturated queue answers 429 with Retry-After before model work", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousTokens = process.env.CLAP_FAKE_WORKER_TOKENS;
    const previousInflight = process.env.CLAP_MAX_INFLIGHT;
    const previousDepth = process.env.CLAP_QUEUE_DEPTH;
    const dir = await mkdtemp(join(tmpdir(), "clap-limiter-test-"));
    const modelPath = join(dir, "limits.Q4_K_M.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_TOKENS = JSON.stringify(Array.from({ length: 60 }, () => "x"));
      process.env.CLAP_MAX_INFLIGHT = "1";
      process.env.CLAP_QUEUE_DEPTH = "1";
      await writeFile(modelPath, "gguf bytes");
      const app = createServer();

      const fire = () => app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "hey" }] }),
      });
      const [r1, r2, r3] = await Promise.all([fire(), fire(), fire()]);
      const statuses = [r1.status, r2.status, r3.status].sort();
      expect(statuses).toEqual([200, 200, 429]);

      const overloaded = [r1, r2, r3].find((response) => response.status === 429);
      expect(overloaded?.headers.get("retry-after")).toMatch(/^\d+$/);
      const body = await overloaded?.json() as { error: { code: string; type: string } };
      expect(body.error.code).toBe("server_overloaded");
      expect(body.error.type).toBe("rate_limit_error");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_TOKENS", previousTokens);
      restoreEnv("CLAP_MAX_INFLIGHT", previousInflight);
      restoreEnv("CLAP_QUEUE_DEPTH", previousDepth);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("model remove endpoint 404s for unknown models", async () => {
    const response = await createServer().request("/clap/v1/models/remove", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "no-such/model-xyz" }),
    });
    expect(response.status).toBe(404);
    const body = await response.json() as { error: { code: string } };
    expect(body.error.code).toBe("not_cached");
  });

  test("selects parser families from model ids", () => {
    const request = { model: "qwen", messages: [{ role: "user" as const, content: "hi" }], stream: false };
    expect(selectParser("/tmp/ambiguous.gguf", request, { familyHints: ["qwen"] }).name).toBe("qwen");
    expect(selectParser("qwen-ish", request, { familyHints: ["mistral"] }).name).toBe("mistral");
    expect(selectParser("Qwen/Qwen2.5-7B-Instruct", request).name).toBe("qwen");
    expect(selectParser("deepseek-ai/deepseek-r1", request).name).toBe("deepseek");
    expect(selectParser("NousResearch/Hermes-3", request).name).toBe("hermes");
    expect(selectParser("mistralai/Mistral-7B", request).name).toBe("mistral");
    expect(selectParser("meta-llama/Llama-3.1", request).name).toBe("llama");
    expect(selectParser("google/functiongemma", request).name).toBe("gemma");
    expect(selectParser("/tmp/model.Q4_K_M.gguf", request).name).toBe("generic");
  });

  test("template-selected parsers parse ambiguous model ids", () => {
    const request = { model: "/tmp/ambiguous.gguf", messages: [{ role: "user" as const, content: "hi" }], stream: false, tools: [{ type: "function" as const, function: { name: "search" } }] };
    const qwen = parseAssistantOutput('<|tool_call_start|>{"name":"search","arguments":{"query":"qwen"}}<|tool_call_end|>', request, { familyHints: ["qwen"] });
    expect(qwen.finishReason).toBe("tool_calls");
    expect(qwen.toolCalls?.[0]?.function.arguments).toBe(JSON.stringify({ query: "qwen" }));

    const mistral = parseAssistantOutput('[TOOL_CALLS]search{"query":"mistral"}', request, { familyHints: ["mistral"] });
    expect(mistral.finishReason).toBe("tool_calls");
    expect(mistral.toolCalls?.[0]?.function.arguments).toBe(JSON.stringify({ query: "mistral" }));
  });

  test("family names inside tokenizer vocabulary do not select the wrong parser", () => {
    // gemma's tokenizer.json vocabulary contains the plain words "harmony"
    // and "Hermes" as tokens; only template/config metadata may vote by name.
    const vocab = '{"vocab":{"harmony":1234,"Hermes":5678,"mistral":9012}}';
    const nameMeta = '{"model_type":"gemma3_text","chat_template":"..."}';
    expect(inferParserFamilies(vocab.toLowerCase(), nameMeta.toLowerCase())).toEqual(["gemma"]);
    // Distinctive protocol markers still count no matter where they appear.
    expect(inferParserFamilies("<|channel|>", "")).toEqual(["harmony"]);
    expect(inferParserFamilies("<tool_call>", "")).toEqual(["hermes"]);
    expect(inferParserFamilies("", '{"model_type":"qwen3_moe","enable_thinking":true}')).toEqual(["qwen"]);
  });

  test("uses cached tokenizer template metadata for ambiguous local models", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-template-info-test-"));
    const model = join(dir, "ambiguous-local-model.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = '<|tool_call_start|>{"name":"search","arguments":"{\\"query\\":\\"qwen\\"}"}<|tool_call_end|>';
      await writeFile(model, "gguf bytes");
      await writeFile(join(dir, "config.json"), JSON.stringify({ model_type: "unknown_arch" }));
      await writeFile(join(dir, "tokenizer_config.json"), JSON.stringify({ chat_template: "{% if tools %}<|tool_call_start|>{{ tool_calls }}<|tool_call_end|>{% endif %}" }));

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "search" }], tools: [{ type: "function", function: { name: "search" } }] }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices[0].finish_reason).toBe("tool_calls");
      expect(body.choices[0].message.tool_calls[0]).toMatchObject({ function: { name: "search", arguments: JSON.stringify({ query: "qwen" }) } });
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
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

  test("defaults omitted max_tokens to 4096 while preserving explicit values", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const previousEcho = process.env.CLAP_FAKE_WORKER_ECHO_MAX_TOKENS;
    const dir = await mkdtemp(join(tmpdir(), "clap-max-tokens-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "ok";
      process.env.CLAP_FAKE_WORKER_ECHO_MAX_TOKENS = "1";

      const app = createServer();
      const defaulted = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "default" }] }),
      });
      const explicit = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, max_tokens: 77, messages: [{ role: "user", content: "explicit" }] }),
      });

      expect(defaulted.status).toBe(200);
      expect(explicit.status).toBe(200);
      expect((await defaulted.json()).choices[0].message.content).toBe("4096");
      expect((await explicit.json()).choices[0].message.content).toBe("77");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      restoreEnv("CLAP_FAKE_WORKER_ECHO_MAX_TOKENS", previousEcho);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("infers args-only JSON as an update_todos tool call", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-args-only-tool-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    const tools = [{
      type: "function",
      function: {
        name: "update_todos",
        parameters: { type: "object", properties: { todos: { type: "array" } }, required: ["todos"] },
      },
    }];
    const args = { todos: [{ step: "Draft README.md content based on provided details", status: "in_progress" }] };
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = `I need to update the todo list. I will call update_todos with the current next step.\n${JSON.stringify(args)}`;

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "continue" }], tools }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices[0].finish_reason).toBe("tool_calls");
      expect(body.choices[0].message.content).toBeNull();
      expect(JSON.stringify(body.choices[0].message)).not.toContain('{"todos"');
      expect(body.choices[0].message.tool_calls[0]).toMatchObject({
        id: "call_0_update_todos",
        type: "function",
        function: { name: "update_todos", arguments: JSON.stringify(args) },
      });

      process.env.CLAP_FAKE_WORKER_OUTPUT = `<think>Need to update todos.</think>\n${JSON.stringify(args)}`;
      const stream = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, stream: true, messages: [{ role: "user", content: "continue" }], tools }),
      });
      expect(stream.status).toBe(200);
      const events = await sseData(stream);
      const reasoningDelta = events.find((event) => event.choices?.[0]?.delta?.reasoning_content);
      const toolDelta = events.find((event) => event.choices?.[0]?.delta?.tool_calls);
      expect(reasoningDelta.choices[0].delta.reasoning).toBe("Need to update todos.");
      expect(reasoningDelta.choices[0].delta.reasoning_content).toBe("Need to update todos.");
      expect(toolDelta.choices[0].delta.tool_calls[0]).toMatchObject({ function: { name: "update_todos", arguments: JSON.stringify(args) } });
      expect(JSON.stringify(events)).not.toContain('{"todos"');
      expect(events.at(-1).choices[0].finish_reason).toBe("tool_calls");

      process.env.CLAP_FAKE_WORKER_OUTPUT = JSON.stringify({ todos: [] });
      const json = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "json" }], tools, response_format: { type: "json_object" } }),
      });
      const jsonBody = await json.json();
      expect(jsonBody.choices[0].finish_reason).toBe("stop");
      expect(jsonBody.choices[0].message.content).toBe(JSON.stringify({ todos: [] }));
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("normalizes Harmony reasoning and tool call markers", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-harmony-tool-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "<|channel|>thought I should inspect the README.\n<|tool_call|>call::read{path: <|'|>README.md<|'|>}<|/tool_call|>";

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: "read" }],
          tools: [{ type: "function", function: { name: "read", parameters: { type: "object" } } }],
        }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      const message = body.choices[0].message;
      expect(body.choices[0].finish_reason).toBe("tool_calls");
      expect(message.content).toBeNull();
      expect(message.reasoning).toContain("inspect the README");
      expect(message.reasoning_content).toBe(message.reasoning);
      expect(JSON.stringify(message)).not.toContain("<|channel|>");
      expect(JSON.stringify(message)).not.toContain("<|tool_call|>");
      expect(message.tool_calls[0]).toMatchObject({
        id: "call_0_read",
        type: "function",
        function: { name: "read", arguments: JSON.stringify({ path: "README.md" }) },
      });
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("normalizes Harmony function message calls and final content", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-harmony-function-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "<|channel|>analysis Need a search.\nto=functions.search <|message|>{\"q\":\"clap\"}<|call|>";
      const tool = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "search" }], tools: [{ type: "function", function: { name: "search" } }] }),
      });
      const toolBody = await tool.json();
      expect(toolBody.choices[0].message.content).toBeNull();
      expect(toolBody.choices[0].message.reasoning).toContain("Need a search");
      expect(toolBody.choices[0].message.tool_calls[0]).toMatchObject({
        id: "call_0_search",
        function: { name: "search", arguments: JSON.stringify({ q: "clap" }) },
      });

      process.env.CLAP_FAKE_WORKER_OUTPUT = "<|channel|>analysis private notes\n<|channel|>final Visible answer.";
      const final = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "answer" }] }),
      });
      const finalBody = await final.json();
      expect(finalBody.choices[0].message.content).toBe("Visible answer.");
      expect(finalBody.choices[0].message.reasoning).toBe("private notes");
      expect(finalBody.choices[0].message.content).not.toContain("<|channel|>");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("suppresses malformed Harmony marker variants without leaking raw protocol", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-harmony-malformed-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);

      process.env.CLAP_FAKE_WORKER_OUTPUT = "<channel>thought\nI need to use a tool.";
      const thought = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "think" }] }),
      });
      const thoughtBody = await thought.json();
      expect(thoughtBody.choices[0].message.content).toBe("");
      expect(thoughtBody.choices[0].message.reasoning).toContain("use a tool");
      expect(JSON.stringify(thoughtBody)).not.toContain("<channel");

      process.env.CLAP_FAKE_WORKER_OUTPUT = '<channel> <tool_call>call::update_todos{"todos":[{"step":"x"}';
      const incomplete = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "todos" }], tools: [{ type: "function", function: { name: "update_todos" } }] }),
      });
      const incompleteBody = await incomplete.json();
      expect(incompleteBody.choices[0].message.content).toBeNull();
      expect(incompleteBody.choices[0].finish_reason).toBe("tool_calls");
      expect(incompleteBody.choices[0].message.tool_calls[0]).toMatchObject({
        function: { name: "update_todos", arguments: JSON.stringify({ todos: [{ step: "x" }] }) },
      });
      expect(JSON.stringify(incompleteBody)).not.toContain("<channel");
      expect(JSON.stringify(incompleteBody)).not.toContain("<tool_call");
      expect(JSON.stringify(incompleteBody)).not.toContain("call::");

      process.env.CLAP_FAKE_WORKER_OUTPUT = '<|channel>thought\nNeed to update todos.<channel|><|tool_call>call:update_todos{"todos":[{"step":"unfinished"';
      const leadingPipe = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "todos" }], tools: [{ type: "function", function: { name: "update_todos" } }] }),
      });
      const leadingPipeBody = await leadingPipe.json();
      expect(leadingPipeBody.choices[0].message.content).toBeNull();
      expect(leadingPipeBody.choices[0].message.reasoning).toContain("Need to update todos");
      expect(leadingPipeBody.choices[0].finish_reason).toBe("tool_calls");
      expect(leadingPipeBody.choices[0].message.tool_calls[0]).toMatchObject({
        function: { name: "update_todos" },
      });
      expect(JSON.stringify(leadingPipeBody)).not.toContain("<|channel");
      expect(JSON.stringify(leadingPipeBody)).not.toContain("<channel|");
      expect(JSON.stringify(leadingPipeBody)).not.toContain("<|tool_call");
      expect(JSON.stringify(leadingPipeBody)).not.toContain("call:update_todos");

      process.env.CLAP_FAKE_WORKER_OUTPUT = '<tool_call>call::update_todos{"todos":[{"step":"x"}]}';
      const unclosed = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "todos" }], tools: [{ type: "function", function: { name: "update_todos" } }] }),
      });
      const unclosedBody = await unclosed.json();
      expect(unclosedBody.choices[0].finish_reason).toBe("tool_calls");
      expect(unclosedBody.choices[0].message.content).toBeNull();
      expect(unclosedBody.choices[0].message.tool_calls[0]).toMatchObject({
        id: "call_0_update_todos",
        function: { name: "update_todos", arguments: JSON.stringify({ todos: [{ step: "x" }] }) },
      });
      expect(JSON.stringify(unclosedBody)).not.toContain("<tool_call");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("normalizes local model tool-call formats without leaking delimiters", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-local-tool-formats-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    const fixtures = [
      { label: "Hermes", output: '<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call><|im_end|>', name: "get_weather", args: { location: "SF" }, delimiter: "<tool_call>" },
      { label: "Hermes multiple", output: '<tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call><tool_call>{"name":"search","arguments":"{\\"query\\":\\"qwen\\"}"}</tool_call>', name: "search", args: { query: "qwen" }, index: 1, delimiter: "<tool_call>" },
      { label: "Qwen2", output: '<|tool_call_start|>{"name":"search","arguments":"{\\"query\\":\\"qwen\\"}"}<|tool_call_end|><|im_end|>', name: "search", args: { query: "qwen" }, delimiter: "<|tool_call_start|>" },
      { label: "Qwen3 think", output: '<think>Need weather.</think><tool_call>{"name":"get_weather","arguments":{"location":"SF"}}</tool_call>', name: "get_weather", args: { location: "SF" }, reasoning: "Need weather.", delimiter: "<think>" },
      { label: "DeepSeek", output: '<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function get_weather\n```json\n{"location":"SF"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>', name: "get_weather", args: { location: "SF" }, delimiter: "<｜tool▁calls▁begin｜>" },
      { label: "Llama JSON", output: '<|python_tag|>{"name":"get_weather","parameters":{"location":"SF"}}', name: "get_weather", args: { location: "SF" }, delimiter: "<|python_tag|>" },
      { label: "Llama pythonic", output: '<|python_tag|>get_weather(location="SF", unit="celsius")', name: "get_weather", args: { location: "SF", unit: "celsius" }, delimiter: "<|python_tag|>" },
      { label: "Mistral compact", output: '[TOOL_CALLS]get_weather{"location":"Paris"}</s>', name: "get_weather", args: { location: "Paris" }, delimiter: "[TOOL_CALLS]" },
      { label: "Mistral array", output: '[TOOL_CALLS][{"name":"get_weather","arguments":{"location":"Paris"}}]', name: "get_weather", args: { location: "Paris" }, delimiter: "[TOOL_CALLS]" },
      { label: "FunctionGemma", output: 'call:get_weather{location: "Paris" unit: "celsius" }', name: "get_weather", args: { location: "Paris", unit: "celsius" }, delimiter: "call:get_weather" },
      { label: "Fenced JSON", output: '```json\n[{"name":"get_weather","arguments":{"location":"SF"}}]\n```', name: "get_weather", args: { location: "SF" }, delimiter: "```" },
    ];
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      for (const fixture of fixtures) {
        process.env.CLAP_FAKE_WORKER_OUTPUT = fixture.output;
        const response = await createServer().request("/v1/chat/completions", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model, messages: [{ role: "user", content: fixture.label }], tools: [{ type: "function", function: { name: fixture.name } }] }),
        });
        expect(response.status).toBe(200);
        const body = await response.json();
        const message = body.choices[0].message;
        const call = message.tool_calls[fixture.index ?? 0];
        expect(body.choices[0].finish_reason).toBe("tool_calls");
        expect(message.content).toBeNull();
        if (fixture.reasoning) expect(message.reasoning).toBe(fixture.reasoning);
        expect(call.function.name).toBe(fixture.name);
        expect(call.function.arguments).toBe(JSON.stringify(fixture.args));
        expect(JSON.stringify(message)).not.toContain(fixture.delimiter);
      }

      process.env.CLAP_FAKE_WORKER_OUTPUT = '{"answer":true}<|eot_id|>';
      const json = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "json" }], response_format: { type: "json_object" } }),
      });
      const jsonBody = await json.json();
      expect(jsonBody.choices[0].message.content).toBe(JSON.stringify({ answer: true }));
      expect(jsonBody.choices[0].finish_reason).toBe("stop");
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

  test("streams reasoning deltas without leaking Harmony markers", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-reasoning-stream-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "<|channel|>analysis hidden chain\n<|channel|>final visible text";

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, stream: true, messages: [{ role: "user", content: "answer" }] }),
      });

      expect(response.status).toBe(200);
      const events = await sseData(response);
      const reasoningDelta = events.find((event) => event.choices?.[0]?.delta?.reasoning);
      const contentDelta = events.find((event) => event.choices?.[0]?.delta?.content);
      expect(reasoningDelta.choices[0].delta.reasoning).toBe("hidden chain");
      expect(reasoningDelta.choices[0].delta.reasoning_content).toBe("hidden chain");
      expect(contentDelta.choices[0].delta.content).toBe("visible text");
      expect(JSON.stringify(events)).not.toContain("<|channel|>");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("streams incremental content deltas token by token", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousTokens = process.env.CLAP_FAKE_WORKER_TOKENS;
    const dir = await mkdtemp(join(tmpdir(), "clap-token-stream-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_TOKENS = JSON.stringify(["Hel", "lo", " wor", "ld!"]);

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, stream: true, messages: [{ role: "user", content: "hi" }] }),
      });

      expect(response.status).toBe(200);
      const events = await sseData(response);
      expect(events[0].choices[0].delta.role).toBe("assistant");
      const contentDeltas = events
        .map((event) => event.choices?.[0]?.delta?.content)
        .filter((content): content is string => typeof content === "string");
      expect(contentDeltas.length).toBeGreaterThan(1);
      expect(contentDeltas.join("")).toBe("Hello world!");
      expect(events.at(-1).choices[0].finish_reason).toBe("stop");
      expect(events.at(-1).usage.total_tokens).toBeGreaterThan(0);
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_TOKENS", previousTokens);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("reports real worker usage and finish_reason when provided", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const previousDone = process.env.CLAP_FAKE_WORKER_DONE;
    const dir = await mkdtemp(join(tmpdir(), "clap-usage-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "truncated answer";
      process.env.CLAP_FAKE_WORKER_DONE = JSON.stringify({ finish_reason: "length", usage: { prompt_tokens: 42, completion_tokens: 7 } });

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.usage).toEqual({ prompt_tokens: 42, completion_tokens: 7, total_tokens: 49 });
      expect(body.choices[0].finish_reason).toBe("length");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_OUTPUT", previousOutput);
      restoreEnv("CLAP_FAKE_WORKER_DONE", previousDone);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("streams Ollama chat and generate ndjson deltas", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousTokens = process.env.CLAP_FAKE_WORKER_TOKENS;
    const dir = await mkdtemp(join(tmpdir(), "clap-ollama-stream-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_TOKENS = JSON.stringify(["Hel", "lo", " wor", "ld!"]);

      const chat = await createServer().request("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }),
      });
      expect(chat.status).toBe(200);
      expect(chat.headers.get("content-type")).toContain("ndjson");
      const chatLines = (await chat.text()).trim().split("\n").map((line) => JSON.parse(line));
      const chatDeltas = chatLines.filter((line) => line.done === false && line.message?.content).map((line) => line.message.content);
      expect(chatDeltas.length).toBeGreaterThan(1);
      expect(chatDeltas.join("")).toBe("Hello world!");
      expect(chatLines.at(-1)).toMatchObject({ done: true, done_reason: "stop" });

      const generate = await createServer().request("/api/generate", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, prompt: "hi" }),
      });
      expect(generate.status).toBe(200);
      const generateLines = (await generate.text()).trim().split("\n").map((line) => JSON.parse(line));
      const generateDeltas = generateLines.filter((line) => line.done === false && line.response).map((line) => line.response);
      expect(generateDeltas.length).toBeGreaterThan(1);
      expect(generateDeltas.join("")).toBe("Hello world!");
      expect(generateLines.at(-1)).toMatchObject({ done: true, done_reason: "stop" });
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_TOKENS", previousTokens);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("does not parse tool JSON from reasoning blocks", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-reasoning-order-test-"));
    const model = join(dir, "qwen-model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = '<think>{"tool_calls":[{"name":"search","arguments":{"q":"hidden"}}]}</think>Final answer.';

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, messages: [{ role: "user", content: "answer" }], tools: [{ type: "function", function: { name: "search" } }] }),
      });

      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.choices[0].finish_reason).toBe("stop");
      expect(body.choices[0].message.tool_calls).toBeUndefined();
      expect(body.choices[0].message.content).toBe("Final answer.");
      expect(body.choices[0].message.reasoning).toContain("tool_calls");
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

  test("serves Responses API reasoning items for Harmony output", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousOutput = process.env.CLAP_FAKE_WORKER_OUTPUT;
    const dir = await mkdtemp(join(tmpdir(), "clap-responses-reasoning-test-"));
    const model = join(dir, "model.Q4_K_M.gguf");
    try {
      await writeFile(model, "gguf bytes");
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_OUTPUT = "<|channel|>thought inspect docs\n<|tool_call|>call::read{path: <|'|>README.md<|'|>}<|/tool_call|>";
      const response = await createServer().request("/v1/responses", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model, input: "read", tools: [{ type: "function", function: { name: "read" } }] }),
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.output_text).toBe("");
      expect(body.output[0]).toMatchObject({ type: "reasoning", content: [{ type: "reasoning_text", text: "inspect docs" }] });
      expect(body.output[1]).toMatchObject({ type: "function_call", name: "read", arguments: JSON.stringify({ path: "README.md" }) });
      expect(JSON.stringify(body)).not.toContain("<|tool_call|>");
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

  test("resumes interrupted downloads with HTTP Range and removes models", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-resume-test-"));
    const content = "0123456789".repeat(100);
    const seenRanges: string[] = [];
    const sha256 = new Bun.CryptoHasher("sha256").update(content).digest("hex");
    const hf = mockHuggingFaceServer({
      "acme/resume": { "model.Q4_K_M.gguf": content },
    }, { seenRanges, sha256: { "model.Q4_K_M.gguf": sha256 } });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const repoDir = join(dir, "models", "huggingface", "acme--resume");
      const target = join(repoDir, "model.Q4_K_M.gguf");
      await mkdir(repoDir, { recursive: true });
      await writeFile(`${target}.part`, content.slice(0, 400));

      const result = await pullModel({ model: "acme/resume", file: "model.Q4_K_M.gguf" });
      expect(seenRanges).toEqual(["bytes=400-"]);
      expect(await readFile(target, "utf8")).toBe(content);
      expect(existsSync(`${target}.part`)).toBe(false);
      expect(result.modelPath).toBe(target);

      const removed = await removeModel("acme/resume");
      expect(removed).toEqual([target]);
      expect(existsSync(repoDir)).toBe(false);
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("rejects downloads whose sha256 does not match Hugging Face metadata", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-checksum-test-"));
    const hf = mockHuggingFaceServer({
      "acme/corrupt": { "model.Q4_K_M.gguf": "corrupted bytes" },
    }, { sha256: { "model.Q4_K_M.gguf": "0".repeat(64) } });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const target = join(dir, "models", "huggingface", "acme--corrupt", "model.Q4_K_M.gguf");

      expect(pullModel({ model: "acme/corrupt", file: "model.Q4_K_M.gguf" })).rejects.toThrow(/checksum mismatch/);
      await Bun.sleep(10);
      expect(existsSync(target)).toBe(false);
      expect(existsSync(`${target}.part`)).toBe(false);
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

  test("resolves runnable model options and recommends MLX on macOS arm64", async () => {
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const previousPlatform = process.env.CLAP_TEST_PLATFORM;
    const hf = mockHuggingFaceServer({
      "mlx-community/gemma-4-e4b-it-4bit": {
        "config.json": "{}",
        "tokenizer_config.json": "{}",
        "model.safetensors": "mlx weights",
      },
      "google/gemma-4-e4b-it": {
        "model.safetensors": "source weights",
      },
    });
    try {
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      process.env.CLAP_TEST_PLATFORM = "darwin-arm64";
      const response = await createServer().request("/clap/v1/models/resolve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "mlx-community/gemma-4-e4b-it-4bit" }),
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.selected).toMatchObject({ backend: "mlx", format: "mlx", supported: true, recommended: true });

      const source = await createServer().request("/clap/v1/models/resolve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "google/gemma-4-e4b-it" }),
      });
      const sourceBody = await source.json();
      expect(sourceBody.selected).toBeUndefined();
      expect(sourceBody.options[0]).toMatchObject({ format: "safetensors", supported: false });
      expect(sourceBody.options[0].unsupportedReason).toContain("not directly runnable");
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      restoreEnv("CLAP_TEST_PLATFORM", previousPlatform);
    }
  });

  test("selects a recommended GGUF quant without requiring --file", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const previousPlatform = process.env.CLAP_TEST_PLATFORM;
    const dir = await mkdtemp(join(tmpdir(), "clap-resolve-gguf-test-"));
    const hf = mockHuggingFaceServer({
      "acme/multi-gguf": {
        "model-Q8_0.gguf": "large",
        "model-Q4_K_M.gguf": "recommended",
        "model-Q3_K_M.gguf": "small",
        "mmproj-model-f16.gguf": "vision projector",
        "model-imatrix.gguf": "quantization data",
        "mtp-model-Q4_0.gguf": "draft model",
      },
    });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      process.env.CLAP_TEST_PLATFORM = "linux-x64";
      const resolve = await createServer().request("/clap/v1/models/resolve", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/multi-gguf" }),
      });
      const resolveBody = await resolve.json();
      expect(resolveBody.selected).toMatchObject({ backend: "gguf", file: "model-Q4_K_M.gguf", quantization: "Q4_K_M" });
      expect(resolveBody.options.map((option: { file?: string }) => option.file)).toEqual([
        "model-Q4_K_M.gguf",
        "model-Q3_K_M.gguf",
        "model-Q8_0.gguf",
      ]);

      const pull = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/multi-gguf" }),
      });
      expect(pull.status).toBe(200);
      const pullBody = await pull.json();
      expect(pullBody.download.selected.file).toBe("model-Q4_K_M.gguf");
      const download = await waitForDownload(pullBody.download.id, "completed");
      expect(download.modelPath).toEndWith("model-Q4_K_M.gguf");
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      restoreEnv("CLAP_TEST_PLATFORM", previousPlatform);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("power-user file override selects the requested artifact", async () => {
    const previousHome = process.env.CLAP_HOME;
    const previousEndpoint = process.env.CLAP_HF_ENDPOINT;
    const dir = await mkdtemp(join(tmpdir(), "clap-resolve-override-test-"));
    const hf = mockHuggingFaceServer({
      "acme/override-gguf": {
        "model-Q4_K_M.gguf": "recommended",
        "model-Q5_K_M.gguf": "bigger",
      },
    });
    try {
      process.env.CLAP_HOME = dir;
      process.env.CLAP_HF_ENDPOINT = `http://${hf.hostname}:${hf.port}`;
      const response = await createServer().request("/clap/v1/models/pull", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: "acme/override-gguf", backend: "gguf", file: "model-Q5_K_M.gguf" }),
      });
      expect(response.status).toBe(200);
      const body = await response.json();
      expect(body.download.file).toBe("model-Q5_K_M.gguf");
      const download = await waitForDownload(body.download.id, "completed");
      expect(download.modelPath).toEndWith("model-Q5_K_M.gguf");
    } finally {
      hf.stop(true);
      restoreEnv("CLAP_HOME", previousHome);
      restoreEnv("CLAP_HF_ENDPOINT", previousEndpoint);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test.skipIf(!mlxSupported)("pulls an MLX repo directory into the Hugging Face cache", async () => {
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

  test("resident worker errors fail chat instead of returning partial or empty content", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousError = process.env.CLAP_FAKE_WORKER_ERROR;
    const previousToken = process.env.CLAP_FAKE_WORKER_PARTIAL_TOKEN;
    const dir = await mkdtemp(join(tmpdir(), "clap-worker-error-test-"));
    const modelPath = join(dir, "ornith.Q5_K_M.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_ERROR = "llama_decode failed";
      process.env.CLAP_FAKE_WORKER_PARTIAL_TOKEN = "_pk";
      await writeFile(modelPath, "gguf bytes");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "hey" }] }),
      });

      expect(response.status).toBe(503);
      const body = await response.json();
      expect(body.error.message).toContain("llama_decode failed");
      expect(body.error.message).toContain("Q4_K_M");
      expect(body.error.message).toContain("CLAP_LLAMA_GPU_LAYERS");
      expect(JSON.stringify(body)).not.toContain("_pk");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_ERROR", previousError);
      restoreEnv("CLAP_FAKE_WORKER_PARTIAL_TOKEN", previousToken);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("resident worker context overflow returns structured 400 with actionable guidance", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousError = process.env.CLAP_FAKE_WORKER_ERROR;
    const previousCode = process.env.CLAP_FAKE_WORKER_ERROR_CODE;
    const dir = await mkdtemp(join(tmpdir(), "clap-worker-context-test-"));
    const modelPath = join(dir, "gemma.Q3_K_S.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_ERROR = "prompt exceeds context window; prompt tokens=11236, context=4096, reserved output tokens=32. Increase CLAP_LLAMA_CONTEXT or reduce the prompt/tool history.";
      process.env.CLAP_FAKE_WORKER_ERROR_CODE = "context_length_exceeded";
      await writeFile(modelPath, "gguf bytes");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "long prompt" }] }),
      });

      expect(response.status).toBe(400);
      const body = await response.json();
      expect(body.error.type).toBe("invalid_request_error");
      expect(body.error.code).toBe("context_length_exceeded");
      expect(body.error.message).toContain("prompt exceeds context window");
      expect(body.error.message).toContain("CLAP_LLAMA_CONTEXT");
      expect(JSON.stringify(body)).not.toContain("choices");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_ERROR", previousError);
      restoreEnv("CLAP_FAKE_WORKER_ERROR_CODE", previousCode);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("resident worker exits during chat fail instead of returning empty content", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousExit = process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT;
    const dir = await mkdtemp(join(tmpdir(), "clap-worker-exit-test-"));
    const modelPath = join(dir, "exit.Q4_K_M.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT = "134";
      await writeFile(modelPath, "gguf bytes");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "hey" }] }),
      });

      expect(response.status).toBe(503);
      const body = await response.json();
      expect(body.error.message).toContain("resident worker exited with code 134");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_EXIT_ON_CHAT", previousExit);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("watchdog auto-restarts a crashed worker after backoff and reports crash counters", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousExit = process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT;
    const dir = await mkdtemp(join(tmpdir(), "clap-worker-watchdog-test-"));
    const modelPath = join(dir, "watchdog.Q4_K_M.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT = "134";
      await writeFile(modelPath, "gguf bytes");
      const app = createServer();

      const crashed = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "hey" }] }),
      });
      expect(crashed.status).toBe(503);

      const loaded = await app.request("/clap/v1/runtime/models");
      const entry = (await loaded.json()).models.find((m: { id: string }) => m.id === modelPath);
      expect(entry.worker.crashes).toBe(1);
      expect(entry.worker.lastCrashAt).toBeString();

      // Worker no longer crashes; the next request must recover automatically
      // after the backoff window without any manual restart.
      delete process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT;
      const recovered = await app.request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, messages: [{ role: "user", content: "hey" }] }),
      });
      expect(recovered.status).toBe(200);
      const body = await recovered.json();
      expect(body.choices[0].message.content).toBe("ok");
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_EXIT_ON_CHAT", previousExit);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test("streaming resident worker errors emit an error event instead of empty completion", async () => {
    const previousWorker = process.env.CLAP_LLAMA_WORKER;
    const previousError = process.env.CLAP_FAKE_WORKER_ERROR;
    const dir = await mkdtemp(join(tmpdir(), "clap-worker-stream-error-test-"));
    const modelPath = join(dir, "stream-error.Q4_K_M.gguf");
    try {
      process.env.CLAP_LLAMA_WORKER = await fakeWorker(dir);
      process.env.CLAP_FAKE_WORKER_ERROR = "llama_decode failed";
      await writeFile(modelPath, "gguf bytes");

      const response = await createServer().request("/v1/chat/completions", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ model: modelPath, stream: true, messages: [{ role: "user", content: "hey" }] }),
      });

      expect(response.status).toBe(200);
      const events = await sseData(response);
      expect(events[0].choices[0].delta).toEqual({ role: "assistant" });
      const errorEvent = events.find((event) => event.error);
      expect(errorEvent.error.message).toContain("llama_decode failed");
      expect(errorEvent.error.message).toContain("Q4_K_M");
      expect(events.some((event) => event.choices?.[0]?.delta?.content)).toBe(false);
    } finally {
      restoreEnv("CLAP_LLAMA_WORKER", previousWorker);
      restoreEnv("CLAP_FAKE_WORKER_ERROR", previousError);
      await rm(dir, { recursive: true, force: true });
    }
  });

  test.skipIf(!mlxSupported)("routes cached Hugging Face repo ids to local worker paths", async () => {
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
    if (process.env.CLAP_FAKE_WORKER_ERROR) {
      if (process.env.CLAP_FAKE_WORKER_PARTIAL_TOKEN) console.log(JSON.stringify({ id: request.id, token: process.env.CLAP_FAKE_WORKER_PARTIAL_TOKEN }));
      const extra = process.env.CLAP_FAKE_WORKER_ERROR_CODE ? { code: process.env.CLAP_FAKE_WORKER_ERROR_CODE } : {};
      console.log(JSON.stringify({ id: request.id, error: process.env.CLAP_FAKE_WORKER_ERROR, ...extra }));
      continue;
    }
    if (process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT) {
      process.exit(Number(process.env.CLAP_FAKE_WORKER_EXIT_ON_CHAT));
    }
    if (process.env.CLAP_FAKE_WORKER_TOKENS) {
      for (const token of JSON.parse(process.env.CLAP_FAKE_WORKER_TOKENS)) {
        console.log(JSON.stringify({ id: request.id, token }));
        await Bun.sleep(2);
      }
      const doneExtras = process.env.CLAP_FAKE_WORKER_DONE ? JSON.parse(process.env.CLAP_FAKE_WORKER_DONE) : {};
      console.log(JSON.stringify({ id: request.id, done: true, ...doneExtras }));
      continue;
    }
    const content = process.env.CLAP_FAKE_WORKER_ECHO_MAX_TOKENS ? String(request.max_tokens) : (process.env.CLAP_FAKE_WORKER_OUTPUT ?? "ok");
    console.log(JSON.stringify({ id: request.id, content }));
    const doneExtras = process.env.CLAP_FAKE_WORKER_DONE ? JSON.parse(process.env.CLAP_FAKE_WORKER_DONE) : {};
    console.log(JSON.stringify({ id: request.id, done: true, ...doneExtras }));
  }
}
`);
  return `/usr/bin/env bun ${path}`;
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

function mockHuggingFaceServer(repos: Record<string, Record<string, string>>, options: { requireAuth?: boolean; seenAuth?: string[]; seenRanges?: string[]; chunkDelayMs?: number; requestCounts?: Record<string, number>; sha256?: Record<string, string> } = {}) {
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
          siblings: Object.entries(files).map(([rfilename, content]) => ({
            rfilename,
            size: content.length,
            ...(options.sha256?.[rfilename] ? { lfs: { sha256: options.sha256[rfilename], size: content.length } } : {}),
          })),
        });
      }

      const resolveMatch = url.pathname.match(/^\/([^/]+\/[^/]+)\/resolve\/main\/(.+)$/);
      if (resolveMatch) {
        const repo = resolveMatch[1]!;
        const file = decodeURIComponent(resolveMatch[2]!);
        const content = repos[repo]?.[file];
        if (content === undefined) return new Response("not found", { status: 404 });
        const range = request.headers.get("range");
        if (range) {
          options.seenRanges?.push(range);
          const offset = Number(range.match(/^bytes=(\d+)-$/)?.[1] ?? 0);
          if (offset >= content.length) return new Response(null, { status: 416 });
          return new Response(content.slice(offset), {
            status: 206,
            headers: { "content-range": `bytes ${offset}-${content.length - 1}/${content.length}` },
          });
        }
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
