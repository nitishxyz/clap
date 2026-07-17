#!/usr/bin/env bun
import { ClapApiError, createClapClient, defaultBaseURL, type ChatCompletionRequest, type Download, type ModelResolveOption, type ModelResolveResponse } from "@clap/api";
import { deleteStoredHfToken, hfAuthGuidance, hfAuthStatus, isHfAuthError, removeModel, storeHfToken } from "@clap/models";
import { startServer } from "@clap/server";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { formatModelList, formatModelStatus } from "./models-output";
import { formatBytes, formatDownloadProgress } from "./progress";
import { installPullCancellationHandler } from "./pull-cancellation";
import { chooseOptionByInput, findOptionByQuant, formatResolveOptions, supportedOptions } from "./resolver-output";
import {
  baseURLFromEnv,
  currentCliCommand,
  findListeningPids,
  healthCheck,
  isLivePid,
  launchdPlist,
  portFromBaseURL,
  readServerMetadata,
  removeServerMetadata,
  serverPaths,
  systemdService,
  tailFile,
  waitForHealthy,
  waitForUnhealthy,
  withServerStartLock,
  writeServerMetadata,
  type ServerMetadata,
} from "./server-lifecycle";

const args = process.argv.slice(2);
const command = args[0] ?? "help";

try {
  if (command === "serve") {
    await serve();
  } else if (command === "auth") {
    await authCommand(args.slice(1));
  } else if (command === "server") {
    await serverCommand(args.slice(1));
  } else if (command === "models") {
    await modelsCommand(args.slice(1));
  } else if (command === "load") {
    await loadCommand(args.slice(1));
  } else if (command === "unload") {
    await unloadCommand(args.slice(1));
  } else if (command === "chat") {
    await chat(args.slice(1));
  } else if (command === "run") {
    await run(args.slice(1));
  } else if (command === "pull") {
    await pull(args.slice(1));
  } else if (command === "rm") {
    await rmCommand(args.slice(1));
  } else if (command === "resolve") {
    await resolveCommand(args.slice(1));
  } else {
    help();
    process.exit(command === "help" || command === "--help" || command === "-h" ? 0 : 1);
  }
} catch (error) {
  printError(error);
  process.exit(1);
}

async function serve() {
  const server = startServer();
  console.log(`clap server listening on http://${server.hostname}:${server.port}`);
  await new Promise(() => undefined);
}

async function serverStatus() {
  const metadata = await readServerMetadata();
  const baseURL = metadata?.baseURL ?? baseURLFromEnv();
  const health = await healthCheck(baseURL);

  if (!health) {
    console.log("status: stopped");
    console.log(`baseURL: ${baseURL}`);
    if (metadata) {
      console.log(`pid: ${metadata.pid}`);
      console.log("health: unavailable");
    }
    return;
  }

  const client = createClapClient({ baseURL });
  const [runtime, backends, models] = await Promise.all([
    client.runtime(),
    client.backends(),
    client.clapModels(),
  ]);

  console.log("status: running");
  console.log(`health: ${health.status}`);
  console.log(`baseURL: ${baseURL}`);
  if (metadata) {
    console.log(`pid: ${metadata.pid}`);
    console.log(`startedAt: ${metadata.startedAt}`);
  }
  console.log(`version: ${health.version}`);
  console.log(`uptimeMs: ${health.uptimeMs}`);
  console.log(`runtime: ${runtime.runtime} ${runtime.bunVersion} (${runtime.platform}/${runtime.arch})`);
  console.log("backends:");
  for (const backend of backends.backends) {
    console.log(`  - ${backend.id}: ${backend.status}${backend.reason ? ` (${backend.reason})` : ""}`);
  }
  console.log("models:");
  for (const model of models.models) {
    console.log(`  - ${model.id}: ${model.status} (${model.backend}/${model.format})`);
  }
}

async function serverCommand(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const action = rest[0] ?? "status";
  const force = flags.force === "true";
  if (action === "start") {
    await startBackgroundServer();
  } else if (action === "stop") {
    await stopBackgroundServer({ force });
  } else if (action === "status") {
    await serverStatus();
  } else if (action === "restart") {
    await stopBackgroundServer({ quiet: true, force });
    await startBackgroundServer();
  } else if (action === "logs") {
    await serverLogs(Number(rest[1] ?? "80"));
  } else if (action === "install") {
    await installServiceTemplate();
  } else {
    throw new Error("usage: clap server <start|stop|status|restart|logs|install> [--force]");
  }
}

async function modelsCommand(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const action = rest[0] ?? "list";
  if (action !== "list") {
    throw new Error("usage: clap models [list] [--aliases] [--json] [--active]");
  }

  await startBackgroundServer({ quiet: true });
  const client = createClapClient();
  const [models, aliases, loaded] = await Promise.all([
    client.clapModels(),
    flags.aliases === "true" ? client.clapAliases() : Promise.resolve(undefined),
    flags.active === "true" ? client.loadedModels() : Promise.resolve(undefined),
  ]);
  if (flags.json === "true") {
    console.log(JSON.stringify({ models: models.models, ...(aliases ? { aliases: aliases.models } : {}), ...(loaded ? { loaded: loaded.models } : {}) }, null, 2));
    return;
  }
  if (loaded) {
    console.log(formatModelStatus(models.models, loaded.models, aliases?.models));
    return;
  }
  console.log(formatModelList(models.models, aliases?.models));
}

async function loadCommand(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const model = rest[0];
  if (!model) {
    throw new Error("usage: clap load <model|alias|path> [--backend mlx|gguf] [--keep-alive 15m|1h|always]");
  }
  await startBackgroundServer({ quiet: true });
  const response = await createClapClient().loadModel({
    model,
    backend: parseBackend(flags.backend),
    keepAlive: flags.keepAlive,
  });
  console.log(`loaded ${response.model.id} (${response.model.backend}/${response.model.format})`);
  console.log(`state: ${response.model.state}`);
  console.log(`keepAlive: ${response.model.keepAlive}`);
  console.log(`expiresAt: ${response.model.expiresAt ?? "never"}`);
  console.log(`worker: ${response.model.worker.state}${response.model.worker.pid ? ` pid ${response.model.worker.pid}` : ""}`);
  if (response.model.worker.limitation) console.log(`note: ${response.model.worker.limitation}`);
}

async function unloadCommand(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const model = rest[0];
  if (!model) {
    throw new Error("usage: clap unload <model|alias|path> [--backend mlx|gguf]");
  }
  await startBackgroundServer({ quiet: true });
  const response = await createClapClient().unloadModel({ model, backend: parseBackend(flags.backend) });
  if (response.unloaded) {
    console.log(`unloaded ${response.model?.id ?? model}`);
    return;
  }
  if (response.model) {
    console.log(`unload pending for ${response.model.id}; activeRequests: ${response.model.activeRequests}`);
    return;
  }
  console.log(`not loaded: ${model}`);
}

async function startBackgroundServer({ quiet = false } = {}): Promise<ServerMetadata> {
  return withServerStartLock(async () => {
    const paths = serverPaths();
    const baseURL = baseURLFromEnv();
    const existingHealth = await healthCheck(baseURL);
    const existingMetadata = await readServerMetadata(paths);
    if (existingHealth?.status === "ok" && existingMetadata && isLivePid(existingMetadata.pid)) {
      if (!quiet) console.log(`clap server already running at ${baseURL} (pid ${existingMetadata.pid})`);
      return existingMetadata;
    }
    if (existingHealth?.status === "ok") {
      if (existingMetadata) await removeServerMetadata(paths);
      if (!quiet) {
        console.log(`clap server already healthy at ${baseURL}, but it is not managed by this CLI`);
        console.log("use `clap server stop --force` to stop the process listening on the Clap port");
      }
      return {
        pid: 0,
        port: portFromBaseURL(baseURL),
        baseURL,
        startedAt: new Date().toISOString(),
        managed: false,
      };
    }

    await mkdir(paths.home, { recursive: true });
    const stdout = Bun.file(paths.stdoutLog);
    const stderr = Bun.file(paths.stderrLog);
    const command = [...currentCliCommand(), "serve"];
    const proc = Bun.spawn(command, {
      env: { ...process.env, CLAP_BASE_URL: baseURL, PORT: String(portFromBaseURL(baseURL)) },
      stdout,
      stderr,
    });
    proc.unref();

    const health = await waitForHealthy(baseURL);
    if (!health) {
      proc.kill();
      throw new Error(`server did not become healthy at ${baseURL}; see ${paths.stderrLog}`);
    }

    const metadata = {
      pid: proc.pid,
      port: portFromBaseURL(baseURL),
      baseURL,
      startedAt: new Date().toISOString(),
      managed: true,
    };
    await writeServerMetadata(metadata, paths);
    if (!quiet) console.log(`clap server started at ${baseURL} (pid ${metadata.pid})`);
    return metadata;
  });
}

async function stopBackgroundServer({ quiet = false, force = false } = {}) {
  const paths = serverPaths();
  const metadata = await readServerMetadata(paths);
  const baseURL = metadata?.baseURL ?? baseURLFromEnv();
  const health = await healthCheck(baseURL);

  if (metadata?.pid && isLivePid(metadata.pid)) {
    try {
      process.kill(metadata.pid, "SIGTERM");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }

    const stopped = await waitForUnhealthy(baseURL);
    await removeServerMetadata(paths);
    if (!stopped) throw new Error(`clap server pid ${metadata.pid} did not stop at ${baseURL}`);
    if (!quiet) console.log("clap server stopped");
    return;
  }

  if (!health) {
    if (metadata) await removeServerMetadata(paths);
    if (!quiet) console.log(metadata ? "clap server is not running (removed stale metadata)" : "clap server is not running");
    return;
  }

  if (!force) {
    const detail = metadata?.pid ? `metadata pid ${metadata.pid} is not live` : "no managed server metadata exists";
    throw new Error(`clap server is healthy at ${baseURL}, but ${detail}; use \`clap server stop --force\` or \`clap server restart --force\` to kill the process listening on port ${portFromBaseURL(baseURL)}`);
  }

  await stopListeningProcesses(baseURL);
  await removeServerMetadata(paths);
  if (!quiet) console.log("clap server stopped with --force");
}

async function stopListeningProcesses(baseURL: string) {
  const port = portFromBaseURL(baseURL);
  const pids = await findListeningPids(port);
  if (pids.length === 0) throw new Error(`no process found listening on Clap port ${port}`);
  for (const pid of pids) {
    try {
      process.kill(pid, "SIGTERM");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
  }
  if (await waitForUnhealthy(baseURL)) return;
  for (const pid of pids) {
    try {
      process.kill(pid, "SIGKILL");
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ESRCH") throw error;
    }
  }
  if (!await waitForUnhealthy(baseURL)) throw new Error(`process on Clap port ${port} did not stop`);
}

async function serverLogs(lineCount: number) {
  const paths = serverPaths();
  const stdout = await tailFile(paths.stdoutLog, lineCount);
  const stderr = await tailFile(paths.stderrLog, lineCount);
  console.log(`==> ${paths.stdoutLog} <==`);
  console.log(stdout || "(empty)");
  console.log(`==> ${paths.stderrLog} <==`);
  console.log(stderr || "(empty)");
}

async function authCommand(argv: string[]) {
  const action = argv[0] ?? "status";
  if (action === "login") {
    await authLogin(argv.slice(1));
  } else if (action === "logout") {
    const status = await deleteStoredHfToken();
    console.log(`status: logged_out`);
    console.log(`detail: ${status.detail}`);
  } else if (action === "status") {
    await printAuthStatus();
  } else {
    throw new Error("usage: clap auth <login|logout|status>");
  }
}

async function authLogin(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const token = flags.token ?? rest[0] ?? await readTokenForLogin();
  const status = await storeHfToken(token);
  console.log("status: logged_in");
  console.log(`source: ${status.source}`);
  if (status.tokenPreview) console.log(`token: ${status.tokenPreview}`);
  if (status.detail) console.log(`detail: ${status.detail}`);
}

async function printAuthStatus() {
  const status = await hfAuthStatus();
  console.log(`status: ${status.authenticated ? "logged_in" : "logged_out"}`);
  console.log(`source: ${status.source}`);
  if (status.envVar) console.log(`env: ${status.envVar}`);
  if (status.tokenPreview) console.log(`token: ${status.tokenPreview}`);
  if (status.detail) console.log(`detail: ${status.detail}`);
}

async function installServiceTemplate() {
  const paths = serverPaths();
  await mkdir(paths.home, { recursive: true });
  const command = currentCliCommand();
  if (process.platform === "darwin") {
    const plistPath = join(process.env.HOME ?? ".", "Library", "LaunchAgents", "dev.clap.server.plist");
    await mkdir(dirname(plistPath), { recursive: true });
    await writeFile(plistPath, launchdPlist(command, paths));
    console.log(`installed ${plistPath}`);
    console.log(`start: launchctl load ${plistPath}`);
    console.log(`stop: launchctl unload ${plistPath}`);
    return;
  }
  if (process.platform === "linux") {
    const servicePath = join(process.env.HOME ?? ".", ".config", "systemd", "user", "clap.service");
    await mkdir(dirname(servicePath), { recursive: true });
    await writeFile(servicePath, systemdService(command, paths));
    console.log(`installed ${servicePath}`);
    console.log("start: systemctl --user daemon-reload && systemctl --user enable --now clap");
    console.log("logs: journalctl --user -u clap -f");
    return;
  }
  console.log("service install is supported on macOS launchd and Linux systemd user services");
}

async function chat(argv: string[]) {
  await startBackgroundServer({ quiet: true });
  const { flags, rest } = parseFlags(argv);
  const model = flags.model ?? process.env.CLAP_DEFAULT_MODEL;
  if (!model) {
    throw new Error("usage: clap chat --model <model|alias|path> [--backend mlx|gguf] [--no-stream] [prompt]");
  }
  await chatSession(model, rest.join(" "), flags);
}

async function run(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const model = rest[0];
  if (!model) {
    throw new Error("usage: clap run <model> [--backend mlx|gguf] [--no-stream] [prompt]");
  }

  await startBackgroundServer({ quiet: true });
  await chatSession(model, rest.slice(1).join(" "), flags);
}

async function chatSession(model: string, promptArg: string, flags: Record<string, string>) {
  const client = createClapClient();
  const backend = parseBackend(flags.backend);
  const stream = flags.stream !== "false";
  let prompt = promptArg;
  if (!prompt && !process.stdin.isTTY) {
    const piped = (await new Response(Bun.stdin.stream()).text()).trim();
    if (!piped) throw new Error("no prompt given; pass a prompt argument or pipe one on stdin");
    prompt = piped;
  }

  if (prompt) {
    await completeTurn(client, model, [{ role: "user", content: prompt }], backend, stream, flags);
    return;
  }

  console.log(`chatting with ${model} (type /bye to exit)`);
  const lines = createStdinLineReader();
  const messages: ChatCompletionRequest["messages"] = [];
  while (true) {
    process.stdout.write(">>> ");
    const line = await lines.readLine();
    if (line === null) {
      process.stdout.write("\n");
      break;
    }
    const input = line.trim();
    if (!input) continue;
    if (input === "/bye" || input === "/exit" || input === "/quit") break;
    if (input === "/clear") {
      messages.length = 0;
      console.log("(history cleared)");
      continue;
    }
    messages.push({ role: "user", content: input });
    try {
      const content = await completeTurn(client, model, messages, backend, stream, flags);
      messages.push({ role: "assistant", content });
    } catch (error) {
      messages.pop();
      printError(error);
    }
  }
  process.exit(0);
}

async function completeTurn(
  client: ReturnType<typeof createClapClient>,
  model: string,
  messages: ChatCompletionRequest["messages"],
  backend: "gguf" | "mlx" | undefined,
  stream: boolean,
  flags: Record<string, string>,
): Promise<string> {
  const request: ChatCompletionRequest = { model, messages, stream, backend };
  try {
    return await streamOrComplete(client, request);
  } catch (error) {
    if (!isModelNotCachedError(error)) throw error;
    console.error(`clap: model ${model} is not cached locally; pulling it now...`);
    await executePull(client, model, { backend, file: flags.file, quant: flags.quant, yes: true, force: false });
    return streamOrComplete(client, request);
  }
}

async function streamOrComplete(client: ReturnType<typeof createClapClient>, request: ChatCompletionRequest): Promise<string> {
  if (request.stream) {
    const tokens: string[] = [];
    for await (const token of client.streamChatCompletions(request)) {
      process.stdout.write(token);
      tokens.push(token);
    }
    process.stdout.write("\n");
    return tokens.join("");
  }
  const response = await client.chatCompletions(request);
  const content = typeof response.choices[0]?.message.content === "string" ? response.choices[0].message.content : "";
  console.log(content);
  return content;
}

function isModelNotCachedError(error: unknown): boolean {
  if (!(error instanceof ClapApiError)) return false;
  const code = typeof error.body === "object" && error.body && "error" in error.body
    ? (error.body as { error?: { code?: unknown } }).error?.code
    : undefined;
  return code === "not_downloaded" || code === "not_found" || code === "not_cached";
}

function createStdinLineReader(): { readLine(): Promise<string | null> } {
  const reader = Bun.stdin.stream().getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let ended = false;
  return {
    async readLine() {
      while (true) {
        const newline = buffer.indexOf("\n");
        if (newline >= 0) {
          const line = buffer.slice(0, newline);
          buffer = buffer.slice(newline + 1);
          return line;
        }
        if (ended) {
          if (!buffer) return null;
          const line = buffer;
          buffer = "";
          return line;
        }
        const chunk = await reader.read();
        if (chunk.done) {
          ended = true;
          continue;
        }
        buffer += decoder.decode(chunk.value, { stream: true });
      }
    },
  };
}

async function resolveCommand(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const model = rest[0];
  if (!model) throw new Error("usage: clap resolve <owner/model|alias> [--backend mlx|gguf] [--file model.gguf] [--quant Q4_K_M]");
  await startBackgroundServer({ quiet: true });
  const response = await createClapClient().resolveModel({ model, file: flags.file, backend: parseBackend(flags.backend), force: false });
  console.log(formatResolveOptions(response));
}

async function pull(argv: string[]) {
  const { flags, rest } = parseFlags(argv);
  const model = rest[0];
  if (!model) {
    throw new Error("usage: clap pull <owner/model> [--file model.gguf]");
  }

  await startBackgroundServer({ quiet: true });
  const client = createClapClient();
  await executePull(client, model, {
    backend: parseBackend(flags.backend),
    file: flags.file,
    quant: flags.quant,
    yes: flags.yes === "true",
    force: flags.force === "true",
  });
}

async function rmCommand(argv: string[]) {
  const { rest } = parseFlags(argv);
  const model = rest[0];
  if (!model) {
    throw new Error("usage: clap rm <owner/model|alias|owner/model:file.gguf>");
  }
  const health = await healthCheck(baseURLFromEnv());
  if (health?.status === "ok") {
    try {
      await createClapClient().unloadModel({ model });
    } catch {
      // model was not loaded; nothing to unload
    }
  }
  const removed = await removeModel(model);
  if (!removed.length) {
    throw new Error(`no cached files found for ${model}; see clap models`);
  }
  for (const path of removed) console.log(`removed ${path}`);
}

async function executePull(
  client: ReturnType<typeof createClapClient>,
  model: string,
  flags: { backend?: "gguf" | "mlx"; file?: string; quant?: string; yes: boolean; force: boolean },
) {
  const backend = flags.backend;
  const resolve = await client.resolveModel({ model, file: flags.file, backend, force: false });
  const selected = await selectPullOption(resolve, { backend, file: flags.file, quant: flags.quant, yes: flags.yes });
  const request = { model, file: selected?.file ?? flags.file, backend: selected?.backend ?? backend, force: flags.force };
  if (selected) console.log(`selected: ${selected.backend}/${selected.format}${selected.quantization ? ` ${selected.quantization}` : ""}${selected.file ? ` ${selected.file}` : ""}`);
  console.log(`pulling ${model}${request.file ? ` (${request.file})` : ""}${request.backend ? ` via ${request.backend}` : ""}...`);
  let activeDownloadId: string | undefined;
  const disposeCancellationHandler = installPullCancellationHandler(async () => {
    if (activeDownloadId) await client.cancelDownload(activeDownloadId);
  });
  try {
    const response = await pullWithAuthRetry(client, request);
    activeDownloadId = response.download.id;
    let download = await waitForDownload(client, activeDownloadId);
    if (download.status === "failed" && isHfAuthError(new Error(download.error ?? ""))) {
      if (!process.stdin.isTTY) {
        throw new Error(`${download.error} Non-interactive shell: ${hfAuthGuidance()}`);
      }
      console.error("Hugging Face authentication is required for this repo.");
      const token = await readSecret("Hugging Face token: ");
      const status = await storeHfToken(token);
      console.error(`Saved Hugging Face token (${status.source}${status.tokenPreview ? `, ${status.tokenPreview}` : ""}); retrying pull...`);
      const retry = await client.pullModel(request);
      activeDownloadId = retry.download.id;
      download = await waitForDownload(client, activeDownloadId);
    }
    console.log(`status: ${download.status}`);
    if (download.modelPath) console.log(`model: ${download.modelPath}`);
    if (download.bytesReceived) console.log(`bytes: ${formatBytes(download.bytesReceived)}`);
    if (download.status === "cancelled") throw new Error("pull cancelled");
    if (download.error) throw new Error(download.error);
  } finally {
    disposeCancellationHandler();
  }
}

async function selectPullOption(response: ModelResolveResponse, flags: { backend?: "gguf" | "mlx"; file?: string; quant?: string; yes?: boolean }): Promise<ModelResolveOption | undefined> {
  const options = supportedOptions(response);
  if (flags.quant) {
    const match = findOptionByQuant(options, flags.quant);
    if (!match) throw new Error(`no supported GGUF option found for --quant ${flags.quant}`);
    return match;
  }
  if (flags.backend || flags.file || options.length <= 1 || !process.stdin.isTTY || !process.stdout.isTTY || flags.yes) {
    return response.selected ?? options[0];
  }
  console.log(formatResolveOptions(response));
  process.stdout.write(`Select option [${options.findIndex((option) => option.recommended) + 1 || 1}]: `);
  return chooseOptionByInput(options, await readLineFromStdin());
}

async function waitForDownload(client: ReturnType<typeof createClapClient>, id: string): Promise<Download> {
  let tick = 0;
  let lastLine = "";
  while (true) {
    const download = (await client.downloads()).downloads.find((entry) => entry.id === id);
    if (!download) throw new Error(`download not found: ${id}`);
    const line = formatDownloadProgress(download, tick++);
    if (process.stderr.isTTY) {
      process.stderr.write(`\r${line}${" ".repeat(Math.max(0, lastLine.length - line.length))}`);
      lastLine = line;
    } else {
      process.stderr.write(`${line}\n`);
    }
    if (download.status === "completed" || download.status === "failed" || download.status === "cancelled") {
      if (process.stderr.isTTY) process.stderr.write("\n");
      return download;
    }
    await Bun.sleep(250);
  }
}

async function pullWithAuthRetry(client: ReturnType<typeof createClapClient>, request: { model: string; file?: string; backend?: "gguf" | "mlx"; force: boolean }) {
  try {
    return await client.pullModel(request);
  } catch (error) {
    if (!(error instanceof ClapApiError) || !isHfAuthError(error)) throw error;
    if (!process.stdin.isTTY) {
      throw new Error(`${error.message} Non-interactive shell: ${hfAuthGuidance()}`);
    }
    console.error("Hugging Face authentication is required for this repo.");
    const token = await readSecret("Hugging Face token: ");
    const status = await storeHfToken(token);
    console.error(`Saved Hugging Face token (${status.source}${status.tokenPreview ? `, ${status.tokenPreview}` : ""}); retrying pull...`);
    return client.pullModel(request);
  }
}

function parseFlags(argv: string[]) {
  const flags: Record<string, string> = {};
  const rest: string[] = [];

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--stream") {
      flags.stream = "true";
    } else if (arg === "--no-stream") {
      flags.stream = "false";
    } else if (arg === "--aliases") {
      flags.aliases = "true";
    } else if (arg === "--json") {
      flags.json = "true";
    } else if (arg === "--active") {
      flags.active = "true";
    } else if (arg === "--force") {
      flags.force = "true";
    } else if (arg === "--yes" || arg === "-y") {
      flags.yes = "true";
    } else if (arg === "--model" || arg === "-m") {
      flags.model = argv[++index];
    } else if (arg === "--file" || arg === "-f") {
      flags.file = argv[++index];
    } else if (arg === "--backend") {
      flags.backend = argv[++index];
    } else if (arg === "--quant") {
      flags.quant = argv[++index];
    } else if (arg === "--keep-alive") {
      flags.keepAlive = argv[++index];
    } else if (arg === "--token") {
      flags.token = argv[++index];
    } else {
      rest.push(arg);
    }
  }

  return { flags, rest };
}

function parseBackend(value: string | undefined): "gguf" | "mlx" | undefined {
  if (value === undefined) return undefined;
  if (value === "gguf" || value === "mlx") return value;
  throw new Error(`--backend must be mlx or gguf, got: ${value}`);
}

async function readTokenForLogin(): Promise<string> {
  if (!process.stdin.isTTY) {
    const text = await new Response(Bun.stdin.stream()).text();
    if (text.trim()) return text.trim();
    throw new Error("usage: clap auth login [--token hf_xxx] (or pipe token on stdin)");
  }
  return readSecret("Hugging Face token: ");
}

async function readSecret(prompt: string): Promise<string> {
  process.stderr.write(prompt);
  if (!process.stdin.isTTY) {
    const text = await new Response(Bun.stdin.stream()).text();
    return text.trim();
  }
  await runTtyCommand("stty -echo");
  try {
    const token = await readLineFromStdin();
    process.stderr.write("\n");
    return token.trim();
  } finally {
    await runTtyCommand("stty echo");
  }
}

async function readLineFromStdin(): Promise<string> {
  const reader = Bun.stdin.stream().getReader();
  const decoder = new TextDecoder();
  let value = "";
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      value += decoder.decode(chunk.value);
      const newline = value.indexOf("\n");
      if (newline >= 0) return value.slice(0, newline);
    }
    return value;
  } finally {
    reader.releaseLock();
  }
}

async function runTtyCommand(command: string): Promise<void> {
  const proc = Bun.spawn(["/bin/sh", "-lc", command], { stdin: "inherit", stdout: "ignore", stderr: "ignore" });
  await proc.exited;
}

function printError(error: unknown) {
  if (error instanceof ClapApiError) {
    console.error(`clap: server request failed (${error.status}): ${error.message}`);
    console.error(`clap: is the server running? try: clap server start`);
    return;
  }
  console.error(`clap: ${error instanceof Error ? error.message : String(error)}`);
}

function help() {
  console.log(`Usage:
  clap serve
  clap auth login|logout|status
  clap server start|stop|status|restart|logs|install [--force]
  clap models [list] [--aliases] [--json] [--active]
  clap load <model|alias|path> [--backend mlx|gguf] [--keep-alive 15m|1h|always]
  clap unload <model|alias|path> [--backend mlx|gguf]
  clap resolve <owner/model|alias> [--backend mlx|gguf] [--file model.gguf] [--quant Q4_K_M]
  clap pull <owner/model|alias> [--backend mlx|gguf] [--file model.gguf] [--quant Q4_K_M] [--yes] [--force]
  clap rm <owner/model|alias|owner/model:file.gguf>
  clap chat --model <model|alias|path> [--backend mlx|gguf] [--no-stream] [prompt]
  clap run <model|alias> [--backend mlx|gguf] [--no-stream] [prompt]

Without a prompt, run/chat open an interactive session (/bye to exit, /clear to reset history).
Models that are not cached locally are pulled automatically before the first reply.

Environment:
  CLAP_HOME      State/log directory (default: ~/.clap)
  CLAP_BASE_URL  Server URL (default: ${defaultBaseURL})
  CLAP_DEFAULT_MODEL  Default model for clap chat
  CLAP_HF_ENDPOINT  Hugging Face endpoint (default: https://huggingface.co)
  CLAP_HF_TOKEN, HF_TOKEN, HUGGINGFACE_HUB_TOKEN, HUGGINGFACE_TOKEN  Hugging Face token`);
}
