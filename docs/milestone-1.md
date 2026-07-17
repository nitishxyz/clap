# Milestone 1 Working Slice

This repository now contains a minimal Bun/TypeScript workspace for the Clap local model server.

## Packages

- `apps/cli` - user-facing `clap` CLI.
- `packages/api` - Zod schemas, OpenAPI document generation, and a generated-SDK-shaped TypeScript client.
- `packages/server` - Hono local server.
- `packages/models` - model inventory, aliases, and Hugging Face cache resolution.
- `packages/runtime-router` - backend selection surface.
- `packages/runtime-llama` - llama.cpp backend worker integration.
- `packages/runtime-mlx` - Swift MLX backend worker integration.

## Run

```bash
bun install
bun run generate:openapi
bun run serve
```

In another shell:

```bash
bun run cli server status
bun run cli chat --model llama3.2:3b --backend gguf "hello"
bun run cli chat --model llama3.2:3b --backend gguf --stream "hello"
bun run cli run llama3.2:3b --backend gguf "hello"
```

The server listens on `http://localhost:11435` by default. Set `CLAP_BASE_URL` to point the CLI/client at another server.

## Implemented API

Management API:

- `GET /clap/v1/health`
- `GET /clap/v1/runtime`
- `GET /clap/v1/backends`
- `GET /clap/v1/models`

OpenAI-compatible API:

- `GET /v1/models`
- `POST /v1/chat/completions`

`POST /v1/chat/completions` requires a real cached model, alias, local `.gguf` file, or MLX directory. When `stream: true`, it returns OpenAI-style Server-Sent Events and terminates with `data: [DONE]`.

## OpenAPI and SDK

Zod schemas in `packages/api/src/schemas.ts` are the source of truth. `packages/api/src/openapi.ts` generates `packages/api/openapi.json` and the server exposes the same document at `/openapi.json`.

The TypeScript client in `packages/api/src/client.ts` is intentionally shaped like a generated SDK while the project bootstraps. A future Hey API pipeline can consume `packages/api/openapi.json` and replace the manual client without changing CLI call sites.
