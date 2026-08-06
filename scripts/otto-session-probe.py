#!/usr/bin/env python3
"""Replay an otto-shaped coding session against a clap server.

The cache probe sends a synthetic block of filler text. This sends what an
agent harness actually sends: a long system prompt with tool schemas, then
alternating user/assistant/tool turns that grow the transcript. It reports
what the cache did per turn so agent-shaped traffic can be told apart from
probe-shaped traffic.

usage:
  otto-session-probe.py <base-url> <model> [--turns N] [--key K]
"""

import argparse
import json
import time
import urllib.request

SYSTEM = """You are otto, a CLI coding agent. Use tools to complete software tasks accurately.

## Priorities
Follow conversation instructions first, then project AGENTS.md, then defaults.
Treat current file contents as truth. Inspect code and dependencies before
changing them; preserve local style and solve root causes. Be concise and
direct. Reference code as path:line and avoid repeating tool output.

## Execution
Unless the user asks only for advice, implement the requested change
end-to-end. Infer routine details from the repository. Ask one focused
question only when blocked by ambiguity, a destructive action, or missing
credentials. Prefer the smallest correct change. Keep unrelated user changes
intact. For non-trivial work: inspect, plan, implement, run focused checks,
then review the diff. Never commit unless asked.

## Workspace safety
Never revert changes you did not make, amend commits, or run destructive git
commands unless explicitly requested. Use non-interactive git commands.
Verify a dependency exists before using it.

## Available tools
- read(path, startLine?, endLine?): read a text file from the workspace
- write(path, content): create or overwrite a file
- edit(path, oldString, newString): replace one exact block in a file
- search(query, mode, path, glob?): regex content search or glob file search
- shell(cmd, cwd?, timeout?): run a non-interactive command
- ls(path): list a directory
- tree(path, depth?): render a directory tree with line counts
- git_status(): show working tree status
- update_todos(todos): maintain a task plan for multi-step work

Tool results arrive as tool messages. Never emit raw harness syntax.
"""

# A plausible repository context block, the kind a harness pins into the
# system prompt so the model does not have to rediscover layout every turn.
CONTEXT = """
## Repository context

packages/server/src/index.ts (4210 lines) - HTTP surface, OpenAI-compatible
packages/server/src/config.ts (312 lines) - config schema and env mapping
packages/server/src/process-usage.ts (98 lines) - CPU and memory sampling
packages/runtime-router/src/process/resident-worker-process.ts (612 lines)
packages/runtime-llama/src/index.ts (140 lines) - worker discovery
native/llama/src/worker.cpp (430 lines) - worker event loop
native/llama/src/cache-executor.cpp (368 lines) - slot admission
native/llama/src/model-runtime.cpp (192 lines) - model load and capabilities
native/cache/crates/clap-cache-core/src/lib.rs (1799 lines) - coordinator
apps/cli/src/index.ts (210 lines) - CLI entry
"""

TURNS = [
    "Where does the server decide how much system memory is available?",
    "Read that function and tell me whether it is container-aware.",
    "What would you change to make it respect a cgroup limit?",
    "Which tests would you add for that change?",
    "Summarize the change you would make in two sentences.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument("--key", default=None)
    ap.add_argument("--session", default=None)
    args = ap.parse_args()

    run = args.session or f"otto-{int(time.time())}"

    def call(path, data=None, method=None):
        req = urllib.request.Request(args.base + path, data=data, method=method)
        req.add_header("User-Agent", "otto-session-probe/1.0")
        if args.key:
            req.add_header("Authorization", "Bearer " + args.key)
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = resp.read()
        return json.loads(body) if body else {}

    messages = [{"role": "system", "content": SYSTEM + CONTEXT}]
    print(f"model={args.model}  session={run}")
    for turn in range(min(args.turns, len(TURNS))):
        messages.append({"role": "user", "content": TURNS[turn]})
        started = time.monotonic()
        resp = call("/v1/chat/completions", json.dumps({
            "model": args.model, "messages": messages, "stream": False,
            "max_tokens": 400, "temperature": 0,
            "cache": {"session": run, "project": "otto-session"},
        }).encode())
        elapsed = time.monotonic() - started
        message = resp["choices"][0]["message"]
        reply = message.get("content") or ""
        messages.append({"role": "assistant", "content": reply or "ok"})
        usage = resp["usage"]
        print(f"\n--- turn {turn + 1}  {elapsed:.1f}s  "
              f"prompt={usage['prompt_tokens']} completion={usage['completion_tokens']}")
        print(f"  Q: {TURNS[turn]}")
        print(f"  A: {' '.join(reply.split())[:300] or '(empty)'}")

    dash = call("/clap/v1/dashboard")
    needle = args.model.split("/")[-1].lower()
    rows = [r for r in sorted(dash.get("requests", []), key=lambda x: x.get("startedAt") or 0)
            if (r.get("source") or "live") == "live"
            and needle in str(r.get("model", "")).lower()
            and (r.get("promptTokens") or 0) > 0][-args.turns:]

    print("\nper-request cache decisions:")
    for i, r in enumerate(rows, 1):
        reused, prompt = r.get("reusedTokens") or 0, r.get("promptTokens") or 0
        pct = 100 if reused >= prompt else int(100 * reused / prompt)
        ttft = r.get("ttftMs")
        print(f"  {i:2d}: hit={str(r.get('cacheHit')):5s} kind={str(r.get('reuseKind')):7s} "
              f"reused={reused}/{prompt} ({pct}%) residual={prompt - reused:5d} "
              f"ttft={(ttft / 1000 if ttft else 0):.1f}s")


if __name__ == "__main__":
    main()
