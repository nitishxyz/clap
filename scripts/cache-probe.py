#!/usr/bin/env python3
"""Measure KV cache reuse against a running clap server.

Sends a large shared system prompt across one or more sessions and reports what
the cache actually did per request, then the server's own totals. Use this to
tell the two failure modes apart:

  low hit rate   -> prefixes are not matching (prompt layout, isolation)
  high hit rate  -> prefixes match but anchors cannot be retained
  + low depth       (check pressureState / evictionCount / retainedBytes)

usage:
  cache-probe.py <base-url> <model> [--sessions N] [--turns N] [--key K]
"""

import argparse
import json
import time
import urllib.request

PARA = (
    "Reference manual section: the coordinator plans key-value reuse against "
    "authenticated identities, generations, and leases. Adapters execute only "
    "validated physical copy, trim, and clear operations. Prefill budgets "
    "bound aggregate prompt ingest per scheduler step, and autofit sizes the "
    "pool to measured available memory. "
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("model")
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--paragraphs", type=int, default=110, help="system prompt size")
    ap.add_argument("--key", default=None, help="API key when enforcement is on")
    ap.add_argument("--reset", action="store_true", help="reset dashboard totals first")
    args = ap.parse_args()

    system = ("You are a terse assistant.\n\n" + PARA * args.paragraphs
              + "\nAnswer in one short sentence.")
    run = str(int(time.time()))

    def call(path, data=None, method=None):
        req = urllib.request.Request(args.base + path, data=data, method=method)
        if args.key:
            req.add_header("Authorization", "Bearer " + args.key)
        if data:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = resp.read()
        return json.loads(body) if body else {}

    if args.reset:
        call("/clap/v1/dashboard", method="DELETE")

    convs = {s: [{"role": "system", "content": system}] for s in range(args.sessions)}
    print(f"model={args.model}  sessions={args.sessions}  turns={args.turns}")
    for turn in range(1, args.turns + 1):
        for s in range(args.sessions):
            convs[s].append({"role": "user", "content": f"Turn {turn}: state one short fact."})
            started = time.monotonic()
            resp = call("/v1/chat/completions", json.dumps({
                "model": args.model, "messages": convs[s], "stream": False,
                "max_tokens": 12, "temperature": 0,
                "cache": {"session": f"probe-{s}-{run}", "project": "cache-probe"},
            }).encode())
            elapsed = time.monotonic() - started
            convs[s].append({"role": "assistant",
                             "content": resp["choices"][0]["message"].get("content") or "ok"})
            flag = "COLD" if elapsed > 20 else ""
            print(f"  turn{turn} session{s}: {elapsed:7.1f}s  "
                  f"prompt_tokens={resp['usage']['prompt_tokens']} {flag}")

    dash = call("/clap/v1/dashboard")
    needle = args.model.split("/")[-1].lower()
    rows = [r for r in sorted(dash.get("requests", []), key=lambda x: x.get("startedAt") or 0)
            if (r.get("source") or "live") == "live"
            and needle in str(r.get("model", "")).lower()
            and (r.get("promptTokens") or 0) > 0]

    print("\nper-request cache decisions:")
    for i, r in enumerate(rows, 1):
        reused, prompt = r.get("reusedTokens") or 0, r.get("promptTokens") or 0
        # Reserve 100% for complete reuse; a multi-turn request always
        # re-prefills its newest message.
        pct = 100 if reused >= prompt else int(100 * reused / prompt)
        ttft = r.get("ttftMs")
        print(f"  {i:2d}: hit={str(r.get('cacheHit')):5s} kind={str(r.get('reuseKind')):7s} "
              f"reused={reused}/{prompt} ({pct}%) residual={prompt - reused:5d} "
              f"ttft={(ttft / 1000 if ttft else 0):.1f}s")

    totals = dash["totals"]
    hits, misses = totals["physicalCacheHits"], totals["physicalCacheMisses"]
    print()
    if hits + misses:
        print(f"hit rate          {100 * hits / (hits + misses):.1f}%  ({hits} hit / {misses} miss)")
    if totals["physicalPromptTokens"]:
        saved = 100 * totals["physicalReusedTokens"] / totals["physicalPromptTokens"]
        print(f"prompt work saved {saved:.1f}%  "
              f"({totals['physicalReusedTokens']:,} of {totals['physicalPromptTokens']:,} tokens)")

    # Retention pressure explains a high hit rate with shallow reuse.
    for model in call("/clap/v1/runtime/models").get("models", []):
        retention = (model.get("worker") or {}).get("retention") or {}
        if not retention:
            continue
        mib = 1024 * 1024
        print(f"\nretention [{model['id']}]: pressure={retention.get('pressureState')} "
              f"evictions={retention.get('evictionCount')} "
              f"anchors={retention.get('retainedAnchors')} "
              f"retained={(retention.get('retainedBytes') or 0) / mib:.0f} MiB "
              f"of high={(retention.get('highWatermarkBytes') or 0) / mib:.0f} MiB")


if __name__ == "__main__":
    main()
