#!/usr/bin/env python3
"""Print the live request log from a clap dashboard, with cache and error detail.

The dashboard UI truncates error text and hides slot/fallback fields. When a
session misbehaves (reuse collapsing, TTFT spikes, requests erroring) this
prints the per-request facts needed to explain it.

usage:
  dashboard-inspect.py <base-url> [--key K] [--last N]
"""

import argparse
import json
import time
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("--key", default=None)
    ap.add_argument("--last", type=int, default=0, help="only the last N live requests")
    args = ap.parse_args()

    req = urllib.request.Request(args.base.rstrip("/") + "/clap/v1/dashboard")
    req.add_header("User-Agent", "clap-dashboard-inspect/1.0")
    if args.key:
        req.add_header("Authorization", "Bearer " + args.key)
    with urllib.request.urlopen(req, timeout=120) as resp:
        dashboard = json.loads(resp.read())

    rows = sorted(dashboard.get("requests", []), key=lambda row: row.get("startedAt") or 0)
    live = [row for row in rows if (row.get("source") or "live") == "live"]
    if args.last:
        live = live[-args.last:]

    print(f"{len(live)} live requests")
    for row in live:
        prompt = row.get("promptTokens") or 0
        reused = row.get("reusedTokens") or 0
        pct = f"{100 * reused // prompt}%" if prompt else "-"
        started = row.get("startedAt")
        # startedAt is epoch millis on some builds, an ISO string on others.
        if isinstance(started, (int, float)):
            started = time.strftime("%H:%M:%S", time.localtime(started / 1000))
        else:
            started = str(started or "")[11:19]
        print("\n{}  {}  in={} reused={} ({})  slot={}  kind={}".format(
            started, row.get("status"),
            prompt, reused, pct, row.get("cacheTargetSlot"), row.get("reuseKind")))
        print("    ttft={:.1f}s queue={:.1f}s total={:.1f}s finish={}".format(
            (row.get("ttftMs") or 0) / 1000, (row.get("queueMs") or 0) / 1000,
            (row.get("totalMs") or 0) / 1000, row.get("finishReason")))
        for key in ("error", "errorMessage", "errorCode", "cacheFallback",
                    "cacheReuseScope", "cancelReason"):
            value = row.get(key)
            if value:
                print(f"    {key}: {json.dumps(value)[:600]}")

    totals = dashboard.get("totals") or {}
    print("\ntotals:", json.dumps({k: v for k, v in totals.items() if "hysical" in k or "ancel" in k}, indent=2))


if __name__ == "__main__":
    main()
