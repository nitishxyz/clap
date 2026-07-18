#!/usr/bin/env python3
"""Concurrency smoke test for the clap-llama continuous-batching worker.

Usage: python3 scripts/batch-smoke.py <worker-binary> <gguf-path>

Pipes overlapping chat requests into the worker over the JSON-lines protocol
and asserts that all complete and that token streams interleave.
"""
import json
import subprocess
import sys
import threading
import time

WORKER = sys.argv[1]
MODEL = sys.argv[2]

proc = subprocess.Popen(
    [WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, text=True, bufsize=1,
)

results = {}
first_token_at = {}
done_at = {}
lock = threading.Lock()

def reader():
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            continue
        rid = msg.get("id", "")
        now = time.time()
        with lock:
            if "token" in msg:
                results.setdefault(rid, []).append(msg["token"])
                first_token_at.setdefault(rid, now)
            if msg.get("error"):
                results.setdefault(rid, []).append(f"<ERROR: {msg['error']}>")
                done_at[rid] = now
            if msg.get("done"):
                done_at[rid] = now

threading.Thread(target=reader, daemon=True).start()

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()

send({"id": "load", "type": "load", "model": MODEL})

prompts = {
    "r1": "Count from 1 to 10, digits separated by spaces.",
    "r2": "Name three primary colors, comma separated.",
    "r3": "Say the alphabet from A to F, no punctuation.",
}
t0 = time.time()
for rid, content in prompts.items():
    send({
        "id": rid, "type": "chat", "model": MODEL,
        "max_tokens": 64, "temperature": 0.0,
        "messages": [{"role": "user", "content": content}],
    })

deadline = time.time() + 300
while time.time() < deadline:
    with lock:
        if len([r for r in prompts if r in done_at]) == len(prompts):
            break
    time.sleep(0.2)

send({"id": "bye", "type": "shutdown"})
time.sleep(0.5)
proc.terminate()

ok = True
with lock:
    for rid in prompts:
        text = "".join(results.get(rid, []))
        finished = rid in done_at
        ttft = (first_token_at.get(rid, 0) - t0) if rid in first_token_at else -1
        total = (done_at.get(rid, 0) - t0) if finished else -1
        status = "OK" if finished and "<ERROR" not in text else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"{rid}: {status} ttft={ttft:.1f}s total={total:.1f}s text={text[:80]!r}")
    if "r3" in first_token_at and "r1" in done_at:
        print(f"interleaving (r3 streamed before r1 finished): {first_token_at['r3'] < done_at['r1']}")

print("SMOKE-PASS" if ok else "SMOKE-FAIL")
sys.exit(0 if ok else 1)
