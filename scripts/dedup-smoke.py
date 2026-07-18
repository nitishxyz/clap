#!/usr/bin/env python3
"""Shared-prefix dedup smoke test for the clap-llama worker.

Usage: python3 scripts/dedup-smoke.py <worker-binary> <gguf-path>

Runs two sessions with an identical large system prompt and asserts the
second session reuses the shared prefix (cache.reused_tokens > 0) without
having a same-session transcript match.
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
    stderr=subprocess.STDOUT if "-v" in sys.argv else subprocess.DEVNULL,
    text=True, bufsize=1,
)

done = {}
cache = {}
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
        with lock:
            if msg.get("cache"):
                cache[rid] = msg["cache"]
            if msg.get("done") or msg.get("error"):
                done[rid] = msg.get("error") or "ok"

threading.Thread(target=reader, daemon=True).start()

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()

def wait(rid, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with lock:
            if rid in done:
                return done[rid]
        time.sleep(0.1)
    return "timeout"

SYSTEM = "You are a precise assistant for a large organization. " + " ".join(
    f"Policy {i}: always verify facts, cite sources, keep answers short, respect user preferences."
    for i in range(80)
)

send({"id": "load", "type": "load", "model": MODEL})
send({"id": "s1", "type": "chat", "model": MODEL, "max_tokens": 8, "temperature": 0.0,
      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Name one ocean."}]})
print("s1:", wait("s1"), "cache:", cache.get("s1"))

send({"id": "s2", "type": "chat", "model": MODEL, "max_tokens": 8, "temperature": 0.0,
      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Name one mountain."}]})
print("s2:", wait("s2"), "cache:", cache.get("s2"))

send({"id": "s3", "type": "chat", "model": MODEL, "max_tokens": 8, "temperature": 0.0,
      "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Name one river."}]})
print("s3:", wait("s3"), "cache:", cache.get("s3"))

send({"id": "bye", "type": "shutdown"})
time.sleep(0.3)
proc.terminate()

reused2 = (cache.get("s2") or {}).get("reused_tokens", 0)
reused3 = (cache.get("s3") or {}).get("reused_tokens", 0)
ok = done.get("s1") == "ok" and done.get("s2") == "ok" and done.get("s3") == "ok" and (reused2 > 100 or reused3 > 100)
print(f"s2 reused={reused2} s3 reused={reused3}")
print("DEDUP-PASS" if ok else "DEDUP-FAIL")
sys.exit(0 if ok else 1)
