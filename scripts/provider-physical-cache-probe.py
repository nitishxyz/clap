#!/usr/bin/env python3
"""Run real shared-prefix, branch, and multi-turn workloads against Clap."""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("CLAP_BASE_URL", "http://127.0.0.1:11435").rstrip("/")
MODEL = os.environ.get("CLAP_VALIDATION_MODEL")
API_KEY = os.environ.get("CLAP_API_KEY")
NAMESPACE = os.environ.get("CLAP_VALIDATION_NAMESPACE", "phase567-validation")
HARNESS_TOKENS = int(os.environ.get("CLAP_VALIDATION_HARNESS_WORDS", "2048"))
REPEATS = int(os.environ.get("CLAP_VALIDATION_REPEATS", "3"))
EXPECT_KIND = os.environ.get("CLAP_EXPECT_ADAPTER_KIND")
EXPECT_FORMAT = os.environ.get("CLAP_EXPECT_CACHE_FORMAT")
EXPECT_ACCOUNTING = os.environ.get("CLAP_EXPECT_BYTE_ACCOUNTING")

if not MODEL:
    raise SystemExit("CLAP_VALIDATION_MODEL is required")


def request(path: str, body: dict | None = None, timeout: int = 600):
    data = None if body is None else json.dumps(body).encode()
    headers = {"accept": "application/json", "user-agent": "clap-phase567-probe/1"}
    if body is not None:
        headers["content-type"] = "application/json"
    if API_KEY:
        headers["authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            if "json" in response.headers.get("content-type", ""):
                return json.loads(payload)
            return payload.decode()
    except urllib.error.HTTPError as error:
        payload = error.read().decode(errors="replace")
        raise RuntimeError(f"{path}: HTTP {error.code}: {payload}") from error


def find_adapter(value):
    if isinstance(value, dict):
        candidate = value.get("adapter")
        if isinstance(candidate, dict) and candidate.get("contract_version") == 1:
            return candidate
        for child in value.values():
            found = find_adapter(child)
            if found:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_adapter(child)
            if found:
                return found
    return None


def latest_record(previous_ids: set[str]):
    deadline = time.time() + 15
    while time.time() < deadline:
        dashboard = request("/clap/v1/dashboard")
        records = dashboard.get("requests", []) + dashboard.get("active", [])
        for record in records:
            if record.get("id") not in previous_ids:
                return record
        time.sleep(0.025)
    raise RuntimeError("dashboard did not publish the completed request")


def dashboard_ids():
    dashboard = request("/clap/v1/dashboard")
    return {item.get("id") for item in dashboard.get("requests", []) + dashboard.get("active", [])}


harness = " ".join(f"shared-rule-{index % 97}" for index in range(HARNESS_TOKENS))
system = {"role": "system", "content": (
    "You are validating immutable KV prefix sharing. Follow this exact synthetic harness and "
    "answer every user with only the requested short marker.\n" + harness
)}


def chat(session: str, messages: list[dict], marker: str):
    before = dashboard_ids()
    started = time.perf_counter()
    response = request("/v1/chat/completions", {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 8,
        "stream": False,
        "cache": {
            "namespace": NAMESPACE,
            "project": "phase567",
            "harness": "shared-provider-harness",
            "session": session,
            "priority": "normal",
        },
    })
    elapsed_ms = (time.perf_counter() - started) * 1000
    content = response["choices"][0]["message"].get("content") or ""
    record = latest_record(before)
    return {
        "marker": marker,
        "session": session,
        "elapsed_ms": elapsed_ms,
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "usage": response.get("usage"),
        "cache": record.get("cache"),
        "timing": record.get("timing"),
        "routing": record.get("routing"),
    }, content


health = request("/clap/v1/health")
if health.get("status") != "ok":
    raise RuntimeError(f"unexpected health response: {health}")
if "<title>clap</title>" not in request("/"):
    raise RuntimeError("public dashboard HTML was not served")

seed_prompt = [system, {"role": "user", "content": "Return marker ALPHA."}]
results = []
seed, seed_content = chat("session-a", seed_prompt, "seed-a")
results.append(seed)
for index in range(REPEATS):
    repeat, _ = chat(f"repeat-{index}", seed_prompt, f"repeat-{index}")
    results.append(repeat)

history = seed_prompt + [{"role": "assistant", "content": seed_content}]
follow_prompt = history + [{"role": "user", "content": "Return marker BETA."}]
follow, follow_content = chat("session-a", follow_prompt, "follow-a")
results.append(follow)

branch_prompt = history + [{"role": "user", "content": "Return marker GAMMA."}]
branch, _ = chat("session-b", branch_prompt, "branch-b")
results.append(branch)

third_prompt = follow_prompt + [
    {"role": "assistant", "content": follow_content},
    {"role": "user", "content": "Return marker DELTA."},
]
third, _ = chat("session-a", third_prompt, "third-a")
results.append(third)

runtime = request("/clap/v1/dashboard")
adapter = find_adapter(runtime)
if not adapter:
    raise RuntimeError("loaded worker did not publish a physical cache adapter")
if EXPECT_KIND and adapter.get("kind") != EXPECT_KIND:
    raise RuntimeError(f"adapter kind mismatch: expected {EXPECT_KIND}, observed {adapter.get('kind')}")
if EXPECT_FORMAT and adapter.get("format", {}).get("cache_format") != EXPECT_FORMAT:
    raise RuntimeError(f"cache format mismatch: expected {EXPECT_FORMAT}, observed {adapter.get('format')}")
if EXPECT_ACCOUNTING and adapter.get("constraints", {}).get("byte_accounting") != EXPECT_ACCOUNTING:
    raise RuntimeError("adapter byte accounting mismatch")

metrics_text = request("/metrics")
summary = {
    "status": "ok",
    "base_url": BASE_URL,
    "model": MODEL,
    "harness_words": HARNESS_TOKENS,
    "adapter": adapter,
    "requests": results,
    "elapsed_ms": {
        "min": min(item["elapsed_ms"] for item in results),
        "median": statistics.median(item["elapsed_ms"] for item in results),
        "max": max(item["elapsed_ms"] for item in results),
    },
    "metrics_sha256": hashlib.sha256(metrics_text.encode()).hexdigest(),
    "retention": runtime.get("loadedModels") or runtime.get("models"),
}
print(json.dumps(summary, indent=2, sort_keys=True))
