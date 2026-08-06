#!/usr/bin/env python3
"""Exercise Clap Phase 2 routing against a live multi-replica deployment."""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("CLAP_BASE_URL", "http://127.0.0.1:11435").rstrip("/")
MODEL = os.environ.get("CLAP_VALIDATION_MODEL")
API_KEY = os.environ.get("CLAP_API_KEY")
NAMESPACE = os.environ.get("CLAP_VALIDATION_NAMESPACE", "phase2-public-fixture")
SEED_SESSIONS = int(os.environ.get("CLAP_ROUTING_SEED_SESSIONS", "12"))

if not MODEL:
    raise SystemExit("CLAP_VALIDATION_MODEL is required")


def request(path: str, body: dict | None = None, timeout: int = 300):
    data = None if body is None else json.dumps(body).encode()
    headers = {"accept": "application/json"}
    if body is not None:
        headers["content-type"] = "application/json"
    if API_KEY:
        headers["authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content_type = response.headers.get("content-type", "")
            payload = response.read()
            return json.loads(payload) if "json" in content_type else payload.decode()
    except urllib.error.HTTPError as error:
        payload = error.read().decode(errors="replace")
        raise RuntimeError(f"{path}: HTTP {error.code}: {payload}") from error


SYSTEM_MESSAGE = {"role": "system", "content": "Phase 2 public routing fixture. Answer briefly."}


def chat(session: str, label: str, max_tokens: int = 8, padding: str = "",
         history: list[dict] | None = None):
    messages = list(history or [SYSTEM_MESSAGE])
    messages.append({"role": "user", "content": f"{label}: answer with one word. {padding}"})
    return request("/v1/chat/completions", {
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "cache": {
            "namespace": NAMESPACE,
            "project": "provider-routing-validation",
            "session": session,
            "priority": "normal",
        },
    })


def dashboard_ids() -> set[str]:
    dashboard = request("/clap/v1/dashboard")
    records = dashboard.get("active", []) + dashboard.get("requests", [])
    return {record["id"] for record in records}


def dashboard_record_after(previous_ids: set[str], active: bool | None = None):
    deadline = time.time() + 10
    while time.time() < deadline:
        dashboard = request("/clap/v1/dashboard")
        records = dashboard.get("active", []) + dashboard.get("requests", [])
        for record in records:
            if record.get("id") in previous_ids:
                continue
            if active is not None and (record.get("status") == "active") != active:
                continue
            return record
        time.sleep(0.025)
    raise RuntimeError("new dashboard record was not observed")


health = request("/clap/v1/health")
if health.get("status") != "ok":
    raise RuntimeError(f"unexpected health response: {health}")
root = request("/")
if "<title>clap</title>" not in root:
    raise RuntimeError("public dashboard HTML was not served")

seed_routes: dict[str, dict] = {}
session_histories: dict[str, list[dict]] = {}
for index in range(SEED_SESSIONS):
    label = f"phase2-seed-{index}"
    session = f"session-{index}"
    previous_ids = dashboard_ids()
    response = chat(session, label)
    record = dashboard_record_after(previous_ids, active=False)
    routing = record.get("routing")
    if not routing:
        raise RuntimeError(f"request {label} did not record routing telemetry")
    seed_routes[session] = routing
    session_histories[session] = [
        SYSTEM_MESSAGE,
        {"role": "user", "content": f"{label}: answer with one word. "},
        {"role": "assistant", "content": response["choices"][0]["message"].get("content") or "ok"},
    ]

worker_ids = {routing["workerId"] for routing in seed_routes.values()}
if len(worker_ids) < 2:
    raise RuntimeError(f"expected at least two routed workers, observed {sorted(worker_ids)}")

router = request("/clap/v1/router")
if router.get("localReplicas", 0) < 2:
    raise RuntimeError(f"router is not configured for multiple replicas: {router}")
if len(router.get("workers", [])) < 2:
    raise RuntimeError(f"router did not publish two worker heartbeats: {router}")
if not router.get("locations"):
    raise RuntimeError("router did not publish any soft session locations")

session, initial_route = next(iter(seed_routes.items()))
owner = initial_route["workerId"]
follow_label = "phase2-follow-up"
previous_ids = dashboard_ids()
follow_response = chat(session, follow_label, history=session_histories[session])
follow_route = dashboard_record_after(previous_ids, active=False).get("routing") or {}
if follow_route.get("workerId") != owner or follow_route.get("reason") != "session_locality":
    raise RuntimeError(f"follow-up did not preserve useful session locality: {follow_route}")
session_histories[session].extend([
    {"role": "user", "content": f"{follow_label}: answer with one word. "},
    {"role": "assistant", "content": follow_response["choices"][0]["message"].get("content") or "ok"},
])

blocker_label = "phase2-overload-blocker"
blocker_error: list[BaseException] = []


def run_blocker():
    try:
        chat(session, blocker_label, max_tokens=512,
             padding="Continue producing short words until the output limit. " * 8,
             history=session_histories[session])
    except BaseException as error:  # surfaced on the main thread below
        blocker_error.append(error)


blocker = threading.Thread(target=run_blocker, daemon=True)
previous_ids = dashboard_ids()
blocker.start()
blocker_record = dashboard_record_after(previous_ids, active=True)
if (blocker_record.get("routing") or {}).get("workerId") != owner:
    raise RuntimeError(f"blocker did not occupy the cached owner: {blocker_record.get('routing')}")

overload_label = "phase2-overload-follow-up"
previous_ids = dashboard_ids()
chat(session, overload_label, history=session_histories[session])
overload_route = dashboard_record_after(previous_ids, active=False).get("routing") or {}
blocker.join(timeout=300)
if blocker.is_alive():
    raise RuntimeError("overload blocker did not finish")
if blocker_error:
    raise blocker_error[0]
if overload_route.get("workerId") == owner:
    raise RuntimeError(f"overloaded cached worker did not lose to recomputation: {overload_route}")
if overload_route.get("reason") != "lowest_cost":
    raise RuntimeError(f"overload route did not report a cost decision: {overload_route}")

request("/clap/v1/models/unload", {"model": MODEL})
after_unload = request("/clap/v1/router")
if after_unload.get("workers") or after_unload.get("locations"):
    raise RuntimeError(f"model unload left stale routing state: {after_unload}")

restart_label = "phase2-after-restart"
previous_ids = dashboard_ids()
chat(session, restart_label, history=session_histories[session])
restart_route = dashboard_record_after(previous_ids, active=False).get("routing") or {}
if restart_route.get("directoryHit"):
    raise RuntimeError(f"restart incorrectly trusted an invalidated location: {restart_route}")

print(json.dumps({
    "status": "ok",
    "base_url": BASE_URL,
    "model": MODEL,
    "seed_sessions": SEED_SESSIONS,
    "workers": sorted(worker_ids),
    "session_owner": owner,
    "follow_up": follow_route,
    "overload": overload_route,
    "after_restart": restart_route,
    "router_limits": router.get("limits"),
}, indent=2, sort_keys=True))
