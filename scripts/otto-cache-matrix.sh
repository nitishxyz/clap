#!/usr/bin/env bash
# Drives real otto sessions across multiple projects against a remote clap
# server, then reports clap's KV cache KPIs for the whole run.
#
# usage: otto-cache-matrix.sh <base-url> <api-key> <model>
set -uo pipefail

BASE="${1:-http://100.102.133.98:11435}"
KEY="${2:?api key required}"
MODEL="${3:-unsloth/gemma-4-E4B-it-GGUF}"
LAB="${LAB:-/tmp/otto-cache-lab}"
PROJECTS=(proj-alpha proj-beta proj-gamma)

auth=(-H "Authorization: Bearer ${KEY}")

metric() { curl -s -m 10 "${auth[@]}" "${BASE}/metrics" | awk -v k="$1" '$1==k{print $2}'; }

echo "== reset clap state =="
curl -s -m 15 -X DELETE "${auth[@]}" "${BASE}/clap/v1/dashboard"; echo
echo "hits=$(metric 'clap_kv_cache_total{outcome="hit"}') misses=$(metric 'clap_kv_cache_total{outcome="miss"}')"

declare -a SESSIONS=()

run_turn() { # project, session-id-or-empty, prompt -> prints seconds, echoes session id
  local project="$1" session="$2" prompt="$3"
  local args=(ask --provider clap --model "$MODEL" --project "${LAB}/${project}" -y)
  [[ -n "$session" ]] && args+=(--session "$session")
  local start end out
  start=$(python3 -c 'import time;print(time.monotonic())')
  out=$(cd "${LAB}/${project}" && otto "${args[@]}" "$prompt" 2>&1)
  end=$(python3 -c 'import time;print(time.monotonic())')
  local secs
  secs=$(python3 -c "print(f'{${end}-${start}:.1f}')")
  local sid
  sid=$(printf '%s\n' "$out" | sed -n 's/^new session \([0-9a-f-]*\).*/\1/p' | head -1)
  [[ -z "$sid" ]] && sid="$session"
  printf '  %-11s %-8s %6ss  %s\n' "$project" "${sid:0:8}" "$secs" \
    "$(printf '%s' "$out" | tail -2 | head -1 | cut -c1-46)"
  SESSION_OUT="$sid"
}

echo
echo "== round 1: open 2 sessions per project =="
for p in "${PROJECTS[@]}"; do
  for s in 1 2; do
    run_turn "$p" "" "Reply with exactly: ready-${s}"
    SESSIONS+=("${p}|${SESSION_OUT}")
  done
done

for round in 2 3; do
  echo
  echo "== round ${round}: continue every session =="
  for entry in "${SESSIONS[@]}"; do
    run_turn "${entry%%|*}" "${entry##*|}" "Round ${round}: reply with one short sentence."
  done
done

echo
echo "== clap KV cache result =="
h=$(metric 'clap_kv_cache_total{outcome="hit"}')
m=$(metric 'clap_kv_cache_total{outcome="miss"}')
el=$(metric 'clap_kv_cache_eligibility_total{eligibility="eligible"}')
ne=$(metric 'clap_kv_cache_eligibility_total{eligibility="not_eligible"}')
act=$(metric 'clap_requests_active')
echo "hits=${h} misses=${m} eligible=${el} not_eligible=${ne} active_gauge=${act}"
python3 -c "
h=${h:-0}; m=${m:-0}
print(f'hit rate: {100*h/(h+m):.1f}%' if h+m else 'no eligible cache lookups')
"
