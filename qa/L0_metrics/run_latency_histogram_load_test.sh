#!/usr/bin/env bash
# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Compare identity-model throughput with latency histograms off and on.
# summary_latencies=simplified enables the lock-free histograms. That run must
# stay within 80% of the counter-only baseline. summary_latencies=true is the
# historical CKMS path and is not the fast mode under test.
#
# Usage:
#   run_latency_histogram_load_test.sh /path/to/tritonserver [/path/to/backends]
#
# Optional environment:
#   LOAD_SECONDS (default 20)
#   LOAD_CONCURRENCY (default 1000)
#   HTTP_PORT (default 8000)
#   METRICS_PORT (default 8002)

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/tritonserver [/path/to/backends]" >&2
  exit 2
fi

TRITONSERVER="$1"
BACKEND_DIR="${2:-}"
if [[ -z "${BACKEND_DIR}" ]]; then
  bin_dir="$(cd "$(dirname "${TRITONSERVER}")" && pwd)"
  if [[ -d "${bin_dir}/../backends" ]]; then
    BACKEND_DIR="$(cd "${bin_dir}/../backends" && pwd)"
  fi
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOAD_SECONDS="${LOAD_SECONDS:-20}"
LOAD_CONCURRENCY="${LOAD_CONCURRENCY:-1000}"
HTTP_PORT="${HTTP_PORT:-8000}"
METRICS_PORT="${METRICS_PORT:-8002}"
WORKDIR="$(mktemp -d /tmp/triton_latency_hist.XXXXXX)"
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  rm -rf "${WORKDIR}"
}
trap cleanup EXIT

ulimit -n 8192 || true

LOADGEN="${WORKDIR}/latency_loadgen"
c++ -std=c++17 -O2 -pthread \
  "${SCRIPT_DIR}/latency_loadgen.cc" -o "${LOADGEN}"

MODEL_REPO="${WORKDIR}/models"
mkdir -p "${MODEL_REPO}/identity/1"
cp "${SCRIPT_DIR}/identity_latency/config.pbtxt" "${MODEL_REPO}/identity/config.pbtxt"

start_server() {
  local extra_args=("$@")
  local log="${WORKDIR}/server.log"
  local cmd=("${TRITONSERVER}"
    --model-repository="${MODEL_REPO}"
    --http-port="${HTTP_PORT}"
    --allow-http=true
    --allow-metrics=true
    --metrics-port="${METRICS_PORT}"
    --log-verbose=0)
  if [[ -n "${BACKEND_DIR}" ]]; then
    cmd+=(--backend-directory="${BACKEND_DIR}")
  fi
  cmd+=("${extra_args[@]}")
  "${cmd[@]}" >"${log}" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${HTTP_PORT}/v2/health/ready" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "tritonserver exited early" >&2
      cat "${log}" >&2
      exit 1
    fi
    sleep 1
  done
  echo "tritonserver did not become ready" >&2
  cat "${log}" >&2
  exit 1
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}

run_load() {
  local out="$1"
  "${LOADGEN}" --host 127.0.0.1 --port "${HTTP_PORT}" --model identity \
    --concurrency "${LOAD_CONCURRENCY}" --seconds "${LOAD_SECONDS}" | tee "${out}"
}

rps_of() {
  awk '/^rps / { print $2 }' "$1"
}

echo "=== baseline: summary_latencies=false histogram_latencies=false ==="
start_server --metrics-config summary_latencies=false --metrics-config histogram_latencies=false
run_load "${WORKDIR}/baseline.txt"
BASE_RPS="$(rps_of "${WORKDIR}/baseline.txt")"
stop_server

echo "=== histograms: summary_latencies=simplified ==="
start_server --metrics-config summary_latencies=simplified
python3 "${SCRIPT_DIR}/latency_histogram_e2e_test.py" \
  --http "http://127.0.0.1:${HTTP_PORT}" \
  --metrics "http://127.0.0.1:${METRICS_PORT}/metrics" \
  --model identity --requests 50
run_load "${WORKDIR}/histogram.txt"
HIST_RPS="$(rps_of "${WORKDIR}/histogram.txt")"
curl -sf "http://127.0.0.1:${METRICS_PORT}/metrics" >"${WORKDIR}/metrics.txt"
stop_server

python3 - "${WORKDIR}/histogram.txt" "${WORKDIR}/metrics.txt" <<'PY'
import re, sys
success = None
for line in open(sys.argv[1]):
    if line.startswith("success "):
        success = float(line.split()[1])
metrics = open(sys.argv[2]).read()
match = re.search(
    r'nv_inference_request_histogram_us_count\{[^}]*model="identity"[^}]*\} (\d+)',
    metrics,
)
if match is None or success is None:
    raise SystemExit("could not compare histogram count with successful requests")
count = float(match.group(1))
# The functional check above also recorded 50 requests.
if count + 1 < success:
    raise SystemExit(f"histogram count {count} is far below successes {success}")
print(f"histogram_count={int(count)} load_success={int(success)}")
PY

python3 - <<PY
base = float("${BASE_RPS}")
hist = float("${HIST_RPS}")
ratio = hist / base if base else 0.0
print(f"baseline_rps={base:.1f} histogram_rps={hist:.1f} ratio={ratio:.3f}")
if base <= 0 or hist < 0.8 * base:
    raise SystemExit("histogram throughput is below 80% of the counter-only baseline")
PY

echo "latency histogram load test passed"
