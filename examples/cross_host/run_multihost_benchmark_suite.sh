#!/usr/bin/env bash
# Copyright (c) 2026, TensorCast Team.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

: "${TC_SEED_PROC:?set TC_SEED_PROC (seed orchestratorctl process id)}"
: "${TC_SEED_IP:?set TC_SEED_IP (seed advertise ip)}"
: "${TC_GET_PROCS:?set TC_GET_PROCS (comma-separated getter process ids)}"
: "${TC_GET_IPS:?set TC_GET_IPS (comma-separated getter advertise ips)}"
: "${TC_GS_ADDR:?set TC_GS_ADDR (global store host:port)}"

TC_DAEMON_CONFIG="${TC_DAEMON_CONFIG:-examples/config/store_daemon_config_cross_host_bench.yaml}"
TC_OUT_DIR="${TC_OUT_DIR:-/data/tc_cross_rerun/results_multi_host_scaleout}"
TC_RUN_ID="${TC_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TC_PORT_BASE="${TC_PORT_BASE:-62800}"
TC_REMOTE_TIMEOUT_SEC="${TC_REMOTE_TIMEOUT_SEC:-900}"
TC_DAEMON_START_TIMEOUT_SEC="${TC_DAEMON_START_TIMEOUT_SEC:-600}"
TC_STOP_TIMEOUT_SEC="${TC_STOP_TIMEOUT_SEC:-240}"
TC_VISIBILITY_TIMEOUT_SEC="${TC_VISIBILITY_TIMEOUT_SEC:-120}"
TC_VISIBILITY_RETRY_SEC="${TC_VISIBILITY_RETRY_SEC:-0.05}"
TC_GET_PINNED_ALLOCATION_TIMEOUT_MS="${TC_GET_PINNED_ALLOCATION_TIMEOUT_MS:-0}"
TC_FAILURE_DIAG="${TC_FAILURE_DIAG:-1}"
TC_FAILURE_DIAG_TIMEOUT_SEC="${TC_FAILURE_DIAG_TIMEOUT_SEC:-45}"
TC_RUNTIME_ROOT="${TC_RUNTIME_ROOT:-/data/tc_cross_rerun/runtime}"
TC_EARLY_GATE_ENABLE="${TC_EARLY_GATE_ENABLE:-1}"
TC_EARLY_GATE_HARD_FAIL="${TC_EARLY_GATE_HARD_FAIL:-1}"
TC_EARLY_GATE_TARGET_SIZE_MIB="${TC_EARLY_GATE_TARGET_SIZE_MIB:-1024}"
TC_EARLY_GATE_ROUND_WORKERS="${TC_EARLY_GATE_ROUND_WORKERS:-3,4}"
TC_EARLY_GATE_MIN_CLUSTER_SCALE_RATIO="${TC_EARLY_GATE_MIN_CLUSTER_SCALE_RATIO:-1.10}"
TC_EARLY_GATE_MIN_P2P_RATIO="${TC_EARLY_GATE_MIN_P2P_RATIO:-1.0}"
TC_EARLY_GATE_MIN_GET_SUCCESS_RATE="${TC_EARLY_GATE_MIN_GET_SUCCESS_RATE:-1.0}"
TC_EARLY_GATE_MIN_WAVE2_OVER_WAVE1="${TC_EARLY_GATE_MIN_WAVE2_OVER_WAVE1:-0.75}"
TC_EARLY_GATE_BASELINE_LINK_GIBPS="${TC_EARLY_GATE_BASELINE_LINK_GIBPS:-0}"
TC_EARLY_GATE_MIN_BASELINE_RATIO="${TC_EARLY_GATE_MIN_BASELINE_RATIO:-0.60}"
TC_IPERF3_PROBE_ENABLE="${TC_IPERF3_PROBE_ENABLE:-1}"
TC_IPERF3_PROBE_HARD_FAIL="${TC_IPERF3_PROBE_HARD_FAIL:-0}"
TC_IPERF3_AUTOFILL_EARLY_BASELINE="${TC_IPERF3_AUTOFILL_EARLY_BASELINE:-1}"
TC_SOURCE_SETTLE_SEC="${TC_SOURCE_SETTLE_SEC:-2}"
TC_PUT_SEED_BASE="${TC_PUT_SEED_BASE:-25000}"
TC_PHASE="${TC_PHASE:-all}"
TC_SCALE_WORKERS="${TC_SCALE_WORKERS:-2,4,8,16,32}"
TC_SCALE_SIZES_MIB="${TC_SCALE_SIZES_MIB:-1024,8192}"
TC_QUOTA_PREFLIGHT_ENABLE="${TC_QUOTA_PREFLIGHT_ENABLE:-1}"
TC_QUOTA_CHARGED_GROUP="${TC_QUOTA_CHARGED_GROUP:-tensorcast-dev}"
TC_XLARGE_STABLE_PREFLIGHT_ENABLE="${TC_XLARGE_STABLE_PREFLIGHT_ENABLE:-1}"
TC_XLARGE_STABLE_OVERLAP_VERSIONS="${TC_XLARGE_STABLE_OVERLAP_VERSIONS:-2}"
TC_XLARGE_STABLE_PREFLIGHT_MARGIN_RATIO="${TC_XLARGE_STABLE_PREFLIGHT_MARGIN_RATIO:-1.05}"
TC_WAVE_ASSIGNMENT="${TC_WAVE_ASSIGNMENT:-rotate}"
TC_WAVE_ASSIGNMENT_SEED="${TC_WAVE_ASSIGNMENT_SEED:-20260227}"
TC_CLEANUP_LEAK_SENTINEL="${TC_CLEANUP_LEAK_SENTINEL:-1}"
TC_CLEANUP_LEAK_THRESHOLD_BYTES="${TC_CLEANUP_LEAK_THRESHOLD_BYTES:-0}"
TC_CLEANUP_LEAK_STREAK_THRESHOLD="${TC_CLEANUP_LEAK_STREAK_THRESHOLD:-2}"
TC_TEARDOWN_STRICT="${TC_TEARDOWN_STRICT:-1}"
TC_TEARDOWN_VERIFY_TIMEOUT_SEC="${TC_TEARDOWN_VERIFY_TIMEOUT_SEC:-30}"

IPERF3_PROBE_JSON=""
QUOTA_GPU_USED=""
QUOTA_GPU_TOTAL=""
QUOTA_GPU_AVAILABLE=""
IPERF3_PROBE_SAMPLE_GETTERS=2
IPERF3_PROBE_DURATION_SEC=20
IPERF3_PROBE_PARALLEL=20
IPERF3_PROBE_PORT_BASE=63900
IPERF3_PROBE_REMOTE_TIMEOUT_SEC=240
IPERF3_PROBE_STARTUP_WAIT_SEC=1.0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      TC_PHASE="${2:-}"
      shift 2
      ;;
    --phase=*)
      TC_PHASE="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--phase small|medium|large|xlarge|all]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TC_PHASE}" ]]; then
  echo "TC_PHASE cannot be empty" >&2
  exit 1
fi
if [[ "${TC_PHASE}" != "small" && "${TC_PHASE}" != "medium" && "${TC_PHASE}" != "large" && "${TC_PHASE}" != "xlarge" && "${TC_PHASE}" != "all" ]]; then
  echo "Invalid phase '${TC_PHASE}'. Expected: small|medium|large|xlarge|all" >&2
  exit 1
fi

IFS=',' read -r -a GET_PROCS_ARR <<< "${TC_GET_PROCS}"
IFS=',' read -r -a GET_IPS_ARR <<< "${TC_GET_IPS}"
if [[ "${#GET_PROCS_ARR[@]}" -ne "${#GET_IPS_ARR[@]}" ]]; then
  echo "TC_GET_PROCS count != TC_GET_IPS count" >&2
  exit 1
fi
GETTER_COUNT="${#GET_PROCS_ARR[@]}"

declare -a SCALE_WORKERS_ARR=()
IFS=',' read -r -a RAW_SCALE_WORKERS <<< "${TC_SCALE_WORKERS}"
for raw in "${RAW_SCALE_WORKERS[@]}"; do
  value="$(echo "${raw}" | xargs)"
  if [[ -z "${value}" || ! "${value}" =~ ^[0-9]+$ ]]; then
    continue
  fi
  if [[ "${value}" -lt 2 ]]; then
    continue
  fi
  seen=0
  for existing in "${SCALE_WORKERS_ARR[@]}"; do
    if [[ "${existing}" -eq "${value}" ]]; then
      seen=1
      break
    fi
  done
  if [[ "${seen}" -eq 0 ]]; then
    SCALE_WORKERS_ARR+=("${value}")
  fi
done
if [[ "${#SCALE_WORKERS_ARR[@]}" -eq 0 ]]; then
  SCALE_WORKERS_ARR=(2 4 8 16 32)
fi

declare -a SCALE_SIZES_ARR=()
IFS=',' read -r -a RAW_SCALE_SIZES <<< "${TC_SCALE_SIZES_MIB}"
for raw in "${RAW_SCALE_SIZES[@]}"; do
  value="$(echo "${raw}" | xargs)"
  if [[ -z "${value}" || ! "${value}" =~ ^[0-9]+$ ]]; then
    continue
  fi
  if [[ "${value}" -le 0 ]]; then
    continue
  fi
  seen=0
  for existing in "${SCALE_SIZES_ARR[@]}"; do
    if [[ "${existing}" -eq "${value}" ]]; then
      seen=1
      break
    fi
  done
  if [[ "${seen}" -eq 0 ]]; then
    SCALE_SIZES_ARR+=("${value}")
  fi
done
if [[ "${#SCALE_SIZES_ARR[@]}" -eq 0 ]]; then
  SCALE_SIZES_ARR=(1024 8192)
fi

join_first_n() {
  local -n arr_ref=$1
  local n=$2
  local out=()
  local i
  for ((i = 0; i < n; ++i)); do
    out+=("${arr_ref[$i]}")
  done
  local joined
  joined="$(IFS=','; echo "${out[*]}")"
  echo "${joined}"
}

require_getters() {
  local min_count=$1
  local phase_label=$2
  if [[ "${GETTER_COUNT}" -lt "${min_count}" ]]; then
    echo "[suite] phase=${phase_label} requires >=${min_count} getters, got ${GETTER_COUNT}" >&2
    exit 1
  fi
}

append_skip_manifest() {
  local phase=$1
  local case_name=$2
  local reason=$3
  local detail=${4:-""}
  local escaped_phase escaped_case_name escaped_reason escaped_detail
  escaped_phase="$(json_escape "${phase}")"
  escaped_case_name="$(json_escape "${case_name}")"
  escaped_reason="$(json_escape "${reason}")"
  escaped_detail="$(json_escape "${detail}")"
  cat >> "${SKIP_MANIFEST_FILE}" <<EOF
{"phase":"${escaped_phase}","case_name":"${escaped_case_name}","reason":"${escaped_reason}","detail":"${escaped_detail}"}
EOF
}

json_escape() {
  local text=${1:-}
  text="${text//\\/\\\\}"
  text="${text//\"/\\\"}"
  text="${text//$'\n'/\\n}"
  text="${text//$'\r'/\\r}"
  text="${text//$'\t'/\\t}"
  printf '%s' "${text}"
}

probe_quota_gpu_available() {
  if [[ "${TC_QUOTA_PREFLIGHT_ENABLE}" != "1" ]]; then
    return 0
  fi
  if ! command -v orchestratorctl >/dev/null 2>&1; then
    echo "[suite] quota preflight skipped: orchestratorctl not found"
    return 0
  fi

  local output
  output="$(
    orchestratorctl launch \
      --charged-group "${TC_QUOTA_CHARGED_GROUP}" \
      --gpu 1 --cpu 1 --memory 1024 \
      --predict-only 2>&1 || true
  )"

  if [[ "${output}" =~ gpu[[:space:]]*:[[:space:]]*([0-9]+)/([0-9]+) ]]; then
    local consumed_with_request="${BASH_REMATCH[1]}"
    local total="${BASH_REMATCH[2]}"
    local used=$((consumed_with_request - 1))
    if [[ "${used}" -lt 0 ]]; then
      used=0
    fi
    local available=$((total - used))
    if [[ "${available}" -lt 0 ]]; then
      available=0
    fi
    QUOTA_GPU_USED="${used}"
    QUOTA_GPU_TOTAL="${total}"
    QUOTA_GPU_AVAILABLE="${available}"
    echo "[suite] quota preflight charged_group=${TC_QUOTA_CHARGED_GROUP} gpu_used=${QUOTA_GPU_USED} gpu_total=${QUOTA_GPU_TOTAL} gpu_available=${QUOTA_GPU_AVAILABLE}"
    return 0
  fi

  if [[ "${output}" =~ fail\ to\ pass\ quota\ check ]]; then
    echo "[suite] quota preflight could not parse gpu usage from output" >&2
    echo "${output}" >&2
  else
    echo "[suite] quota preflight predict output did not include quota counters"
  fi
}

resolve_daemon_stable_bytes() {
  python - "${TC_DAEMON_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

path = Path(sys.argv[1]).expanduser().resolve()
if not path.exists():
    print(0)
    raise SystemExit(0)
try:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
except Exception:
    print(0)
    raise SystemExit(0)
engine = payload.get("engine", {}) if isinstance(payload, dict) else {}
tiers = engine.get("memory_tiers", {}) if isinstance(engine, dict) else {}
raw = tiers.get("stable_bytes", 0) if isinstance(tiers, dict) else 0
units = {
    "b": 1,
    "kb": 1024,
    "mb": 1024**2,
    "gb": 1024**3,
    "tb": 1024**4,
}
text = str(raw).strip().lower()
if not text:
    print(0)
    raise SystemExit(0)
if text.isdigit():
    print(int(text))
    raise SystemExit(0)
for suffix, mul in units.items():
    if text.endswith(suffix):
        value = text[:-len(suffix)].strip()
        try:
            parsed = float(value)
        except Exception:
            print(0)
            raise SystemExit(0)
        print(int(parsed * mul))
        raise SystemExit(0)
print(0)
PY
}

append_case_manifest() {
  local mode=$1
  local case_name=$2
  local getters=$3
  local size_mib=$4
  local conn=$5
  local buffers=$6
  local maxw=$7
  local egc=$8
  local warmup=$9
  local iterations=${10}
  local wave_size=${11}
  cat >> "${MANIFEST_FILE}" <<EOF
{"case_name":"${case_name}","mode":"${mode}","getters":${getters},"size_mib":${size_mib},"conn":${conn},"buffers":${buffers},"maxw":${maxw},"expected_gpu_channels":${egc},"warmup":${warmup},"iterations":${iterations},"wave_size":${wave_size},"output_json":"${TC_OUT_DIR}/${case_name}.json"}
EOF
}

run_cascade_case() {
  local case_name=$1
  local getter_count=$2
  local seed_grpc=$3
  local seed_p2p=$4
  local get_grpc=$5
  local get_p2p=$6
  local put_seed=$7

  echo "[suite] running cascade case=${case_name} getters=${getter_count}"
  append_case_manifest \
    "cascade" \
    "${case_name}" \
    "${getter_count}" \
    1024 \
    20 \
    16 \
    16 \
    0 \
    0 \
    1 \
    0
  python examples/cross_host/cross_host_fanout_runner.py \
    --mode cascade \
    --case-name "${case_name}" \
    --seed-proc "${TC_SEED_PROC}" \
    --seed-adv-ip "${TC_SEED_IP}" \
    --seed-grpc-port "${seed_grpc}" \
    --seed-p2p-port "${seed_p2p}" \
    --get-procs "$(join_first_n GET_PROCS_ARR "${getter_count}")" \
    --get-adv-ips "$(join_first_n GET_IPS_ARR "${getter_count}")" \
    --get-grpc-port "${get_grpc}" \
    --get-p2p-port "${get_p2p}" \
    --get-port-step 10 \
    --gs-addr "${TC_GS_ADDR}" \
    --conn 20 \
    --buffers 16 \
    --maxw 16 \
    --expected-gpu-channels 0 \
    --size-mib 1024 \
    --put-seed-base "${put_seed}" \
    --source-retire-mode deregister \
    --source-stop-settle-sec "${TC_SOURCE_SETTLE_SEC}" \
    --runtime-root "${TC_RUNTIME_ROOT}" \
    --require-p2p \
    --require-vram-source \
    --daemon-config "${TC_DAEMON_CONFIG}" \
    --daemon-start-timeout-sec "${TC_DAEMON_START_TIMEOUT_SEC}" \
    --remote-timeout-sec "${TC_REMOTE_TIMEOUT_SEC}" \
    --stop-timeout-sec "${TC_STOP_TIMEOUT_SEC}" \
    --out-dir "${TC_OUT_DIR}"
}

run_fanout_case() {
  local case_name=$1
  local getters=$2
  local seed_grpc=$3
  local seed_p2p=$4
  local get_grpc=$5
  local get_p2p=$6
  local conn=$7
  local buffers=$8
  local maxw=$9
  local egc=${10}
  local size_mib=${11}
  local warmup=${12}
  local iterations=${13}
  local wave_size=${14}
  local put_seed=${15}
  local payload_verify=${16}

  echo "[suite] running fanout case=${case_name} getters=${getters}"
  append_case_manifest \
    "fanout" \
    "${case_name}" \
    "${getters}" \
    "${size_mib}" \
    "${conn}" \
    "${buffers}" \
    "${maxw}" \
    "${egc}" \
    "${warmup}" \
    "${iterations}" \
    "${wave_size}"
  python examples/cross_host/cross_host_fanout_runner.py \
    --mode fanout \
    --case-name "${case_name}" \
    --seed-proc "${TC_SEED_PROC}" \
    --seed-adv-ip "${TC_SEED_IP}" \
    --seed-grpc-port "${seed_grpc}" \
    --seed-p2p-port "${seed_p2p}" \
    --get-procs "$(join_first_n GET_PROCS_ARR "${getters}")" \
    --get-adv-ips "$(join_first_n GET_IPS_ARR "${getters}")" \
    --get-grpc-port "${get_grpc}" \
    --get-p2p-port "${get_p2p}" \
    --get-port-step 10 \
    --gs-addr "${TC_GS_ADDR}" \
    --conn "${conn}" \
    --buffers "${buffers}" \
    --maxw "${maxw}" \
    --expected-gpu-channels "${egc}" \
    --size-mib "${size_mib}" \
    --warmup "${warmup}" \
    --iterations "${iterations}" \
    --wave-size "${wave_size}" \
    --wave-assignment "${TC_WAVE_ASSIGNMENT}" \
    --wave-assignment-seed "${TC_WAVE_ASSIGNMENT_SEED}" \
    --put-seed-base "${put_seed}" \
    --visibility-timeout-sec "${TC_VISIBILITY_TIMEOUT_SEC}" \
    --visibility-retry-sec "${TC_VISIBILITY_RETRY_SEC}" \
    --get-pinned-allocation-timeout-ms "${TC_GET_PINNED_ALLOCATION_TIMEOUT_MS}" \
    --failure-diag-timeout-sec "${TC_FAILURE_DIAG_TIMEOUT_SEC}" \
    --runtime-root "${TC_RUNTIME_ROOT}" \
    --require-p2p \
    --teardown-verify-timeout-sec "${TC_TEARDOWN_VERIFY_TIMEOUT_SEC}" \
    --daemon-config "${TC_DAEMON_CONFIG}" \
    --daemon-start-timeout-sec "${TC_DAEMON_START_TIMEOUT_SEC}" \
    --remote-timeout-sec "${TC_REMOTE_TIMEOUT_SEC}" \
    --stop-timeout-sec "${TC_STOP_TIMEOUT_SEC}" \
    $([[ "${TC_TEARDOWN_STRICT}" == "1" ]] && echo "--teardown-strict" || echo "--no-teardown-strict") \
    $([[ "${TC_CLEANUP_LEAK_SENTINEL}" == "1" ]] && echo "--cleanup-leak-sentinel" || echo "--no-cleanup-leak-sentinel") \
    --cleanup-leak-threshold-bytes "${TC_CLEANUP_LEAK_THRESHOLD_BYTES}" \
    --cleanup-leak-streak-threshold "${TC_CLEANUP_LEAK_STREAK_THRESHOLD}" \
    $([[ "${TC_FAILURE_DIAG}" == "1" ]] && echo "--failure-diag" || echo "--no-failure-diag") \
    "${payload_verify}" \
    --out-dir "${TC_OUT_DIR}"
}

run_phase_small() {
  require_getters 2 "small"
  local cascade_getters=2
  if [[ "${GETTER_COUNT}" -ge 3 ]]; then
    cascade_getters=3
  fi

  run_cascade_case \
    "suite_${TC_RUN_ID}_small_cascade_${cascade_getters}g_deregister" \
    "${cascade_getters}" \
    "$((TC_PORT_BASE + 1))" \
    "$((TC_PORT_BASE + 1001))" \
    "$((TC_PORT_BASE + 11))" \
    "$((TC_PORT_BASE + 1011))" \
    "$((TC_PUT_SEED_BASE + 1))"

  run_fanout_case \
    "suite_${TC_RUN_ID}_small_fanout_3n_c20b16w16_g0_s1024" \
    2 \
    "$((TC_PORT_BASE + 101))" \
    "$((TC_PORT_BASE + 1101))" \
    "$((TC_PORT_BASE + 111))" \
    "$((TC_PORT_BASE + 1111))" \
    20 16 16 0 1024 1 3 1 \
    "$((TC_PUT_SEED_BASE + 101))" \
    "--payload-sample-verify"

  if [[ "${GETTER_COUNT}" -ge 3 ]]; then
    run_fanout_case \
      "suite_${TC_RUN_ID}_small_fanout_4n_c20b16w16_g0_s1024" \
      3 \
      "$((TC_PORT_BASE + 151))" \
      "$((TC_PORT_BASE + 1151))" \
      "$((TC_PORT_BASE + 161))" \
      "$((TC_PORT_BASE + 1161))" \
      20 16 16 0 1024 1 3 1 \
      "$((TC_PUT_SEED_BASE + 151))" \
      "--payload-sample-verify"
  fi
}

run_phase_medium() {
  require_getters 5 "medium"
  run_fanout_case \
    "suite_${TC_RUN_ID}_medium_fanout_6n_c20b16w16_g0_s1024" \
    5 \
    "$((TC_PORT_BASE + 201))" \
    "$((TC_PORT_BASE + 1201))" \
    "$((TC_PORT_BASE + 211))" \
    "$((TC_PORT_BASE + 1211))" \
    20 16 16 0 1024 1 4 2 \
    "$((TC_PUT_SEED_BASE + 201))" \
    "--payload-sample-verify"
}

run_phase_large() {
  require_getters 7 "large"
  run_fanout_case \
    "suite_${TC_RUN_ID}_large_fanout_8n_c20b16w16_g0_s1024" \
    7 \
    "$((TC_PORT_BASE + 301))" \
    "$((TC_PORT_BASE + 1301))" \
    "$((TC_PORT_BASE + 311))" \
    "$((TC_PORT_BASE + 1311))" \
    20 16 16 0 1024 1 4 3 \
    "$((TC_PUT_SEED_BASE + 301))" \
    "--no-payload-sample-verify"

  run_fanout_case \
    "suite_${TC_RUN_ID}_large_fanout_8n_c20b16w16_g0_s2048" \
    7 \
    "$((TC_PORT_BASE + 401))" \
    "$((TC_PORT_BASE + 1401))" \
    "$((TC_PORT_BASE + 411))" \
    "$((TC_PORT_BASE + 1411))" \
    20 16 16 0 2048 1 3 3 \
    "$((TC_PUT_SEED_BASE + 401))" \
    "--no-payload-sample-verify"

  if [[ "${GETTER_COUNT}" -ge 8 ]]; then
    run_fanout_case \
      "suite_${TC_RUN_ID}_large_fanout_9n_c20b16w16_g0_s1024" \
      8 \
      "$((TC_PORT_BASE + 501))" \
      "$((TC_PORT_BASE + 1501))" \
      "$((TC_PORT_BASE + 511))" \
      "$((TC_PORT_BASE + 1511))" \
      20 16 16 0 1024 1 4 4 \
      "$((TC_PUT_SEED_BASE + 501))" \
      "--no-payload-sample-verify"

    run_fanout_case \
      "suite_${TC_RUN_ID}_large_fanout_9n_c20b16w16_g0_s2048" \
      8 \
      "$((TC_PORT_BASE + 601))" \
      "$((TC_PORT_BASE + 1601))" \
      "$((TC_PORT_BASE + 611))" \
      "$((TC_PORT_BASE + 1611))" \
      20 16 16 0 2048 1 3 4 \
      "$((TC_PUT_SEED_BASE + 601))" \
      "--no-payload-sample-verify"
  else
    echo "[suite] skip 9n large cases: getters=${GETTER_COUNT} (<8)"
    append_skip_manifest \
      "large" \
      "suite_${TC_RUN_ID}_large_fanout_9n" \
      "insufficient_getters" \
      "getters=${GETTER_COUNT} required=8"
  fi
}

run_phase_xlarge() {
  local case_idx=0
  local ran_any=0
  local stable_bytes=0
  local overlap_versions="${TC_XLARGE_STABLE_OVERLAP_VERSIONS}"
  local margin_ratio="${TC_XLARGE_STABLE_PREFLIGHT_MARGIN_RATIO}"
  if [[ "${TC_XLARGE_STABLE_PREFLIGHT_ENABLE}" == "1" ]]; then
    stable_bytes="$(resolve_daemon_stable_bytes)"
    if [[ -z "${stable_bytes}" || ! "${stable_bytes}" =~ ^[0-9]+$ ]]; then
      stable_bytes=0
    fi
    echo "[suite] xlarge stable preflight: config=${TC_DAEMON_CONFIG} stable_bytes=${stable_bytes} overlap_versions=${overlap_versions} margin_ratio=${margin_ratio}"
  fi
  local worker_target
  local size_mib
  for worker_target in "${SCALE_WORKERS_ARR[@]}"; do
    if [[ "${worker_target}" -lt 16 ]]; then
      continue
    fi
    local getters=$((worker_target - 1))
    if [[ "${getters}" -gt "${GETTER_COUNT}" ]]; then
      local quota_hint="unknown"
      if [[ -n "${QUOTA_GPU_AVAILABLE}" ]]; then
        quota_hint="${QUOTA_GPU_AVAILABLE}"
      fi
      local skip_detail="getters=${GETTER_COUNT} required=${getters} quota_gpu_available=${quota_hint}"
      echo "[suite] skip xlarge target workers=${worker_target}: ${skip_detail}"
      append_skip_manifest \
        "xlarge" \
        "suite_${TC_RUN_ID}_xlarge_fanout_${worker_target}n_skipped" \
        "insufficient_getters" \
        "${skip_detail}"
      continue
    fi
    for size_mib in "${SCALE_SIZES_ARR[@]}"; do
      case_idx=$((case_idx + 1))
      local warmup=2
      local iterations=6
      local payload_verify="--no-payload-sample-verify"
      if [[ "${size_mib}" -ge 8192 ]]; then
        warmup=1
        iterations=3
      fi
      local wave_size=$((getters / 2))
      if [[ "${wave_size}" -lt 1 ]]; then
        wave_size=1
      fi
      local port_offset=$((700 + case_idx * 100))
      local case_name="suite_${TC_RUN_ID}_xlarge_fanout_${worker_target}n_c20b16w16_g0_s${size_mib}"

      if [[ "${TC_XLARGE_STABLE_PREFLIGHT_ENABLE}" == "1" && "${stable_bytes}" -gt 0 ]]; then
        local required_stable_bytes
        required_stable_bytes="$(
          python - "${size_mib}" "${overlap_versions}" "${margin_ratio}" <<'PY'
import math
import sys
size_mib = int(sys.argv[1])
overlap = max(1, int(sys.argv[2]))
margin = float(sys.argv[3])
base = int(size_mib) * 1024 * 1024 * overlap
print(int(math.ceil(base * margin)))
PY
        )"
        if [[ -n "${required_stable_bytes}" && "${required_stable_bytes}" =~ ^[0-9]+$ ]]; then
          if [[ "${required_stable_bytes}" -gt "${stable_bytes}" ]]; then
            local detail="stable_bytes=${stable_bytes} required_stable_bytes=${required_stable_bytes} size_mib=${size_mib} overlap_versions=${overlap_versions} margin_ratio=${margin_ratio}"
            echo "[suite] skip ${case_name}: ${detail}"
            append_skip_manifest "xlarge" "${case_name}" "stable_budget_insufficient" "${detail}"
            continue
          fi
        fi
      fi

      run_fanout_case \
        "${case_name}" \
        "${getters}" \
        "$((TC_PORT_BASE + port_offset + 1))" \
        "$((TC_PORT_BASE + port_offset + 1001))" \
        "$((TC_PORT_BASE + port_offset + 11))" \
        "$((TC_PORT_BASE + port_offset + 1011))" \
        20 16 16 0 "${size_mib}" "${warmup}" "${iterations}" "${wave_size}" \
        "$((TC_PUT_SEED_BASE + port_offset + 1))" \
        "${payload_verify}"
      ran_any=1
    done
  done
  if [[ "${ran_any}" -eq 0 ]]; then
    local detail="getters=${GETTER_COUNT} scale_workers=$(IFS=,; echo "${SCALE_WORKERS_ARR[*]}") scale_sizes_mib=$(IFS=,; echo "${SCALE_SIZES_ARR[*]}")"
    echo "[suite] no xlarge case eligible: ${detail}"
    append_skip_manifest "xlarge" "phase_xlarge" "no_eligible_case" "${detail}"
    return
  fi
}

run_iperf3_probe() {
  if [[ "${TC_IPERF3_PROBE_ENABLE}" != "1" ]]; then
    echo "[suite] skip iperf3 probe: TC_IPERF3_PROBE_ENABLE=${TC_IPERF3_PROBE_ENABLE}"
    append_skip_manifest "preflight" "iperf3_probe" "disabled" "TC_IPERF3_PROBE_ENABLE=${TC_IPERF3_PROBE_ENABLE}"
    return
  fi
  if [[ "${GETTER_COUNT}" -lt 1 ]]; then
    echo "[suite] skip iperf3 probe: no getter available"
    append_skip_manifest "preflight" "iperf3_probe" "no_getter_available" "getters=${GETTER_COUNT}"
    return
  fi
  local probe_json="${TC_OUT_DIR}/suite_${TC_RUN_ID}_iperf3_probe.json"
  local probe_md="${TC_OUT_DIR}/suite_${TC_RUN_ID}_iperf3_probe.md"
  local sample_getters="${IPERF3_PROBE_SAMPLE_GETTERS}"
  if [[ "${sample_getters}" -gt "${GETTER_COUNT}" ]]; then
    sample_getters="${GETTER_COUNT}"
  fi
  if [[ "${sample_getters}" -le 0 ]]; then
    echo "[suite] skip iperf3 probe: sample_getters=${sample_getters}"
    append_skip_manifest "preflight" "iperf3_probe" "invalid_sample_getters" "sample_getters=${sample_getters}"
    return
  fi
  local getter_procs_csv getter_ips_csv
  getter_procs_csv="$(join_first_n GET_PROCS_ARR "${sample_getters}")"
  getter_ips_csv="$(join_first_n GET_IPS_ARR "${sample_getters}")"

  local -a cmd=(
    python examples/cross_host/cross_host_iperf3_probe.py
    --seed-proc "${TC_SEED_PROC}"
    --seed-ip "${TC_SEED_IP}"
    --get-procs "${getter_procs_csv}"
    --get-ips "${getter_ips_csv}"
    --sample-getters "${sample_getters}"
    --duration-sec "${IPERF3_PROBE_DURATION_SEC}"
    --parallel "${IPERF3_PROBE_PARALLEL}"
    --port-base "${IPERF3_PROBE_PORT_BASE}"
    --remote-timeout-sec "${IPERF3_PROBE_REMOTE_TIMEOUT_SEC}"
    --startup-wait-sec "${IPERF3_PROBE_STARTUP_WAIT_SEC}"
    --out-json "${probe_json}"
    --out-md "${probe_md}"
  )
  echo "[suite] iperf3 probe start: sample_getters=${sample_getters} duration=${IPERF3_PROBE_DURATION_SEC}s parallel=${IPERF3_PROBE_PARALLEL}"
  if "${cmd[@]}"; then
    IPERF3_PROBE_JSON="${probe_json}"
    echo "[suite] iperf3 probe done: json=${probe_json}"
  else
    echo "[suite] iperf3 probe failed" >&2
    if [[ "${TC_IPERF3_PROBE_HARD_FAIL}" == "1" ]]; then
      exit 1
    fi
    append_skip_manifest "preflight" "iperf3_probe" "probe_failed_soft" "hard_fail=0"
  fi
}

run_early_gate() {
  if [[ "${TC_EARLY_GATE_ENABLE}" != "1" ]]; then
    echo "[suite] skip early gate: TC_EARLY_GATE_ENABLE=${TC_EARLY_GATE_ENABLE}"
    append_skip_manifest "preflight" "early_gate" "disabled" "TC_EARLY_GATE_ENABLE=${TC_EARLY_GATE_ENABLE}"
    return
  fi
  if [[ "${GETTER_COUNT}" -lt 3 ]]; then
    echo "[suite] skip early gate: requires >=3 getters to complete two startup rounds"
    append_skip_manifest "preflight" "early_gate" "insufficient_getters" "getters=${GETTER_COUNT} required=3"
    return
  fi
  local gate_json="${TC_OUT_DIR}/suite_${TC_RUN_ID}_early_gate.json"
  local gate_md="${TC_OUT_DIR}/suite_${TC_RUN_ID}_early_gate.md"
  local baseline_link_gibps="${TC_EARLY_GATE_BASELINE_LINK_GIBPS}"
  local baseline_source="env"
  if [[ "${TC_IPERF3_AUTOFILL_EARLY_BASELINE}" == "1" ]]; then
    if python - "${baseline_link_gibps}" <<'PY'
import sys
try:
    value = float(sys.argv[1])
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if value > 0.0 else 2)
PY
    then
      :
    else
      if [[ -n "${IPERF3_PROBE_JSON}" && -f "${IPERF3_PROBE_JSON}" ]]; then
        local auto_baseline
        auto_baseline="$(
          python - "${IPERF3_PROBE_JSON}" <<'PY'
import json
import sys
p = sys.argv[1]
payload = json.load(open(p, "r", encoding="utf-8"))
summary = payload.get("summary", {})
value = summary.get("single_link_ref_gibps")
try:
    parsed = float(value)
except Exception:
    raise SystemExit(1)
if parsed <= 0.0:
    raise SystemExit(2)
print(f"{parsed:.6f}")
PY
        )" || auto_baseline=""
        if [[ -n "${auto_baseline}" ]]; then
          baseline_link_gibps="${auto_baseline}"
          baseline_source="iperf3_probe"
        fi
      fi
    fi
  fi
  echo "[suite] early gate baseline_link_gibps=${baseline_link_gibps} source=${baseline_source}"
  local -a gate_cmd=(
    python examples/cross_host/scaleout_early_gate.py
    --fanout-dir "${TC_OUT_DIR}"
    --run-id "${TC_RUN_ID}"
    --target-size-mib "${TC_EARLY_GATE_TARGET_SIZE_MIB}"
    --round-workers "${TC_EARLY_GATE_ROUND_WORKERS}"
    --min-p2p-ratio "${TC_EARLY_GATE_MIN_P2P_RATIO}"
    --min-get-success-rate "${TC_EARLY_GATE_MIN_GET_SUCCESS_RATE}"
    --min-wave2-over-wave1 "${TC_EARLY_GATE_MIN_WAVE2_OVER_WAVE1}"
    --min-cluster-scale-ratio "${TC_EARLY_GATE_MIN_CLUSTER_SCALE_RATIO}"
    --baseline-link-gibps "${baseline_link_gibps}"
    --min-baseline-ratio "${TC_EARLY_GATE_MIN_BASELINE_RATIO}"
    --out-json "${gate_json}"
    --out-md "${gate_md}"
    --no-fail-on-gate
  )
  if [[ -n "${IPERF3_PROBE_JSON}" && -f "${IPERF3_PROBE_JSON}" ]]; then
    gate_cmd+=(--iperf-json "${IPERF3_PROBE_JSON}")
  fi
  echo "[suite] early gate start: rounds=${TC_EARLY_GATE_ROUND_WORKERS} size_mib=${TC_EARLY_GATE_TARGET_SIZE_MIB}"
  if "${gate_cmd[@]}"; then
    :
  else
    echo "[suite] early gate runner failed unexpectedly" >&2
    if [[ "${TC_EARLY_GATE_HARD_FAIL}" == "1" ]]; then
      exit 1
    fi
    append_skip_manifest "preflight" "early_gate" "runner_failed_soft" "hard_fail=0"
  fi
  if ! python - "${gate_json}" <<'PY'
import json
import sys
path = sys.argv[1]
payload = json.load(open(path, "r", encoding="utf-8"))
if bool(payload.get("pass")):
    print(f"[suite] early gate PASS json={path}")
    raise SystemExit(0)
print(f"[suite] early gate FAIL json={path}")
for check in payload.get("checks", []):
    if bool(check.get("passed")):
        continue
    name = str(check.get("name", "unknown"))
    detail = str(check.get("detail", ""))
    print(f"[suite][early-gate][fail] {name}: {detail}")
raise SystemExit(2)
PY
  then
    if [[ "${TC_EARLY_GATE_HARD_FAIL}" == "1" ]]; then
      echo "[suite] stop due to early gate failure (TC_EARLY_GATE_HARD_FAIL=1)" >&2
      exit 2
    fi
    echo "[suite] continue despite early gate failure (TC_EARLY_GATE_HARD_FAIL=0)"
    append_skip_manifest "preflight" "early_gate" "gate_failed_soft_continue" "hard_fail=0 gate_json=${gate_json}"
  fi
}

mkdir -p "${TC_OUT_DIR}"

META_FILE="${TC_OUT_DIR}/suite_${TC_RUN_ID}_meta.json"
MANIFEST_FILE="${TC_OUT_DIR}/suite_${TC_RUN_ID}_cases.jsonl"
SKIP_MANIFEST_FILE="${TC_OUT_DIR}/suite_${TC_RUN_ID}_skips.jsonl"
: > "${MANIFEST_FILE}"
: > "${SKIP_MANIFEST_FILE}"

probe_quota_gpu_available
cat > "${META_FILE}" <<EOF
{
  "run_id": "${TC_RUN_ID}",
  "phase": "${TC_PHASE}",
  "seed_process": "${TC_SEED_PROC}",
  "seed_ip": "${TC_SEED_IP}",
  "getter_count": ${GETTER_COUNT},
  "global_store": "${TC_GS_ADDR}",
  "daemon_config": "${TC_DAEMON_CONFIG}",
  "visibility_timeout_sec": ${TC_VISIBILITY_TIMEOUT_SEC},
  "visibility_retry_sec": ${TC_VISIBILITY_RETRY_SEC},
  "get_pinned_allocation_timeout_ms": ${TC_GET_PINNED_ALLOCATION_TIMEOUT_MS},
  "failure_diag": ${TC_FAILURE_DIAG},
  "failure_diag_timeout_sec": ${TC_FAILURE_DIAG_TIMEOUT_SEC},
  "runtime_root": "${TC_RUNTIME_ROOT}",
  "wave_assignment": "${TC_WAVE_ASSIGNMENT}",
  "wave_assignment_seed": ${TC_WAVE_ASSIGNMENT_SEED},
  "cleanup_leak_sentinel": ${TC_CLEANUP_LEAK_SENTINEL},
  "cleanup_leak_threshold_bytes": ${TC_CLEANUP_LEAK_THRESHOLD_BYTES},
  "cleanup_leak_streak_threshold": ${TC_CLEANUP_LEAK_STREAK_THRESHOLD},
  "teardown_strict": ${TC_TEARDOWN_STRICT},
  "teardown_verify_timeout_sec": ${TC_TEARDOWN_VERIFY_TIMEOUT_SEC},
  "quota_preflight_enable": ${TC_QUOTA_PREFLIGHT_ENABLE},
  "quota_charged_group": "${TC_QUOTA_CHARGED_GROUP}",
  "quota_gpu_used": "${QUOTA_GPU_USED}",
  "quota_gpu_total": "${QUOTA_GPU_TOTAL}",
  "quota_gpu_available": "${QUOTA_GPU_AVAILABLE}",
  "xlarge_stable_preflight_enable": ${TC_XLARGE_STABLE_PREFLIGHT_ENABLE},
  "xlarge_stable_overlap_versions": ${TC_XLARGE_STABLE_OVERLAP_VERSIONS},
  "xlarge_stable_preflight_margin_ratio": ${TC_XLARGE_STABLE_PREFLIGHT_MARGIN_RATIO},
  "iperf3_probe_enable": ${TC_IPERF3_PROBE_ENABLE},
  "iperf3_probe_hard_fail": ${TC_IPERF3_PROBE_HARD_FAIL},
  "iperf3_probe_profile": "default_v1",
  "iperf3_sample_getters": ${IPERF3_PROBE_SAMPLE_GETTERS},
  "iperf3_duration_sec": ${IPERF3_PROBE_DURATION_SEC},
  "iperf3_parallel": ${IPERF3_PROBE_PARALLEL},
  "iperf3_port_base": ${IPERF3_PROBE_PORT_BASE},
  "iperf3_remote_timeout_sec": ${IPERF3_PROBE_REMOTE_TIMEOUT_SEC},
  "iperf3_startup_wait_sec": ${IPERF3_PROBE_STARTUP_WAIT_SEC},
  "iperf3_autofill_early_baseline": ${TC_IPERF3_AUTOFILL_EARLY_BASELINE},
  "iperf3_probe_json": "${TC_OUT_DIR}/suite_${TC_RUN_ID}_iperf3_probe.json",
  "iperf3_probe_md": "${TC_OUT_DIR}/suite_${TC_RUN_ID}_iperf3_probe.md",
  "early_gate_enable": ${TC_EARLY_GATE_ENABLE},
  "early_gate_hard_fail": ${TC_EARLY_GATE_HARD_FAIL},
  "early_gate_target_size_mib": ${TC_EARLY_GATE_TARGET_SIZE_MIB},
  "early_gate_round_workers": "${TC_EARLY_GATE_ROUND_WORKERS}",
  "early_gate_min_cluster_scale_ratio": ${TC_EARLY_GATE_MIN_CLUSTER_SCALE_RATIO},
  "early_gate_min_p2p_ratio": ${TC_EARLY_GATE_MIN_P2P_RATIO},
  "early_gate_min_get_success_rate": ${TC_EARLY_GATE_MIN_GET_SUCCESS_RATE},
  "early_gate_min_wave2_over_wave1": ${TC_EARLY_GATE_MIN_WAVE2_OVER_WAVE1},
  "early_gate_baseline_link_gibps": ${TC_EARLY_GATE_BASELINE_LINK_GIBPS},
  "early_gate_min_baseline_ratio": ${TC_EARLY_GATE_MIN_BASELINE_RATIO},
  "scale_workers": "$(IFS=,; echo "${SCALE_WORKERS_ARR[*]}")",
  "scale_sizes_mib": "$(IFS=,; echo "${SCALE_SIZES_ARR[*]}")",
  "manifest_file": "${MANIFEST_FILE}",
  "skip_manifest_file": "${SKIP_MANIFEST_FILE}"
}
EOF

echo "[suite] phase=${TC_PHASE} run_id=${TC_RUN_ID} getters=${GETTER_COUNT}"
echo "[suite] meta=${META_FILE}"
echo "[suite] manifest=${MANIFEST_FILE}"
echo "[suite] skip_manifest=${SKIP_MANIFEST_FILE}"

run_iperf3_probe

case "${TC_PHASE}" in
  small)
    run_phase_small
    run_early_gate
    ;;
  medium)
    run_phase_medium
    ;;
  large)
    run_phase_large
    ;;
  xlarge)
    run_phase_xlarge
    ;;
  all)
    run_phase_small
    run_early_gate
    if [[ "${GETTER_COUNT}" -ge 5 ]]; then
      run_phase_medium
    else
      echo "[suite] skip medium phase: getters=${GETTER_COUNT} (<5)"
      append_skip_manifest "medium" "phase_medium" "insufficient_getters" "getters=${GETTER_COUNT} required=5"
    fi
    if [[ "${GETTER_COUNT}" -ge 7 ]]; then
      run_phase_large
    else
      echo "[suite] skip large phase: getters=${GETTER_COUNT} (<7)"
      append_skip_manifest "large" "phase_large" "insufficient_getters" "getters=${GETTER_COUNT} required=7"
    fi
    if [[ "${GETTER_COUNT}" -ge 15 ]]; then
      run_phase_xlarge
    else
      echo "[suite] skip xlarge phase: getters=${GETTER_COUNT} (<15)"
      append_skip_manifest "xlarge" "phase_xlarge" "insufficient_getters" "getters=${GETTER_COUNT} required=15"
    fi
    ;;
esac

echo "[suite] completed. outputs in ${TC_OUT_DIR}"
