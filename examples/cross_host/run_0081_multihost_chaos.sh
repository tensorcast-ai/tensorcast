#!/usr/bin/env bash
# Copyright (c) 2026, TensorCast Team.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

PHASE="${TC0081_PHASE:-all}"
OUT_ROOT="${TC0081_OUT_ROOT:-/tmp/tc_cross_20260222/results_chaos_0081_fixed}"
RUN_LABEL="${TC0081_RUN_LABEL:-$(date +%Y%m%d-%H%M%S)}"
GS_ADDR="${TC0081_GS_ADDR:-}"
WORKER_COUNT="${TC0081_WORKER_COUNT:-0}"

CHARGED_GROUP="${TC0081_CHARGED_GROUP:-tensorcast-dev}"
PRIVATE_MACHINE="${TC0081_PRIVATE_MACHINE:-group}"
GPU="${TC0081_GPU:-1}"
CPU="${TC0081_CPU:-4}"
MEMORY_MIB="${TC0081_MEMORY_MIB:-106400}"
MAX_WAIT_DURATION="${TC0081_MAX_WAIT_DURATION:-20m}"
POSITIVE_TAGS="${TC0081_POSITIVE_TAGS:-}"
COMMENT_PREFIX="${TC0081_COMMENT_PREFIX:-tc-0081-chaos-fixed}"
CLEANUP_WORKERS="${TC0081_CLEANUP_WORKERS:-true}"

DAEMON_CONFIG="${TC0081_DAEMON_CONFIG:-examples/config/store_daemon_config_cross_host_bench.yaml}"
REMOTE_TIMEOUT_SEC="${TC0081_REMOTE_TIMEOUT_SEC:-900}"
CHAOS_SEED="${TC0081_CHAOS_SEED:-7}"
GATE_MAX_RECOVER_TIME_SEC="${TC0081_GATE_MAX_RECOVER_TIME_SEC:-300}"

WAIT_RUNNING_TIMEOUT_SEC="${TC0081_WAIT_RUNNING_TIMEOUT_SEC:-1500}"
RUN_WORKDIR="${TC0081_RUN_WORKDIR:-./}"

usage() {
  cat <<'EOF'
Usage:
  bash examples/cross_host/run_0081_multihost_chaos.sh [options]

Options:
  --phase <small|medium|large|all>     Phase to run (default: all)
  --out-root <path>                     Output root (default: /tmp/tc_cross_20260222/results_chaos_0081_fixed)
  --run-label <label>                   Run label (default: timestamp)
  --gs-addr <host:port>                 Global Store address (default: from tensorcast-cli global status --json)
  --workers <n>                         Worker count to launch (default: auto by phase)
  --charged-group <name>                orchestratorctl charged group (default: tensorcast-dev)
  --private-machine <mode>              orchestratorctl private-machine (default: group)
  --gpu <n>                             GPU per worker (default: 1)
  --cpu <n>                             CPU per worker (default: 4)
  --memory-mib <n>                      Memory per worker MiB (default: 106400)
  --max-wait-duration <duration>        orchestratorctl max wait (default: 20m)
  --positive-tags <csv>                 Optional orchestratorctl positive-tags
  --daemon-config <path>                Daemon config path
  --remote-timeout-sec <n>              Runner remote timeout seconds (default: 900)
  --chaos-seed <n>                      Chaos seed (default: 7)
  --gate-max-recover-time-sec <n>       Gate recover threshold seconds (default: 300)
  --keep-workers                        Keep launched workers after run
  --cleanup-workers                     Cleanup launched workers after run (default)
  -h, --help                            Show help

Environment variables:
  TC0081_* variables with same names as defaults can override behavior.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phase)
      PHASE="${2:-}"
      shift 2
      ;;
    --out-root)
      OUT_ROOT="${2:-}"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="${2:-}"
      shift 2
      ;;
    --gs-addr)
      GS_ADDR="${2:-}"
      shift 2
      ;;
    --workers)
      WORKER_COUNT="${2:-}"
      shift 2
      ;;
    --charged-group)
      CHARGED_GROUP="${2:-}"
      shift 2
      ;;
    --private-machine)
      PRIVATE_MACHINE="${2:-}"
      shift 2
      ;;
    --gpu)
      GPU="${2:-}"
      shift 2
      ;;
    --cpu)
      CPU="${2:-}"
      shift 2
      ;;
    --memory-mib)
      MEMORY_MIB="${2:-}"
      shift 2
      ;;
    --max-wait-duration)
      MAX_WAIT_DURATION="${2:-}"
      shift 2
      ;;
    --positive-tags)
      POSITIVE_TAGS="${2:-}"
      shift 2
      ;;
    --daemon-config)
      DAEMON_CONFIG="${2:-}"
      shift 2
      ;;
    --remote-timeout-sec)
      REMOTE_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --chaos-seed)
      CHAOS_SEED="${2:-}"
      shift 2
      ;;
    --gate-max-recover-time-sec)
      GATE_MAX_RECOVER_TIME_SEC="${2:-}"
      shift 2
      ;;
    --keep-workers)
      CLEANUP_WORKERS="false"
      shift
      ;;
    --cleanup-workers)
      CLEANUP_WORKERS="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ "${PHASE}" != "small" && "${PHASE}" != "medium" && "${PHASE}" != "large" && "${PHASE}" != "all" ]]; then
  echo "Invalid phase '${PHASE}', expected small|medium|large|all" >&2
  exit 1
fi

required_getters_for_phase() {
  local phase_name="$1"
  case "${phase_name}" in
    small) echo 2 ;;
    medium) echo 5 ;;
    large) echo 7 ;;
    all) echo 7 ;;
    *) return 1 ;;
  esac
}

required_workers_for_phase() {
  local phase_name="$1"
  local getters
  getters="$(required_getters_for_phase "${phase_name}")"
  echo $((getters + 1))
}

if [[ -z "${GS_ADDR}" ]]; then
  if ! status_json="$(tensorcast-cli global status --json 2>/dev/null)"; then
    echo "Failed to get Global Store status. Set --gs-addr explicitly." >&2
    exit 1
  fi
  GS_ADDR="$(printf '%s' "${status_json}" | jq -r '.address // .health.advertise_address // empty')"
fi
if [[ -z "${GS_ADDR}" ]]; then
  echo "Global Store address is empty. Set --gs-addr or start GS session first." >&2
  exit 1
fi

required_workers="$(required_workers_for_phase "${PHASE}")"
if [[ "${WORKER_COUNT}" == "0" ]]; then
  WORKER_COUNT="${required_workers}"
fi
if [[ "${WORKER_COUNT}" -lt "${required_workers}" ]]; then
  echo "phase=${PHASE} requires at least ${required_workers} workers (1 seed + ${required_workers}-1 getters), got ${WORKER_COUNT}" >&2
  exit 1
fi

RUN_ROOT="${OUT_ROOT}/${RUN_LABEL}"
SCHEMA_DIR="${RUN_ROOT}/schemas"
RESULTS_DIR="${RUN_ROOT}/results"
META_DIR="${RUN_ROOT}/meta"
mkdir -p "${SCHEMA_DIR}" "${RESULTS_DIR}" "${META_DIR}"

declare -a WORKER_IDS=()
declare -a WORKER_IPS=()
declare -a WORKER_HOSTS=()

cleanup_workers() {
  if [[ "${CLEANUP_WORKERS}" != "true" ]]; then
    return 0
  fi
  if [[ "${#WORKER_IDS[@]}" -eq 0 ]]; then
    return 0
  fi
  echo "[0081-fixed] cleaning up workers..."
  local pid
  for pid in "${WORKER_IDS[@]}"; do
    orchestratorctl delete process "${pid}" -n tensorcast >/dev/null 2>&1 || true
  done
}

trap cleanup_workers EXIT

wait_worker_running() {
  local pid="$1"
  local deadline=$((SECONDS + WAIT_RUNNING_TIMEOUT_SEC))
  while (( SECONDS < deadline )); do
    local status_line
    status_line="$(orchestratorctl get process "${pid}" -n tensorcast | awk 'NR==2{print $4" "$5}')"
    if [[ "${status_line}" == "1/1 Running" ]]; then
      return 0
    fi
    case "${status_line}" in
      *Failed*|*Error*|*CrashLoopBackOff*|*Stopped*)
        echo "[0081-fixed] worker ${pid} entered bad state: ${status_line}" >&2
        return 1
        ;;
    esac
    sleep 5
  done
  echo "[0081-fixed] timeout waiting worker ${pid} to become Running" >&2
  return 1
}

join_slice_csv() {
  local -n ref=$1
  local start="$2"
  local count="$3"
  local items=()
  local i
  local idx
  for ((i = 0; i < count; ++i)); do
    idx=$((start + i))
    items+=("${ref[$idx]}")
  done
  local joined
  joined="$(IFS=','; echo "${items[*]}")"
  echo "${joined}"
}

json_array_from_bash_array() {
  local -n ref=$1
  local out="["
  local idx=0
  local value escaped
  for value in "${ref[@]}"; do
    escaped="${value//\\/\\\\}"
    escaped="${escaped//\"/\\\"}"
    if [[ "${idx}" -gt 0 ]]; then
      out+=", "
    fi
    out+="\"${escaped}\""
    idx=$((idx + 1))
  done
  out+="]"
  echo "${out}"
}

write_phase_schema() {
  local phase_name="$1"
  local getter_count="$2"
  local cleanup_artifacts="$3"
  local output_path="${SCHEMA_DIR}/${phase_name}.json"
  local get_procs
  local get_ips
  get_procs="$(join_slice_csv WORKER_IDS 1 "${getter_count}")"
  get_ips="$(join_slice_csv WORKER_IPS 1 "${getter_count}")"
  cat > "${output_path}" <<EOF
{
  "defaults": {
    "fanout_args": {
      "mode": "fanout",
      "seed_proc": "${WORKER_IDS[0]}",
      "seed_adv_ip": "${WORKER_IPS[0]}",
      "get_procs": "${get_procs}",
      "get_adv_ips": "${get_ips}",
      "gs_addr": "${GS_ADDR}",
      "daemon_config": "${DAEMON_CONFIG}",
      "seed_grpc_port": 62001,
      "seed_p2p_port": 63001,
      "get_grpc_port": 62001,
      "get_p2p_port": 63001,
      "get_port_step": 10,
      "lookup_mode": "key",
      "put_policy": "pinned",
      "put_device": "cuda:0",
      "get_device": "cuda:0",
      "size_mib": 256,
      "warmup": 1,
      "iterations": 2,
      "wave_size": 1,
      "conn": 20,
      "buffers": 16,
      "maxw": 16,
      "expected_gpu_channels": 0,
      "payload_sample_verify": true,
      "require_p2p": true,
      "cleanup_artifacts": ${cleanup_artifacts},
      "remote_timeout_sec": ${REMOTE_TIMEOUT_SEC},
      "daemon_start_timeout_sec": 600,
      "stop_timeout_sec": 240
    }
  },
  "cases": [
    {
      "name": "${phase_name}_positive_stable",
      "expected_outcome": "success",
      "chaos_events": [
        {
          "offset_sec": 1,
          "target_role": "seed",
          "action": "sleep",
          "duration_sec": 0.2,
          "expected_impact": "no-op"
        }
      ]
    },
    {
      "name": "${phase_name}_negative_expected_fail",
      "expected_outcome": "failure",
      "expected_error_pattern": "Process exited with code 2 during startup",
      "fanout_args": {
        "buffers": 0
      },
      "chaos_events": []
    }
  ]
}
EOF
}

echo "[0081-fixed] run_label=${RUN_LABEL} phase=${PHASE} gs=${GS_ADDR}"
echo "[0081-fixed] preflight orchestratorctl..."
orchestratorctl version >/dev/null 2>&1
orchestratorctl options >/dev/null 2>&1
orchestratorctl launch \
  --charged-group="${CHARGED_GROUP}" \
  --gpu "${GPU}" \
  --cpu "${CPU}" \
  --memory "${MEMORY_MIB}" \
  --private-machine "${PRIVATE_MACHINE}" \
  --max-wait-duration="${MAX_WAIT_DURATION}" \
  --predict-only >/dev/null 2>&1

echo "[0081-fixed] launching ${WORKER_COUNT} workers..."
for idx in $(seq 1 "${WORKER_COUNT}"); do
  launch_cmd=(
    orchestratorctl launch -d
    --charged-group="${CHARGED_GROUP}"
    --gpu "${GPU}"
    --cpu "${CPU}"
    --memory "${MEMORY_MIB}"
    --private-machine "${PRIVATE_MACHINE}"
    --max-wait-duration="${MAX_WAIT_DURATION}"
    --comment "${COMMENT_PREFIX}-${RUN_LABEL}-${idx}"
  )
  if [[ -n "${POSITIVE_TAGS}" ]]; then
    launch_cmd+=(--positive-tags "${POSITIVE_TAGS}")
  fi
  launch_cmd+=(-- bash -lc "echo START; hostname; nvidia-smi -L | head -n 2; sleep infinity")
  pid="$("${launch_cmd[@]}")"
  WORKER_IDS+=("${pid}")
  echo "[0081-fixed] launched worker ${idx}/${WORKER_COUNT}: ${pid}"
done

echo "[0081-fixed] waiting workers running..."
for pid in "${WORKER_IDS[@]}"; do
  wait_worker_running "${pid}"
done

echo "[0081-fixed] collecting worker inventory and health check..."
inventory_file="${META_DIR}/worker_inventory.jsonl"
: > "${inventory_file}"
for pid in "${WORKER_IDS[@]}"; do
  pod_ip="$(orchestratorctl describe process/"${pid}" -n tensorcast | awk -F': ' '/Pod IP/{gsub(/^ +/,"",$2); print $2; exit}')"
  if [[ -z "${pod_ip}" ]]; then
    echo "[0081-fixed] failed to get Pod IP for ${pid}" >&2
    exit 1
  fi
  host_name="$(orchestratorctl get process "${pid}" -n tensorcast | awk 'NR==2{print $2}')"
  orchestratorctl exec process/"${pid}" -n tensorcast -- bash -lc \
    "set -euo pipefail; nvidia-smi -L | head -n 1 >/dev/null; test -d ${RUN_WORKDIR}; test -d ${RUN_WORKDIR}/.venv"
  WORKER_IPS+=("${pod_ip}")
  WORKER_HOSTS+=("${host_name}")
  printf '{"process_id":"%s","pod_ip":"%s","hostname":"%s"}\n' "${pid}" "${pod_ip}" "${host_name}" >> "${inventory_file}"
done

available_getters=$((WORKER_COUNT - 1))
if [[ "${available_getters}" -ge 2 ]]; then
  write_phase_schema "small" 2 "true"
fi
if [[ "${available_getters}" -ge 5 ]]; then
  write_phase_schema "medium" 5 "false"
fi
if [[ "${available_getters}" -ge 7 ]]; then
  write_phase_schema "large" 7 "false"
fi

cat > "${META_DIR}/launcher_meta.json" <<EOF
{
  "run_label": "${RUN_LABEL}",
  "phase": "${PHASE}",
  "global_store": "${GS_ADDR}",
  "worker_count": ${WORKER_COUNT},
  "worker_ids": $(json_array_from_bash_array WORKER_IDS),
  "worker_ips": $(json_array_from_bash_array WORKER_IPS),
  "schemas_dir": "${SCHEMA_DIR}",
  "results_dir": "${RESULTS_DIR}",
  "cleanup_workers": ${CLEANUP_WORKERS}
}
EOF

declare -A RUN_DIR_BY_PHASE=()

run_phase() {
  local phase_name="$1"
  local run_id="phase-${phase_name}-${RUN_LABEL}"
  echo "[0081-fixed] running phase=${phase_name} run_id=${run_id}"
  TC_CASE_SCHEMA="${SCHEMA_DIR}/${phase_name}.json" \
  TC_OUT_DIR="${RESULTS_DIR}" \
  TC_RUN_ID="${run_id}" \
  TC_CHAOS_SEED="${CHAOS_SEED}" \
  TC_REMOTE_TIMEOUT_SEC="${REMOTE_TIMEOUT_SEC}" \
  TC_GATE_MAX_RECOVER_TIME_SEC="${GATE_MAX_RECOVER_TIME_SEC}" \
    bash examples/cross_host/run_multihost_chaos_suite.sh
  RUN_DIR_BY_PHASE["${phase_name}"]="${RESULTS_DIR}/${run_id}"
}

case "${PHASE}" in
  small)
    run_phase small
    ;;
  medium)
    run_phase medium
    ;;
  large)
    run_phase large
    ;;
  all)
    run_phase small
    run_phase medium
    run_phase large
    ;;
esac

if [[ -n "${RUN_DIR_BY_PHASE[small]:-}" && -n "${RUN_DIR_BY_PHASE[medium]:-}" && -n "${RUN_DIR_BY_PHASE[large]:-}" ]]; then
  echo "[0081-fixed] aggregating phase gates..."
  python examples/cross_host/chaos_phase_gate_review.py \
    --small-run-dir "${RUN_DIR_BY_PHASE[small]}" \
    --medium-run-dir "${RUN_DIR_BY_PHASE[medium]}" \
    --large-run-dir "${RUN_DIR_BY_PHASE[large]}" \
    --output "${RESULTS_DIR}/phase_gate_review.json" \
    --markdown-output "${RESULTS_DIR}/phase_gate_review.md"
fi

echo "[0081-fixed] completed."
echo "[0081-fixed] meta: ${META_DIR}/launcher_meta.json"
echo "[0081-fixed] inventory: ${inventory_file}"
echo "[0081-fixed] results: ${RESULTS_DIR}"
if [[ "${CLEANUP_WORKERS}" != "true" ]]; then
  echo "[0081-fixed] keep-workers enabled, worker ids:"
  printf '  - %s\n' "${WORKER_IDS[@]}"
fi
