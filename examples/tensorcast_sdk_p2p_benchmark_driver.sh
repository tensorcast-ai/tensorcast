#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  echo "Missing virtualenv at ${ROOT_DIR}/.venv" >&2
  exit 1
fi

source "${ROOT_DIR}/.venv/bin/activate"

BASE_DIR="${BASE_DIR:-/tmp/tc-bench-driver}"
RESULTS_DIR="${RESULTS_DIR:-}"
PUT_HOME="${PUT_HOME:-}"
GET_HOME="${GET_HOME:-}"
PUT_STORAGE="${PUT_STORAGE:-}"
GET_STORAGE="${GET_STORAGE:-}"

GLOBAL_ADDR="${GLOBAL_ADDR:-127.0.0.1:50051}"
PUT_ADDR="${PUT_ADDR:-}"
GET_ADDR="${GET_ADDR:-}"
P2P_PUT_PORT="${P2P_PUT_PORT:-}"
P2P_GET_PORT="${P2P_GET_PORT:-}"
RUN_ID="${RUN_ID:-}"
PUT_SESSION="${PUT_SESSION:-}"
GET_SESSION="${GET_SESSION:-}"
PUT_DAEMON_ID="${PUT_DAEMON_ID:-}"
GET_DAEMON_ID="${GET_DAEMON_ID:-}"
ADVERTISE_HOST="${ADVERTISE_HOST:-}"
DAEMON_CONFIG="${DAEMON_CONFIG:-}"
EXTRA_DAEMON_SETS=()

CONN_COUNT="${CONN_COUNT:-20}"
BUFFERS_PER_FLOW="${BUFFERS_PER_FLOW:-16}"
HEARTBEAT_SEC="${HEARTBEAT_SEC:-5}"
SYNC_SEC="${SYNC_SEC:-5}"
SIZE_MIB="${SIZE_MIB:-2048}"
WARMUP="${WARMUP:-2}"
ITERATIONS="${ITERATIONS:-8}"
PUT_POLICY="${PUT_POLICY:-pinned}"
PUT_DEVICE="${PUT_DEVICE:-cuda:0}"
GET_DEVICE="${GET_DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
LOOKUP_MODE="${LOOKUP_MODE:-key}"
CLEANUP_ARTIFACTS="${CLEANUP_ARTIFACTS:-false}"
LOG_LEVEL="${LOG_LEVEL:-warn}"
VISIBILITY_TIMEOUT_SEC="${VISIBILITY_TIMEOUT_SEC:-20}"
VISIBILITY_RETRY_INTERVAL_SEC="${VISIBILITY_RETRY_INTERVAL_SEC:-0.05}"

for arg in "$@"; do
  case "${arg}" in
    --conn-count=*) CONN_COUNT="${arg#*=}" ;;
    --buffers-per-flow=*) BUFFERS_PER_FLOW="${arg#*=}" ;;
    --heartbeat-sec=*) HEARTBEAT_SEC="${arg#*=}" ;;
    --sync-sec=*) SYNC_SEC="${arg#*=}" ;;
    --size-mib=*) SIZE_MIB="${arg#*=}" ;;
    --warmup=*) WARMUP="${arg#*=}" ;;
    --iterations=*) ITERATIONS="${arg#*=}" ;;
    --put-policy=*) PUT_POLICY="${arg#*=}" ;;
    --put-device=*) PUT_DEVICE="${arg#*=}" ;;
    --get-device=*) GET_DEVICE="${arg#*=}" ;;
    --dtype=*) DTYPE="${arg#*=}" ;;
    --lookup-mode=*) LOOKUP_MODE="${arg#*=}" ;;
    --advertise-host=*) ADVERTISE_HOST="${arg#*=}" ;;
    --cleanup-artifacts) CLEANUP_ARTIFACTS="true" ;;
    --no-cleanup-artifacts) CLEANUP_ARTIFACTS="false" ;;
    --daemon-config=*) DAEMON_CONFIG="${arg#*=}" ;;
    --log-level=*) LOG_LEVEL="${arg#*=}" ;;
    --set=*) EXTRA_DAEMON_SETS+=("${arg#*=}") ;;
    --visibility-timeout-sec=*) VISIBILITY_TIMEOUT_SEC="${arg#*=}" ;;
    --visibility-retry-interval-sec=*) VISIBILITY_RETRY_INTERVAL_SEC="${arg#*=}" ;;
    --global-addr=*) GLOBAL_ADDR="${arg#*=}" ;;
    --put-addr=*) PUT_ADDR="${arg#*=}" ;;
    --get-addr=*) GET_ADDR="${arg#*=}" ;;
    --run-id=*) RUN_ID="${arg#*=}" ;;
    --put-session=*) PUT_SESSION="${arg#*=}" ;;
    --get-session=*) GET_SESSION="${arg#*=}" ;;
    --put-daemon-id=*) PUT_DAEMON_ID="${arg#*=}" ;;
    --get-daemon-id=*) GET_DAEMON_ID="${arg#*=}" ;;
    --results-dir=*) RESULTS_DIR="${arg#*=}" ;;
    --base-dir=*) BASE_DIR="${arg#*=}" ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RESULTS_DIR}" ]]; then
  RESULTS_DIR="${BASE_DIR}/results"
fi
if [[ -z "${PUT_HOME}" ]]; then
  PUT_HOME="${BASE_DIR}/put-home"
fi
if [[ -z "${GET_HOME}" ]]; then
  GET_HOME="${BASE_DIR}/get-home"
fi
if [[ -z "${PUT_STORAGE}" ]]; then
  PUT_STORAGE="${BASE_DIR}/put-storage"
fi
if [[ -z "${GET_STORAGE}" ]]; then
  GET_STORAGE="${BASE_DIR}/get-storage"
fi

mkdir -p "${RESULTS_DIR}" "${PUT_HOME}" "${GET_HOME}" "${PUT_STORAGE}" "${GET_STORAGE}"

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(date +%Y%m%d%H%M%S)-${RANDOM}"
fi
if [[ -z "${PUT_SESSION}" ]]; then
  PUT_SESSION="bench-put-${RUN_ID}"
fi
if [[ -z "${GET_SESSION}" ]]; then
  GET_SESSION="bench-get-${RUN_ID}"
fi
if [[ -z "${PUT_DAEMON_ID}" ]]; then
  PUT_DAEMON_ID="${PUT_SESSION}"
fi
if [[ -z "${GET_DAEMON_ID}" ]]; then
  GET_DAEMON_ID="${GET_SESSION}"
fi

port_in_use() {
  local port="$1"
  ss -ltn | awk 'NR > 1 {print $4}' | grep -Eq "(^|:)${port}$"
}

pick_free_port() {
  local candidate=""
  local attempt=0
  while (( attempt < 200 )); do
    candidate=$((20000 + RANDOM % 40000))
    if ! port_in_use "${candidate}"; then
      echo "${candidate}"
      return 0
    fi
    attempt=$((attempt + 1))
  done
  echo "failed to find a free TCP port after 200 attempts" >&2
  return 1
}

if [[ -z "${PUT_ADDR}" ]]; then
  PUT_ADDR="127.0.0.1:$(pick_free_port)"
fi
if [[ -z "${GET_ADDR}" ]]; then
  while true; do
    candidate_port="$(pick_free_port)"
    if [[ "${candidate_port}" != "${PUT_ADDR##*:}" ]]; then
      GET_ADDR="127.0.0.1:${candidate_port}"
      break
    fi
  done
fi
if [[ -z "${P2P_PUT_PORT}" ]]; then
  while true; do
    candidate_port="$(pick_free_port)"
    if [[ "${candidate_port}" != "${PUT_ADDR##*:}" && "${candidate_port}" != "${GET_ADDR##*:}" ]]; then
      P2P_PUT_PORT="${candidate_port}"
      break
    fi
  done
fi
if [[ -z "${P2P_GET_PORT}" ]]; then
  while true; do
    candidate_port="$(pick_free_port)"
    if [[ "${candidate_port}" != "${PUT_ADDR##*:}" && "${candidate_port}" != "${GET_ADDR##*:}" && "${candidate_port}" != "${P2P_PUT_PORT}" ]]; then
      P2P_GET_PORT="${candidate_port}"
      break
    fi
  done
fi

resolve_advertise_host() {
  if [[ -n "${ADVERTISE_HOST}" ]]; then
    echo "${ADVERTISE_HOST}"
    return
  fi
  local detected=""
  detected="$(ip -4 -o addr show scope global up 2>/dev/null | awk '{split($4, a, "/"); if (a[1] != "127.0.0.1") {print a[1]; exit}}')"
  if [[ -n "${detected}" ]]; then
    echo "${detected}"
    return
  fi
  detected="$(hostname -I 2>/dev/null | awk '{for (i = 1; i <= NF; i++) if ($i != "127.0.0.1") {print $i; exit}}')"
  if [[ -n "${detected}" ]]; then
    echo "${detected}"
    return
  fi
  echo "127.0.0.1"
}

ADVERTISE_HOST="$(resolve_advertise_host)"

if [[ "${ADVERTISE_HOST}" == "127.0.0.1" ]]; then
  echo "[driver] warning: advertise_host resolved to loopback (127.0.0.1). Global Store registration can fail on strict routability checks."
fi

total_put_mib=$((SIZE_MIB * (WARMUP + ITERATIONS)))
if [[ "${CLEANUP_ARTIFACTS}" != "true" && "${total_put_mib}" -gt 65536 ]]; then
  echo "[driver] warning: expected put volume per run is ${total_put_mib} MiB (> 65536 MiB default stable tier)."
  echo "[driver] warning: use --cleanup-artifacts or reduce --size-mib / --iterations / --warmup."
fi

echo "[driver] run_id=${RUN_ID} put_session=${PUT_SESSION} get_session=${GET_SESSION}"
echo "[driver] put_addr=${PUT_ADDR} get_addr=${GET_ADDR} p2p_put_port=${P2P_PUT_PORT} p2p_get_port=${P2P_GET_PORT}"

stop_daemon() {
  local home="$1"
  TENSORCAST_HOME="${home}" tensorcast-cli daemon stop >/dev/null 2>&1 || true
}

cleanup() {
  stop_daemon "${PUT_HOME}"
  stop_daemon "${GET_HOME}"
}
trap cleanup EXIT

echo "[driver] starting put daemon"
daemon_config_args=()
if [[ -n "${DAEMON_CONFIG}" ]]; then
  daemon_config_args=(-c "${DAEMON_CONFIG}")
fi
daemon_set_args=()
if [[ ${#EXTRA_DAEMON_SETS[@]} -gt 0 ]]; then
  for daemon_set in "${EXTRA_DAEMON_SETS[@]}"; do
    daemon_set_args+=(--set "${daemon_set}")
  done
fi
TENSORCAST_HOME="${PUT_HOME}" LD_LIBRARY_PATH=/data/cuda/compat \
  tensorcast-cli daemon start \
    "${daemon_config_args[@]}" \
    "${daemon_set_args[@]}" \
    --session "${PUT_SESSION}" \
    --global-store-mode connect \
    --global-store-address "${GLOBAL_ADDR}" \
    --set daemon_id="${PUT_DAEMON_ID}" \
    --set server.storage_path="${PUT_STORAGE}" \
    --set server.listen.port="${PUT_ADDR##*:}" \
    --set server.p2p_listen.port="${P2P_PUT_PORT}" \
    --set server.advertise.host="${ADVERTISE_HOST}" \
    --set high_availability.enabled=true \
    --set high_availability.heartbeat_interval="${HEARTBEAT_SEC}s" \
    --set high_availability.periodic_sync_interval="${SYNC_SEC}s" \
    --set communicator.transport.tcp_conn_count="${CONN_COUNT}" \
    --set communicator.stager.buffers_per_flow="${BUFFERS_PER_FLOW}" \
    --set observability.logging.level="${LOG_LEVEL}" \
    --json

echo "[driver] starting get daemon"
TENSORCAST_HOME="${GET_HOME}" LD_LIBRARY_PATH=/data/cuda/compat \
  tensorcast-cli daemon start \
    "${daemon_config_args[@]}" \
    "${daemon_set_args[@]}" \
    --session "${GET_SESSION}" \
    --global-store-mode connect \
    --global-store-address "${GLOBAL_ADDR}" \
    --set daemon_id="${GET_DAEMON_ID}" \
    --set server.storage_path="${GET_STORAGE}" \
    --set server.listen.port="${GET_ADDR##*:}" \
    --set server.p2p_listen.port="${P2P_GET_PORT}" \
    --set server.advertise.host="${ADVERTISE_HOST}" \
    --set high_availability.enabled=true \
    --set high_availability.heartbeat_interval="${HEARTBEAT_SEC}s" \
    --set high_availability.periodic_sync_interval="${SYNC_SEC}s" \
    --set communicator.transport.tcp_conn_count="${CONN_COUNT}" \
    --set communicator.stager.buffers_per_flow="${BUFFERS_PER_FLOW}" \
    --set observability.logging.level="${LOG_LEVEL}" \
    --json

OUT_JSON="${RESULTS_DIR}/p2p_hb${HEARTBEAT_SEC}_sync${SYNC_SEC}_tcp${CONN_COUNT}_buf${BUFFERS_PER_FLOW}_${SIZE_MIB}m.json"

echo "[driver] running benchmark -> ${OUT_JSON}"
if [[ "${LOOKUP_MODE}" != "key" ]]; then
  echo "[driver] warning: lookup_mode=${LOOKUP_MODE} bypasses key-mapping/index resolution and should only be used for A/B control."
fi
cleanup_args=()
if [[ "${CLEANUP_ARTIFACTS}" == "true" ]]; then
  cleanup_args+=(--cleanup-artifacts)
fi
uv run "${ROOT_DIR}/examples/tensorcast_sdk_p2p_put_get_benchmark.py" \
  --put-daemon "${PUT_ADDR}" \
  --get-daemon "${GET_ADDR}" \
  --put-device "${PUT_DEVICE}" \
  --get-device "${GET_DEVICE}" \
  --dtype "${DTYPE}" \
  --lookup-mode "${LOOKUP_MODE}" \
  --size-mib "${SIZE_MIB}" \
  --tensor-count 1 \
  --warmup "${WARMUP}" \
  --iterations "${ITERATIONS}" \
  --put-policy "${PUT_POLICY}" \
  --prefer p2p \
  --no-allow-disk \
  --require-p2p \
  --sync-cuda \
  --export-policy force \
  --visibility-timeout-sec "${VISIBILITY_TIMEOUT_SEC}" \
  --visibility-retry-interval-sec "${VISIBILITY_RETRY_INTERVAL_SEC}" \
  "${cleanup_args[@]}" \
  --json-out "${OUT_JSON}"

echo "[driver] done"
