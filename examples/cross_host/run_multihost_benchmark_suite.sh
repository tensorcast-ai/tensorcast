#!/usr/bin/env bash
# Copyright (c) 2026, TensorCast Team.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

: "${TC_SEED_PROC:?set TC_SEED_PROC (seed brainctl process id)}"
: "${TC_SEED_IP:?set TC_SEED_IP (seed advertise ip)}"
: "${TC_GET_PROCS:?set TC_GET_PROCS (comma-separated getter process ids)}"
: "${TC_GET_IPS:?set TC_GET_IPS (comma-separated getter advertise ips)}"
: "${TC_GS_ADDR:?set TC_GS_ADDR (global store host:port)}"

TC_DAEMON_CONFIG="${TC_DAEMON_CONFIG:-examples/config/store_daemon_config_cross_host_bench.yaml}"
TC_OUT_DIR="${TC_OUT_DIR:-/tmp/tc_cross_20260222/results_multi_host}"
TC_RUN_ID="${TC_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TC_PORT_BASE="${TC_PORT_BASE:-62800}"
TC_REMOTE_TIMEOUT_SEC="${TC_REMOTE_TIMEOUT_SEC:-900}"
TC_DAEMON_START_TIMEOUT_SEC="${TC_DAEMON_START_TIMEOUT_SEC:-600}"
TC_STOP_TIMEOUT_SEC="${TC_STOP_TIMEOUT_SEC:-240}"
TC_SOURCE_SETTLE_SEC="${TC_SOURCE_SETTLE_SEC:-2}"
TC_PUT_SEED_BASE="${TC_PUT_SEED_BASE:-25000}"
TC_PHASE="${TC_PHASE:-all}"

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
      echo "Usage: $0 [--phase small|medium|large|all]" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${TC_PHASE}" ]]; then
  echo "TC_PHASE cannot be empty" >&2
  exit 1
fi
if [[ "${TC_PHASE}" != "small" && "${TC_PHASE}" != "medium" && "${TC_PHASE}" != "large" && "${TC_PHASE}" != "all" ]]; then
  echo "Invalid phase '${TC_PHASE}'. Expected: small|medium|large|all" >&2
  exit 1
fi

IFS=',' read -r -a GET_PROCS_ARR <<< "${TC_GET_PROCS}"
IFS=',' read -r -a GET_IPS_ARR <<< "${TC_GET_IPS}"
if [[ "${#GET_PROCS_ARR[@]}" -ne "${#GET_IPS_ARR[@]}" ]]; then
  echo "TC_GET_PROCS count != TC_GET_IPS count" >&2
  exit 1
fi
GETTER_COUNT="${#GET_PROCS_ARR[@]}"

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

run_cascade_case() {
  local case_name=$1
  local getter_count=$2
  local seed_grpc=$3
  local seed_p2p=$4
  local get_grpc=$5
  local get_p2p=$6
  local put_seed=$7

  echo "[suite] running cascade case=${case_name} getters=${getter_count}"
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
    --put-seed-base "${put_seed}" \
    --require-p2p \
    --daemon-config "${TC_DAEMON_CONFIG}" \
    --daemon-start-timeout-sec "${TC_DAEMON_START_TIMEOUT_SEC}" \
    --remote-timeout-sec "${TC_REMOTE_TIMEOUT_SEC}" \
    --stop-timeout-sec "${TC_STOP_TIMEOUT_SEC}" \
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
  fi
}

mkdir -p "${TC_OUT_DIR}"

META_FILE="${TC_OUT_DIR}/suite_${TC_RUN_ID}_meta.json"
cat > "${META_FILE}" <<EOF
{
  "run_id": "${TC_RUN_ID}",
  "phase": "${TC_PHASE}",
  "seed_process": "${TC_SEED_PROC}",
  "seed_ip": "${TC_SEED_IP}",
  "getter_count": ${GETTER_COUNT},
  "global_store": "${TC_GS_ADDR}"
}
EOF

echo "[suite] phase=${TC_PHASE} run_id=${TC_RUN_ID} getters=${GETTER_COUNT}"
echo "[suite] meta=${META_FILE}"

case "${TC_PHASE}" in
  small)
    run_phase_small
    ;;
  medium)
    run_phase_medium
    ;;
  large)
    run_phase_large
    ;;
  all)
    run_phase_small
    if [[ "${GETTER_COUNT}" -ge 5 ]]; then
      run_phase_medium
    else
      echo "[suite] skip medium phase: getters=${GETTER_COUNT} (<5)"
    fi
    if [[ "${GETTER_COUNT}" -ge 7 ]]; then
      run_phase_large
    else
      echo "[suite] skip large phase: getters=${GETTER_COUNT} (<7)"
    fi
    ;;
esac

echo "[suite] completed. outputs in ${TC_OUT_DIR}"
