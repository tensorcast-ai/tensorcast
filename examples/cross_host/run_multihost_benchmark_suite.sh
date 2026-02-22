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

IFS=',' read -r -a GET_PROCS_ARR <<< "${TC_GET_PROCS}"
IFS=',' read -r -a GET_IPS_ARR <<< "${TC_GET_IPS}"
if [[ "${#GET_PROCS_ARR[@]}" -ne "${#GET_IPS_ARR[@]}" ]]; then
  echo "TC_GET_PROCS count != TC_GET_IPS count" >&2
  exit 1
fi
GETTER_COUNT="${#GET_PROCS_ARR[@]}"
if [[ "${GETTER_COUNT}" -lt 7 ]]; then
  echo "Need at least 7 getters in TC_GET_PROCS/TC_GET_IPS for full suite" >&2
  exit 1
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

run_cascade_case() {
  local case_name=$1
  local seed_grpc=$2
  local seed_p2p=$3
  local get_grpc=$4
  local get_p2p=$5
  local put_seed=$6

  echo "[suite] running cascade case=${case_name}"
  python examples/cross_host/cross_host_fanout_runner.py \
    --mode cascade \
    --case-name "${case_name}" \
    --seed-proc "${TC_SEED_PROC}" \
    --seed-adv-ip "${TC_SEED_IP}" \
    --seed-grpc-port "${seed_grpc}" \
    --seed-p2p-port "${seed_p2p}" \
    --get-procs "$(join_first_n GET_PROCS_ARR 3)" \
    --get-adv-ips "$(join_first_n GET_IPS_ARR 3)" \
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
    --out-dir "${TC_OUT_DIR}"
}

mkdir -p "${TC_OUT_DIR}"

run_cascade_case \
  "suite_${TC_RUN_ID}_cascade_4n_deregister" \
  "$((TC_PORT_BASE + 1))" \
  "$((TC_PORT_BASE + 1001))" \
  "$((TC_PORT_BASE + 11))" \
  "$((TC_PORT_BASE + 1011))" \
  "$((TC_PUT_SEED_BASE + 1))"

run_fanout_case \
  "suite_${TC_RUN_ID}_fanout_6n_c20b16w16_g0_s1024" \
  5 \
  "$((TC_PORT_BASE + 101))" \
  "$((TC_PORT_BASE + 1101))" \
  "$((TC_PORT_BASE + 111))" \
  "$((TC_PORT_BASE + 1111))" \
  20 16 16 0 1024 1 4 2 \
  "$((TC_PUT_SEED_BASE + 101))"

run_fanout_case \
  "suite_${TC_RUN_ID}_fanout_8n_c20b16w16_g0_s1024" \
  7 \
  "$((TC_PORT_BASE + 201))" \
  "$((TC_PORT_BASE + 1201))" \
  "$((TC_PORT_BASE + 211))" \
  "$((TC_PORT_BASE + 1211))" \
  20 16 16 0 1024 1 4 3 \
  "$((TC_PUT_SEED_BASE + 201))"

run_fanout_case \
  "suite_${TC_RUN_ID}_fanout_8n_c24b12w24_g0_s1024" \
  7 \
  "$((TC_PORT_BASE + 301))" \
  "$((TC_PORT_BASE + 1301))" \
  "$((TC_PORT_BASE + 311))" \
  "$((TC_PORT_BASE + 1311))" \
  24 12 24 0 1024 1 4 3 \
  "$((TC_PUT_SEED_BASE + 301))"

run_fanout_case \
  "suite_${TC_RUN_ID}_fanout_8n_c20b16w16_g8_s1024" \
  7 \
  "$((TC_PORT_BASE + 401))" \
  "$((TC_PORT_BASE + 1401))" \
  "$((TC_PORT_BASE + 411))" \
  "$((TC_PORT_BASE + 1411))" \
  20 16 16 8 1024 1 4 3 \
  "$((TC_PUT_SEED_BASE + 401))"

run_fanout_case \
  "suite_${TC_RUN_ID}_fanout_8n_c20b16w16_g0_s2048" \
  7 \
  "$((TC_PORT_BASE + 501))" \
  "$((TC_PORT_BASE + 1501))" \
  "$((TC_PORT_BASE + 511))" \
  "$((TC_PORT_BASE + 1511))" \
  20 16 16 0 2048 1 3 3 \
  "$((TC_PUT_SEED_BASE + 501))"

if [[ "${GETTER_COUNT}" -ge 8 ]]; then
  run_fanout_case \
    "suite_${TC_RUN_ID}_fanout_9n_c20b16w16_g0_s1024" \
    8 \
    "$((TC_PORT_BASE + 601))" \
    "$((TC_PORT_BASE + 1601))" \
    "$((TC_PORT_BASE + 611))" \
    "$((TC_PORT_BASE + 1611))" \
    20 16 16 0 1024 1 4 4 \
    "$((TC_PUT_SEED_BASE + 601))"

  run_fanout_case \
    "suite_${TC_RUN_ID}_fanout_9n_c20b16w16_g0_s2048" \
    8 \
    "$((TC_PORT_BASE + 701))" \
    "$((TC_PORT_BASE + 1701))" \
    "$((TC_PORT_BASE + 711))" \
    "$((TC_PORT_BASE + 1711))" \
    20 16 16 0 2048 1 3 4 \
    "$((TC_PUT_SEED_BASE + 701))"
else
  echo "[suite] skip 9n cases: getters=${GETTER_COUNT} (<8)"
fi

echo "[suite] completed. outputs in ${TC_OUT_DIR}"
