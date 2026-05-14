#!/usr/bin/env bash
# Copyright (c) 2026, TensorCast Team.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

: "${TC_WP_PUBLISHER_PROC:?set TC_WP_PUBLISHER_PROC (publisher brainctl process id)}"
: "${TC_WP_RECEIVER_PROCS:?set TC_WP_RECEIVER_PROCS (comma-separated receiver process ids)}"
: "${TC_GS_ADDR:?set TC_GS_ADDR (global store host:port)}"

TC_DAEMON_CONFIG="${TC_DAEMON_CONFIG:-examples/config/store_daemon_config_cross_host_bench.yaml}"
TC_DAEMON_CONNECT_ADDRESS="${TC_DAEMON_CONNECT_ADDRESS:-127.0.0.1:50052}"
TC_OUT_DIR="${TC_OUT_DIR:-/data/tc_cross_rerun/results_weight_publisher}"
TC_RUN_ID="${TC_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TC_MODEL_NAME_PREFIX="${TC_MODEL_NAME_PREFIX:-wp-multihost-suite}"
TC_START_VERSION="${TC_START_VERSION:-1}"
TC_NUM_VERSIONS="${TC_NUM_VERSIONS:-6}"
TC_KEEP_LAST="${TC_KEEP_LAST:-2}"
TC_PUBLISH_INTERVAL_S="${TC_PUBLISH_INTERVAL_S:-60}"
TC_PROGRESS_POLL_S="${TC_PROGRESS_POLL_S:-10}"
TC_POLL_INTERVAL_S="${TC_POLL_INTERVAL_S:-0.5}"
TC_RECEIVER_TIMEOUT_S="${TC_RECEIVER_TIMEOUT_S:-95}"
TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST="${TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST:-1}"
TC_WP_KEEP_LAST_AUTO_ADJUST="${TC_WP_KEEP_LAST_AUTO_ADJUST:-1}"
TC_MAX_PUBLISH_TO_APPLY_S="${TC_MAX_PUBLISH_TO_APPLY_S:-30}"
TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST="${TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST:-1}"
TC_WP_PRE_PUBLISH_TRIM_MARGIN="${TC_WP_PRE_PUBLISH_TRIM_MARGIN:-1}"
TC_RETENTION_TIMEOUT_S="${TC_RETENTION_TIMEOUT_S:-90}"
TC_REMOTE_TIMEOUT_SEC="${TC_REMOTE_TIMEOUT_SEC:-1800}"
TC_RECEIVER_HOLD_AFTER_FINISH_S="${TC_RECEIVER_HOLD_AFTER_FINISH_S:-25}"
TC_PUBLISHER_HOLD_AFTER_FINISH_S="${TC_PUBLISHER_HOLD_AFTER_FINISH_S:-30}"
TC_RECEIVER_WARMUP_S="${TC_RECEIVER_WARMUP_S:-8}"
TC_PUBLISH_DEVICE="${TC_PUBLISH_DEVICE:-cpu}"
TC_MATERIALIZE_DEVICE="${TC_MATERIALIZE_DEVICE:-cuda:0}"
TC_ALLOW_RECEIVER_SKIPS="${TC_ALLOW_RECEIVER_SKIPS:-0}"
TC_WP_SCALE_RECEIVER_COUNTS="${TC_WP_SCALE_RECEIVER_COUNTS:-${TC_SCALE_RECEIVER_COUNTS:-1,2,4,8,16,31}}"
TC_SCALE_NUM_VERSIONS="${TC_SCALE_NUM_VERSIONS:-10}"
TC_SCALE_PUBLISH_INTERVAL_S="${TC_SCALE_PUBLISH_INTERVAL_S:-20}"
TC_WP_PAYLOAD_MODE="${TC_WP_PAYLOAD_MODE:-tp_ranked}"
TC_WP_TP_WORLD_SIZE="${TC_WP_TP_WORLD_SIZE:-1}"
TC_WP_TP_TOTAL_BYTES="${TC_WP_TP_TOTAL_BYTES:-0}"
TC_WP_TP_DEVICE_MAP_POLICY="${TC_WP_TP_DEVICE_MAP_POLICY:-auto}"
TC_WP_MAX_CONCURRENCY="${TC_WP_MAX_CONCURRENCY:-1}"
TC_WP_PUBLISHER_DAEMON_CONFIG="${TC_WP_PUBLISHER_DAEMON_CONFIG:-${TC_DAEMON_CONFIG}}"
TC_WP_RECEIVER_DAEMON_CONFIG="${TC_WP_RECEIVER_DAEMON_CONFIG:-${TC_DAEMON_CONFIG}}"
TC_WP_RECEIVER_PREFLIGHT_TRANSIENT_OVERLAP="${TC_WP_RECEIVER_PREFLIGHT_TRANSIENT_OVERLAP:-1}"
TC_WP_P0_EARLY_STOP="${TC_WP_P0_EARLY_STOP:-1}"
TC_WP_P0_EARLY_STOP_GRACE_S="${TC_WP_P0_EARLY_STOP_GRACE_S:-20}"
TC_WP_THROUGHPUT_SAMPLE_INTERVAL_S="${TC_WP_THROUGHPUT_SAMPLE_INTERVAL_S:-1.0}"
TC_WP_THROUGHPUT_MAX_SAMPLES="${TC_WP_THROUGHPUT_MAX_SAMPLES:-20000}"
TC_SINGLE_HOST_FUNCTIONAL_ENABLE="${TC_SINGLE_HOST_FUNCTIONAL_ENABLE:-1}"
TC_SINGLE_HOST_DAEMON_CONFIG="${TC_SINGLE_HOST_DAEMON_CONFIG:-${TC_DAEMON_CONFIG}}"
TC_WP_TP_AUTO_KEEP_LAST_ONE="${TC_WP_TP_AUTO_KEEP_LAST_ONE:-1}"
TC_LONG_RUN_ENABLE="${TC_LONG_RUN_ENABLE:-1}"
TC_LONG_RUN_RECEIVER_COUNT="${TC_LONG_RUN_RECEIVER_COUNT:-0}"
TC_LONG_RUN_NUM_VERSIONS="${TC_LONG_RUN_NUM_VERSIONS:-20}"
TC_LONG_RUN_TARGET_DURATION_S="${TC_LONG_RUN_TARGET_DURATION_S:-900}"
TC_LONG_RUN_PUBLISH_INTERVAL_S="${TC_LONG_RUN_PUBLISH_INTERVAL_S:-}"
TC_LONG_RUN_RECEIVER_TIMEOUT_S="${TC_LONG_RUN_RECEIVER_TIMEOUT_S:-}"
TC_RUN_AS_USER="${TC_RUN_AS_USER:-$(id -un)}"

if [[ -z "${TC_RUN_AS_USER}" || "${TC_RUN_AS_USER}" == "root" ]]; then
  echo "TC_RUN_AS_USER must resolve to a non-root user" >&2
  exit 2
fi

if [[ "${TC_WP_TP_AUTO_KEEP_LAST_ONE}" == "1" && "${TC_WP_PAYLOAD_MODE}" == "tp_ranked" && "${TC_WP_TP_TOTAL_BYTES}" -gt 0 && "${TC_KEEP_LAST}" -gt 1 ]]; then
  echo "[wp-suite] auto-adjust keep_last ${TC_KEEP_LAST} -> 1 for tp_ranked payload memory safety"
  TC_KEEP_LAST=1
fi

IFS=',' read -r -a RECEIVER_PROCS_ARR <<< "${TC_WP_RECEIVER_PROCS}"
if [[ "${#RECEIVER_PROCS_ARR[@]}" -lt 1 ]]; then
  echo "TC_WP_RECEIVER_PROCS must contain at least one process id" >&2
  exit 1
fi

RUN_DIR="${TC_OUT_DIR}/${TC_RUN_ID}"
mkdir -p "${RUN_DIR}"
SKIP_MANIFEST_FILE="${RUN_DIR}/suite_skips.jsonl"
: > "${SKIP_MANIFEST_FILE}"

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

array_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "${item}" == "${needle}" ]]; then
      return 0
    fi
  done
  return 1
}

append_skip_manifest() {
  local case_name=$1
  local reason=$2
  local detail=${3:-""}
  local escaped_case_name escaped_reason escaped_detail
  escaped_case_name="$(json_escape "${case_name}")"
  escaped_reason="$(json_escape "${reason}")"
  escaped_detail="$(json_escape "${detail}")"
  cat >> "${SKIP_MANIFEST_FILE}" <<EOF
{"case_name":"${escaped_case_name}","reason":"${escaped_reason}","detail":"${escaped_detail}"}
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

remote_exec_as_user() {
  local process_id="$1"
  local inner_cmd="$2"
  local inner_b64
  inner_b64="$(printf '%s' "${inner_cmd}" | base64 | tr -d '\n')"
  brainctl exec process/"${process_id}" -n shai-core -- bash -lc "
set -euo pipefail
run_as_user='${TC_RUN_AS_USER}'
if [ \"\${run_as_user}\" = \"root\" ]; then
  echo 'refuse to run as root' >&2
  exit 2
fi
getent passwd \"\${run_as_user}\" >/dev/null
decoded_cmd=\"\$(printf '%s' '${inner_b64}' | base64 -d)\"
if [ \"\$(id -un)\" = \"\${run_as_user}\" ]; then
  bash -lc \"\${decoded_cmd}\"
else
  su - \"\${run_as_user}\" -s /bin/bash -c \"\${decoded_cmd}\"
fi
"
}

run_single_host_functional() {
  local model_name="${TC_MODEL_NAME_PREFIX}-${TC_RUN_ID}-single"
  local remote_summary="/data/tc_cross_rerun/results_weight_publisher/${TC_RUN_ID}/single_host_summary.json"
  local local_summary="${RUN_DIR}/single_host_summary.json"
  local daemon_session="wp-suite-single-${TC_RUN_ID}"
  local daemon_port="${TC_DAEMON_CONNECT_ADDRESS##*:}"
  echo "[wp-suite] functional single-host smoke on publisher process=${TC_WP_PUBLISHER_PROC}"
  local single_host_cmd
  single_host_cmd="$(cat <<EOF
set -euo pipefail
cd ${REPO_ROOT}
source .venv/bin/activate
export LD_LIBRARY_PATH=/data/cuda/compat:\${LD_LIBRARY_PATH:-}
cleanup() {
  tensorcast-cli daemon stop --session ${daemon_session} >/dev/null 2>&1 || true
  tensorcast-cli daemon stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT
tensorcast-cli daemon stop --force >/dev/null 2>&1 || true
for pid in \$(pgrep -f '[t]ensorcast_daemon --config=' || true); do
  kill -TERM "\${pid}" >/dev/null 2>&1 || true
done
sleep 1
python -c "from tensorcast.cli_utils.paths import runtime_state_path; from tensorcast.cli_utils.process import clear_runtime_global_store; p=runtime_state_path(); clear_runtime_global_store(p, preserve_cluster_token=False) if p.exists() else None" >/dev/null 2>&1 || true
tensorcast-cli daemon start \
  --config ${TC_SINGLE_HOST_DAEMON_CONFIG} \
  --session ${daemon_session} \
  --global-store-mode connect \
  --global-store-address ${TC_GS_ADDR} \
  --set high_availability.enabled=true \
  --set high_availability.heartbeat_interval=5s \
  --set high_availability.periodic_sync_interval=5s \
  --set server.listen.host=0.0.0.0 \
  --set server.listen.port=${daemon_port} \
  --json >/dev/null
mkdir -p \$(dirname "${remote_summary}")
python ./tensorcast/tools/weight_publisher_e2e.py single-host \
  --init-mode connect \
  --connect-address ${TC_DAEMON_CONNECT_ADDRESS} \
  --model-name ${model_name} \
  --start-version 1 \
  --num-versions 4 \
  --keep-last 2 \
  --publish-interval-s 2 \
  --poll-interval-s ${TC_POLL_INTERVAL_S} \
  --receiver-timeout-s ${TC_RECEIVER_TIMEOUT_S} \
  --retention-timeout-s ${TC_RETENTION_TIMEOUT_S} \
  --publish-device ${TC_PUBLISH_DEVICE} \
  --materialize-device ${TC_MATERIALIZE_DEVICE} \
  --payload-mode ${TC_WP_PAYLOAD_MODE} \
  --tp-world-size ${TC_WP_TP_WORLD_SIZE} \
  --tp-total-bytes ${TC_WP_TP_TOTAL_BYTES} \
  --tp-device-map-policy ${TC_WP_TP_DEVICE_MAP_POLICY} \
  --transport-group-namespace ${model_name}:single \
  --transport-group-total-parts ${TC_WP_TP_WORLD_SIZE} \
  --weights-root /data/tc_cross_rerun/weights_single_host \
  --run-id ${TC_RUN_ID}-single \
  --output-json ${remote_summary}
EOF
)"
  remote_exec_as_user "${TC_WP_PUBLISHER_PROC}" "${single_host_cmd}"
  remote_exec_as_user "${TC_WP_PUBLISHER_PROC}" "cat ${remote_summary}" > "${local_summary}"
}

run_case() {
  local case_name=$1
  local receiver_count=$2
  local num_versions=$3
  local publish_interval_s=$4
  local receiver_timeout_s=$5
  local receiver_procs
  receiver_procs="$(join_first_n RECEIVER_PROCS_ARR "${receiver_count}")"
  local model_name="${TC_MODEL_NAME_PREFIX}-${case_name}"
  local -a cmd=(
    python ./examples/cross_host/cross_host_weight_publisher_runner.py
    --case-name "${case_name}"
    --publisher-proc "${TC_WP_PUBLISHER_PROC}"
    --receiver-procs "${receiver_procs}"
    --gs-addr "${TC_GS_ADDR}"
    --daemon-config "${TC_DAEMON_CONFIG}"
    --publisher-daemon-config "${TC_WP_PUBLISHER_DAEMON_CONFIG}"
    --receiver-daemon-config "${TC_WP_RECEIVER_DAEMON_CONFIG}"
    --daemon-connect-address "${TC_DAEMON_CONNECT_ADDRESS}"
    --model-name "${model_name}"
    --start-version "${TC_START_VERSION}"
    --num-versions "${num_versions}"
    --keep-last "${TC_KEEP_LAST}"
    --pre-publish-trim-margin "${TC_WP_PRE_PUBLISH_TRIM_MARGIN}"
    --publish-interval-s "${publish_interval_s}"
    --poll-interval-s "${TC_POLL_INTERVAL_S}"
    --receiver-timeout-s "${receiver_timeout_s}"
    --progress-poll-s "${TC_PROGRESS_POLL_S}"
    --max-publish-to-apply-s "${TC_MAX_PUBLISH_TO_APPLY_S}"
    --retention-timeout-s "${TC_RETENTION_TIMEOUT_S}"
    --publish-device "${TC_PUBLISH_DEVICE}"
    --materialize-device "${TC_MATERIALIZE_DEVICE}"
    --receiver-hold-after-finish-s "${TC_RECEIVER_HOLD_AFTER_FINISH_S}"
    --publisher-hold-after-finish-s "${TC_PUBLISHER_HOLD_AFTER_FINISH_S}"
    --receiver-warmup-s "${TC_RECEIVER_WARMUP_S}"
    --remote-timeout-sec "${TC_REMOTE_TIMEOUT_SEC}"
    --payload-mode "${TC_WP_PAYLOAD_MODE}"
    --tp-world-size "${TC_WP_TP_WORLD_SIZE}"
    --tp-total-bytes "${TC_WP_TP_TOTAL_BYTES}"
    --tp-device-map-policy "${TC_WP_TP_DEVICE_MAP_POLICY}"
    --max-concurrency "${TC_WP_MAX_CONCURRENCY}"
    --receiver-preflight-transient-overlap "${TC_WP_RECEIVER_PREFLIGHT_TRANSIENT_OVERLAP}"
    --throughput-sample-interval-s "${TC_WP_THROUGHPUT_SAMPLE_INTERVAL_S}"
    --throughput-max-samples "${TC_WP_THROUGHPUT_MAX_SAMPLES}"
    --p0-early-stop-grace-s "${TC_WP_P0_EARLY_STOP_GRACE_S}"
    --out-dir "${RUN_DIR}"
  )
  if [[ "${TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST}" == "1" ]]; then
    cmd+=(--receiver-timeout-auto-adjust)
  else
    cmd+=(--no-receiver-timeout-auto-adjust)
  fi
  if [[ "${TC_WP_KEEP_LAST_AUTO_ADJUST}" == "1" ]]; then
    cmd+=(--keep-last-auto-adjust)
  else
    cmd+=(--no-keep-last-auto-adjust)
  fi
  if [[ "${TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST}" == "1" ]]; then
    cmd+=(--max-publish-to-apply-auto-adjust)
  else
    cmd+=(--no-max-publish-to-apply-auto-adjust)
  fi
  if [[ "${TC_WP_P0_EARLY_STOP}" == "1" ]]; then
    cmd+=(--p0-early-stop)
  else
    cmd+=(--no-p0-early-stop)
  fi
  if [[ "${TC_ALLOW_RECEIVER_SKIPS}" == "1" ]]; then
    cmd+=(--allow-receiver-skips)
  fi
  echo "[wp-suite] running case=${case_name} receivers=${receiver_count} num_versions=${num_versions} publish_interval_s=${publish_interval_s} receiver_timeout_s=${receiver_timeout_s}"
  "${cmd[@]}"
}

echo "[wp-suite] run_id=${TC_RUN_ID} publisher=${TC_WP_PUBLISHER_PROC} receivers=${#RECEIVER_PROCS_ARR[@]}"
echo "[wp-suite] daemon_config=${TC_DAEMON_CONFIG} daemon_connect=${TC_DAEMON_CONNECT_ADDRESS} gs_addr=${TC_GS_ADDR}"
echo "[wp-suite] payload_mode=${TC_WP_PAYLOAD_MODE} tp_world_size=${TC_WP_TP_WORLD_SIZE} tp_total_bytes=${TC_WP_TP_TOTAL_BYTES} tp_device_map_policy=${TC_WP_TP_DEVICE_MAP_POLICY} max_concurrency=${TC_WP_MAX_CONCURRENCY}"
echo "[wp-suite] receiver_timeout_s=${TC_RECEIVER_TIMEOUT_S} receiver_timeout_auto_adjust=${TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST} keep_last_auto_adjust=${TC_WP_KEEP_LAST_AUTO_ADJUST} max_publish_to_apply_auto_adjust=${TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST} pre_publish_trim_margin=${TC_WP_PRE_PUBLISH_TRIM_MARGIN}"
STRICT_NO_SKIP="true"
if [[ "${TC_ALLOW_RECEIVER_SKIPS}" == "1" ]]; then
  STRICT_NO_SKIP="false"
fi
echo "[wp-suite] strict_no_skip=${STRICT_NO_SKIP} progress_poll_s=${TC_PROGRESS_POLL_S}"

declare -a SCALE_CASE_COUNTS=()
IFS=',' read -r -a RAW_SCALE_COUNTS <<< "${TC_WP_SCALE_RECEIVER_COUNTS}"
for raw in "${RAW_SCALE_COUNTS[@]}"; do
  value="$(echo "${raw}" | xargs)"
  if [[ -z "${value}" ]]; then
    continue
  fi
  if ! [[ "${value}" =~ ^[0-9]+$ ]]; then
    append_skip_manifest \
      "suite_${TC_RUN_ID}_scale_receivers_${value}" \
      "invalid_scale_receiver_count" \
      "value=${value} available=${#RECEIVER_PROCS_ARR[@]}"
    continue
  fi
  if [[ "${value}" -lt 1 || "${value}" -gt "${#RECEIVER_PROCS_ARR[@]}" ]]; then
    append_skip_manifest \
      "suite_${TC_RUN_ID}_scale_receivers_${value}" \
      "out_of_range_scale_receiver_count" \
      "value=${value} available=${#RECEIVER_PROCS_ARR[@]}"
    continue
  fi
  if ! array_contains "${value}" "${SCALE_CASE_COUNTS[@]}"; then
    SCALE_CASE_COUNTS+=("${value}")
  fi
done
if [[ "${#SCALE_CASE_COUNTS[@]}" -eq 0 ]]; then
  SCALE_CASE_COUNTS+=("1")
  if [[ "${#RECEIVER_PROCS_ARR[@]}" -ge 2 ]]; then
    SCALE_CASE_COUNTS+=("2")
  fi
fi
echo "[wp-suite] staged_receiver_counts=${SCALE_CASE_COUNTS[*]} (available=${#RECEIVER_PROCS_ARR[@]})"

declare -a EXECUTED_CASES=()

if [[ "${TC_SINGLE_HOST_FUNCTIONAL_ENABLE}" == "1" ]]; then
  run_single_host_functional
else
  append_skip_manifest \
    "suite_${TC_RUN_ID}_single_host_functional" \
    "single_host_functional_disabled" \
    "TC_SINGLE_HOST_FUNCTIONAL_ENABLE=${TC_SINGLE_HOST_FUNCTIONAL_ENABLE}"
fi
for receiver_count in "${SCALE_CASE_COUNTS[@]}"; do
  case_name="suite_${TC_RUN_ID}_r${receiver_count}_group_realization"
  run_case \
    "${case_name}" \
    "${receiver_count}" \
    "${TC_SCALE_NUM_VERSIONS}" \
    "${TC_SCALE_PUBLISH_INTERVAL_S}" \
    "${TC_RECEIVER_TIMEOUT_S}"
  EXECUTED_CASES+=("${case_name}")
done

if [[ "${TC_LONG_RUN_ENABLE}" == "1" ]]; then
  long_receiver_count="${TC_LONG_RUN_RECEIVER_COUNT}"
  if [[ "${long_receiver_count}" -eq 0 || "${long_receiver_count}" -gt "${#RECEIVER_PROCS_ARR[@]}" ]]; then
    long_receiver_count="${#RECEIVER_PROCS_ARR[@]}"
  fi
  if [[ "${long_receiver_count}" -ge 1 ]]; then
    long_publish_interval_s="${TC_LONG_RUN_PUBLISH_INTERVAL_S}"
    if [[ -z "${long_publish_interval_s}" ]]; then
      long_publish_interval_s="$(
        awk -v target="${TC_LONG_RUN_TARGET_DURATION_S}" -v n="${TC_LONG_RUN_NUM_VERSIONS}" \
          'BEGIN { if (n <= 1) { printf "0"; } else { printf "%.2f", target / (n - 1); } }'
      )"
    fi
    long_receiver_timeout_s="${TC_LONG_RUN_RECEIVER_TIMEOUT_S}"
    if [[ -z "${long_receiver_timeout_s}" ]]; then
      long_receiver_timeout_s="$(
        awk -v interval="${long_publish_interval_s}" \
          'BEGIN { printf "%.2f", interval + 35.0; }'
      )"
    fi
    long_case_name="suite_${TC_RUN_ID}_long_r${long_receiver_count}_v${TC_LONG_RUN_NUM_VERSIONS}"
    echo "[wp-suite] long-run target_duration_s=${TC_LONG_RUN_TARGET_DURATION_S} computed_publish_interval_s=${long_publish_interval_s}"
    run_case \
      "${long_case_name}" \
      "${long_receiver_count}" \
      "${TC_LONG_RUN_NUM_VERSIONS}" \
      "${long_publish_interval_s}" \
      "${long_receiver_timeout_s}"
    EXECUTED_CASES+=("${long_case_name}")
  else
    echo "[wp-suite] skip long-run: no available receiver process"
    append_skip_manifest \
      "suite_${TC_RUN_ID}_long_run" \
      "no_available_receiver_process" \
      "requested=${TC_LONG_RUN_RECEIVER_COUNT} available=${#RECEIVER_PROCS_ARR[@]}"
  fi
else
  append_skip_manifest \
    "suite_${TC_RUN_ID}_long_run" \
    "long_run_disabled" \
    "TC_LONG_RUN_ENABLE=${TC_LONG_RUN_ENABLE}"
fi

cat > "${RUN_DIR}/suite_meta.json" <<EOF
{
  "run_id": "${TC_RUN_ID}",
  "publisher_proc": "${TC_WP_PUBLISHER_PROC}",
  "receiver_procs": "${TC_WP_RECEIVER_PROCS}",
  "gs_addr": "${TC_GS_ADDR}",
  "run_as_user": "${TC_RUN_AS_USER}",
  "daemon_config": "${TC_DAEMON_CONFIG}",
  "publisher_daemon_config": "${TC_WP_PUBLISHER_DAEMON_CONFIG}",
  "receiver_daemon_config": "${TC_WP_RECEIVER_DAEMON_CONFIG}",
  "single_host_functional_enable": ${TC_SINGLE_HOST_FUNCTIONAL_ENABLE},
  "single_host_daemon_config": "${TC_SINGLE_HOST_DAEMON_CONFIG}",
  "tp_auto_keep_last_one": ${TC_WP_TP_AUTO_KEEP_LAST_ONE},
  "start_version": ${TC_START_VERSION},
  "num_versions": ${TC_NUM_VERSIONS},
  "keep_last": ${TC_KEEP_LAST},
  "pre_publish_trim_margin": ${TC_WP_PRE_PUBLISH_TRIM_MARGIN},
  "publish_interval_s": ${TC_PUBLISH_INTERVAL_S},
  "receiver_timeout_s": ${TC_RECEIVER_TIMEOUT_S},
  "receiver_timeout_auto_adjust": ${TC_WP_RECEIVER_TIMEOUT_AUTO_ADJUST},
  "keep_last_auto_adjust": ${TC_WP_KEEP_LAST_AUTO_ADJUST},
  "max_publish_to_apply_auto_adjust": ${TC_WP_MAX_PUBLISH_TO_APPLY_AUTO_ADJUST},
  "progress_poll_s": ${TC_PROGRESS_POLL_S},
  "scale_receiver_counts": $(json_array_from_bash_array SCALE_CASE_COUNTS),
  "payload_mode": "${TC_WP_PAYLOAD_MODE}",
  "tp_world_size": ${TC_WP_TP_WORLD_SIZE},
  "tp_total_bytes": ${TC_WP_TP_TOTAL_BYTES},
  "tp_device_map_policy": "${TC_WP_TP_DEVICE_MAP_POLICY}",
  "max_concurrency": ${TC_WP_MAX_CONCURRENCY},
  "receiver_preflight_transient_overlap": ${TC_WP_RECEIVER_PREFLIGHT_TRANSIENT_OVERLAP},
  "p0_early_stop": ${TC_WP_P0_EARLY_STOP},
  "p0_early_stop_grace_s": ${TC_WP_P0_EARLY_STOP_GRACE_S},
  "throughput_sample_interval_s": ${TC_WP_THROUGHPUT_SAMPLE_INTERVAL_S},
  "throughput_max_samples": ${TC_WP_THROUGHPUT_MAX_SAMPLES},
  "scale_num_versions": ${TC_SCALE_NUM_VERSIONS},
  "scale_publish_interval_s": ${TC_SCALE_PUBLISH_INTERVAL_S},
  "long_run_enable": ${TC_LONG_RUN_ENABLE},
  "long_run_receiver_count": ${TC_LONG_RUN_RECEIVER_COUNT},
  "long_run_num_versions": ${TC_LONG_RUN_NUM_VERSIONS},
  "long_run_target_duration_s": ${TC_LONG_RUN_TARGET_DURATION_S},
  "long_run_publish_interval_s": "${TC_LONG_RUN_PUBLISH_INTERVAL_S}",
  "long_run_receiver_timeout_s": "${TC_LONG_RUN_RECEIVER_TIMEOUT_S}",
  "max_publish_to_apply_s": ${TC_MAX_PUBLISH_TO_APPLY_S},
  "allow_receiver_skips": ${TC_ALLOW_RECEIVER_SKIPS},
  "skip_manifest_file": "${SKIP_MANIFEST_FILE}",
  "executed_cases": $(json_array_from_bash_array EXECUTED_CASES)
}
EOF

echo "[wp-suite] done. artifacts: ${RUN_DIR}"
