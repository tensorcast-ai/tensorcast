#!/usr/bin/env bash
# Copyright (c) 2026, TensorCast Team.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

: "${TC_CASE_SCHEMA:?set TC_CASE_SCHEMA (path to chaos case schema json)}"

TC_OUT_DIR="${TC_OUT_DIR:-/tmp/tc_cross_20260222/results_chaos}"
TC_RUN_ID="${TC_RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
TC_CHAOS_SEED="${TC_CHAOS_SEED:-7}"
TC_REMOTE_TIMEOUT_SEC="${TC_REMOTE_TIMEOUT_SEC:-900}"

# Optional orchestratorctl workflow hooks (scriptized phase 5.2).
# Provide full commands when you want this script to own worker lifecycle.
TC_ORCHESTRATOR_PREDICT_CMD="${TC_ORCHESTRATOR_PREDICT_CMD:-}"
TC_ORCHESTRATOR_LAUNCH_CMD="${TC_ORCHESTRATOR_LAUNCH_CMD:-}"
TC_ORCHESTRATOR_CLEANUP_CMD="${TC_ORCHESTRATOR_CLEANUP_CMD:-}"
TC_ORCHESTRATOR_ENABLE_LAUNCH="${TC_ORCHESTRATOR_ENABLE_LAUNCH:-false}"
TC_ORCHESTRATOR_CHARGED_GROUP="${TC_ORCHESTRATOR_CHARGED_GROUP:-tensorcast-dev}"
TC_ORCHESTRATOR_GPU="${TC_ORCHESTRATOR_GPU:-1}"
TC_ORCHESTRATOR_CPU="${TC_ORCHESTRATOR_CPU:-4}"
TC_ORCHESTRATOR_MEMORY="${TC_ORCHESTRATOR_MEMORY:-106400}"
TC_ORCHESTRATOR_MAX_WAIT_DURATION="${TC_ORCHESTRATOR_MAX_WAIT_DURATION:-15m}"
TC_ORCHESTRATOR_POSITIVE_TAGS="${TC_ORCHESTRATOR_POSITIVE_TAGS:-}"
TC_ORCHESTRATOR_MOUNT="${TC_ORCHESTRATOR_MOUNT:-}"
TC_ORCHESTRATOR_PRIVATE_MACHINE="${TC_ORCHESTRATOR_PRIVATE_MACHINE:-}"
TC_ORCHESTRATOR_COMMENT="${TC_ORCHESTRATOR_COMMENT:-multihost-chaos-suite}"
TC_ORCHESTRATOR_REMOTE_SMOKE_CMD="${TC_ORCHESTRATOR_REMOTE_SMOKE_CMD:-echo START; hostname; nvidia-smi -L | head -n 4; echo DONE}"
TC_ORCHESTRATOR_CLEANUP_PROCESS_IDS="${TC_ORCHESTRATOR_CLEANUP_PROCESS_IDS:-}"
TC_GATE_MAX_RECOVER_TIME_SEC="${TC_GATE_MAX_RECOVER_TIME_SEC:-180}"
TC_GATE_REQUIRE_ALL_GET_COMPLETE="${TC_GATE_REQUIRE_ALL_GET_COMPLETE:-true}"
TC_GATE_REQUIRE_SOURCE_CARDINALITY="${TC_GATE_REQUIRE_SOURCE_CARDINALITY:-true}"
TC_GATE_REQUIRE_EXPECTED_FAILURE_PASS="${TC_GATE_REQUIRE_EXPECTED_FAILURE_PASS:-true}"
TC_GATE_REQUIRE_COMM_ERRORS_ZERO="${TC_GATE_REQUIRE_COMM_ERRORS_ZERO:-true}"
TC_GATE_ALLOW_FAILURE_EXIT="${TC_GATE_ALLOW_FAILURE_EXIT:-false}"

RUN_DIR="${TC_OUT_DIR}/${TC_RUN_ID}"
META_DIR="${RUN_DIR}/meta"
mkdir -p "${META_DIR}"

META_JSONL="${META_DIR}/orchestratorctl_steps.jsonl"
: > "${META_JSONL}"

json_escape() {
  printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'
}

record_step() {
  local step="$1"
  local cmd="$2"
  local rc="$3"
  local started_epoch="$4"
  local ended_epoch="$5"
  local log_file="$6"
  printf '{"step":"%s","cmd":"%s","returncode":%s,"started_epoch":%s,"ended_epoch":%s,"log_file":"%s"}\n' \
    "$(json_escape "${step}")" \
    "$(json_escape "${cmd}")" \
    "${rc}" \
    "${started_epoch}" \
    "${ended_epoch}" \
    "$(json_escape "${log_file}")" >> "${META_JSONL}"
}

run_step() {
  local step="$1"
  local cmd="$2"
  local required="$3"
  local log_file="${META_DIR}/${step}.log"
  local started_epoch
  local ended_epoch
  local rc=0
  started_epoch="$(date +%s.%N)"
  set +e
  bash -lc "${cmd}" >"${log_file}" 2>&1
  rc=$?
  set -e
  ended_epoch="$(date +%s.%N)"
  record_step "${step}" "${cmd}" "${rc}" "${started_epoch}" "${ended_epoch}" "${log_file}"
  if [[ "${required}" == "true" && "${rc}" -ne 0 ]]; then
    echo "[chaos-suite] step failed: ${step} rc=${rc}" >&2
    tail -n 80 "${log_file}" >&2 || true
    exit "${rc}"
  fi
}

run_optional_step() {
  local step="$1"
  local cmd="$2"
  if [[ -z "${cmd}" ]]; then
    return 0
  fi
  run_step "${step}" "${cmd}" false
}

run_cleanup_if_configured() {
  if [[ -n "${TC_ORCHESTRATOR_CLEANUP_PROCESS_IDS}" ]]; then
    local cleanup_cmd=""
    IFS=',' read -r -a pid_arr <<< "${TC_ORCHESTRATOR_CLEANUP_PROCESS_IDS}"
    for pid in "${pid_arr[@]}"; do
      pid="$(echo "${pid}" | xargs)"
      if [[ -z "${pid}" ]]; then
        continue
      fi
      cleanup_cmd+="orchestratorctl delete process ${pid} -n tensorcast || true; "
    done
    if [[ -n "${cleanup_cmd}" ]]; then
      run_step "cleanup_processes" "${cleanup_cmd}" false
    fi
  fi
  if [[ -n "${TC_ORCHESTRATOR_CLEANUP_CMD}" ]]; then
    run_step "cleanup" "${TC_ORCHESTRATOR_CLEANUP_CMD}" false
  fi
}

trap run_cleanup_if_configured EXIT

run_step "preflight_version" "orchestratorctl version" true
run_step "preflight_options" "orchestratorctl options" true

if [[ "${TC_ORCHESTRATOR_ENABLE_LAUNCH}" == "true" ]]; then
  ORCHESTRATOR_BASE_CMD="orchestratorctl launch --charged-group=${TC_ORCHESTRATOR_CHARGED_GROUP} --gpu ${TC_ORCHESTRATOR_GPU} --cpu ${TC_ORCHESTRATOR_CPU} --memory ${TC_ORCHESTRATOR_MEMORY} --max-wait-duration=${TC_ORCHESTRATOR_MAX_WAIT_DURATION}"
  if [[ -n "${TC_ORCHESTRATOR_MOUNT}" ]]; then
    ORCHESTRATOR_BASE_CMD+=" --mount='${TC_ORCHESTRATOR_MOUNT}'"
  fi
  if [[ -n "${TC_ORCHESTRATOR_PRIVATE_MACHINE}" ]]; then
    ORCHESTRATOR_BASE_CMD+=" --private-machine ${TC_ORCHESTRATOR_PRIVATE_MACHINE}"
  fi
  if [[ -n "${TC_ORCHESTRATOR_POSITIVE_TAGS}" ]]; then
    ORCHESTRATOR_BASE_CMD+=" --positive-tags ${TC_ORCHESTRATOR_POSITIVE_TAGS}"
  fi
  if [[ -z "${TC_ORCHESTRATOR_PREDICT_CMD}" ]]; then
    TC_ORCHESTRATOR_PREDICT_CMD="${ORCHESTRATOR_BASE_CMD} --predict-only"
  fi
  if [[ -z "${TC_ORCHESTRATOR_LAUNCH_CMD}" ]]; then
    TC_ORCHESTRATOR_LAUNCH_CMD="${ORCHESTRATOR_BASE_CMD} --comment '${TC_ORCHESTRATOR_COMMENT}' -- bash -lc '${TC_ORCHESTRATOR_REMOTE_SMOKE_CMD}'"
  fi
fi

run_optional_step "predict" "${TC_ORCHESTRATOR_PREDICT_CMD}"
run_optional_step "launch" "${TC_ORCHESTRATOR_LAUNCH_CMD}"

SUITE_META_JSON="${META_DIR}/suite_meta.json"
cat > "${SUITE_META_JSON}" <<EOF
{
  "run_id": "${TC_RUN_ID}",
  "case_schema": "${TC_CASE_SCHEMA}",
  "charged_group": "${TC_ORCHESTRATOR_CHARGED_GROUP}",
  "predict_cmd": "$(json_escape "${TC_ORCHESTRATOR_PREDICT_CMD}")",
  "launch_cmd": "$(json_escape "${TC_ORCHESTRATOR_LAUNCH_CMD}")",
  "cleanup_cmd": "$(json_escape "${TC_ORCHESTRATOR_CLEANUP_CMD}")",
  "cleanup_process_ids": "${TC_ORCHESTRATOR_CLEANUP_PROCESS_IDS}",
  "gate_max_recover_time_sec": "${TC_GATE_MAX_RECOVER_TIME_SEC}",
  "gate_require_all_get_complete": "${TC_GATE_REQUIRE_ALL_GET_COMPLETE}",
  "gate_require_source_cardinality": "${TC_GATE_REQUIRE_SOURCE_CARDINALITY}",
  "gate_require_expected_failure_pass": "${TC_GATE_REQUIRE_EXPECTED_FAILURE_PASS}",
  "gate_require_comm_errors_zero": "${TC_GATE_REQUIRE_COMM_ERRORS_ZERO}",
  "gate_allow_failure_exit": "${TC_GATE_ALLOW_FAILURE_EXIT}",
  "gate_review_json": "${RUN_DIR}/gate_review.json",
  "gate_review_md": "${RUN_DIR}/gate_review.md",
  "meta_steps_jsonl": "${META_JSONL}"
}
EOF

CHAOS_CMD=(
  "python"
  "examples/cross_host/cross_host_chaos_runner.py"
  "--case-schema" "${TC_CASE_SCHEMA}"
  "--out-dir" "${TC_OUT_DIR}"
  "--run-id" "${TC_RUN_ID}"
  "--chaos-seed" "${TC_CHAOS_SEED}"
  "--remote-timeout-sec" "${TC_REMOTE_TIMEOUT_SEC}"
)

echo "[chaos-suite] run_id=${TC_RUN_ID} schema=${TC_CASE_SCHEMA}"
echo "[chaos-suite] metadata=${META_JSONL}"
if [[ -f "${SUITE_META_JSON}" ]]; then
  echo "[chaos-suite] suite_meta=${SUITE_META_JSON}"
fi
CHAOS_RC=0
set +e
"${CHAOS_CMD[@]}"
CHAOS_RC=$?
set -e

GATE_CMD=(
  "python"
  "examples/cross_host/chaos_gate_review.py"
  "--run-dir" "${RUN_DIR}"
  "--max-recover-time-sec" "${TC_GATE_MAX_RECOVER_TIME_SEC}"
)
if [[ "${TC_GATE_REQUIRE_ALL_GET_COMPLETE}" != "true" ]]; then
  GATE_CMD+=("--no-require-all-get-complete")
fi
if [[ "${TC_GATE_REQUIRE_SOURCE_CARDINALITY}" != "true" ]]; then
  GATE_CMD+=("--no-require-source-cardinality")
fi
if [[ "${TC_GATE_REQUIRE_EXPECTED_FAILURE_PASS}" != "true" ]]; then
  GATE_CMD+=("--no-require-expected-failure-pass")
fi
if [[ "${TC_GATE_REQUIRE_COMM_ERRORS_ZERO}" != "true" ]]; then
  GATE_CMD+=("--no-require-comm-errors-zero")
fi
if [[ "${TC_GATE_ALLOW_FAILURE_EXIT}" == "true" ]]; then
  GATE_CMD+=("--allow-failure-exit")
fi

GATE_RC=0
set +e
"${GATE_CMD[@]}"
GATE_RC=$?
set -e

if [[ "${CHAOS_RC}" -ne 0 ]]; then
  echo "[chaos-suite] chaos run failed rc=${CHAOS_RC}" >&2
fi
if [[ "${GATE_RC}" -ne 0 ]]; then
  echo "[chaos-suite] gate review failed rc=${GATE_RC}" >&2
fi
if [[ "${CHAOS_RC}" -ne 0 || "${GATE_RC}" -ne 0 ]]; then
  exit 1
fi

echo "[chaos-suite] completed. output=${RUN_DIR}"
