#!/usr/bin/env bash

set -euo pipefail

namespace="${NAMESPACE:-shai-core}"
charged_group="${CHARGED_GROUP:-tensorcast_dev}"
positive_tags="${POSITIVE_TAGS:-H800,ib}"
worker_gpu_count="${WORKER_GPU_COUNT:-8}"
worker_cpu_count="${WORKER_CPU_COUNT:-32}"
worker_memory_mb="${WORKER_MEMORY_MB:-512000}"
max_wait_duration="${MAX_WAIT_DURATION:-20m}"
ready_poll_interval_sec="${READY_POLL_INTERVAL_SEC:-5}"
describe_poll_rounds="${DESCRIBE_POLL_ROUNDS:-2}"
scheduler_retry_backoff_sec="${SCHEDULER_RETRY_BACKOFF_SEC:-45}"
quota_grace_sec="${QUOTA_GRACE_SEC:-240}"
launch_retry_limit="${LAUNCH_RETRY_LIMIT:-3}"
pair_launch_retry_limit="${PAIR_LAUNCH_RETRY_LIMIT:-2}"
pair_topology_retry_limit="${PAIR_TOPOLOGY_RETRY_LIMIT:-3}"
pair_topology_retry_backoff_sec="${PAIR_TOPOLOGY_RETRY_BACKOFF_SEC:-30}"
repo_path="${REPO_PATH:-$(pwd)}"
transfer_gpu_count="${TRANSFER_GPU_COUNT:-8}"
min_common_hca_spare="${MIN_COMMON_HCA_SPARE:-0}"
transfer_port="${TRANSFER_PORT:-19099}"
transfer_chunk="${TRANSFER_CHUNK:-1}"
transfer_count="${TRANSFER_COUNT:-16777216}"
per_link_min_gbps="${PER_LINK_MIN_GBPS:-120}"
per_link_min_of_peak_ratio="${PER_LINK_MIN_OF_PEAK_RATIO:-0.75}"
enforce_bandwidth="${ENFORCE_BANDWIDTH:-1}"
client_timeout_sec="${CLIENT_TIMEOUT_SEC:-600}"
hca_selection_mode="${HCA_SELECTION_MODE:-first}"
ib_hca_override="${IB_HCA_OVERRIDE:-}"
keep_workers="${KEEP_WORKERS:-0}"
comment_prefix="${COMMENT_PREFIX:-topology-guided-routing-smoke}"
extra_launch_flags="${EXTRA_LAUNCH_FLAGS:-}"
max_transfer_attempts="${MAX_TRANSFER_ATTEMPTS:-3}"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_dir="${LOG_DIR:-/data/tensorcast/tests/topology_guided_routing_${timestamp}}"

declare -a extra_launch_args=()
if [[ -n "${extra_launch_flags}" ]]; then
  read -r -a extra_launch_args <<< "${extra_launch_flags}"
fi

if (( transfer_gpu_count < 8 )); then
  echo "TRANSFER_GPU_COUNT must be >= 8 to validate RDMA 8-rail affinity." >&2
  exit 2
fi
if (( transfer_chunk != 1 )); then
  echo "TRANSFER_CHUNK must be 1. Current verification logic expects one tensor per GPU key." >&2
  exit 2
fi
if (( worker_gpu_count < 8 )); then
  echo "WORKER_GPU_COUNT must be >= 8 because RDMA verification requires explicit 8-GPU launch." >&2
  exit 2
fi
if (( worker_gpu_count < transfer_gpu_count )); then
  echo "WORKER_GPU_COUNT(${worker_gpu_count}) must be >= TRANSFER_GPU_COUNT(${transfer_gpu_count})." >&2
  exit 2
fi
if (( client_timeout_sec <= 0 )); then
  echo "CLIENT_TIMEOUT_SEC must be > 0." >&2
  exit 2
fi
if (( ready_poll_interval_sec <= 0 )); then
  echo "READY_POLL_INTERVAL_SEC must be > 0." >&2
  exit 2
fi
if (( max_transfer_attempts <= 0 )); then
  echo "MAX_TRANSFER_ATTEMPTS must be > 0." >&2
  exit 2
fi
if (( describe_poll_rounds <= 0 )); then
  echo "DESCRIBE_POLL_ROUNDS must be > 0." >&2
  exit 2
fi
if (( scheduler_retry_backoff_sec <= 0 )); then
  echo "SCHEDULER_RETRY_BACKOFF_SEC must be > 0." >&2
  exit 2
fi
if (( quota_grace_sec <= 0 )); then
  echo "QUOTA_GRACE_SEC must be > 0." >&2
  exit 2
fi
if (( launch_retry_limit <= 0 )); then
  echo "LAUNCH_RETRY_LIMIT must be > 0." >&2
  exit 2
fi
if (( pair_launch_retry_limit <= 0 )); then
  echo "PAIR_LAUNCH_RETRY_LIMIT must be > 0." >&2
  exit 2
fi
if (( pair_topology_retry_limit <= 0 )); then
  echo "PAIR_TOPOLOGY_RETRY_LIMIT must be > 0." >&2
  exit 2
fi
if (( pair_topology_retry_backoff_sec <= 0 )); then
  echo "PAIR_TOPOLOGY_RETRY_BACKOFF_SEC must be > 0." >&2
  exit 2
fi
if (( min_common_hca_spare < 0 )); then
  echo "MIN_COMMON_HCA_SPARE must be >= 0." >&2
  exit 2
fi
if [[ "${enforce_bandwidth}" != "0" && "${enforce_bandwidth}" != "1" ]]; then
  echo "ENFORCE_BANDWIDTH must be 0 or 1." >&2
  exit 2
fi
if ! awk -v v="${per_link_min_gbps}" 'BEGIN { exit !(v + 0 > 0) }'; then
  echo "PER_LINK_MIN_GBPS must be > 0." >&2
  exit 2
fi
if ! awk -v v="${per_link_min_of_peak_ratio}" 'BEGIN { exit !(v + 0 > 0 && v + 0 <= 1) }'; then
  echo "PER_LINK_MIN_OF_PEAK_RATIO must be in (0, 1]." >&2
  exit 2
fi

if ! command -v brainctl >/dev/null 2>&1; then
  echo "brainctl is required but not found in PATH." >&2
  exit 2
fi
if ! command -v bazel >/dev/null 2>&1; then
  echo "bazel is required but not found in PATH." >&2
  exit 2
fi

run_as_user="$(id -un)"
if [[ "${run_as_user}" == "root" ]]; then
  echo "refuse to run remote workload as root" >&2
  exit 2
fi

server_worker_pid=""
client_worker_pid=""
wait_last_reason=""
wait_last_detail=""
launch_last_reason=""
launch_last_detail=""

duration_to_seconds() {
  local duration="$1"
  if [[ "${duration}" =~ ^([0-9]+)([smhd])$ ]]; then
    local value="${BASH_REMATCH[1]}"
    local unit="${BASH_REMATCH[2]}"
    case "${unit}" in
      s) echo "${value}" ;;
      m) echo $((value * 60)) ;;
      h) echo $((value * 3600)) ;;
      d) echo $((value * 86400)) ;;
      *)
        break
        ;;
    esac
    return 0
  fi
  echo "MAX_WAIT_DURATION must be in <number><unit> format, e.g. 20m/600s/1h." >&2
  return 1
}

if ! max_wait_sec="$(duration_to_seconds "${max_wait_duration}")"; then
  exit 2
fi
max_wait_rounds=$(((max_wait_sec + ready_poll_interval_sec - 1) / ready_poll_interval_sec))
if (( max_wait_rounds <= 0 )); then
  echo "derived max wait rounds must be > 0." >&2
  exit 2
fi

brain_exec_as_user() {
  local pid="$1"
  local remote_cmd="$2"
  local cmd_b64
  cmd_b64="$(printf '%s' "${remote_cmd}" | base64 | tr -d '\n')"

  brainctl exec "process/${pid}" -n "${namespace}" -- env \
    RUN_AS_USER="${run_as_user}" \
    REPO_PATH="${repo_path}" \
    REMOTE_CMD_B64="${cmd_b64}" \
    bash -lc '
set -euo pipefail
if [ "${RUN_AS_USER}" = "root" ]; then
  echo "refuse to run remote workload as root" >&2
  exit 2
fi
getent passwd "${RUN_AS_USER}" >/dev/null
remote_cmd="$(printf "%s" "${REMOTE_CMD_B64}" | base64 -d)"
inner="$(cat <<EOF
set -euo pipefail
cd "${REPO_PATH}"
source .venv/bin/activate
${remote_cmd}
EOF
)"
if [ "$(id -un)" = "${RUN_AS_USER}" ]; then
  bash -lc "${inner}"
else
  su - "${RUN_AS_USER}" -s /bin/bash -c "${inner}"
fi
'
}

print_process_status() {
  local pid="$1"
  brainctl get process "${pid}" -n "${namespace}"
  brainctl describe "process/${pid}" -n "${namespace}" | sed -n '1,80p'
}

delete_process_if_exists() {
  local pid="$1"
  if [[ -z "${pid}" ]]; then
    return 0
  fi
  brainctl delete process "${pid}" -n "${namespace}" >/dev/null 2>&1 || true
  brainctl get process "${pid}" -n "${namespace}" 2>&1 | grep -q "NotFound" || true
}

launch_worker_once() {
  local role="$1"
  local pair_attempt="$2"
  local submit_attempt="$3"
  local role_upper
  role_upper="$(printf '%s' "${role}" | tr '[:lower:]' '[:upper:]')"
  local launch_output
  local pid

  if ! launch_output="$(brainctl launch -d \
    --charged-group="${charged_group}" \
    --gpu "${worker_gpu_count}" \
    --cpu "${worker_cpu_count}" \
    --memory "${worker_memory_mb}" \
    --private-machine group \
    --positive-tags "${positive_tags}" \
    "${extra_launch_args[@]}" \
    --max-wait-duration="${max_wait_duration}" \
    --comment "${comment_prefix}-${role}-${timestamp}-p${pair_attempt}-s${submit_attempt}" \
    -- bash -lc "echo ${role_upper}_READY; hostname; nvidia-smi -L | sed -n '1,16p'; sleep 7200" 2>&1)"; then
    echo "[launch] brainctl launch failed for ${role}: ${launch_output}" >&2
    return 1
  fi

  pid="$(printf '%s\n' "${launch_output}" | awk '
    /^ws-[[:alnum:]-]+$/ { print $1; exit }
    /^ws-[[:alnum:]-]+[[:space:]]/ { print $1; exit }
  ')"
  if [[ -z "${pid}" ]]; then
    pid="$(printf '%s\n' "${launch_output}" | awk 'NF {last=$1} END {print last}')"
  fi
  if [[ -z "${pid}" ]]; then
    echo "[launch] failed to parse ${role} worker pid from output: ${launch_output}" >&2
    return 1
  fi
  echo "${pid}"
}

classify_scheduling_event() {
  local describe_text="$1"
  if printf '%s\n' "${describe_text}" | grep -qi "insufficient group quota: gpu"; then
    echo "quota"
    return 0
  fi
  if printf '%s\n' "${describe_text}" | grep -qi "Waiting for resources to be preempted"; then
    echo "preempt"
    return 0
  fi
  if printf '%s\n' "${describe_text}" | grep -Eqi "FailedScheduling|StartProcessActionFailed"; then
    echo "scheduler"
    return 0
  fi
  echo "none"
}

wait_for_process_ready() {
  local pid="$1"
  local max_rounds="${2:-120}"
  local interval_sec="${3:-5}"
  local round=0
  local preempt_noted=0
  local quota_first_round=0
  local quota_grace_rounds=$(((quota_grace_sec + interval_sec - 1) / interval_sec))
  local describe_text=""

  wait_last_reason=""
  wait_last_detail=""

  while (( round < max_rounds )); do
    round=$((round + 1))
    local row
    row="$(brainctl get process "${pid}" -n "${namespace}" --no-headers 2>/dev/null || true)"
    local ready status
    ready="$(printf '%s\n' "${row}" | awk '{print $4}')"
    status="$(printf '%s\n' "${row}" | awk '{print $5}')"
    echo "[wait] pid=${pid} round=${round} ready=${ready:-unknown} status=${status:-unknown}" >&2

    if [[ "${ready}" == "1/1" && "${status}" == "Running" ]]; then
      wait_last_reason="running"
      return 0
    fi

    if [[ "${status}" == "Failed" || "${status}" == "Error" || "${status}" == "Succeeded" ]]; then
      wait_last_reason="terminal"
      wait_last_detail="status=${status}"
      break
    fi

    if (( round == 1 || round % describe_poll_rounds == 0 )); then
      describe_text="$(brainctl describe "process/${pid}" -n "${namespace}" 2>&1 || true)"
      case "$(classify_scheduling_event "${describe_text}")" in
        quota)
          if (( quota_first_round == 0 )); then
            quota_first_round="${round}"
            echo "[scheduler] process ${pid} hit GPU quota limit; start grace window ${quota_grace_sec}s" >&2
          fi
          if (( round - quota_first_round >= quota_grace_rounds )); then
            wait_last_reason="quota_timeout"
            wait_last_detail="quota_grace_exhausted_after=${quota_grace_sec}s"
            break
          fi
          ;;
        preempt)
          if (( preempt_noted == 0 )); then
            echo "[scheduler] process ${pid} is waiting for preemption; continuing to wait" >&2
            preempt_noted=1
          fi
          ;;
        scheduler)
          wait_last_reason="scheduler_pending"
          wait_last_detail="failed scheduling/start action observed while pending"
          ;;
        none)
          ;;
      esac
    fi
    sleep "${interval_sec}"
  done

  if [[ -z "${wait_last_reason}" ]]; then
    wait_last_reason="timeout"
    wait_last_detail="max_rounds=${max_rounds} interval_sec=${interval_sec}"
  fi

  echo "process ${pid} did not become ready in time" >&2
  print_process_status "${pid}" >&2
  return 1
}

launch_worker_with_retries() {
  local role="$1"
  local pair_attempt="$2"
  local out_pid_name="${3:-}"
  if [[ -z "${out_pid_name}" ]]; then
    echo "launch_worker_with_retries requires output pid variable name." >&2
    return 2
  fi
  local -n out_pid_ref="${out_pid_name}"
  local submit_attempt=0
  local pid=""
  local sleep_time=0

  out_pid_ref=""
  launch_last_reason=""
  launch_last_detail=""

  while (( submit_attempt < launch_retry_limit )); do
    submit_attempt=$((submit_attempt + 1))
    echo "[launch] ${role} submit ${submit_attempt}/${launch_retry_limit} (pair attempt ${pair_attempt})" >&2
    if ! pid="$(launch_worker_once "${role}" "${pair_attempt}" "${submit_attempt}")"; then
      launch_last_reason="launch_submit_failed"
      launch_last_detail="brainctl launch failed before pid ready"
      if (( submit_attempt < launch_retry_limit )); then
        sleep_time=$((scheduler_retry_backoff_sec * submit_attempt))
        echo "[launch] ${role} submit failed; retry in ${sleep_time}s" >&2
        sleep "${sleep_time}"
        continue
      fi
      return 1
    fi

    echo "[launch] ${role} worker pid: ${pid}" >&2
    print_process_status "${pid}" >&2 || true
    if wait_for_process_ready "${pid}" "${max_wait_rounds}" "${ready_poll_interval_sec}"; then
      out_pid_ref="${pid}"
      return 0
    fi

    launch_last_reason="${wait_last_reason}"
    launch_last_detail="${wait_last_detail}"
    echo "[launch] ${role} worker ${pid} not ready: reason=${launch_last_reason} detail=${launch_last_detail}" >&2
    delete_process_if_exists "${pid}"
    pid=""

    if (( submit_attempt < launch_retry_limit )); then
      sleep_time=$((scheduler_retry_backoff_sec * submit_attempt))
      echo "[launch] ${role} retry after cleanup; backoff=${sleep_time}s" >&2
      sleep "${sleep_time}"
    fi
  done
  return 1
}

launch_worker_pair() {
  local pair_attempt=0
  local backoff_sec=0

  while (( pair_attempt < pair_launch_retry_limit )); do
    pair_attempt=$((pair_attempt + 1))
    echo "[launch] pair attempt ${pair_attempt}/${pair_launch_retry_limit}" >&2

    server_worker_pid=""
    client_worker_pid=""

    if ! launch_worker_with_retries "server" "${pair_attempt}" server_worker_pid; then
      echo "[launch] server worker launch failed: reason=${launch_last_reason} detail=${launch_last_detail}" >&2
      if (( pair_attempt < pair_launch_retry_limit )); then
        backoff_sec=$((scheduler_retry_backoff_sec * pair_attempt))
        echo "[launch] retry full pair in ${backoff_sec}s" >&2
        sleep "${backoff_sec}"
        continue
      fi
      return 1
    fi

    if ! launch_worker_with_retries "client" "${pair_attempt}" client_worker_pid; then
      echo "[launch] client worker launch failed: reason=${launch_last_reason} detail=${launch_last_detail}" >&2
      delete_process_if_exists "${server_worker_pid}"
      server_worker_pid=""
      if (( pair_attempt < pair_launch_retry_limit )); then
        backoff_sec=$((scheduler_retry_backoff_sec * pair_attempt))
        echo "[launch] retry full pair in ${backoff_sec}s" >&2
        sleep "${backoff_sec}"
        continue
      fi
      return 1
    fi

    echo "[launch] both workers ready: server=${server_worker_pid} client=${client_worker_pid}" >&2
    return 0
  done

  return 1
}

cleanup_workers() {
  if [[ "${keep_workers}" == "1" ]]; then
    echo "KEEP_WORKERS=1; skip worker deletion."
    if [[ -n "${server_worker_pid}" ]]; then
      echo "server worker: ${server_worker_pid}"
    fi
    if [[ -n "${client_worker_pid}" ]]; then
      echo "client worker: ${client_worker_pid}"
    fi
    return
  fi

  if [[ -n "${server_worker_pid}" ]]; then
    brain_exec_as_user "${server_worker_pid}" "
if [ -f \"${log_dir}/server.pid\" ]; then
  kill \$(cat \"${log_dir}/server.pid\") >/dev/null 2>&1 || true
fi
" || true
  fi

  for pid in "${server_worker_pid}" "${client_worker_pid}"; do
    if [[ -z "${pid}" ]]; then
      continue
    fi
    brainctl delete process "${pid}" -n "${namespace}" || true
    brainctl get process "${pid}" -n "${namespace}" 2>&1 | grep -q "NotFound" || true
  done
}

on_exit() {
  local status=$?
  set +e
  cleanup_workers
  return "${status}"
}
trap on_exit EXIT

trim_number() {
  printf '%s' "$1" | tr -cd '0-9'
}

emit_rdma_environment_hints() {
  cat >&2 <<EOF
RDMA verbs preflight failed.
This smoke requires:
  1) /dev/infiniband to be mounted in the worker
  2) ibv_devinfo -l to report mlx5 HCAs
  3) at least ${transfer_gpu_count} common HCAs between server/client workers

Current launch knobs:
  POSITIVE_TAGS=${positive_tags}
  EXTRA_LAUNCH_FLAGS=${extra_launch_flags:-<empty>}

If your cluster needs explicit RDMA passthrough/device plugin flags, pass them via:
  EXTRA_LAUNCH_FLAGS='...'
EOF
}

collect_verbs_hcas() {
  local pid="$1"
  local role="$2"
  local output

  if ! output="$(brain_exec_as_user "${pid}" "
if [ ! -d /dev/infiniband ]; then
  echo '[rdma-preflight] missing /dev/infiniband'
  ls -ld /dev/infiniband 2>/dev/null || true
  exit 20
fi
if ! command -v ibv_devinfo >/dev/null 2>&1; then
  echo '[rdma-preflight] ibv_devinfo not found in PATH'
  exit 21
fi
echo '[rdma-preflight] /dev/infiniband is present'
echo '[rdma-preflight] ibv_devinfo -l:'
ibv_devinfo -l || true
verbs_hcas=\$(ibv_devinfo -l 2>/dev/null | awk '{
  for (i = 1; i <= NF; ++i) {
    if (\$i ~ /^mlx5_[0-9]+$/) {
      print \$i;
    }
  }
}' | sort -Vu || true)
if [ -z \"\${verbs_hcas}\" ]; then
  echo '[rdma-preflight] no verbs-visible mlx5 HCAs'
  ls -1 /sys/class/infiniband 2>/dev/null | sort -V || true
  exit 22
fi
echo \"\${verbs_hcas}\"
" 2>&1)"; then
    echo "[rdma-preflight] ${role} worker ${pid} failed:" >&2
    printf '%s\n' "${output}" >&2
    emit_rdma_environment_hints
    exit 1
  fi

  printf '%s\n' "${output}" | awk '/^mlx5/{print}' | sort -Vu
}

collect_engine_usable_hcas() {
  local pid="$1"
  local role="$2"
  local hca_csv="$3"
  local output

  if ! output="$(brain_exec_as_user "${pid}" "
if [ ! -x ./bazel-bin/core/communicator/gpu_ce_test_binary ]; then
  bazel build //core/communicator:gpu_ce_test_binary \
    --noshow_progress --noshow_loading_progress \
    --ui_event_filters=warning,error
fi
export TENSORCAST_IB_HCA=\"${hca_csv}\"
probe_port=\$((29000 + RANDOM % 1000))
set +e
probe_output=\$(timeout --signal=TERM 12 \
  ./bazel-bin/core/communicator/gpu_ce_test_binary \
    --actor server \
    --port \"\${probe_port}\" \
    --gpu 1 \
    --chunk 1 \
    --count 1024 \
    --rdma 2>&1)
probe_status=\$?
set -e
printf '%s\n' \"\${probe_output}\"
if [ \"\${probe_status}\" -ne 0 ] && [ \"\${probe_status}\" -ne 124 ] && [ \"\${probe_status}\" -ne 143 ]; then
  echo \"[rdma-probe] ${role} probe exited with status \${probe_status}\" >&2
fi
" 2>&1)"; then
    echo "[rdma-probe] ${role} worker ${pid} probe failed:" >&2
    printf '%s\n' "${output}" >&2
    exit 1
  fi

  printf '%s\n' "${output}" | sed -n \
    -e 's/.*Dev: \(mlx5_[0-9]\+\) added.*/\1/p' \
    -e 's/.*RDMA candidate accepted: dev=\(mlx5_[0-9]\+\).*/\1/p' \
    | sort -Vu
}

collect_no_regmr_read_cost_us() {
  local pid="$1"
  local log_path="$2"
  brain_exec_as_user "${pid}" "
sed -n 's/^no regmr result: key=gpu-ce-test-tensor-\\([0-9]\\+\\)-0, status=0, .*rdma_read=\\([0-9]\\+\\).*/\\1 \\2/p' \"${log_path}\" \
  | awk '\$2 + 0 > 0 { print \$1, \$2 }' \
  | sort -n -k1,1
"
}

collect_latest_gpu_nic_map() {
  local pid="$1"
  local log_path="$2"
  brain_exec_as_user "${pid}" "
grep 'read tensor:' \"${log_path}\" \
  | sed -n 's/.*key=gpu-ce-test-tensor-\\([0-9]\\+\\)-0 .*net_dev=\\([^ ,]*\\).*/\\1 \\2/p' \
  | awk '\$2 != \"none\" { nic[\$1] = \$2 } END { for (gpu in nic) { print gpu, nic[gpu] } }' \
  | sort -n -k1,1
"
}

echo "[preflight] brainctl version"
brainctl version
echo "[preflight] brainctl options"
brainctl options
echo "[preflight] my processes in ${namespace}"
brainctl get process -n "${namespace}" --no-headers \
  | awk -v creator="${run_as_user}" '$3==creator {print $1, $2, $5, $7}' \
  | sed -n '1,20p'

echo "[build] bazel build //core/communicator:gpu_ce_test_binary"
bazel build //core/communicator:gpu_ce_test_binary \
  --noshow_progress --noshow_loading_progress \
  --ui_event_filters=warning,error

echo "[predict] capacity check"
brainctl launch \
  --charged-group="${charged_group}" \
  --gpu "${worker_gpu_count}" \
  --cpu "${worker_cpu_count}" \
  --memory "${worker_memory_mb}" \
  --private-machine group \
  --positive-tags "${positive_tags}" \
  --predict-only

required_common_hcas=$(( transfer_gpu_count + min_common_hca_spare ))
topology_pair_attempt=0
declare -a server_hcas=()
declare -a client_hcas=()
declare -a server_engine_hcas=()
declare -a client_engine_hcas=()
declare -a common_hcas=()

while (( topology_pair_attempt < pair_topology_retry_limit )); do
  topology_pair_attempt=$((topology_pair_attempt + 1))
  echo "[launch] server+client workers with scheduler-aware retries (topology attempt ${topology_pair_attempt}/${pair_topology_retry_limit})"
  if ! launch_worker_pair; then
    echo "failed to launch 2x${worker_gpu_count} workers under scheduler constraints." >&2
    exit 1
  fi

  echo "[remote] validate user + workspace"
  for pid in "${server_worker_pid}" "${client_worker_pid}"; do
    brainctl exec "process/${pid}" -n "${namespace}" -- bash -lc "getent passwd ${run_as_user} >/dev/null"
    brain_exec_as_user "${pid}" "
pwd
test -d \"${repo_path}\"
if [ ! -x ./bazel-bin/core/communicator/gpu_ce_test_binary ]; then
  bazel build //core/communicator:gpu_ce_test_binary \
    --noshow_progress --noshow_loading_progress \
    --ui_event_filters=warning,error
fi
"
  done

  echo "[remote] RDMA verbs preflight + discover usable RNICs"
  server_hcas_raw="$(collect_verbs_hcas "${server_worker_pid}" "server")"
  client_hcas_raw="$(collect_verbs_hcas "${client_worker_pid}" "client")"

  mapfile -t server_hcas < <(printf '%s\n' "${server_hcas_raw}" | awk '/^mlx5/{print}' | sort -Vu)
  mapfile -t client_hcas < <(printf '%s\n' "${client_hcas_raw}" | awk '/^mlx5/{print}' | sort -Vu)
  if (( ${#server_hcas[@]} == 0 || ${#client_hcas[@]} == 0 )); then
    echo "failed to discover verbs-visible RNIC devices on remote workers" >&2
    echo "server RNICs: ${server_hcas_raw}" >&2
    echo "client RNICs: ${client_hcas_raw}" >&2
    emit_rdma_environment_hints
    exit 1
  fi

  server_verbs_csv="$(IFS=,; echo "${server_hcas[*]}")"
  client_verbs_csv="$(IFS=,; echo "${client_hcas[*]}")"

  echo "[remote] RDMA communicator probe (engine-usable RNICs)"
  server_engine_hcas_raw="$(collect_engine_usable_hcas "${server_worker_pid}" "server" "${server_verbs_csv}")"
  client_engine_hcas_raw="$(collect_engine_usable_hcas "${client_worker_pid}" "client" "${client_verbs_csv}")"
  mapfile -t server_engine_hcas < <(printf '%s\n' "${server_engine_hcas_raw}" | awk '/^mlx5/{print}' | sort -Vu)
  mapfile -t client_engine_hcas < <(printf '%s\n' "${client_engine_hcas_raw}" | awk '/^mlx5/{print}' | sort -Vu)
  if (( ${#server_engine_hcas[@]} == 0 || ${#client_engine_hcas[@]} == 0 )); then
    echo "failed to discover communicator-usable RNIC devices on remote workers" >&2
    echo "server communicator RNICs: ${server_engine_hcas_raw}" >&2
    echo "client communicator RNICs: ${client_engine_hcas_raw}" >&2
    exit 1
  fi

  mapfile -t common_hcas < <(
    comm -12 \
      <(printf '%s\n' "${server_engine_hcas[@]}" | LC_ALL=C sort -u) \
      <(printf '%s\n' "${client_engine_hcas[@]}" | LC_ALL=C sort -u) \
      | sort -V
  )

  echo "[remote] server communicator RNICs (${#server_engine_hcas[@]}): ${server_engine_hcas[*]}"
  echo "[remote] client communicator RNICs (${#client_engine_hcas[@]}): ${client_engine_hcas[*]}"
  echo "[remote] common communicator RNICs (${#common_hcas[@]}): ${common_hcas[*]}"
  echo "[remote] spare common RNICs beyond TRANSFER_GPU_COUNT(${transfer_gpu_count}): $(( ${#common_hcas[@]} - transfer_gpu_count ))"

  if (( ${#common_hcas[@]} < transfer_gpu_count )); then
    echo "need at least ${transfer_gpu_count} common communicator-usable RNICs, got ${#common_hcas[@]}" >&2
    echo "server communicator RNICs: ${server_engine_hcas[*]}" >&2
    echo "client communicator RNICs: ${client_engine_hcas[*]}" >&2
    exit 1
  fi

  if (( ${#common_hcas[@]} >= required_common_hcas )); then
    break
  fi

  echo "[topology] common RNIC count ${#common_hcas[@]} does not satisfy required ${required_common_hcas} (TRANSFER_GPU_COUNT=${transfer_gpu_count} + MIN_COMMON_HCA_SPARE=${min_common_hca_spare})." >&2
  if (( topology_pair_attempt >= pair_topology_retry_limit )); then
    echo "[topology] exhausted pair reselection attempts while looking for spare common RNICs." >&2
    exit 1
  fi

  echo "[topology] recycle worker pair and retry topology selection after ${pair_topology_retry_backoff_sec}s." >&2
  delete_process_if_exists "${server_worker_pid}"
  delete_process_if_exists "${client_worker_pid}"
  server_worker_pid=""
  client_worker_pid=""
  sleep "${pair_topology_retry_backoff_sec}"
done

declare -A excluded_hcas=()
declare -a excluded_hca_order=()
declare -A observed_failed_hcas=()
declare -a selected_hcas=()
hca_failover_enabled=0

select_hca_subset() {
  local -n out_hcas="$1"
  local -a filtered_hcas=()
  local dev

  for dev in "${common_hcas[@]}"; do
    if [[ -n "${excluded_hcas[${dev}]:-}" ]]; then
      continue
    fi
    filtered_hcas+=("${dev}")
  done

  if (( ${#filtered_hcas[@]} < transfer_gpu_count )); then
    return 1
  fi

  case "${hca_selection_mode}" in
    first)
      out_hcas=("${filtered_hcas[@]:0:${transfer_gpu_count}}")
      ;;
    last)
      local start_index=$(( ${#filtered_hcas[@]} - transfer_gpu_count ))
      out_hcas=("${filtered_hcas[@]:${start_index}:${transfer_gpu_count}}")
      ;;
    *)
      echo "HCA_SELECTION_MODE must be one of: first, last" >&2
      exit 1
      ;;
  esac
}

rotate_selected_hca_with_spare() {
  local reason="${1:-}"
  local -A selected_hca_set=()
  local -a spare_candidates=()
  local -A bw_by_gpu=()
  local -A nic_usage=()
  local -A nic_min_bw=()
  local dev gpu nic bw usage min_bw
  local remove_dev=""
  local remove_usage=-1
  local remove_min_bw="999999"
  local add_dev=""
  local idx

  for dev in "${selected_hcas[@]}"; do
    selected_hca_set["${dev}"]=1
  done
  for dev in "${common_hcas[@]}"; do
    if [[ -n "${selected_hca_set[${dev}]:-}" ]]; then
      continue
    fi
    spare_candidates+=("${dev}")
  done
  if (( ${#spare_candidates[@]} == 0 )); then
    return 1
  fi

  while read -r gpu bw; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if ! awk -v v="${bw}" 'BEGIN { exit !(v + 0 >= 0) }'; then
      continue
    fi
    bw_by_gpu["${gpu}"]="${bw}"
  done <<< "${bandwidth_table}"

  while read -r gpu nic; do
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
      continue
    fi
    if [[ ! "${nic}" =~ ^mlx5_[0-9]+$ ]]; then
      continue
    fi
    nic_usage["${nic}"]=$(( ${nic_usage["${nic}"]:-0} + 1 ))
    bw="${bw_by_gpu["${gpu}"]:-0}"
    if [[ -z "${nic_min_bw["${nic}"]:-}" ]] || \
      awk -v lhs="${bw}" -v rhs="${nic_min_bw["${nic}"]}" 'BEGIN { exit !(lhs + 0 < rhs + 0) }'; then
      nic_min_bw["${nic}"]="${bw}"
    fi
  done <<< "${gpu_nic_map_raw}"

  for dev in "${selected_hcas[@]}"; do
    usage="${nic_usage["${dev}"]:-0}"
    min_bw="${nic_min_bw["${dev}"]:-999999}"
    if (( usage > remove_usage )); then
      remove_dev="${dev}"
      remove_usage="${usage}"
      remove_min_bw="${min_bw}"
      continue
    fi
    if (( usage == remove_usage )) && \
      awk -v lhs="${min_bw}" -v rhs="${remove_min_bw}" 'BEGIN { exit !(lhs + 0 < rhs + 0) }'; then
      remove_dev="${dev}"
      remove_min_bw="${min_bw}"
    fi
  done

  if [[ -z "${remove_dev}" ]]; then
    return 1
  fi
  if (( remove_usage <= 1 )) && [[ "${reason}" != "bandwidth_imbalance" ]]; then
    return 1
  fi

  add_dev="${spare_candidates[0]}"
  for idx in "${!selected_hcas[@]}"; do
    if [[ "${selected_hcas[${idx}]}" != "${remove_dev}" ]]; then
      continue
    fi
    selected_hcas[${idx}]="${add_dev}"
    break
  done

  echo "[retry] rotate selected RNIC via spare: remove=${remove_dev}(usage=${remove_usage},min_gbps=${remove_min_bw}) add=${add_dev}"
  echo "[retry] preserve existing working lanes; patch only suspected weak lane (no full reset)"
  return 0
}

if [[ -n "${ib_hca_override}" ]]; then
  mapfile -t selected_hcas < <(printf '%s' "${ib_hca_override}" | tr ',' '\n' | awk '/^mlx5_[0-9]+$/{print}' | sort -Vu)
  if (( ${#selected_hcas[@]} != transfer_gpu_count )); then
    echo "IB_HCA_OVERRIDE must contain exactly ${transfer_gpu_count} mlx5_* entries." >&2
    exit 1
  fi
  for dev in "${selected_hcas[@]}"; do
    if ! printf '%s\n' "${common_hcas[@]}" | grep -qx "${dev}"; then
      echo "IB_HCA_OVERRIDE contains non-common HCA: ${dev}" >&2
      echo "common RNICs: ${common_hcas[*]}" >&2
      exit 1
    fi
  done
else
  hca_failover_enabled=1
  if ! select_hca_subset selected_hcas; then
    echo "failed to select ${transfer_gpu_count} RNICs from common set." >&2
    exit 1
  fi
fi
ib_hca_csv="$(IFS=,; echo "${selected_hcas[*]}")"
echo "[remote] selected RNICs: ${ib_hca_csv}"

server_ip_raw="$(brain_exec_as_user "${server_worker_pid}" "hostname -I | awk '{for (i = 1; i <= NF; ++i) if (\$i ~ /^[0-9.]+$/) {print \$i; exit}}'")"
server_ip="$(printf '%s\n' "${server_ip_raw}" | awk '/^[0-9]+\./{print; exit}')"
if [[ -z "${server_ip}" ]]; then
  echo "failed to resolve server IP, raw output: ${server_ip_raw}" >&2
  exit 1
fi
echo "[remote] server ip: ${server_ip}"

min_unique_nics=$(( transfer_gpu_count - 1 ))
if (( min_unique_nics < 1 )); then
  min_unique_nics=1
fi

transfer_passed=0
final_attempt=0
final_server_log=""
final_client_log=""

for (( attempt = 1; attempt <= max_transfer_attempts; ++attempt )); do
  attempt_suffix=""
  if (( max_transfer_attempts > 1 )); then
    attempt_suffix="_attempt${attempt}"
  fi
  server_log_path="${log_dir}/server${attempt_suffix}.log"
  client_log_path="${log_dir}/client${attempt_suffix}.log"

  ib_hca_csv="$(IFS=,; echo "${selected_hcas[*]}")"
  echo "[remote] transfer attempt ${attempt}/${max_transfer_attempts} with RNICs: ${ib_hca_csv}"

  echo "[remote] start server transfer (attempt ${attempt})"
  brain_exec_as_user "${server_worker_pid}" "
mkdir -p \"${log_dir}\"
if [ -f \"${log_dir}/server.pid\" ]; then
  kill \$(cat \"${log_dir}/server.pid\") >/dev/null 2>&1 || true
  rm -f \"${log_dir}/server.pid\"
fi
export TENSORCAST_IB_HCA=\"${ib_hca_csv}\"
nohup ./bazel-bin/core/communicator/gpu_ce_test_binary \
  --actor server \
  --port ${transfer_port} \
  --gpu ${transfer_gpu_count} \
  --chunk ${transfer_chunk} \
  --count ${transfer_count} \
  --rdma \
  > \"${server_log_path}\" 2>&1 &
echo \$! > \"${log_dir}/server.pid\"
sleep 3
ps -p \$(cat \"${log_dir}/server.pid\") -o pid,cmd
"

  echo "[remote] run client transfer (attempt ${attempt})"
  client_command_ok=1
  if ! brain_exec_as_user "${client_worker_pid}" "
mkdir -p \"${log_dir}\"
export TENSORCAST_IB_HCA=\"${ib_hca_csv}\"
timeout --signal=TERM ${client_timeout_sec} \
  ./bazel-bin/core/communicator/gpu_ce_test_binary \
  --actor client \
  --ip ${server_ip} \
  --port ${transfer_port} \
  --gpu ${transfer_gpu_count} \
  --chunk ${transfer_chunk} \
  --count ${transfer_count} \
  --rdma \
  > \"${client_log_path}\" 2>&1
"; then
    client_command_ok=0
  fi

  brain_exec_as_user "${server_worker_pid}" "
if [ -f \"${log_dir}/server.pid\" ]; then
  kill \$(cat \"${log_dir}/server.pid\") >/dev/null 2>&1 || true
  rm -f \"${log_dir}/server.pid\"
fi
" || true

  echo "[verify] collect metrics (attempt ${attempt})"
  with_regmr_ok_raw="$(brain_exec_as_user "${client_worker_pid}" "grep -Ec '^with regmr result: key=gpu-ce-test-tensor-[0-9]+-0, status=0' \"${client_log_path}\" || true")"
  no_regmr_ok_raw="$(brain_exec_as_user "${client_worker_pid}" "grep -Ec '^no regmr result: key=gpu-ce-test-tensor-[0-9]+-0, status=0' \"${client_log_path}\" || true")"
  read_path_pairs_raw="$(brain_exec_as_user "${client_worker_pid}" "grep 'read tensor:' \"${client_log_path}\" | sed -n 's/.*key=gpu-ce-test-tensor-\\([0-9]\\+\\)-0 .*net_dev=\\([^ ,]*\\).*/\\1->\\2/p' | grep -v -- '->none$' | sort -u | wc -l || true")"
  unique_read_nics_raw="$(brain_exec_as_user "${client_worker_pid}" "grep 'read tensor:' \"${client_log_path}\" | sed -n 's/.*net_dev=\\([^ ,]*\\).*/\\1/p' | grep -v '^none$' | sort -u | wc -l || true")"
  client_ready_pairs_raw="$(brain_exec_as_user "${client_worker_pid}" "grep '\\[rdma_handshake\\].* -> ready' \"${client_log_path}\" | sed -n 's/.*dev=\\([^ ]*\\) peer=\\([^ ]*\\).*/\\1->\\2/p' | sort -u | wc -l || true")"
  server_connect_pairs_raw="$(brain_exec_as_user "${server_worker_pid}" "grep 'recv rdma connect' \"${server_log_path}\" | sed -n 's/.*net_dev=\\([^ ]*\\).*/\\1/p' | sort -u | wc -l || true")"
  no_regmr_read_costs_raw="$(collect_no_regmr_read_cost_us "${client_worker_pid}" "${client_log_path}" || true)"
  gpu_nic_map_raw="$(collect_latest_gpu_nic_map "${client_worker_pid}" "${client_log_path}" || true)"

  with_regmr_ok="$(trim_number "${with_regmr_ok_raw}")"
  no_regmr_ok="$(trim_number "${no_regmr_ok_raw}")"
  read_path_pairs="$(trim_number "${read_path_pairs_raw}")"
  unique_read_nics="$(trim_number "${unique_read_nics_raw}")"
  client_ready_pairs="$(trim_number "${client_ready_pairs_raw}")"
  server_connect_pairs="$(trim_number "${server_connect_pairs_raw}")"
  bandwidth_table="$(
    printf '%s\n' "${no_regmr_read_costs_raw}" \
      | awk -v bytes="${transfer_count}" '
          NF == 2 && $2 + 0 > 0 {
            gbps = (bytes * 8.0) / $2 / 1000.0;
            printf "%d %.2f\n", $1, gbps;
          }
        ' \
      | sort -n -k1,1
  )"
  bandwidth_count="$(printf '%s\n' "${bandwidth_table}" | awk 'NF == 2 { c += 1 } END { print c + 0 }')"
  bandwidth_min="$(printf '%s\n' "${bandwidth_table}" | awk 'NF == 2 { if (c == 0 || $2 < min) { min = $2 } c += 1 } END { if (c == 0) { print 0 } else { printf "%.2f", min } }')"
  bandwidth_max="$(printf '%s\n' "${bandwidth_table}" | awk 'NF == 2 { if (c == 0 || $2 > max) { max = $2 } c += 1 } END { if (c == 0) { print 0 } else { printf "%.2f", max } }')"
  bandwidth_avg="$(printf '%s\n' "${bandwidth_table}" | awk 'NF == 2 { sum += $2; c += 1 } END { if (c == 0) { print 0 } else { printf "%.2f", sum / c } }')"
  bandwidth_floor_by_peak="$(awk -v max="${bandwidth_max}" -v ratio="${per_link_min_of_peak_ratio}" 'BEGIN { printf "%.2f", max * ratio }')"

  echo "with_regmr_ok=${with_regmr_ok}"
  echo "no_regmr_ok=${no_regmr_ok}"
  echo "read_path_pairs=${read_path_pairs}"
  echo "unique_read_nics=${unique_read_nics}"
  echo "client_ready_pairs=${client_ready_pairs}"
  echo "server_connect_pairs=${server_connect_pairs}"
  if [[ "${enforce_bandwidth}" == "1" ]]; then
    echo "bandwidth_count=${bandwidth_count}"
    echo "bandwidth_min_gbps=${bandwidth_min}"
    echo "bandwidth_max_gbps=${bandwidth_max}"
    echo "bandwidth_avg_gbps=${bandwidth_avg}"
    echo "bandwidth_floor_by_peak_gbps=${bandwidth_floor_by_peak}"
    if [[ -n "${bandwidth_table}" ]]; then
      echo "[verify] per-gpu no-regmr bandwidth (Gbps)"
      awk '
        FNR == NR {
          if (NF == 2) {
            nic[$1] = $2;
          }
          next;
        }
        NF == 2 {
          gpu = $1;
          gbps = $2;
          nic_name = (gpu in nic) ? nic[gpu] : "unknown";
          printf "  gpu=%s nic=%s gbps=%s\n", gpu, nic_name, gbps;
        }
      ' <(printf '%s\n' "${gpu_nic_map_raw}") <(printf '%s\n' "${bandwidth_table}")
    fi
  fi

  declare -a failure_reasons=()
  if (( client_command_ok == 0 )); then
    failure_reasons+=("client transfer command failed or timed out")
  fi
  if [[ "${with_regmr_ok}" != "${transfer_gpu_count}" ]]; then
    failure_reasons+=("unexpected with-regmr success count: ${with_regmr_ok}")
  fi
  if [[ "${no_regmr_ok}" != "${transfer_gpu_count}" ]]; then
    failure_reasons+=("unexpected no-regmr success count: ${no_regmr_ok}")
  fi
  if [[ "${read_path_pairs}" != "${transfer_gpu_count}" ]]; then
    failure_reasons+=("unexpected read-path pair coverage: ${read_path_pairs}")
  fi
  if (( unique_read_nics < min_unique_nics )); then
    failure_reasons+=("unexpected unique read NIC count: ${unique_read_nics} (min=${min_unique_nics})")
  fi
  if (( client_ready_pairs < min_unique_nics )); then
    failure_reasons+=("client handshake ready pairs too low: ${client_ready_pairs} (min=${min_unique_nics})")
  fi
  if (( server_connect_pairs < min_unique_nics )); then
    failure_reasons+=("server accepted RDMA connect pairs too low: ${server_connect_pairs} (min=${min_unique_nics})")
  fi
  if [[ "${enforce_bandwidth}" == "1" ]]; then
    if [[ "${bandwidth_count}" != "${transfer_gpu_count}" ]]; then
      failure_reasons+=("unexpected no-regmr bandwidth sample count: ${bandwidth_count}")
    fi
    if ! awk -v min_bw="${bandwidth_min}" -v threshold="${per_link_min_gbps}" 'BEGIN { exit !(min_bw + 0 >= threshold + 0) }'; then
      failure_reasons+=(
        "per-link bandwidth below absolute threshold: min=${bandwidth_min}Gbps threshold=${per_link_min_gbps}Gbps")
    fi
    if ! awk -v min_bw="${bandwidth_min}" -v floor_bw="${bandwidth_floor_by_peak}" 'BEGIN { exit !(min_bw + 0 >= floor_bw + 0) }'; then
      failure_reasons+=(
        "per-link bandwidth imbalance: min=${bandwidth_min}Gbps floor_by_peak=${bandwidth_floor_by_peak}Gbps ratio=${per_link_min_of_peak_ratio}")
    fi
  fi

  final_attempt="${attempt}"
  final_server_log="${server_log_path}"
  final_client_log="${client_log_path}"

  if (( ${#failure_reasons[@]} == 0 )); then
    transfer_passed=1
    break
  fi

  echo "[verify] transfer attempt ${attempt} failed" >&2
  for reason in "${failure_reasons[@]}"; do
    echo "  - ${reason}" >&2
  done

  brain_exec_as_user "${client_worker_pid}" "tail -n 120 \"${client_log_path}\"" || true
  brain_exec_as_user "${server_worker_pid}" "tail -n 120 \"${server_log_path}\"" || true

  if (( attempt == max_transfer_attempts )); then
    break
  fi

  if (( hca_failover_enabled == 0 )); then
    echo "[retry] IB_HCA_OVERRIDE is set; keeping current RNIC selection for next attempt." >&2
    continue
  fi

  client_failed_hcas_raw="$(brain_exec_as_user "${client_worker_pid}" "
(
  grep '\\[rdma_handshake\\] transport connect failed:' \"${client_log_path}\" \
    | sed -n 's/.*local_dev=\\([^ ]*\\).*/\\1/p' || true
  grep 'RDMA_CONNECT_FAILED:' \"${client_log_path}\" \
    | sed -n 's/.*RDMA_CONNECT_FAILED: local=\\([^ ]*\\) peer=.*/\\1/p' || true
  suspect_keys=\$(
    (
      grep -E '^(with|no) regmr result: key=gpu-ce-test-tensor-[0-9]+-0, status=14' \"${client_log_path}\" \
        | sed -n 's/.*tensor-\\([0-9]\\+\\)-0.*/\\1/p' || true
      grep 'ibv_post_send failed for request=gpu-ce-test-tensor-' \"${client_log_path}\" \
        | sed -n 's/.*request=gpu-ce-test-tensor-\\([0-9]\\+\\)-0:.*/\\1/p' || true
      grep -E '^(with|no) regmr result: key=gpu-ce-test-tensor-[0-9]+-0, status=0, .*rdma_read=0' \"${client_log_path}\" \
        | sed -n 's/.*tensor-\\([0-9]\\+\\)-0.*/\\1/p' || true
    ) | sort -u
  )
  if [ -n \"\${suspect_keys}\" ]; then
    while IFS= read -r key_idx; do
      if [ -z \"\${key_idx}\" ]; then
        continue
      fi
      grep \"read tensor:.*key=gpu-ce-test-tensor-\${key_idx}-0 \" \"${client_log_path}\" \
        | sed -n 's/.*net_dev=\\([^ ,]*\\).*/\\1/p' | head -n 1
    done <<< \"\${suspect_keys}\"
  fi
) | sort -Vu
" || true)"
  server_failed_hcas_raw="$(brain_exec_as_user "${server_worker_pid}" "grep 'failed to rdma connect from' \"${server_log_path}\" | sed -n 's/.*net_dev=\\([^ ]*\\).*/\\1/p' | sort -Vu || true")"
  mapfile -t failed_hcas < <(printf '%s\n%s\n' "${client_failed_hcas_raw}" "${server_failed_hcas_raw}" | awk '/^mlx5_[0-9]+$/ {print}' | sort -Vu)
  imbalance_reason_present=0
  for reason in "${failure_reasons[@]}"; do
    if [[ "${reason}" == per-link\ bandwidth\ imbalance:* ]]; then
      imbalance_reason_present=1
      break
    fi
  done
  if (( ${#failed_hcas[@]} == 0 )); then
    if (( imbalance_reason_present == 1 )) && rotate_selected_hca_with_spare "bandwidth_imbalance"; then
      ib_hca_csv="$(IFS=,; echo "${selected_hcas[*]}")"
      echo "[retry] next attempt RNICs: ${ib_hca_csv}"
      continue
    fi
    echo "[retry] no failed RNIC extracted from logs; retry with current RNIC set." >&2
    continue
  fi

  max_exclusions=$(( ${#common_hcas[@]} - transfer_gpu_count ))
  declare -A failed_hca_set=()
  for dev in "${failed_hcas[@]}"; do
    observed_failed_hcas["${dev}"]=1
    failed_hca_set["${dev}"]=1
  done
  if (( max_exclusions <= 0 )); then
    declare -A selected_hca_usage=()
    for dev in "${selected_hcas[@]}"; do
      selected_hca_usage["${dev}"]=0
    done
    while read -r gpu_idx nic_name; do
      if [[ -z "${gpu_idx}" || -z "${nic_name}" ]]; then
        continue
      fi
      if [[ -n "${selected_hca_usage[${nic_name}]+x}" ]]; then
        selected_hca_usage["${nic_name}"]=$(( selected_hca_usage["${nic_name}"] + 1 ))
      fi
    done <<< "${gpu_nic_map_raw}"

    slot_swap_applied=0
    for failed_dev in "${failed_hcas[@]}"; do
      failed_idx=-1
      for idx in "${!selected_hcas[@]}"; do
        if [[ "${selected_hcas[${idx}]}" == "${failed_dev}" ]]; then
          failed_idx="${idx}"
          break
        fi
      done
      if (( failed_idx < 0 )); then
        continue
      fi

      best_idx=-1
      best_usage=2147483647
      for idx in "${!selected_hcas[@]}"; do
        if (( idx == failed_idx )); then
          continue
        fi
        candidate_dev="${selected_hcas[${idx}]}"
        if [[ -n "${failed_hca_set[${candidate_dev}]:-}" ]]; then
          continue
        fi
        candidate_usage="${selected_hca_usage[${candidate_dev}]:-0}"
        if (( candidate_usage < best_usage )); then
          best_usage="${candidate_usage}"
          best_idx="${idx}"
        fi
      done
      if (( best_idx < 0 )); then
        continue
      fi

      replacement_dev="${selected_hcas[${best_idx}]}"
      selected_hcas[${best_idx}]="${selected_hcas[${failed_idx}]}"
      selected_hcas[${failed_idx}]="${replacement_dev}"
      slot_swap_applied=1
      echo "[retry] no spare RNIC candidates; remapped slot by usage: ${failed_dev}@idx${failed_idx} <-> ${replacement_dev}@idx${best_idx}(usage=${best_usage})"
      break
    done

    if (( slot_swap_applied == 0 )); then
      echo "[retry] no spare RNIC candidates available for failover or slot remap." >&2
      continue
    fi
    ib_hca_csv="$(IFS=,; echo "${selected_hcas[*]}")"
    cumulative_failed="$(printf '%s\n' "${!observed_failed_hcas[@]}" | sort -V | paste -sd, -)"
    echo "[retry] failed RNICs (attempt-local): ${failed_hcas[*]}"
    echo "[retry] cumulative failed RNICs: ${cumulative_failed:-none}"
    echo "[retry] next attempt RNICs: ${ib_hca_csv}"
    continue
  fi

  applied_exclusions=0
  replaced_exclusions=0
  declare -a replacement_pairs=()
  current_exclusions=${#excluded_hcas[@]}
  for dev in "${failed_hcas[@]}"; do
    if ! printf '%s\n' "${common_hcas[@]}" | grep -qx "${dev}"; then
      continue
    fi
    if [[ -n "${excluded_hcas[${dev}]:-}" ]]; then
      continue
    fi
    if (( current_exclusions < max_exclusions )); then
      excluded_hcas["${dev}"]=1
      excluded_hca_order+=("${dev}")
      current_exclusions=$(( current_exclusions + 1 ))
      applied_exclusions=$(( applied_exclusions + 1 ))
      continue
    fi
    if (( ${#excluded_hca_order[@]} == 0 )); then
      continue
    fi
    evicted="${excluded_hca_order[0]}"
    excluded_hca_order=("${excluded_hca_order[@]:1}")
    unset "excluded_hcas[${evicted}]"
    excluded_hcas["${dev}"]=1
    excluded_hca_order+=("${dev}")
    replaced_exclusions=$(( replaced_exclusions + 1 ))
    applied_exclusions=$(( applied_exclusions + 1 ))
    replacement_pairs+=("${evicted}->${dev}")
  done

  if (( applied_exclusions == 0 )); then
    echo "[retry] failed RNICs already excluded/outside common set/no replacement room: ${failed_hcas[*]}" >&2
    continue
  fi

  if ! select_hca_subset selected_hcas; then
    echo "[retry] cannot keep ${transfer_gpu_count} RNICs after excluding: ${failed_hcas[*]}" >&2
    continue
  fi
  ib_hca_csv="$(IFS=,; echo "${selected_hcas[*]}")"
  current_excluded="$(printf '%s\n' "${!excluded_hcas[@]}" | sort -V | paste -sd, -)"
  cumulative_failed="$(printf '%s\n' "${!observed_failed_hcas[@]}" | sort -V | paste -sd, -)"
  echo "[retry] failed RNICs (attempt-local): ${failed_hcas[*]}"
  if (( replaced_exclusions > 0 )); then
    echo "[retry] exclusion window rotated: $(IFS=,; echo "${replacement_pairs[*]}")"
  fi
  echo "[retry] currently excluded RNICs: ${current_excluded:-none}"
  echo "[retry] cumulative failed RNICs: ${cumulative_failed:-none}"
  echo "[retry] next attempt RNICs: ${ib_hca_csv}"
done

if (( transfer_passed == 0 )); then
  echo "[verify] topology-guided routing RDMA smoke failed after ${final_attempt} attempt(s)." >&2
  exit 1
fi

echo "[verify] topology-guided routing RDMA smoke passed on attempt ${final_attempt}"
echo "logs: ${final_server_log} and ${final_client_log}"
