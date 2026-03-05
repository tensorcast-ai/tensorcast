#!/usr/bin/env bash

set -euo pipefail

namespace="${NAMESPACE:-shai-core}"
charged_group="${CHARGED_GROUP:-tensorcast_dev}"
positive_tags="${POSITIVE_TAGS:-H800,ib}"
worker_gpu_count="${WORKER_GPU_COUNT:-8}"
worker_cpu_count="${WORKER_CPU_COUNT:-32}"
worker_memory_mb="${WORKER_MEMORY_MB:-512000}"
max_wait_duration="${MAX_WAIT_DURATION:-20m}"
repo_path="${REPO_PATH:-$(pwd)}"
transfer_gpu_count="${TRANSFER_GPU_COUNT:-8}"
transfer_port="${TRANSFER_PORT:-19099}"
transfer_chunk="${TRANSFER_CHUNK:-1}"
transfer_count="${TRANSFER_COUNT:-16777216}"
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
if (( max_transfer_attempts <= 0 )); then
  echo "MAX_TRANSFER_ATTEMPTS must be > 0." >&2
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

wait_for_process_ready() {
  local pid="$1"
  local max_rounds="${2:-120}"
  local interval_sec="${3:-5}"
  local round=0
  local preempt_noted=0

  while (( round < max_rounds )); do
    round=$((round + 1))
    local row
    row="$(brainctl get process "${pid}" -n "${namespace}" --no-headers 2>/dev/null || true)"
    local ready status
    ready="$(printf '%s\n' "${row}" | awk '{print $4}')"
    status="$(printf '%s\n' "${row}" | awk '{print $5}')"
    echo "[wait] pid=${pid} round=${round} ready=${ready:-unknown} status=${status:-unknown}"

    if [[ "${ready}" == "1/1" && "${status}" == "Running" ]]; then
      return 0
    fi

    if [[ "${status}" == "Failed" || "${status}" == "Error" || "${status}" == "Succeeded" ]]; then
      break
    fi

    if (( preempt_noted == 0 )) &&
      brainctl describe "process/${pid}" -n "${namespace}" | grep -q "Waiting for resources to be preempted"; then
      echo "[scheduler] process ${pid} is waiting for preemption; sleep 60s before next check"
      preempt_noted=1
      sleep 60
      continue
    fi
    sleep "${interval_sec}"
  done

  echo "process ${pid} did not become ready in time" >&2
  print_process_status "${pid}" >&2
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

  printf '%s\n' "${output}" | sed -n 's/.*Dev: \(mlx5_[0-9]\+\) added.*/\1/p' | sort -Vu
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

echo "[launch] server worker"
server_worker_pid="$(brainctl launch -d \
  --charged-group="${charged_group}" \
  --gpu "${worker_gpu_count}" \
  --cpu "${worker_cpu_count}" \
  --memory "${worker_memory_mb}" \
  --private-machine group \
  --positive-tags "${positive_tags}" \
  "${extra_launch_args[@]}" \
  --max-wait-duration="${max_wait_duration}" \
  --comment "${comment_prefix}-server-${timestamp}" \
  -- bash -lc 'echo SERVER_READY; hostname; nvidia-smi -L | sed -n "1,16p"; sleep 7200')"
echo "server worker pid: ${server_worker_pid}"
print_process_status "${server_worker_pid}"

echo "[launch] client worker"
client_worker_pid="$(brainctl launch -d \
  --charged-group="${charged_group}" \
  --gpu "${worker_gpu_count}" \
  --cpu "${worker_cpu_count}" \
  --memory "${worker_memory_mb}" \
  --private-machine group \
  --positive-tags "${positive_tags}" \
  "${extra_launch_args[@]}" \
  --max-wait-duration="${max_wait_duration}" \
  --comment "${comment_prefix}-client-${timestamp}" \
  -- bash -lc 'echo CLIENT_READY; hostname; nvidia-smi -L | sed -n "1,16p"; sleep 7200')"
echo "client worker pid: ${client_worker_pid}"
print_process_status "${client_worker_pid}"

for pid in "${server_worker_pid}" "${client_worker_pid}"; do
  wait_for_process_ready "${pid}"
done

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
    <(printf '%s\n' "${server_engine_hcas[@]}" | sort -V) \
    <(printf '%s\n' "${client_engine_hcas[@]}" | sort -V)
)
if (( ${#common_hcas[@]} < transfer_gpu_count )); then
  echo "need at least ${transfer_gpu_count} common communicator-usable RNICs, got ${#common_hcas[@]}" >&2
  echo "server communicator RNICs: ${server_engine_hcas[*]}" >&2
  echo "client communicator RNICs: ${client_engine_hcas[*]}" >&2
  exit 1
fi

declare -A excluded_hcas=()
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

  with_regmr_ok="$(trim_number "${with_regmr_ok_raw}")"
  no_regmr_ok="$(trim_number "${no_regmr_ok_raw}")"
  read_path_pairs="$(trim_number "${read_path_pairs_raw}")"
  unique_read_nics="$(trim_number "${unique_read_nics_raw}")"
  client_ready_pairs="$(trim_number "${client_ready_pairs_raw}")"
  server_connect_pairs="$(trim_number "${server_connect_pairs_raw}")"

  echo "with_regmr_ok=${with_regmr_ok}"
  echo "no_regmr_ok=${no_regmr_ok}"
  echo "read_path_pairs=${read_path_pairs}"
  echo "unique_read_nics=${unique_read_nics}"
  echo "client_ready_pairs=${client_ready_pairs}"
  echo "server_connect_pairs=${server_connect_pairs}"

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
  failed_keys=\$(grep -E '^(with|no) regmr result: key=gpu-ce-test-tensor-[0-9]+-0, status=14' \"${client_log_path}\" \
    | sed -n 's/.*tensor-\\([0-9]\\+\\)-0.*/\\1/p' | sort -u || true)
  if [ -n \"\${failed_keys}\" ]; then
    while IFS= read -r key_idx; do
      if [ -z \"\${key_idx}\" ]; then
        continue
      fi
      grep \"read tensor:.*key=gpu-ce-test-tensor-\${key_idx}-0 \" \"${client_log_path}\" \
        | sed -n 's/.*net_dev=\\([^ ,]*\\).*/\\1/p' | head -n 1
    done <<< \"\${failed_keys}\"
  fi
) | sort -Vu
" || true)"
  server_failed_hcas_raw="$(brain_exec_as_user "${server_worker_pid}" "grep 'failed to rdma connect from' \"${server_log_path}\" | sed -n 's/.*net_dev=\\([^ ]*\\).*/\\1/p' | sort -Vu || true")"
  mapfile -t failed_hcas < <(printf '%s\n%s\n' "${client_failed_hcas_raw}" "${server_failed_hcas_raw}" | awk '/^mlx5_[0-9]+$/ {print}' | sort -Vu)
  if (( ${#failed_hcas[@]} == 0 )); then
    echo "[retry] no failed RNIC extracted from logs; retry with current RNIC set." >&2
    continue
  fi

  max_exclusions=$(( ${#common_hcas[@]} - transfer_gpu_count ))
  if (( max_exclusions <= 0 )); then
    echo "[retry] no spare RNIC candidates available for failover." >&2
    continue
  fi

  excluded_hcas=()
  applied_exclusions=0
  for dev in "${failed_hcas[@]}"; do
    if ! printf '%s\n' "${common_hcas[@]}" | grep -qx "${dev}"; then
      continue
    fi
    excluded_hcas["${dev}"]=1
    applied_exclusions=$(( applied_exclusions + 1 ))
    if (( applied_exclusions >= max_exclusions )); then
      break
    fi
  done

  if (( applied_exclusions == 0 )); then
    echo "[retry] failed RNICs are outside common candidate set: ${failed_hcas[*]}" >&2
    continue
  fi

  if ! select_hca_subset selected_hcas; then
    echo "[retry] cannot keep ${transfer_gpu_count} RNICs after excluding: ${failed_hcas[*]}" >&2
    continue
  fi
  ib_hca_csv="$(IFS=,; echo "${selected_hcas[*]}")"
  echo "[retry] excluded failed RNICs (attempt-local): ${failed_hcas[*]}"
  echo "[retry] next attempt RNICs: ${ib_hca_csv}"
done

if (( transfer_passed == 0 )); then
  echo "[verify] topology-guided routing RDMA smoke failed after ${final_attempt} attempt(s)." >&2
  exit 1
fi

echo "[verify] topology-guided routing RDMA smoke passed on attempt ${final_attempt}"
echo "logs: ${final_server_log} and ${final_client_log}"
