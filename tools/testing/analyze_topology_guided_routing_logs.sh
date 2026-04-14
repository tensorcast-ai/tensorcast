#!/usr/bin/env bash

set -euo pipefail

if (( $# != 1 )); then
  echo "Usage: $0 <topology_guided_routing_log_dir>" >&2
  exit 2
fi

log_dir="$1"
if [[ ! -d "${log_dir}" ]]; then
  echo "log directory not found: ${log_dir}" >&2
  exit 2
fi

analyze_client_attempt() {
  local log_path="$1"
  local name
  name="$(basename "${log_path}")"
  echo "===== ${name} ====="

  local key_map usage_table failed_keys zero_read_keys suspect_keys
  key_map="$(
    grep 'read tensor:' "${log_path}" \
      | sed -n 's/.*key=gpu-ce-test-tensor-\([0-9]\+\)-0 .*net_dev=\([^ ,]*\).*/\1 \2/p' \
      | awk '{m[$1]=$2} END {for (k in m) print k, m[k]}' \
      | sort -n -k1,1
  )"
  usage_table="$(
    printf '%s\n' "${key_map}" \
      | awk '{c[$2]+=1} END {for (nic in c) print nic, c[nic]}' \
      | sort -V -k1,1
  )"
  failed_keys="$(
    grep 'ibv_post_send failed for request=gpu-ce-test-tensor-' "${log_path}" \
      | sed -n 's/.*request=gpu-ce-test-tensor-\([0-9]\+\)-0:.*/\1/p' \
      | sort -n -u
  )"
  zero_read_keys="$(
    grep -E '^(with|no) regmr result: key=gpu-ce-test-tensor-[0-9]+-0, status=0, .*rdma_read=0' "${log_path}" \
      | sed -n 's/.*tensor-\([0-9]\+\)-0.*/\1/p' \
      | sort -n -u
  )"
  suspect_keys="$(
    printf '%s\n%s\n' "${failed_keys}" "${zero_read_keys}" \
      | awk 'NF > 0 {print}' \
      | sort -n -u
  )"

  echo "-- key->nic map"
  printf '%s\n' "${key_map}" | sed 's/^/  /'
  echo "-- nic usage"
  printf '%s\n' "${usage_table}" | sed 's/^/  /'
  echo "-- ibv_post_send failed keys: ${failed_keys:-none}"
  echo "-- rdma_read=0 keys: ${zero_read_keys:-none}"
  echo "-- suspect key->nic"
  if [[ -n "${suspect_keys}" ]]; then
    while IFS= read -r key; do
      if [[ -z "${key}" ]]; then
        continue
      fi
      nic="$(printf '%s\n' "${key_map}" | awk -v k="${key}" '$1 == k {print $2; exit}')"
      echo "  ${key} -> ${nic:-unknown}"
    done <<< "${suspect_keys}"
  else
    echo "  none"
  fi
}

for client_log in "${log_dir}"/client_attempt*.log; do
  if [[ -f "${client_log}" ]]; then
    analyze_client_attempt "${client_log}"
  fi
done

for server_log in "${log_dir}"/server_attempt*.log; do
  if [[ ! -f "${server_log}" ]]; then
    continue
  fi
  name="$(basename "${server_log}")"
  connects="$(
    grep 'recv rdma connect' "${server_log}" \
      | sed -n 's/.*net_dev=\([^ ]*\).*/\1/p' \
      | sort -Vu \
      | paste -sd, -
  )"
  echo "===== ${name} ====="
  echo "-- recv rdma connect unique net_dev: ${connects:-none}"
done
