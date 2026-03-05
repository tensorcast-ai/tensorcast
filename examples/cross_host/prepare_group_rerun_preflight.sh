#!/usr/bin/env bash
# Copyright (c) 2026, TensorCast Team.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source .venv/bin/activate

TC_PREP_GS_CONFIG="${TC_PREP_GS_CONFIG:-examples/config/global_store_config_cross_host_bench_32workers_hb60_clustered.yaml}"
TC_PREP_EXPECT_SPREAD_WEIGHT="${TC_PREP_EXPECT_SPREAD_WEIGHT:-2.0}"
TC_PREP_EXPECT_SOFT_CAP_RATIO="${TC_PREP_EXPECT_SOFT_CAP_RATIO:-1.3}"
TC_PREP_EXPECT_MIN_CANDIDATES="${TC_PREP_EXPECT_MIN_CANDIDATES:-3}"
TC_PREP_GS_START="${TC_PREP_GS_START:-1}"
TC_PREP_GS_RESET_DB="${TC_PREP_GS_RESET_DB:-1}"
TC_PREP_REMOTE_CLEANUP="${TC_PREP_REMOTE_CLEANUP:-0}"
TC_PREP_REMOTE_TIMEOUT_SEC="${TC_PREP_REMOTE_TIMEOUT_SEC:-120}"
TC_PREP_LISTEN_HOST="${TC_PREP_LISTEN_HOST:-}"
TC_PREP_LISTEN_PORT="${TC_PREP_LISTEN_PORT:-}"
TC_RUN_AS_USER="${TC_RUN_AS_USER:-$(id -un)}"

if [[ "${TC_RUN_AS_USER}" == "root" ]]; then
  echo "[prep] TC_RUN_AS_USER must be non-root" >&2
  exit 2
fi

CONFIG_SUMMARY_JSON="$(
  python - "${TC_PREP_GS_CONFIG}" \
    "${TC_PREP_EXPECT_SPREAD_WEIGHT}" \
    "${TC_PREP_EXPECT_SOFT_CAP_RATIO}" \
    "${TC_PREP_EXPECT_MIN_CANDIDATES}" <<'PY'
import json
import math
import sys

from tensorcast.global_store.config import GlobalStoreConfig

config_path = str(sys.argv[1]).strip()
expect_spread = float(sys.argv[2])
expect_soft_cap = float(sys.argv[3])
expect_min_candidates = int(sys.argv[4])

cfg = GlobalStoreConfig.from_file(config_path)
dispatch = cfg.transport_scheduler.group_dispatch

actual_spread = float(dispatch.group_source_spread_weight)
actual_soft_cap = float(dispatch.group_source_soft_cap_ratio)
actual_min_candidates = int(dispatch.group_source_min_candidates_for_enforce)

if not math.isclose(actual_spread, expect_spread, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(
        f"group_source_spread_weight mismatch: expected={expect_spread} actual={actual_spread}"
    )
if not math.isclose(actual_soft_cap, expect_soft_cap, rel_tol=0.0, abs_tol=1e-9):
    raise SystemExit(
        f"group_source_soft_cap_ratio mismatch: expected={expect_soft_cap} actual={actual_soft_cap}"
    )
if actual_min_candidates != expect_min_candidates:
    raise SystemExit(
        "group_source_min_candidates_for_enforce mismatch: "
        f"expected={expect_min_candidates} actual={actual_min_candidates}"
    )

print(
    json.dumps(
        {
            "config_path": config_path,
            "db_file": str(cfg.db_file) if cfg.db_file is not None else "",
            "listen_host": str(cfg.listen_host),
            "listen_port": int(cfg.listen_port),
            "transport_scheduler_mode": str(cfg.transport_scheduler.mode),
            "group_source_spread_weight": actual_spread,
            "group_source_soft_cap_ratio": actual_soft_cap,
            "group_source_min_candidates_for_enforce": actual_min_candidates,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
)
PY
)"

echo "[prep] config validated: ${CONFIG_SUMMARY_JSON}"

DB_FILE="$(
  python - "${CONFIG_SUMMARY_JSON}" <<'PY'
import json
import sys

summary = json.loads(sys.argv[1])
print(str(summary.get("db_file", "")).strip())
PY
)"

status_json_or_empty() {
  tensorcast-cli global status --json 2>/dev/null || echo "{}"
}

is_global_running() {
  local payload="$1"
  python - "$payload" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
running = bool(payload.get("health")) and bool(payload.get("state"))
print("1" if running else "0")
PY
}

start_global_store() {
  local -a cmd=(tensorcast-cli global start --config "${TC_PREP_GS_CONFIG}" --json)
  if [[ -n "${TC_PREP_LISTEN_HOST}" ]]; then
    cmd+=(--listen-host "${TC_PREP_LISTEN_HOST}")
  fi
  if [[ -n "${TC_PREP_LISTEN_PORT}" ]]; then
    cmd+=(--listen-port "${TC_PREP_LISTEN_PORT}")
  fi
  "${cmd[@]}" >/dev/null
}

remote_cleanup_process() {
  local process_id="$1"
  local user="$2"
  local timeout_sec="$3"
  local cleanup_cmd wrapped

cleanup_cmd="$(cat <<EOF
set -euo pipefail
cd ${REPO_ROOT}
source .venv/bin/activate
tensorcast-cli daemon stop --force >/dev/null 2>&1 || true
for pid in \$(pgrep -f '[t]ensorcast_daemon --config=' || true); do
  kill -TERM "\${pid}" >/dev/null 2>&1 || true
done
sleep 1
python -c "from tensorcast.cli_utils.paths import runtime_state_path; from tensorcast.cli_utils.process import clear_runtime_global_store; p=runtime_state_path(); clear_runtime_global_store(p, preserve_cluster_token=False) if p.exists() else None" >/dev/null 2>&1 || true
EOF
)"

  wrapped="$(cat <<EOF
set -euo pipefail
run_as_user='${user}'
if [[ "\${run_as_user}" == "root" ]]; then
  echo "refuse to run as root" >&2
  exit 2
fi
if ! getent passwd "\${run_as_user}" >/dev/null 2>&1; then
  echo "remote run-as user not found: \${run_as_user}" >&2
  exit 97
fi
decoded_cmd=\$(printf '%s' '$(printf '%s' "${cleanup_cmd}" | base64 | tr -d '\n')' | base64 -d)
if [[ "\$(id -un)" == "\${run_as_user}" ]]; then
  bash -lc "\${decoded_cmd}"
else
  su - "\${run_as_user}" -s /bin/bash -c "\${decoded_cmd}"
fi
EOF
)"

  timeout "${timeout_sec}s" \
    brainctl exec "process/${process_id}" -n shai-core -- bash -lc "${wrapped}" >/dev/null
}

if [[ "${TC_PREP_REMOTE_CLEANUP}" == "1" ]]; then
  if ! command -v brainctl >/dev/null 2>&1; then
    echo "[prep] TC_PREP_REMOTE_CLEANUP=1 but brainctl is not available" >&2
    exit 1
  fi
  declare -A seen=()
  declare -a process_ids=()
  for single in "${TC_WP_PUBLISHER_PROC:-}" "${TC_SEED_PROC:-}"; do
    value="$(echo "${single}" | xargs)"
    if [[ -n "${value}" && -z "${seen[${value}]:-}" ]]; then
      seen["${value}"]=1
      process_ids+=("${value}")
    fi
  done
  for csv in "${TC_WP_RECEIVER_PROCS:-}" "${TC_GET_PROCS:-}"; do
    IFS=',' read -r -a entries <<<"${csv}"
    for raw in "${entries[@]}"; do
      value="$(echo "${raw}" | xargs)"
      if [[ -n "${value}" && -z "${seen[${value}]:-}" ]]; then
        seen["${value}"]=1
        process_ids+=("${value}")
      fi
    done
  done
  if [[ "${#process_ids[@]}" -eq 0 ]]; then
    echo "[prep] TC_PREP_REMOTE_CLEANUP=1 but no process ids found in env" >&2
    exit 1
  fi
  echo "[prep] remote daemon cleanup begin processes=${process_ids[*]}"
  for process_id in "${process_ids[@]}"; do
    remote_cleanup_process "${process_id}" "${TC_RUN_AS_USER}" "${TC_PREP_REMOTE_TIMEOUT_SEC}"
  done
  echo "[prep] remote daemon cleanup done"
else
  echo "[prep] remote daemon cleanup skipped (TC_PREP_REMOTE_CLEANUP=0)"
fi

if [[ "${TC_PREP_GS_START}" == "1" ]]; then
  before_json="$(status_json_or_empty)"
  before_running="$(is_global_running "${before_json}")"
  echo "[prep] global status before restart running=${before_running}"
  tensorcast-cli global stop >/dev/null 2>&1 || true

  if [[ "${TC_PREP_GS_RESET_DB}" == "1" && -n "${DB_FILE}" && -f "${DB_FILE}" ]]; then
    backup_path="${DB_FILE}.bak.$(date +%Y%m%d-%H%M%S)"
    mv "${DB_FILE}" "${backup_path}"
    echo "[prep] moved previous db_file to ${backup_path}"
  else
    echo "[prep] db reset skipped (db_file empty, missing, or disabled)"
  fi

  start_global_store

  ready=0
  for _ in $(seq 1 20); do
    current_json="$(status_json_or_empty)"
    running="$(is_global_running "${current_json}")"
    if [[ "${running}" == "1" ]]; then
      ready=1
      break
    fi
    sleep 0.5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "[prep] global store failed to become healthy after restart" >&2
    exit 1
  fi
  echo "[prep] global store restart done"
  tensorcast-cli global status --json
else
  echo "[prep] global restart skipped (TC_PREP_GS_START=0)"
fi

echo "[prep] preflight completed"
