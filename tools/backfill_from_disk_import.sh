#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)/$(basename -- "${BASH_SOURCE[0]}")"
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd -P)"

usage() {
  cat <<'EOF'
Usage: backfill_from_disk_import.sh [options] <artifact_dir>

Backfill import metadata for a disk artifact directory by running one isolated
Store.from_disk(...) import against a temporary local daemon.

This is primarily useful for raw safetensors directories that do not yet have
tensor_index.json / artifact_descriptor.json. The script can auto-escalate to
sudo when the target directory is not writable by the current user.

Options:
  --sudo=auto|always|never   Control privilege escalation. Default: auto
  --verify-checksums         Pass verify_checksums=True to from_disk
  --show-progress            Pass show_progress=True to from_disk
  --keep-runtime             Keep the temporary TENSORCAST_HOME/HOME tree
  --help                     Show this help

Examples:
  bash tools/backfill_from_disk_import.sh /mnt/models/my_hf_dir
  bash tools/backfill_from_disk_import.sh --sudo=always /mnt/models/root_owned_dir
EOF
}

fail() {
  echo "error: $*" >&2
  exit 1
}

SUDO_MODE="auto"
VERIFY_CHECKSUMS="false"
SHOW_PROGRESS="false"
KEEP_RUNTIME="false"
TARGET_PATH=""

while (($# > 0)); do
  case "$1" in
    --sudo=auto|--sudo=always|--sudo=never)
      SUDO_MODE="${1#--sudo=}"
      ;;
    --verify-checksums)
      VERIFY_CHECKSUMS="true"
      ;;
    --show-progress)
      SHOW_PROGRESS="true"
      ;;
    --keep-runtime)
      KEEP_RUNTIME="true"
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --*)
      fail "unknown option: $1"
      ;;
    *)
      if [[ -n "${TARGET_PATH}" ]]; then
        fail "only one artifact_dir may be provided"
      fi
      TARGET_PATH="$1"
      ;;
  esac
  shift
done

if [[ -z "${TARGET_PATH}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -d "${TARGET_PATH}" ]]; then
  fail "artifact_dir is not a directory: ${TARGET_PATH}"
fi

if [[ ! -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
  fail "missing virtualenv at ${ROOT_DIR}/.venv"
fi

should_use_sudo="false"
case "${SUDO_MODE}" in
  auto)
    if [[ "${EUID}" -ne 0 && ! -w "${TARGET_PATH}" ]]; then
      should_use_sudo="true"
    fi
    ;;
  always)
    if [[ "${EUID}" -ne 0 ]]; then
      should_use_sudo="true"
    fi
    ;;
  never)
    should_use_sudo="false"
    ;;
  *)
    fail "invalid --sudo mode: ${SUDO_MODE}"
    ;;
esac

if [[ "${should_use_sudo}" == "true" ]]; then
  sudo_args=("${SCRIPT_PATH}" "--sudo=never")
  if [[ "${VERIFY_CHECKSUMS}" == "true" ]]; then
    sudo_args+=("--verify-checksums")
  fi
  if [[ "${SHOW_PROGRESS}" == "true" ]]; then
    sudo_args+=("--show-progress")
  fi
  if [[ "${KEEP_RUNTIME}" == "true" ]]; then
    sudo_args+=("--keep-runtime")
  fi
  sudo_args+=("${TARGET_PATH}")
  exec sudo "${sudo_args[@]}"
fi

RUNTIME_BASE="$(mktemp -d "${TMPDIR:-/tmp}/tensorcast-backfill.XXXXXX")"
export HOME="${RUNTIME_BASE}/home"
export TENSORCAST_HOME="${RUNTIME_BASE}/runtime"
mkdir -p "${HOME}" "${TENSORCAST_HOME}"

SESSION="backfill-$(date +%Y%m%d-%H%M%S)-$$"
START_JSON="${RUNTIME_BASE}/daemon-start.json"

cleanup() {
  set +e
  cd "${ROOT_DIR}" >/dev/null 2>&1 || true
  if [[ -f "${ROOT_DIR}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${ROOT_DIR}/.venv/bin/activate" >/dev/null 2>&1 || true
    tensorcast-cli daemon stop --session "${SESSION}" >/dev/null 2>&1 || true
  fi
  if [[ "${KEEP_RUNTIME}" != "true" ]]; then
    rm -rf "${RUNTIME_BASE}"
  fi
}
trap cleanup EXIT

descriptor_before="false"
tensor_index_before="false"
if [[ -f "${TARGET_PATH}/artifact_descriptor.json" ]]; then
  descriptor_before="true"
fi
if [[ -f "${TARGET_PATH}/tensor_index.json" ]]; then
  tensor_index_before="true"
fi

cd "${ROOT_DIR}"
# shellcheck disable=SC1091
source "${ROOT_DIR}/.venv/bin/activate"

tensorcast-cli daemon start \
  --session "${SESSION}" \
  --global-store-mode none \
  --log-level warning \
  --set server.listen.port=0 \
  --set server.p2p_listen.port=0 \
  --set engine.artifact_chunk_bytes=64MB \
  --set engine.streaming_buffer_chunks=1 \
  --set engine.memory_tiers.stable_bytes=512MB \
  --set communicator.transport.tcp_conn_count=1 \
  --set communicator.stager.buffers_per_flow=1 \
  --set communicator.stager.max_window_segments=1 \
  --set pinned_memory.classes='[{"name":"engine","slice_bytes":"64MB","pool_bytes":"512MB","rdma_preregister":false},{"name":"comm_gpu","slice_bytes":"4MB","pool_bytes":"32MB","rdma_preregister":false},{"name":"comm_cpu","slice_bytes":"4MB","pool_bytes":"32MB","rdma_preregister":false}]' \
  --json >"${START_JSON}"

DAEMON_ADDRESS="$(
  python - "${START_JSON}" <<'PY'
import json
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
start = text.find("{")
if start < 0:
    raise SystemExit("daemon start output did not contain JSON")
payload = json.loads(text[start:])
print(payload["daemon"]["address"])
PY
)"

python - "${TARGET_PATH}" "${DAEMON_ADDRESS}" "${VERIFY_CHECKSUMS}" "${SHOW_PROGRESS}" "${descriptor_before}" "${tensor_index_before}" <<'PY'
import json
import sys
import time
from pathlib import Path

from tensorcast.api.store import Store

target = Path(sys.argv[1])
daemon_address = sys.argv[2]
verify_checksums = sys.argv[3].lower() == "true"
show_progress = sys.argv[4].lower() == "true"
descriptor_before = sys.argv[5].lower() == "true"
tensor_index_before = sys.argv[6].lower() == "true"

store = Store(daemon_address)
start = time.perf_counter()
artifact = store.from_disk(
    str(target),
    verify_checksums=verify_checksums,
    show_progress=show_progress,
)
latency_s = time.perf_counter() - start
store.close()

result = {
    "artifact_dir": str(target),
    "artifact_id": artifact.artifact_id,
    "daemon_address": daemon_address,
    "latency_s": latency_s,
    "verify_checksums": verify_checksums,
    "show_progress": show_progress,
    "descriptor_before": descriptor_before,
    "tensor_index_before": tensor_index_before,
    "descriptor_after": (target / "artifact_descriptor.json").exists(),
    "tensor_index_after": (target / "tensor_index.json").exists(),
}
print(json.dumps(result, ensure_ascii=False))
PY
