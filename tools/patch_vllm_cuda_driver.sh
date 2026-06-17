#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DEFAULT_VENV_ROOT="${REPO_ROOT}/.venv"
DEFAULT_VLLM_ROOT="/opt/vllm"
DEFAULT_CUDA_DRIVER_DIR="/usr/local/nvidia/lib64"

VENV_ROOT="${VENV_ROOT:-${DEFAULT_VENV_ROOT}}"
VLLM_ROOT="${VLLM_ROOT:-}"
CUDA_DRIVER_DIR="${CUDA_DRIVER_DIR:-${DEFAULT_CUDA_DRIVER_DIR}}"
CUDA_RUNTIME_DIR="${CUDA_RUNTIME_DIR:-}"
VERIFY=1

usage() {
  cat <<'EOF'
Usage: tools/patch_vllm_cuda_driver.sh [options]

Patch vLLM CUDA extensions so they record a direct libcuda.so.1 dependency and
carry a RUNPATH that finds torch, CUDA runtime, and NVIDIA driver libraries
without relying on LD_PRELOAD.

Options:
  --venv-root PATH          Python virtualenv root. Default: <repo>/.venv
  --vllm-root PATH          vllm source root. Auto-detected when possible
  --cuda-driver-dir PATH    Directory that contains libcuda.so.1
                            Default: /usr/local/nvidia/lib64
  --cuda-runtime-dir PATH   Directory that contains libcudart.so.12
                            Auto-detected from nvcc/CUDA install when omitted
  --no-verify               Skip import verification
  -h, --help                Show this help text

Environment overrides:
  VENV_ROOT
  VLLM_ROOT
  CUDA_DRIVER_DIR
  CUDA_RUNTIME_DIR
EOF
}

log() {
  printf '[patch_vllm_cuda_driver] %s\n' "$*"
}

die() {
  printf '[patch_vllm_cuda_driver] error: %s\n' "$*" >&2
  exit 1
}

require_tool() {
  local tool="$1"
  command -v "${tool}" >/dev/null 2>&1 || die "missing required tool: ${tool}"
}

resolve_vllm_root() {
  if [[ -n "${VLLM_ROOT}" ]]; then
    printf '%s\n' "${VLLM_ROOT}"
    return
  fi

  if [[ -x "${VENV_ROOT}/bin/python" ]]; then
    local detected_root
    detected_root="$(
      "${VENV_ROOT}/bin/python" - <<'PY' 2>/dev/null || true
import pathlib

try:
    import vllm
except Exception:
    raise SystemExit(1)

print(pathlib.Path(vllm.__file__).resolve().parents[1])
PY
    )"
    if [[ -n "${detected_root}" ]]; then
      printf '%s\n' "${detected_root}"
      return
    fi
  fi

  printf '%s\n' "${DEFAULT_VLLM_ROOT}"
}

resolve_site_packages_dir() {
  local site_packages
  site_packages="$(find "${VENV_ROOT}/lib" -maxdepth 2 -type d -path '*/site-packages' | sort | head -n 1 || true)"
  [[ -n "${site_packages}" ]] || die "failed to locate site-packages under ${VENV_ROOT}"
  printf '%s\n' "${site_packages}"
}

resolve_torch_lib_dir() {
  local site_packages="$1"
  local torch_lib_dir
  torch_lib_dir="$(find "${site_packages}" -maxdepth 2 -type d -path '*/torch/lib' | sort | head -n 1 || true)"
  [[ -n "${torch_lib_dir}" ]] || die "failed to locate torch/lib under ${site_packages}"
  printf '%s\n' "${torch_lib_dir}"
}

resolve_cuda_runtime_dir() {
  if [[ -n "${CUDA_RUNTIME_DIR}" ]]; then
    printf '%s\n' "${CUDA_RUNTIME_DIR}"
    return
  fi

  if command -v nvcc >/dev/null 2>&1; then
    local nvcc_path cuda_root
    nvcc_path="$(readlink -f "$(command -v nvcc)")"
    cuda_root="$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
    if [[ -f "${cuda_root}/lib64/libcudart.so.12" ]]; then
      printf '%s\n' "${cuda_root}/lib64"
      return
    fi
  fi

  local candidate
  candidate="$(
    find /data/cuda /usr/local -type f -name 'libcudart.so.12' 2>/dev/null | sort | head -n 1 || true
  )"
  [[ -n "${candidate}" ]] || die "failed to locate libcudart.so.12; pass --cuda-runtime-dir"
  dirname "${candidate}"
}

append_unique_path() {
  local candidate="$1"
  [[ -n "${candidate}" ]] || return 0
  [[ -d "${candidate}" ]] || return 0

  local existing
  for existing in "${MERGED_RPATH_ENTRIES[@]:-}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      return 0
    fi
  done

  MERGED_RPATH_ENTRIES+=("${candidate}")
}

merge_rpath() {
  local so_path="$1"
  local existing_rpath
  existing_rpath="$(patchelf --print-rpath "${so_path}" 2>/dev/null || true)"

  MERGED_RPATH_ENTRIES=()

  if [[ -n "${existing_rpath}" ]]; then
    local old_ifs="${IFS}"
    IFS=':'
    local entry
    for entry in ${existing_rpath}; do
      append_unique_path "${entry}"
    done
    IFS="${old_ifs}"
  fi

  append_unique_path "${TORCH_LIB_DIR}"
  append_unique_path "${CUDA_RUNTIME_DIR_RESOLVED}"
  append_unique_path "${CUDA_DRIVER_DIR}"

  local merged_rpath
  local old_ifs="${IFS}"
  IFS=':'
  merged_rpath="${MERGED_RPATH_ENTRIES[*]}"
  IFS="${old_ifs}"

  printf '%s\n' "${merged_rpath}"
}

has_cuda_driver_symbols() {
  local so_path="$1"
  nm -D "${so_path}" 2>/dev/null | grep -Eq '^[[:space:]]+U[[:space:]]+cu[A-Z]'
}

has_libcuda_needed() {
  local so_path="$1"
  patchelf --print-needed "${so_path}" | grep -Fxq 'libcuda.so.1'
}

repair_placeholder_needed() {
  local so_path="$1"
  local placeholder
  placeholder="$(patchelf --print-needed "${so_path}" | grep -E '^X+$' | head -n 1 || true)"
  if [[ -n "${placeholder}" ]]; then
    log "repairing placeholder DT_NEEDED entry in ${so_path}"
    patchelf --replace-needed "${placeholder}" libcuda.so.1 "${so_path}"
  fi
}

patch_shared_object() {
  local so_path="$1"

  if ! has_cuda_driver_symbols "${so_path}"; then
    log "skip ${so_path}: no unresolved CUDA driver symbols"
    return 0
  fi

  local merged_rpath
  merged_rpath="$(merge_rpath "${so_path}")"
  if [[ -n "${merged_rpath}" ]]; then
    patchelf --set-rpath "${merged_rpath}" "${so_path}"
  fi

  if ! has_libcuda_needed "${so_path}"; then
    log "adding libcuda.so.1 dependency to ${so_path}"
    patchelf --add-needed libcuda.so.1 "${so_path}"
  else
    log "libcuda.so.1 already present in ${so_path}"
  fi

  repair_placeholder_needed "${so_path}"
}

verify_imports() {
  local python_bin="${VENV_ROOT}/bin/python"
  [[ -x "${python_bin}" ]] || die "missing Python interpreter: ${python_bin}"

  log "verifying imports without LD_PRELOAD"
  env \
    -u LD_PRELOAD \
    PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -c 'import vllm._C; print("vllm._C ok")'

  env \
    -u LD_PRELOAD \
    PYTHONPATH="${VLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" -c 'from vllm import LLM; print("vllm.LLM ok")'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-root)
      VENV_ROOT="$2"
      shift 2
      ;;
    --vllm-root)
      VLLM_ROOT="$2"
      shift 2
      ;;
    --cuda-driver-dir)
      CUDA_DRIVER_DIR="$2"
      shift 2
      ;;
    --cuda-runtime-dir)
      CUDA_RUNTIME_DIR="$2"
      shift 2
      ;;
    --no-verify)
      VERIFY=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

require_tool find
require_tool nm
require_tool patchelf

[[ -d "${VENV_ROOT}" ]] || die "virtualenv root does not exist: ${VENV_ROOT}"
[[ -d "${CUDA_DRIVER_DIR}" ]] || die "CUDA driver directory does not exist: ${CUDA_DRIVER_DIR}"
[[ -f "${CUDA_DRIVER_DIR}/libcuda.so.1" ]] || die "missing ${CUDA_DRIVER_DIR}/libcuda.so.1"

VLLM_ROOT="$(resolve_vllm_root)"
[[ -d "${VLLM_ROOT}" ]] || die "vLLM root does not exist: ${VLLM_ROOT}"
[[ -d "${VLLM_ROOT}/vllm" ]] || die "missing vllm package directory under ${VLLM_ROOT}"

SITE_PACKAGES_DIR="$(resolve_site_packages_dir)"
TORCH_LIB_DIR="$(resolve_torch_lib_dir "${SITE_PACKAGES_DIR}")"
CUDA_RUNTIME_DIR_RESOLVED="$(resolve_cuda_runtime_dir)"
[[ -d "${CUDA_RUNTIME_DIR_RESOLVED}" ]] || die "CUDA runtime directory does not exist: ${CUDA_RUNTIME_DIR_RESOLVED}"
[[ -f "${CUDA_RUNTIME_DIR_RESOLVED}/libcudart.so.12" ]] || die "missing ${CUDA_RUNTIME_DIR_RESOLVED}/libcudart.so.12"

log "using VENV_ROOT=${VENV_ROOT}"
log "using VLLM_ROOT=${VLLM_ROOT}"
log "using TORCH_LIB_DIR=${TORCH_LIB_DIR}"
log "using CUDA_RUNTIME_DIR=${CUDA_RUNTIME_DIR_RESOLVED}"
log "using CUDA_DRIVER_DIR=${CUDA_DRIVER_DIR}"

declare -a TARGETS=()
while IFS= read -r so_path; do
  TARGETS+=("${so_path}")
done < <(find "${VLLM_ROOT}/vllm" -type f -name '*.so' | sort)

while IFS= read -r so_path; do
  TARGETS+=("${so_path}")
done < <(find "${SITE_PACKAGES_DIR}" -maxdepth 1 -type f -name 'deep_ep_cpp*.so' | sort)

[[ "${#TARGETS[@]}" -gt 0 ]] || die "no candidate shared objects found"

for so_path in "${TARGETS[@]}"; do
  patch_shared_object "${so_path}"
done

if [[ "${VERIFY}" == "1" ]]; then
  verify_imports
fi

log "done"
