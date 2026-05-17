#!/usr/bin/env bash
# tensorcast release pipeline
# ----------------------------
# Subcommands:
#   build [--torch-version X.Y.Z] [--cuda-version cuNNN] [--pypi] [--in-docker] [--skip-uv-sync] [--cache-uv-lock]
#       Build the wheel(s). Defaults to torch 2.11.0 + cu128.
#       --pypi      Strip +local from version string (PyPI-uploadable).
#       --in-docker Run inside the manylinux_2_28 docker image (Stage B). Implied if you
#                   want auditwheel repair → manylinux wheels.
#   post-process WHEEL_PATH
#       Run patchelf/strip/chmod on a wheel; if IN_DOCKER=1 also run auditwheel repair.
#   check
#       Run `twine check dist/*.whl`.
#   publish-test
#       Upload manylinux wheels in dist/ to TestPyPI (filters out linux_x86_64).
#   publish
#       Upload manylinux wheels in dist/ to production PyPI (with confirmation).
#
# Legacy passthrough: status / sync-venv / update-pyproject / cache forward to
# tools/manage_torch_version.py (kept for backwards compat).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( dirname "${SCRIPT_DIR}" )"

# ---- legacy management subcommands (forwarded as-is) -----------------------
case "${1:-}" in
    status|sync-venv|update-pyproject|cache)
        python "${SCRIPT_DIR}/manage_torch_version.py" "$@"
        exit $?
        ;;
esac

# ---- defaults --------------------------------------------------------------
DEFAULT_TORCH="2.11.0"
DEFAULT_CUDA="cu128"
DOCKER_IMAGE="${TENSORCAST_RELEASE_IMAGE:-quay.io/pypa/manylinux_2_28_x86_64}"

usage() {
    cat <<EOF
tensorcast release pipeline

USAGE
  tools/release.sh <command> [options]

COMMANDS
  build [opts]            Build wheel(s) for the current matrix.
  post-process WHEEL      patchelf + strip + chmod (+ auditwheel inside docker).
  check                   twine check dist/*.whl.
  publish-test            Upload manylinux wheels to TestPyPI.
  publish                 Upload manylinux wheels to PyPI (asks for confirmation).
  status | sync-venv | update-pyproject | cache
                          Forwarded to tools/manage_torch_version.py (legacy).

build OPTIONS
  --torch-version X.Y.Z   Pin torch version            (default ${DEFAULT_TORCH})
  --cuda-version  cuNNN   Pin CUDA index               (default ${DEFAULT_CUDA})
  --pypi                  Emit a clean version string  (RELEASE_PYPI=1)
  --in-docker             Build inside ${DOCKER_IMAGE} for manylinux_2_28
  --skip-uv-sync          Skip uv sync (CI / pre-prepared env)
  --cache-uv-lock         Cache resolved uv.lock under tools/uv-lock-cache/

EXAMPLES
  tools/release.sh build                          # Stage A, default matrix
  tools/release.sh build --pypi                   # Stage A, clean version
  tools/release.sh build --in-docker --pypi       # Stage B, manylinux + clean
  tools/release.sh post-process dist/foo.whl
  tools/release.sh check
  tools/release.sh publish-test
EOF
}

# ---- common helpers --------------------------------------------------------
run_in_docker() {
    # Re-invoke this script inside the manylinux docker image with the
    # remaining argv (minus the --in-docker flag the outer call consumed).
    local extra_env=()
    if [[ -n "${BUILD_VERSION:-}" ]]; then
        extra_env+=("-e" "BUILD_VERSION=${BUILD_VERSION}")
    fi
    extra_env+=("-e" "IN_DOCKER=1")
    echo "==> Re-invoking inside ${DOCKER_IMAGE}: $*"
    exec docker run --rm \
        -v "${PROJECT_ROOT}:/io" \
        -w /io \
        "${extra_env[@]}" \
        "${DOCKER_IMAGE}" \
        bash /io/tools/release.sh "$@"
}

ensure_uv() {
    command -v uv >/dev/null 2>&1 || {
        echo "error: uv is required. Install: https://docs.astral.sh/uv/" >&2
        exit 1
    }
}

# Ensure the `release` dependency group (wheel / auditwheel / patchelf / twine)
# is synced into .venv. These tools are not part of the default `uv sync`, so
# any release-only entrypoint (post-process / check / publish) must call this
# before invoking them.
ensure_release_deps() {
    ensure_uv
    cd "${PROJECT_ROOT}"
    echo "==> uv sync --group release --no-install-project"
    uv sync --group release --no-install-project
}

# Filter dist/ for wheels eligible for PyPI upload (manylinux_*, not linux_*).
collect_pypi_wheels() {
    local out=()
    while IFS= read -r -d '' f; do
        out+=("$f")
    done < <(find "${PROJECT_ROOT}/dist" -maxdepth 1 -name "*manylinux*.whl" -print0 2>/dev/null || true)
    if [[ ${#out[@]} -eq 0 ]]; then
        echo "error: no manylinux wheels found under dist/. Run \`tools/release.sh build --in-docker --pypi\` first." >&2
        exit 1
    fi
    printf '%s\n' "${out[@]}"
}

# ---- build -----------------------------------------------------------------
cmd_build() {
    local torch_version="${DEFAULT_TORCH}"
    local cuda_version="${DEFAULT_CUDA}"
    local pypi=0
    local in_docker_flag=0
    local skip_uv_sync=0
    local cache_uv_lock=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --torch-version) torch_version="$2"; shift 2 ;;
            --cuda-version)  cuda_version="$2";  shift 2 ;;
            --pypi)          pypi=1;             shift ;;
            --in-docker)     in_docker_flag=1;   shift ;;
            --skip-uv-sync)  skip_uv_sync=1;     shift ;;
            --cache-uv-lock) cache_uv_lock=1;    shift ;;
            -h|--help)       usage; exit 0 ;;
            *)
                echo "error: unknown option: $1" >&2; usage; exit 2 ;;
        esac
    done

    if [[ ${in_docker_flag} -eq 1 && -z "${IN_DOCKER:-}" ]]; then
        # Re-launch self in docker. Drop --in-docker; pass the rest through.
        local rebuilt=("build")
        [[ "${torch_version}" != "${DEFAULT_TORCH}" ]] && rebuilt+=("--torch-version" "${torch_version}")
        [[ "${cuda_version}" != "${DEFAULT_CUDA}" ]]  && rebuilt+=("--cuda-version" "${cuda_version}")
        [[ ${pypi} -eq 1 ]]          && rebuilt+=("--pypi")
        [[ ${skip_uv_sync} -eq 1 ]]  && rebuilt+=("--skip-uv-sync")
        [[ ${cache_uv_lock} -eq 1 ]] && rebuilt+=("--cache-uv-lock")
        run_in_docker "${rebuilt[@]}"
    fi

    cd "${PROJECT_ROOT}"
    ensure_uv

    echo "==> Build matrix"
    echo "    torch:   ${torch_version}"
    echo "    cuda:    ${cuda_version}"
    echo "    pypi:    ${pypi}"
    echo "    docker:  ${IN_DOCKER:-0}"

    # Use uv's pytorch index for the requested CUDA build.
    if [[ -n "${cuda_version}" ]]; then
        export UV_INDEX_URL="https://download.pytorch.org/whl/${cuda_version}"
        export UV_EXTRA_INDEX_URL="https://pypi.org/simple"
        export PIP_INDEX_URL="${UV_INDEX_URL}"
        export PIP_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL}"
    fi

    # Patch pyproject.toml in place if torch version differs from default;
    # restore on exit. Skipped when running the default matrix.
    local restore_pyproject=0
    if [[ "${torch_version}" != "${DEFAULT_TORCH}" ]]; then
        cp pyproject.toml pyproject.toml.bak
        restore_pyproject=1
        echo "==> Patching pyproject.toml to torch==${torch_version}"
        uv run --no-project python "${SCRIPT_DIR}/update_torch_version.py" "${torch_version}"
        trap '[[ ${restore_pyproject} -eq 1 ]] && mv pyproject.toml.bak pyproject.toml' EXIT
    fi

    # Try to restore a cached uv.lock for this matrix if available.
    if python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from torch_version_manager import restore_uv_lock
sys.exit(0 if restore_uv_lock('${torch_version}', '${cuda_version}' or None) else 1)
" 2>/dev/null; then
        echo "==> Restored cached uv.lock for torch ${torch_version}/${cuda_version}"
    fi

    if [[ ${skip_uv_sync} -eq 0 ]]; then
        echo "==> uv sync"
        uv sync
        if [[ ${cache_uv_lock} -eq 1 ]]; then
            python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from torch_version_manager import cache_uv_lock
cache_uv_lock('${torch_version}', '${cuda_version}' or None)
"
        fi
    fi

    echo "==> Validating torch version consistency"
    if ! uv run --no-project python -c "
import sys; sys.path.insert(0, '${SCRIPT_DIR}')
from torch_version_manager import validate_torch_versions
ok, _ = validate_torch_versions(raise_on_error=False)
sys.exit(0 if ok else 1)
"; then
        echo "error: torch version inconsistent between pyproject.toml / .venv / MODULE.bazel" >&2
        exit 1
    fi

    echo "==> Sync MODULE.bazel http_archives from uv.lock"
    uv run --no-project python "${SCRIPT_DIR}/update_module_http_archives.py" \
        --lockfile uv.lock --module MODULE.bazel

    echo "==> Building wheel"
    local build_env=()
    build_env+=("BUILD_EXTENSION=1" "BUILD_CORE=1")
    if [[ ${pypi} -eq 1 ]]; then
        build_env+=("RELEASE_PYPI=1")
    else
        build_env+=("RELEASE=1")
        # When RELEASE=1 (not RELEASE_PYPI), setup.py needs BUILD_VERSION.
        if [[ -z "${BUILD_VERSION:-}" ]]; then
            build_env+=("BUILD_VERSION=$(cat version.txt)")
        fi
    fi
    # Make Bazel's progress and Fetching@... events stream live to the
    # terminal. The default `uv build --wheel` front-end runs the PEP 517
    # build backend through a captured pipe, which swallows Bazel's curses
    # progress lines and leaves the user staring at a single "Analyzing:
    # 2 targets ..." line for minutes. Going through `setup.py bdist_wheel`
    # directly (the same pattern as `uv run -vvv setup.py build_ext`) lets
    # the bazel subprocess inherit stdio so its output is visible in real
    # time.
    #
    # BAZEL_BUILD_FLAGS: enable curses, lift the progress rate limit, and
    # show only progress + warnings + errors so external fetches are loud.
    export BAZEL_BUILD_FLAGS="--curses=yes --show_progress_rate_limit=0 --ui_event_filters=progress,warning,error,info"

    rm -rf build dist
    env "${build_env[@]}" uv run --no-project -v python setup.py bdist_wheel

    if [[ ${restore_pyproject} -eq 1 ]]; then
        mv pyproject.toml.bak pyproject.toml
        restore_pyproject=0
    fi

    echo
    echo "==> Built wheel(s):"
    ls -la dist/*.whl 2>/dev/null || true

    # Auto-run post-process for each freshly built wheel; auditwheel only
    # fires when IN_DOCKER=1.
    local need_post_process=0
    for whl in dist/*.whl; do
        case "${whl}" in
            *manylinux*) continue ;;
        esac
        need_post_process=1
        break
    done
    if [[ ${need_post_process} -eq 1 ]]; then
        ensure_release_deps
    fi
    for whl in dist/*.whl; do
        # Skip files already labeled "manylinux*" (auditwheel output) to
        # avoid double-processing on rerun.
        case "${whl}" in
            *manylinux*) continue ;;
        esac
        echo
        echo "==> Post-processing ${whl}"
        uv run python "${SCRIPT_DIR}/wheel_post_process.py" "${whl}"
    done
}

# ---- post-process ----------------------------------------------------------
cmd_post_process() {
    local whl="${1:-}"
    if [[ -z "${whl}" ]]; then
        echo "usage: tools/release.sh post-process WHEEL_PATH" >&2; exit 2
    fi
    ensure_release_deps
    uv run --no-project python "${SCRIPT_DIR}/wheel_post_process.py" "${whl}"
}

# ---- check -----------------------------------------------------------------
cmd_check() {
    ensure_release_deps
    echo "==> twine check dist/*.whl"
    uv run --no-project twine check dist/*.whl
}

# ---- publish ---------------------------------------------------------------
cmd_publish_test() {
    ensure_release_deps
    local wheels
    wheels="$(collect_pypi_wheels)"
    echo "==> Wheels to upload to TestPyPI:"
    echo "${wheels}"
    echo
    read -r -p "Continue? [y/N] " confirm
    [[ "${confirm}" == "y" || "${confirm}" == "Y" ]] || { echo "aborted."; exit 1; }
    # shellcheck disable=SC2086
    uv run --no-project twine upload --repository testpypi ${wheels}
}

cmd_publish() {
    ensure_release_deps
    local wheels
    wheels="$(collect_pypi_wheels)"
    echo "==> Wheels to upload to **production PyPI**:"
    echo "${wheels}"
    echo
    echo "WARNING: PyPI does not allow re-uploading the same version."
    read -r -p "Type the project version to confirm (e.g. 0.1.0): " confirm_version
    local actual_version
    actual_version="$(cat "${PROJECT_ROOT}/version.txt")"
    if [[ "${confirm_version}" != "${actual_version}" ]]; then
        echo "error: version mismatch (expected ${actual_version}). Aborted." >&2
        exit 1
    fi
    # shellcheck disable=SC2086
    uv run --no-project twine upload ${wheels}
}

# ---- dispatch --------------------------------------------------------------
case "${1:-}" in
    build)         shift; cmd_build "$@" ;;
    post-process)  shift; cmd_post_process "$@" ;;
    check)         shift; cmd_check "$@" ;;
    publish-test)  shift; cmd_publish_test "$@" ;;
    publish)       shift; cmd_publish "$@" ;;
    -h|--help|"")  usage; exit 0 ;;
    *)             echo "error: unknown command: $1" >&2; usage; exit 2 ;;
esac
