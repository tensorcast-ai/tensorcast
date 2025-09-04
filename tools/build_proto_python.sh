#!/usr/bin/env bash
set -euo pipefail

# Determine repository root based on this script's location
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"
PROTO_DIR="${ROOT_DIR}/proto"

# Determine execution mode: bazel (default) or raw
MODE="${1:-bazel}"

case "${MODE}" in
  bazel)
    echo "[proto] Generating via bazel-run buf in: ${PROTO_DIR}"
    RUN_CMD=(bazel run @rules_buf_toolchains//:buf -- generate --template buf.gen.yaml)
    ;;
  raw)
    echo "[proto] Generating via raw buf in: ${PROTO_DIR}"
    RUN_CMD=(buf generate --template buf.gen.yaml)
    ;;
  *)
    echo "Usage: $(basename "$0") [bazel|raw]" >&2
    exit 1
    ;;
esac

cd "${PROTO_DIR}"
"${RUN_CMD[@]}"

# Post-process generated Python to use package prefix tensorcast.proto.*
PY_OUT_DIR="${PROTO_DIR}/gen/python/tensorcast"
if [[ -d "${PY_OUT_DIR}" ]]; then
  echo "[proto] Rewriting Python imports under ${PY_OUT_DIR} (tensorcast.* -> tensorcast.proto.*)"
  # Replace lines starting with "from tensorcast." but skip ones already using tensorcast.proto.
  # Applies to both .py and .pyi files.
  find "${PY_OUT_DIR}" -type f \( -name "*.py" -o -name "*.pyi" \) \
    -exec sed -i -E \
      -e '/^[[:space:]]*from[[:space:]]+tensorcast\.proto\./b' \
      -e 's/^[[:space:]]*from[[:space:]]+tensorcast\./from tensorcast.proto./' {} +
else
  echo "[proto] Skipping import rewrite: directory not found: ${PY_OUT_DIR}"
fi

echo "[proto] Done."


