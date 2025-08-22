#!/usr/bin/env bash
# Entrypoint script for the StepCast Store image.
#
# Usage examples:
#   docker run stepcast-store store-daemon --config /path/to/config.yaml
#   docker run stepcast-store global-store --port 50051
#
# The first positional argument selects which service to start.
# * store-daemon  -> scstore.cli start
# * global-store -> scstore.global_store
# Any additional arguments are forwarded verbatim to the underlying command.

set -euo pipefail

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <store-daemon|global-store> [args...]" >&2
  exit 1
fi

mode="$1"
shift

case "${mode}" in
  store-daemon)
    exec python -m scstore.cli start "$@"
    ;;
  global-store)
    exec python -m scstore.global_store "$@"
    ;;
  *)
    # Fallback: execute the command as-is (allows users to run arbitrary shells)
    exec "$mode" "$@"
    ;;
esac