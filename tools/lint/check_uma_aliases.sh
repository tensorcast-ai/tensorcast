#!/usr/bin/env bash
set -euo pipefail

# Enforce UMA V3 final naming; fail on legacy includes (error-level).
# Policy: only final headers allowed
#   - core/common/memory/virtual_address_space.h (canonical VS)
#   - core/store/replica/unified_memory_authority.h (canonical UMA)
#   - core/store/replica/memory_export_registry.h (canonical export registry)
#
# This script scans core/store sources (excluding tests) and prints errors for
# legacy includes outside owning files.

ROOT_DIR="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT_DIR"

FAILED=0

function fail_matches() {
  local pattern="$1"; shift
  local allow_regex="$1"; shift
  local files
  files=$(rg --files core/store | rg -v "_test\\.cc$" || true)
  if [[ -z "$files" ]]; then return; fi
  # shellcheck disable=SC2086
  matches=$(rg -n "$pattern" $files | rg -v "$allow_regex" || true)
  if [[ -n "$matches" ]]; then
    echo "[lint:uma-aliases] error: found legacy includes; prefer alias headers:" >&2
    echo "$matches" >&2
    FAILED=1
  fi
}

# DVMP legacy include is forbidden across the tree
fail_matches "#include \"core/common/memory/distributed_virtual_memory_pool.h\"" "(^$)"

# UMA legacy include is forbidden; use unified_memory_authority.h
fail_matches "#include \"core/store/replica/replica_memory_coordinator.h\"" "(^$)"

# Export legacy include is forbidden; use memory_export_registry.h
fail_matches "#include \"core/store/replica/chunk_export_service.h\"" "(^$)"

if [[ "$FAILED" -eq 1 ]]; then
  exit 1
fi
exit 0
