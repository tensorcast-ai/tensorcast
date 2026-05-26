#!/bin/bash
set -euo pipefail

# Resolve repo root and module file
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODULE_FILE="$REPO_ROOT/MODULE.bazel"

# Parse LLVM version from MODULE.bazel (e.g., llvm_version = "20.1.0")
LLVM_VERSION="$(sed -n 's/.*llvm_version\s*=\s*"\([^"]*\)".*/\1/p' "$MODULE_FILE" | head -n1)"
if [[ -z "${LLVM_VERSION:-}" ]]; then
  echo "Error: Failed to parse llvm_version from $MODULE_FILE" >&2
  exit 1
fi

# Parse expected sha256 for linux-x86_64 from MODULE.bazel (64 hex chars)
EXPECTED_SHA256="$(sed -n 's/.*"linux-x86_64":[[:space:]]*"\([0-9a-fA-F]\{64\}\)".*/\1/p' "$MODULE_FILE" | head -n1)"
if [[ -z "${EXPECTED_SHA256:-}" ]]; then
  echo "Error: Failed to parse sha256 for linux-x86_64 from $MODULE_FILE" >&2
  exit 1
fi

TARBALL="LLVM-${LLVM_VERSION}-Linux-X64.tar.xz"
DEST_PATH="$REPO_ROOT/$TARBALL"

# Candidate public download URLs.
URLS=(
  "https://githubfast.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/LLVM-${LLVM_VERSION}-Linux-X64.tar.xz"
  "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_VERSION}/LLVM-${LLVM_VERSION}-Linux-X64.tar.xz"
)

# Compute and verify sha256
compute_sha256() {
  sha256sum "$1" | awk '{print $1}'
}

verify_sha256() {
  local file="$1"
  local expect="$2"
  if [[ ! -f "$file" || ! -s "$file" ]]; then
    return 1
  fi
  local actual
  actual="$(compute_sha256 "$file")" || return 1
  if [[ "$actual" == "$expect" ]]; then
    echo "SHA256 verified: $actual"
    return 0
  else
    echo "SHA256 mismatch for $file" >&2
    echo "  expected: $expect" >&2
    echo "  actual  : $actual" >&2
    return 1
  fi
}

download_with_verification() {
  local file="$1"
  local expect="$2"
  local max_attempts=5
  local attempt=1
  while (( attempt <= max_attempts )); do
    for u in "${URLS[@]}"; do
      echo "Attempt ${attempt}: $u"
      if wget -c -O "$file" "$u"; then
        if verify_sha256 "$file" "$expect"; then
          return 0
        fi
      fi
    done
    attempt=$((attempt + 1))
    sleep 1
  done
  return 1
}

# Ensure we have a correct tarball (verify existing; otherwise download until verified)
if verify_sha256 "$DEST_PATH" "$EXPECTED_SHA256"; then
  echo "Using existing $DEST_PATH (checksum OK)."
else
  echo "Downloading $TARBALL to $DEST_PATH ..."
  if ! download_with_verification "$DEST_PATH" "$EXPECTED_SHA256"; then
    echo "Error: Failed to obtain a valid $TARBALL after multiple attempts." >&2
    exit 1
  fi
fi

echo "Done. Verified local LLVM tarball: $DEST_PATH"
