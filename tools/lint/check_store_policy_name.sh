#!/usr/bin/env bash
set -euo pipefail

matches="$(rg -n "PutPolicy" docs proto || true)"
if [[ -n "${matches}" ]]; then
  echo "Found forbidden PutPolicy references; use StorePolicy instead." >&2
  echo "${matches}" >&2
  exit 1
fi
