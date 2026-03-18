#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")/../.."

body_store_hits="$(rg -n 'body_store_\.(exists|get)\(' daemon/service || true)"
if [[ -n "${body_store_hits}" ]]; then
  while IFS= read -r hit; do
    [[ -z "${hit}" ]] && continue
    file="${hit%%:*}"
    if [[ "${file}" != "daemon/service/byte_artifact_authority_service.cc" ]]; then
      echo "guardrail violation: direct body_store visibility access outside byte_artifact_authority_service.cc"
      echo "${hit}"
      exit 1
    fi
  done <<< "${body_store_hits}"
fi

sha_hits="$(rg -n 'compute_sha256_hex\(' core daemon || true)"
if [[ -n "${sha_hits}" ]]; then
  while IFS= read -r hit; do
    [[ -z "${hit}" ]] && continue
    file="${hit%%:*}"
    case "${file}" in
      core/store/runtime/ingestion/materialization_facade.cc|\
      daemon/service/payload_transport_broker.cc|\
      daemon/service/byte_artifact_body_handle.cc|\
      daemon/service/byte_artifact_body_handle.h)
        ;;
      *)
        echo "guardrail violation: compute_sha256_hex() used outside approved seam/integrity helpers"
        echo "${hit}"
        exit 1
        ;;
    esac
  done <<< "${sha_hits}"
fi

echo "artifact truth guardrails: OK"
