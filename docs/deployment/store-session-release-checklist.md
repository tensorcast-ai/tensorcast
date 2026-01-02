---
title: Store Session Release Checklist
description: End-to-end checklist for shipping the Store-centric session API
sidebar_position: 5
---

# Store Session Release Checklist

Use this checklist when promoting the Store-centric session API to any environment beyond staging. Each stage ensures the Global Store schema, Store Daemon binary, and Python SDK wheel stay in lockstep and that observability continues to match [Design 0010](../designs/0010-opentelemetry-unified-observability-design.md).

## Pre-release alignment

- Verify Global Store migrations are applied and backward compatible. Schema drift blocks rollout.
- Build the Store Daemon with matching MODULE.bazel lockfile revisions and publish the artifact (`bazel build //daemon:tensorcast_daemon`).
- Produce the Python SDK wheel with `BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext` and sanity-check `tensorcast/api/store.py` exports.
- Confirm the release manifest notes the shared version triplet (`tensorcast` Python wheel, daemon binary, Global Store image) and share it with deployment owners.

## Validation gates

- Run targeted integration suites on staging:
  - `uv run pytest tests/python/test_register_lease_in_place_helper.py`
  - `uv run pytest tests/python/test_register_vram_leased_and_dvmp_stream.py`
  - `uv run pytest tests/python/test_store_session_api.py`
  - `bazel test //daemon:session_lifecycle_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Execute smoke benchmarks or representative workloads and record latency deltas versus the Phase 0 baseline.
- Exercise the CLI:
  - `uv run tensorcast daemon status` shows Store sessions with reasonable lease/future counts.
  - Sample workload completes via SDK helpers (e.g., `tensorcast.register` + `tensorcast.artifact(...).tensor_dict()`).

## Observability checks

- Confirm OpenTelemetry exporters are wired and receiving spans named `Store/Register`, `Store/Put`, `Store/Get`, `Store/GetInto`.
- Inspect metrics in Grafana/Prometheus:
  - Histograms `tc_store_operation_latency_seconds` (per verb, per daemon)
 - Counters `tc_store_operation_errors_total`, `tc_store_operation_retries_total`
  - Any custom lease health gauges added during Phase 4
- Compare p95 latency, error, and retry rates against the previous release—flag if deviation exceeds approved SLO guardrails.

## Deployment steps

1. Roll the Global Store image.
2. Roll Store Daemon binaries cluster by cluster. Confirm `uv run tensorcast daemon status` reports the new build string.
3. Roll the Python SDK wheel to client hosts. Restart workers to pick up the new wheel.
4. Announce the rollout window and expectations in the deployment channel.

## Post-release verification

- Monitor rollout dashboards for 30 minutes minimum; ensure retry/error counters remain flat.
- Check the session registry on representative hosts (`~/.tensorcast/store_sessions`) for stale entries.
- Collect feedback from early adopters (inference, training, batch pipelines) and confirm no regressions in their smoke tests.
- Update release notes with the rollout status, including any deviations from the checklist.

## Backout plan

- Re-deploy the previous Python SDK wheel and daemon binary if SLOs degrade. No database rollback is required.
- Clear on-host caches if necessary (`rm ~/.tensorcast/store_sessions/*.json`)—clients will repopulate files on next verb execution.
- Re-run the validation suites to confirm behaviour matches the prior release before re-attempting rollout.
- Document the incident and corrective actions before scheduling the next release attempt.
