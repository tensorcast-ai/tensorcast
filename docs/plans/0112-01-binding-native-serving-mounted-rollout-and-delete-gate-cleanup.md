---
slug: binding-native-serving-mounted-rollout-and-delete-gate-cleanup
title: Binding-Native Serving Mounted Rollout and Delete-Gate Cleanup Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "serving"]
created: 2026-04-10
last_updated: 2026-04-13
related_code:
  - docs/designs/0112-binding-native-serving-realization-and-publication.md
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/state/lip_manager.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - /data/workspace/internal-vllm/vllm/model_executor/model_loader/tensorcast_loader.py
links:
  design: ../designs/0112-binding-native-serving-realization-and-publication.md
---

# Objective

Track the remaining mounted rollout, operator-visible evidence hardening, and
delete-gate cleanup for the binding-native same-binding serving path now that
`0112` is the surviving path owner and this file is its active rollout
checklist.

# Current State & Grounding

- `0112` already owns the shipped same-binding correctness model:
  - public disk ingress,
  - binding-native publication subject,
  - fail-closed canonical-full seal,
  - and the audited Step3p5 same-binding path closure.
- The audited Step3p5 mounted path is already proved:
  - the CLI-started daemon plus `vllm serve ... tensorcast_init_mode=connect`
    operator path reaches serving readiness,
  - executor-side evidence shows collective-dominant same-binding startup on the
    audited Step3p5 run,
  - and publish-hash identity facts are surfaced through typed diagnostics.
- What remains is broader rollout and cleanup, not a reopened architecture cut:
  - harden the operator-visible mounted evidence packet,
  - clean up transitional bridge and delete-gate logic once evidence is
    sufficient,
  - and treat second-family `BINDING_FINALIZE` validation as a rollout-quality
    gate rather than an environment-availability gap now that qwen2.5 is
    locally runnable.
- The operator-visible typed packet now also carries actual collective evidence
  beyond the Step3p5 minimum:
  - collective unique-source bytes,
  - peer-transfer bytes,
  - peak temporary bytes,
  - batch count,
  - dedup savings,
  - and the existing hash backend/location/identity facts.
- The local qwen3 `PURE_TRANSFORM` reference slice is already captured as a
  mounted evidence-hardening sample, but it is not a substitute for a second
  representative `BINDING_FINALIZE` family.
- The provided local `Qwen2.5-32B-Instruct` checkpoint is useful for `0108` /
  `0109` host-local trace and benchmark work, but the current
  `internal-vllm` qwen2 family gate is no longer the blocker:
  - repo-local family readiness now admits qwen2 to the same PURE_TRANSFORM
    serving-only runtime allowlist as qwen3,
  - `2026-04-13` local debugging first showed three distinct outcomes:
    - `TP=1` with generic `vllm serve --load-format dummy` can reach
      readiness when `--kv-cache-memory-bytes=8GiB` is supplied,
    - `TP=1` with `--load-format tensorcast` still fails during
      `CreateBinding` with daemon-side `cudaErrorMemoryAllocation` because the
      same-binding target allocation cannot coexist with the already-materialized
      ~61 GiB runtime model on one H800,
    - `TP>=2` fails before ready even under generic
      `vllm serve --load-format dummy`, with worker death during distributed
      engine startup (`determine_available_memory` / KV-cache init path) and
      allocator corruption signals outside TensorCast,
  - the distributed-runtime instability is now worked around on this host class
    with the safe TP4 serving flags already captured in local debug logs, and
    both `--load-format tensorcast` and `--load-format safetensors` can reach
    `/health=200`,
  - the remaining qwen2.5 blocker is not readiness but evidence quality:
    the first TP4 TensorCast packet showed `planned_collective_admitted_bytes`
    near `1.3MiB` while runtime executed `actual_collective_committed_bytes`
    near `16.38GiB`,
  - repo-local debugging traced that mismatch to copy-plan lowering feeding the
    whole mapped byte space into `compatibility_lowered_map` instead of only the
    compatibility-admitted subset, which in turn over-selected
    `OwnerFileCollectiveExecutor`,
  - fixing that seam exposed a second repo-local bug where
    `executor_generic_data_map` was derived from the narrowed compatibility map
    rather than the full `generic_fallback_map`, so the updated daemon then
    failed closed with `generic_backend_coverage_unproven`,
  - both daemon-side lowering seams are now fixed in repo-local code with
    targeted regression coverage,
  - and `2026-04-13` refreshed mounted qwen2.5 TP4 evidence now shows:
    - `/health=200`, `/v1/models`, and `/v1/completions`,
    - `Tensorcast bootstrap realized SAME_BINDING_FAST_PATH: family=qwen2`,
    - `planned_collective_admitted_bytes=16382928896`,
    - `actual_collective_committed_bytes=16382928896`,
    - `actual_generic_backend_bytes=0`,
    - `dominant_executor=OwnerFileCollectiveExecutor`,
    - `publish_hash_location=seal`,
    - `publish_hash_backend=gpu`,
    - and `publish_hash_identity_forming=True`,
  - but that second representative family is still not cleanup-ready:
    - the same safe TP4 `safetensors` baseline loads in about `6.36s` to
      `6.41s`,
    - while TensorCast remains about `84.6s` to `85.4s`,
    - and the mounted logs still show slow `ResolvePublicDiskSource`
      (~`54s`) plus slow `RefillOwnedBinding` (~`38s` to `47s`) before serving
      readiness.

# Phases & Milestones

- [x] Phase 1: Evidence Surface Hardening
  - [x] Milestone 1.1: surface any remaining operator-visible mounted metrics
    required beyond the Step3p5 minimum packet.
  - [x] Milestone 1.2: make the mounted evidence packet stable enough that
    delete-gate cleanup can depend on it.
  - [x] Milestone 1.3: capture one local mounted reference slice beyond Step3p5
    without treating it as a `BINDING_FINALIZE` substitute.

- [ ] Phase 2: Delete-Gate Cleanup
  - [ ] Milestone 2.1: remove transitional bridge logic and residual-map-based
    gating that only existed to carry the audited closure through rollout.
  - [ ] Milestone 2.2: retire stale active references and historical aliases
    once mounted evidence is sufficient.
  - [ ] Milestone 2.3: remove any duplicate or temporary diagnostics paths that
    exist only to keep old compatibility surfaces alive.

- [ ] Phase 3: Deferred Future Validation
  - [ ] Milestone 3.1: when a representative second `BINDING_FINALIZE` family is
    locally available, add its mounted runbook and evidence packet without
    reopening already-completed cleanup work.

# Tasks

- [x] Audit mounted operator-visible surfaces for any remaining missing metrics:
  - collective unique-source bytes
  - peer-transfer bytes
  - peak temporary bytes
  - batch count
  - residual hash facts not already surfaced
  - hash round, location, bytes, wall time, backend, and identity-forming facts
    wherever hash work still exists

- [ ] Keep one evidence package current for the audited Step3p5 path.
  - mounted run identifiers and linked status files
  - typed execution diagnostics for collective, executor, hash, and identity
    outcomes
  - downstream additive summary or profile output derived from those typed facts
  - baseline comparison against the default loader on the same host class
  - explicit delete-gate conclusion

- [x] Keep one mounted reference slice beyond Step3p5.
  - qwen3 `PURE_TRANSFORM` reference slice is already documented and rerun.

- [ ] Delete transitional bridge or residual-map cleanup only after the mounted
  evidence packet is stable on the currently available mounted paths.

- [ ] Update docs and references when rollout ownership changes.
  - `docs/README.md`
  - historical closure references in TensorCast docs
  - surviving cross-repo references under `internal-vllm`

- [ ] Keep the second representative `BINDING_FINALIZE` mounted packet current
  now that qwen2.5 is locally runnable, but do not treat mere readiness as
  sufficient evidence for delete-gate cleanup.

## Qwen3 Mounted Reference Slice

This slice is complete as a `PURE_TRANSFORM` reference sample. It does not stand
in for a representative second `BINDING_FINALIZE` family, and this plan records
that gap as deferred follow-up instead of treating it as a blocker for the
current cleanup work.

### Local Runbook

Terminal 1:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate
cd /data/workspace/internal-vllm
tensorcast-cli daemon start --blocking \
    --config vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml \
    --global-store-mode start
```

Terminal 2:

```bash
source /data/workspace/internal-vllm/.venv/bin/activate
cd /data/workspace/internal-vllm
LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
NCCL_DEBUG=WARN \
CUDA_VISIBLE_DEVICES=0 \
vllm serve /home/luoyuchu/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
    --port 8011 \
    --served-model-name qwen3-tensorcast-phase2 \
    --load-format tensorcast \
    --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_show_daemon_logs":true,"tensorcast_enable_runtime_binding":true}' \
    --gpu-memory-utilization 0.6 \
    --enforce-eager
```

Verification:

```bash
curl -sS http://127.0.0.1:8011/health
curl -sS http://127.0.0.1:8011/v1/models
curl -sS http://127.0.0.1:8011/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-tensorcast-phase2","prompt":"Say hi in five words.","max_tokens":8,"temperature":0}'
```

### Captured Evidence

- `GET /health` returned `200`.
- `GET /v1/models` returned `200` and exposed
  `id=qwen3-tensorcast-phase2`.
- `POST /v1/completions` returned `200`.
- vLLM mounted startup logged
  `Tensorcast bootstrap realized SAME_BINDING_FAST_PATH: family=qwen3`.
- `realize_execution` surfaced:
  - `collective_handled=False`
  - `actual_collective_committed_bytes=0`
  - `actual_local_typed_bytes=920`
  - `actual_generic_backend_bytes=1192099840`
  - `dominant_executor=SourceOrderedMappedTargetExecutor`
  - `collective_skip_reason=planner_collective_group_missing`
- `publish_hash` surfaced:
  - `hash_rounds=0`
  - `hash_location=seal`
  - `hash_backend=gpu`
  - `hash_bytes=1192100760`
  - `hash_wall_time_ms=165`
  - `hash_identity_forming=True`

# Test / Rollout / Backout

## Acceptance checks

- [ ] The mounted evidence packet for the audited Step3p5 path is rich enough
  that operators do not need internal-only traces to understand executor and
  hash behavior.
- [ ] Transitional bridge logic and residual-map-based gates are deleted only
  after those mounted evidence gates pass.
- [ ] Any unavailable second representative `BINDING_FINALIZE` family remains an
  explicit deferred item rather than being implied complete.

## Test plan

- [ ] Rerun the real Step3p5 operator path whenever cleanup touches daemon or
  runtime behavior.
- [ ] Continue running the targeted repo checks already used for the audited
  same-binding path.
- [ ] Keep the qwen3 mounted reference slice runnable as an evidence-hardening
  sample.

## Rollout

- treat the audited Step3p5 closure as already closed; do not reopen it just to
  carry rollout cleanup,
- use this companion plan as the single active checklist for mounted evidence and
  delete-gates,
- and treat second-family absence as explicit deferral rather than permission to
  substitute an undocumented family ad hoc.

## Backout

- if cleanup removes evidence or reintroduces ambiguity, revert the cleanup
  commits rather than restoring hidden transitional branches,
- do not recreate a second serving startup path or integration-local contract to
  recover from cleanup mistakes.

# Risks & Tracking

- [ ] Risk: mounted evidence remains split across daemon and integration logs in
  a way operators cannot reliably consume.
  - Mitigation: harden typed operator-visible surfaces before deleting the gates
    that still depend on them.

- [ ] Risk: second-family validation stays environment-blocked.
  - Mitigation: keep the deferral explicit and non-blocking for cleanup that is
    already justified by current evidence.

- [ ] Risk: delete-gate cleanup accidentally absorbs strategy or executor work
  that belongs to `0108` or `0109`.
  - Mitigation: keep this plan scoped to mounted rollout and cleanup only.
