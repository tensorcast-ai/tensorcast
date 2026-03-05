---
slug: unified-artifact-binding-kv-runtime
title: Plan - Unified Artifact-Binding Runtime for Weights and Cache Blobs
status: proposed
areas: ["sdk", "daemon", "core", "proto", "integrations", "docs"]
created: 2026-03-04
last_updated: 2026-03-05
related_code:
  - docs/designs/0084-unified-artifact-binding-kv-runtime.md
  - docs/designs/0056-programmable-framework-adv.md
  - docs/designs/0070-mapped-binding-requirements.md
  - docs/architecture/api/region-backed.md
  - tensorcast/common/identity.py
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/README.md
  - proto/tensorcast/plan/v1/plan.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
links:
  design: ../designs/0084-unified-artifact-binding-kv-runtime.md
---

# Objective

Implement one runtime contract for weights and high-cardinality cache blob operations (paged KV motivating case):

- unified object model (`Artifact` + `Binding`/region + mapping/manifest),
- explicit blob `open` -> `sealed` lifecycle boundary,
- batch-first external-target IO primitives with VRAM-first fast paths,
- generic `cache_*` plan actions with `kvcache_*` aliases.

# Current State and Grounding

- `0063` already anchors `Binding` lifecycle and swap safety.
- `0070` mapped binding exists in SDK/daemon implementation and supports `target_write_token` publish flow.
- `0056` currently mixes control-plane evolution with KV data-path semantics and is too broad for phased landing.
- CGID validation grammar blocks encodings that include `:`, requiring doc/code normalization for cache blob examples.

# Phases and Milestones

- [ ] Phase 0: Doc and contract convergence
  - [ ] Milestone 0.1: Land `0084` design and scope boundary updates in `0056`.
  - [ ] Milestone 0.2: Align `0070` and SDK docs on mapped publish semantics.
  - [ ] Milestone 0.3: Normalize cache blob CGID encoding examples to grammar-safe format.
  - [ ] Milestone 0.4: Add explicit namespace override in `0017` that `cgid:cache_blob~...` does not enter GS per-blob artifact/replica catalogs.

- [ ] Phase 1: Identity helper unification
  - [ ] Milestone 1.1: Add shared Python helper for cache blob CGID segment encode/decode/validation.
  - [ ] Milestone 1.2: Add matching C++ helper and cross-language test vectors.
  - [ ] Milestone 1.3: Update callers and docs to use shared helpers only.

- [ ] Phase 2: Batch external-target IO primitives
  - [ ] Milestone 2.1: Define additive proto contracts for local front-door external-target APIs (`batch_exists`, `batch_get_into_region`, `batch_put_if_absent_from_region`, `batch_touch_ttl`).
  - [ ] Milestone 2.2: Implement daemon/controller preflight and per-item outcomes.
  - [ ] Milestone 2.3: Ensure payload transport does not inline above threshold and reuses data-plane handles or region IO.
  - [ ] Milestone 2.4: Regenerate proto code (`bash tools/build_proto_python.sh`).
  - [ ] Milestone 2.5: Define/implement home-scoped fenced daemon-to-daemon batch RPCs (`CacheBlobBatch*` family) distinct from front-door APIs.

- [ ] Phase 3: Plan IR and SDK aliasing
  - [ ] Milestone 3.1: Add generic `cache_*` actions to `plan.proto` and Python plan builder.
  - [ ] Milestone 3.2: Keep `kvcache_*` as aliases mapped to generic actions in IR.
  - [ ] Milestone 3.3: Preserve deterministic idempotency fingerprints across alias paths.

- [ ] Phase 4: Adapter integration and lifecycle correctness
  - [ ] Milestone 4.1: Enforce `open` non-publishable and `sealed` publishable states in engine adapters.
  - [ ] Milestone 4.2: Enforce `PUT_IF_ABSENT_JOIN` invariants on sealed publish.
  - [ ] Milestone 4.3: Keep token->key and block-table updates inside engine adapters only.

- [ ] Phase 5: Performance validation and rollout
  - [ ] Milestone 5.1: Benchmark per-blob loops vs batch primitives on representative cache blob cardinalities (paged KV).
  - [ ] Milestone 5.2: Validate VRAM->VRAM and P2P preferred path with host staging fallback.
  - [ ] Milestone 5.3: Roll out by mode gate and publish acceptance report.

- [ ] Phase 6: Config and fencing consistency
  - [ ] Milestone 6.1: Add cluster-global cache-blob routing invariants to unified runtime config (`S`, hash64 version, inline threshold, lease/staleness policy, retention defaults/limits).
  - [ ] Milestone 6.2: Enforce staleness handling (`unknown freshness -> miss/unavailable`) and no-unproven-hit rule in routing paths.
  - [ ] Milestone 6.3: Define rollout/cutover playbook for incompatible invariant changes (global cache epoch cutover).

# Tasks

- SDK:
  - wire generic `cache_*` operations and alias handling in `tensorcast/api/plan/plan.py`.
  - keep canonical action names (`cache_*`) as source of truth for idempotency/metrics and preserve `kvcache_*` only as alias metadata.
  - keep `Binding`/mapped swap behavior unchanged while switching internals to generic primitives where applicable.
  - update `tensorcast/api/store/README.md` and usage examples.

- Daemon/core:
  - add local front-door batch RPC handlers and route resolution without bypassing shard fencing rules.
  - add home-scoped fenced daemon-to-daemon cache-blob RPC handlers; do not mix with front-door region-write APIs.
  - keep per-item statuses explicit and fail-fast only on invariant/precondition violations.
  - enforce lease-freshness uncertainty handling (miss/unavailable instead of guessed hit).
  - implement cache-blob selection profile branch with fixed digest validation and normalized resolved selection.
  - add progress and no-progress diagnostics for batch operations.

- Integrations:
  - update engine adapters to produce sealed blobs with invariants and invoke generic actions.
  - verify SGLang/vLLM integration points do not require framework-specific core object types.

# Test / Rollout / Backout

Tests:

- Python:
  - `source .venv/bin/activate && pytest tests/python/test_binding.py`
  - `source .venv/bin/activate && pytest tests/python/test_mapped_binding.py`
  - `source .venv/bin/activate && pytest tests/python/test_plan.py`
- C++:
  - `bazel test //daemon:session_lifecycle_test`
  - `bazel test //core/store:store_engine_test`
- Proto hygiene:
  - `bash tools/build_proto_python.sh`
  - `bazel test //proto/... --test_output=errors`

Rollout:

- start with aliases enabled and generic actions shadowed in non-prod.
- gate publish and routing changes by explicit capability/version checks.
- promote only after parity and perf gates pass.

Backout:

- keep old action decoding path for one release window.
- disable generic actions by config gate and continue serving `kvcache_*` legacy handlers.

# Risks and Tracking

- Risk: alias and canonical action ids diverge.
  - tracking: audit operation_id and canonical action labels in logs.
- Risk: batch API adds contention or head-of-line blocking.
  - tracking: queue depth, inflight, and sustained no-progress alerts.
- Risk: staged fallback accidentally becomes default.
  - tracking: source-path counters (vram_to_vram, p2p, host_staging).
