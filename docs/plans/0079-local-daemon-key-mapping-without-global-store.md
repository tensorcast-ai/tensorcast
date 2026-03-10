---
slug: local-daemon-key-mapping-without-global-store
title: Local-Daemon Key Mapping Without Global Store Plan
status: draft
areas: ["sdk", "daemon", "core", "docs", "proto"]
created: 2026-02-14
last_updated: 2026-02-14
related_code:
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/daemon_ctl.py
  - daemon/service/controllers/key_mapping_controller.h
  - daemon/service/controllers/key_mapping_controller.cc
  - daemon/service/grpc_service_impl_publish_replica_key_test.cc
  - core/store/runtime/metadata/metadata_gateway.cc
  - core/store/runtime/context/runtime_context.cc
  - tensorcast/api/store/registration.py
  - tensorcast/api/store/retry.py
  - tensorcast/api/_register.py
  - tensorcast/api/store/__init__.py
  - tests/python/test_store_session_api.py
  - tests/python/global_store/test_binding_key_mapping_rpc.py
  - docs/guides/sdk-startup-user-guide.md
  - docs/architecture/api/api-design.md
links:
  design: ../designs/0079-local-daemon-key-mapping-without-global-store.md
---

# Objective

Deliver local-only key authority without Global Store while enforcing one contract across local and global backends, removing `fail_if_exists`, and eliminating SDK key-path strictness drift.

# Current State and Grounding

Baseline gaps:

- Local-only daemon mode is documented, but key flows can fail when key RPCs require GS-connected metadata gateway.
- Key semantics differ across code paths (generation/cache/conflict behavior and best-effort vs strict SDK calls).
- `fail_if_exists` exists in daemon key publish API shape but is not part of required long-term behavior.

Constraints:

- SDK key operations remain daemon-mediated only.
- No implicit local fallback in global authority mode.
- No key-mapping schema changes.

# Phases and Milestones

- [ ] Phase 1: Contract Freeze First
  - [ ] Milestone 1.1: Encode one normative publish/resolve/swap contract in daemon layer (local == global semantics).
  - [ ] Milestone 1.2: Lock generation semantics (`start=0`, publish no bump, swap bump on value change only).
  - [ ] Milestone 1.3: Lock cache policy semantics (`ALIAS -> ttl=0`, `IMMUTABLE -> backend hint`).

- [ ] Phase 2: Authority Mode Hardening
  - [ ] Milestone 2.1: Add explicit authority mode (`LOCAL_ONLY`, `GLOBAL_STORE`) to key controller.
  - [ ] Milestone 2.2: Determine mode from runtime authority declaration (endpoint and/or injected GS client).
  - [ ] Milestone 2.3: Freeze authority mode for daemon lifetime; no runtime auto-switching.

- [ ] Phase 3: API Simplification (`fail_if_exists` removal)
  - [ ] Milestone 3.1: Remove `fail_if_exists` from daemon proto request.
  - [ ] Milestone 3.2: Remove client plumbing in `tensorcast/daemon_ctl.py`.
  - [ ] Milestone 3.3: Remove/adjust tests and callsites that rely on this field.

- [ ] Phase 4: Local Backend Implementation (Contract-Equivalent)
  - [ ] Milestone 4.1: Implement local row model (`artifact_id`, `generation`, `kind`, `updated_at`).
  - [ ] Milestone 4.2: Implement publish conflict/idempotency behavior.
  - [ ] Milestone 4.3: Implement swap behavior including missing-key alias creation.
  - [ ] Milestone 4.4: Remove controller-local fixed mutation TTL override that conflicts with alias TTL semantics.

- [ ] Phase 5: Global-Mode Guardrails and Error Model
  - [ ] Milestone 5.1: Keep GS path unchanged when connected.
  - [ ] Milestone 5.2: Reject writes when global authority is required but unavailable.
  - [ ] Milestone 5.3: Preserve transport error propagation (`UNAVAILABLE`, deadlines) rather than over-collapsing to one status.

- [ ] Phase 6: SDK Strictness Unification
  - [ ] Milestone 6.1: Keep `Store.put(..., key=...)` strict precheck behavior.
  - [ ] Milestone 6.2: Remove best-effort swallow semantics from other key publish entry points.
  - [ ] Milestone 6.3: Keep error hints actionable and mode-accurate in `retry.py`.

- [ ] Phase 7: Conformance Tests and Docs
  - [ ] Milestone 7.1: Build local/global semantic conformance matrix for publish/resolve/swap.
  - [ ] Milestone 7.2: Add disconnected-global negative tests.
  - [ ] Milestone 7.3: Update startup/API docs with local-only scope and key contract semantics.

# Task Breakdown

- [ ] Proto/client API
  - [ ] Update `proto/tensorcast/daemon/v2/store_daemon.proto` to remove `fail_if_exists`.
  - [ ] Regenerate protobuf artifacts via `bash tools/build_proto_python.sh`.
  - [ ] Update `tensorcast/daemon_ctl.py` request construction.

- [ ] Daemon key controller
  - [ ] Add authority mode and local backend state in `daemon/service/controllers/key_mapping_controller.h`.
  - [ ] Implement contract-equivalent local publish/resolve/swap in `daemon/service/controllers/key_mapping_controller.cc`.
  - [ ] Ensure alias cache TTL behavior is preserved (`ttl=0`) in responses.

- [ ] Core/runtime authority plumbing
  - [ ] Ensure authority detection uses GS authority declaration, not endpoint-only heuristics.
  - [ ] Keep global-disconnected write failures explicit and non-fallback.

- [ ] SDK behavior
  - [ ] Align key publish strictness across `registration.py`, `_register.py`, and `store/__init__.py`.
  - [ ] Keep key operation errors mapped with actionable hints in `retry.py`.

- [ ] Tests
  - [ ] Extend daemon key tests with full matrix:
    - [ ] publish missing key / same key-idempotent / different key-conflict
    - [ ] swap missing without guards creates alias generation 0
    - [ ] swap with guard mismatch conflicts and returns current state
    - [ ] swap no-op keeps generation unchanged
    - [ ] alias resolve ttl is 0
  - [ ] Add global-disconnected negative tests (writes fail, resolve cache-only behavior).
  - [ ] Add Python regression tests for strict key behavior across `put`, `register`, `from_disk` entry paths.

- [ ] Docs
  - [ ] Update `docs/guides/sdk-startup-user-guide.md` with local-only key scope/lifetime.
  - [ ] Update `docs/architecture/api/api-design.md` key-mapping contract section.

# Test, Rollout, and Backout

## Test Plan

C++:

- `bazel test //daemon:grpc_service_impl_publish_replica_key_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`

Python:

- `source .venv/bin/activate && pytest tests/python/test_store_session_api.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_binding_key_mapping_rpc.py`

Targeted acceptance checks:

- [ ] local-only `put(key=...)` succeeds for free key.
- [ ] local/global publish/resolve/swap matrix matches contract outcomes.
- [ ] alias resolve cache TTL is 0 in both backends.
- [ ] global-disconnected writes never create local-authority state.
- [ ] no remaining `fail_if_exists` field usage.
- [ ] SDK key publish entry points share strict conflict behavior.

## DoD Contract Gates

- [ ] Contract-equivalent key semantics verified across local/global backends.
- [ ] No implicit local fallback exists under global authority mode.
- [ ] SDK remains daemon-only for key operations.
- [ ] `fail_if_exists` removed end-to-end (proto/client/controller/tests).
- [ ] startup/API docs reflect updated key contract and local-only limits.

## Rollout

- Merge daemon + sdk + proto + tests + docs in one change set.
- Run conformance matrix tests first; then run broader daemon/python targeted suites.
- Monitor key-mapping metrics by authority mode and conflict/not_found/precondition outcomes.

## Backout

- Full revert of this change set if global authority semantics regress.
- Avoid partial backout; proto/client/controller/test/doc must remain synchronized.

# Risks and Tracking

- [ ] Risk: local implementation drifts from GS semantics over time.
  - Mitigation: conformance matrix in CI and contract-gate ownership.

- [ ] Risk: strict SDK unification surfaces more explicit errors in legacy callers.
  - Mitigation: clear migration notes and consistent error hints.

- [ ] Risk: authority detection regression around injected GS clients.
  - Mitigation: dedicated tests for endpoint-only vs injected-client startup paths.

# Owner Checklist

- [ ] Daemon owner sign-off on authority model and local/global semantic equivalence.
- [ ] SDK owner sign-off on strictness unification and error mapping.
- [ ] Proto owner sign-off on `fail_if_exists` removal.
- [ ] Test owner sign-off on conformance matrix coverage.
- [ ] Docs owner sign-off on startup/API contract wording.
