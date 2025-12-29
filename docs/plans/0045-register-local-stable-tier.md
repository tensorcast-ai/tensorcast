---
slug: register-local-stable-tier
title: Register/Put With Unified StorePolicy and Local Stable Tier (Plan)
links:
  design: ../designs/0045-register-local-stable-tier.md
areas: ["sdk", "daemon", "core", "proto"]
related_code:
  - docs/designs/0044-unified-put-policy-interface.md
  - docs/designs/0041-distributed-persistence-placement.md
  - docs/designs/0021-region‑backed-registration.md
  - tensorcast/api/_config.py
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/registration.py
  - tensorcast/api/store/handles.py
  - tensorcast/daemon_ctl.py
  - daemon/store_policy_resolver.*
  - daemon/service/controllers/registration_controller.cc
  - daemon/persistence_manager.*
  - proto/tensorcast/daemon/v1/store_daemon.proto
  - core/store/components/stable_dram_cache_manager.*
  - core/store/components/stable_dram_cache_manager_test.cc
  - tests/python/api/test_config_models.py
  - tests/python/api/test_public_surface.py
  - tests/python/api/test_persistence_wireup.py
---

# Objective

Implement Design 0045 so that:

- `register` and `put` share a first-class `policy` parameter surface.
- `register` can synchronously satisfy `stable_dram(scope=local)` when requested by policy, without introducing any user-facing “mirror” concept.
- Persistence continues to follow Design 0041/0044, preferring a local stable source when available.

# Latest Status

All phases in this plan are complete (proto + SDK + daemon + core + docs + tests). This plan is now retrospective; the remaining sections are kept for grounding and validation references.

# Current State & Grounding

This section captures the repo state before implementing Design 0045 (kept for historical grounding).

- SDK verbs:
  - `Store.register` uses `PlanType.VRAM_LEASED`, `Store.put` uses `PlanType.DRAM_STABLE` and both route through `RegistrationPipeline._perform_registration(...)`: `tensorcast/api/store/registration.py`.
  - Today the policy is only reachable via `RegisterArtifactOptions(policy=...)` (not first-class in the verb signature): `tensorcast/api/_config.py`, `tensorcast/api/store/registration.py`.
- Policy model:
  - `StorePolicy` + profiles (`cache/durable/ha/cold/pinned`) exist and are serialized to proto: `tensorcast/api/_config.py`.
  - Daemon resolves policies centrally (single source of truth) and already computes `local_requirement`, `local_retention`, and `overflow_policy`: `daemon/store_policy_resolver.*`.
- Persistence wire-up:
  - SDK calls `start_persistence(artifact_id, policy)` post-commit based on `policy_requires_persistence(...)`: `tensorcast/api/store/registration.py`, `tensorcast/api/_config.py`.
  - Daemon persistence is already policy-driven and can start from LIP or stable-DRAM sources (per Design 0044): `daemon/persistence_manager.*`.
- Stable DRAM retention/admission:
  - Implemented in `StableDramCacheManager` and tested: `core/store/components/stable_dram_cache_manager.*`, `core/store/components/stable_dram_cache_manager_test.cc`.

# Scope & Assumptions

- This plan does **not** introduce new tiers; local stable tier is expressed via existing `stable_dram(scope=local)` in `StorePolicy`.
- Local stable tier (`should|must`) is satisfied **synchronously** in `CommitRegisteredArtifact` when needed; remote stable/shared disk remain **asynchronous** via persistence tasks (Design 0041/0044).
- No new runtime knobs: stable tier budget remains `engine.memory_tiers.stable_bytes` (Design 0044).
- Region-backed registration sources must be supported for any lease-based materialization path (Design 0021).

# Phases & Milestones

- [x] Phase 0: Decide profile strategy (ergonomics)
  - [x] If we want a one-word preset for “best-effort local stable”, add profile `warm`:
    - [x] Extend SDK policy profile enum + parsing: `tensorcast/api/_config.py`.
    - [x] Extend proto `PolicyProfile` (and regenerate): `proto/tensorcast/daemon/v1/store_daemon.proto`, then `bash tools/build_proto_python.sh`.
    - [x] Extend daemon profile defaults: `daemon/store_policy_resolver.cc`.
    - [x] Tests:
      - [x] Python parsing/serialization: `tests/python/api/test_config_models.py`.
      - [x] Daemon resolution: `daemon/store_policy_resolver_test.cc`.
  - [x] If we do **not** add a profile, ensure docs/examples use explicit tier specs (N/A; profile `warm` added).

- [x] Phase 1: Unify SDK policy entry points (register + put)
  - [x] Add `policy: StorePolicy | str | None` to `Store.register/register_async`: `tensorcast/api/store/registration.py`.
  - [x] Add `policy: StorePolicy | str | None` to `Store.put/put_async`: `tensorcast/api/store/registration.py`.
  - [x] Precedence rule: reject conflicting `policy` vs `options.policy` after normalization (avoid silent divergence): `tensorcast/api/store/registration.py`.
  - [x] Update module-level wrappers to forward `policy`: `tensorcast/api/store/__init__.py`.
  - [x] Update public-surface tests to assert the new signatures (and that legacy paths still work): `tests/python/api/test_public_surface.py`.

- [x] Phase 2: Proto surface for “local stable tier result” (mirror-free)
  - [x] Extend `CommitRegisteredArtifactResponse` with `local_stable_tier` result (READY/DEGRADED/SKIPPED): `proto/tensorcast/daemon/v1/store_daemon.proto`.
  - [x] Regenerate Python stubs after proto edits: `bash tools/build_proto_python.sh`.
  - [x] Map the proto result into the SDK result shape (prefer nesting under existing `registration_result`/commit result): `tensorcast/api/store/registration.py`, `tensorcast/api/store/handles.py`.
  - [x] Add SDK mapping tests using stubs (patterned after persistence wire-up tests): `tests/python/api/test_persistence_wireup.py` (or a new focused test file).

- [x] Phase 3: Daemon commit-time local stable tier satisfaction
  - [x] Add commit-time hook in registration commit path:
    - [x] Persist policy intent from `BeginRegisterArtifactRequest.policy` into registration meta so `CommitRegisteredArtifact` can enforce local stable semantics for lease registrations:
      - [x] Option A (preferred): store the daemon-resolved `ResolvedStorePolicy` (or just `local_requirement` + derived `StableDramCachePolicy`) in `RegistrationManager::RegMeta`: `daemon/registration_manager.h`, `daemon/service/controllers/registration_controller.cc`.
      - [x] Option B: store the raw `StorePolicy` proto in `RegMeta` and resolve on commit (not selected; Option A implemented).
    - [x] Resolve the effective policy using existing resolver (single source of truth): `daemon/store_policy_resolver.*`.
    - [x] If local requirement `< should`, return `SKIPPED` and perform no extra work.
    - [x] If local requirement is `should|must`, ensure a local stable replica exists:
      - [x] If stable already exists (or plan is `dram_stable`), apply/upgrade retention via `StableDramCacheManager` and return `READY`.
      - [x] Otherwise materialize stable from the committed base replica (LIP/coalesced) using existing materialization/runtime helpers: `daemon/service/controllers/registration_controller.cc`, `core/store/materialization/*`.
  - [x] Admission/retention:
    - [x] Use `StableDramCacheManager` with the resolved `StableDramCachePolicy` (no new knobs): `core/store/components/stable_dram_cache_manager.*`.
    - [x] Enforce `must` as fail-fast (commit fails), `should` as degraded (commit succeeds + `DEGRADED` result).
  - [x] Region-backed lease sources (Design 0021):
    - [x] Resolve segments via `storage_id` and hold region refs for the copy duration: `daemon/lip_manager.*`, `daemon/ipc_region_registry.*`.
  - [x] Tests:
    - [x] Extend daemon registration tests to cover `should` degraded vs `must` failure: `daemon/grpc_service_impl_registration_test.cc` (or a new targeted test).

- [x] Phase 4: Persistence source preference (daemon)
  - [x] Prefer local stable DRAM as persistence source when available; fall back to lease source only when no stable source exists: `daemon/persistence_manager.*`.
  - [x] Tests:
    - [x] Add daemon-level regression coverage for source selection when both lease and stable exist: `daemon/persistence_manager_test.cc`.

- [x] Phase 5: Observability + docs sync
  - [x] Add metrics/logging using “local stable tier” terminology (never “mirror”): daemon metrics infra + structured logs near registration commit.
  - [x] Update module docs impacted by behavior/API changes (doc sync rule):
    - [x] `tensorcast/api/README.md` (register/put policy surface).
    - [x] `daemon/README.md` (commit-time local stable tier satisfaction and persistence source preference).
    - [x] Any internal docs that describe register/put flows (if present).

# Validation

Run the commands in “Test Plan” and confirm the “Acceptance Checks”.

# Acceptance Checks

- `register` and `put` both accept a first-class `policy` argument; conflicting inputs with `options.policy` are rejected deterministically.
- `register` can satisfy `stable_dram(scope=local)` during commit when policy requests it, and reports `READY/DEGRADED/SKIPPED` via `local_stable_tier`.
- Persistence still follows Design 0041/0044; when a local stable replica exists, persistence reads from it (no lease-only coupling).
- No new runtime config knobs are introduced; stable budget remains `engine.memory_tiers.stable_bytes`.

# Test Plan

- Python:
  - `uv run ruff check .`
  - `uv run pytest tests/python/api/test_config_models.py`
  - `uv run pytest tests/python/api/test_public_surface.py`
  - `uv run pytest tests/python/api/test_persistence_wireup.py`
  - `uv run pytest tests/python/test_daemon_ctl_commit_result.py`
- C++:
  - `bazel test //daemon:store_policy_resolver_test`
  - `bazel test //daemon:grpc_service_impl_registration_test`
  - `bazel test //daemon:persistence_manager_test`
  - `bazel test //core/store:stable_dram_cache_manager_test`

# Rollout / Backout

- Rollout: land proto+daemon+SDK changes together, regenerate stubs, then run the full test plan.
- Backout: revert to pre-0045 behavior (no commit-time local stable tier) and keep the SDK `policy` surface gated behind compatibility aliases if needed.

# Risks & Tracking

- Risk: adding a new profile (`warm`) requires strict SDK↔daemon alignment; mitigate with daemon-side resolution tests and shared profile fixtures.
- Risk: commit-time copy latency surprises users; mitigate with profile defaults (`should` for `warm`) and explicit `must` only for `pinned`.
- Risk: region-backed source handling bugs (lifetime/permission); mitigate with targeted daemon tests around region ownership and bounds.
