---
slug: client-generated-artifact-id
title: Client-Generated Artifact ID (Plan)
areas: ["core","daemon","global_store","sdk","proto"]
links:
  design: ../designs/0017-client-generated-artifact-id.md
related_code:
  - core/store/**
  - daemon/**
  - tensorcast/global_store/**
  - tensorcast/api/**
---

# Objective

Deliver client-generated artifact identifiers (CGID) end-to-end so SDKs can provide trusted `cgid:` values, the daemon registers them without server-side hashing, and the Global Store tracks residency and routing with appropriate schema updates while retaining mi2 compatibility.

# Phases & Milestones

- [x] Phase 0: Protocol & Validation Foundations
  - [x] Milestone 0.1: Extend `BeginRegisterArtifactRequest` in `proto/tensorcast/daemon/v1/store_daemon.proto` with optional `client_artifact_id` and plumb through `CommitRegisteredArtifactResponse` / `tensorcast.common.v1.ArtifactDescriptor`; regenerate code via `tools/build_proto_python.sh` and `bazel build //proto/...`.
  - [x] Milestone 0.2: Add shared artifact ID helper (`tensorcast/common/identity.py` for Python, `core/common/artifact_identity.h` for C++) that enforces CGID grammar and is consumed by `tensorcast/api/_register.py`, `daemon/lip_manager.cc`, and `tensorcast/global_store`.

- [x] Phase 1: SDK Path Enablement
  - [x] Milestone 1.1: Update `tensorcast/api/store.py` and `_register.py` so `Store.register()` accepts `artifact_id="cgid:..."`, skips mi2 hashing in `_register_artifact_core`, and plumbs the new identity kind from `CommitResult`.
  - [x] Milestone 1.2: Add CGID-focused coverage in `tests/python/test_register_cgid.py` (new) or extend `tests/python/test_store_session_api.py`, asserting invalid prefix rejection and that `RegisteredArtifact.artifact_id` mirrors the provided CGID.

- [x] Phase 2: Daemon & Core Registration Flow
  - [x] Milestone 2.1: Store CGID intent in `daemon/types.h::LipLeaseEntry`, branch `daemon/lip_manager.cc::CommitLease` to bypass `common::compute_*_multihash` when `client_artifact_id` is set, and populate `CommitLeaseResult` accordingly.
  - [x] Milestone 2.2: Update `daemon/grpc_service_impl.cc` and `daemon/service/controllers/registration_controller.cc` to validate CGID via the shared helper, record `identity_kind` metric labels in `daemon/metrics.*`, and keep mi2 flow untouched.
  - [x] Milestone 2.3: Extend `core/store/store_engine_test.cc` and `daemon/grpc_service_impl_registration_test.cc` to assert both mi2 and CGID commits succeed, including mixed-device scenarios.

- [x] Phase 3: Global Store Persistence & Routing
  - [x] Milestone 3.1: Update `schema.sql` (artifacts table adds `id_kind`, digest columns nullable; `artifact_replicas` gains `expires_at`) and refresh bootstrap `tensorcast/global_store/init.sql`; introduce a DuckDB migration helper module (e.g., `tensorcast/global_store/migrations/0017_cgid.py`) so existing deployments can apply the DDL.
  - [x] Milestone 3.2: Modify `tensorcast/global_store/services/artifact_service.py` and repository layers (`repositories/artifact_repository.py`, `repositories/replica_repository.py`, `repositories/key_mapping_repository.py`) so lookup/write paths persist `id_kind`, omit mi2-only columns for CGID, and expose identity kind through RPC payloads.
  - [x] Milestone 3.3: Extend integration coverage in `tests/python/global_store/test_artifacts.py` (or sibling suite) for CGID registrations, ensuring key mappings (`key_mappings` table) and replica listings work with mixed ID kinds.

- [x] Phase 4: Rollout, Compatibility & Documentation
  - [x] Milestone 4.1: Update relevant READMEs/AGENTS with CGID guidance and rollout guardrails.
  - [x] Milestone 4.2: Finalize production rollout guidance—CGID is enabled by default (no feature flag), document operational metrics, and spell out backout procedures.

# Tasks

- Define `ArtifactIdKind` utilities shared across SDK (`tensorcast/common/identity.py`), daemon (`core/common/artifact_identity.h/.cc`), and GS validation logic.
- Align SDK response models (`tensorcast/types.py::ArtifactDescriptor`, any dataclasses referenced in `tensorcast/api/store.py`) with new `identity_kind` field and nullable digests.
- Adjust daemon RPC handlers (`daemon/service/controllers/registration_controller.cc`, `daemon/grpc_service_impl.cc`) to accept `client_artifact_id`, populate `CommitRegisteredArtifactResponse.artifact_descriptor` with CGID, and only compute mi2 digests when missing; tag OpenTelemetry counters in the same controller with `identity_kind`.
- Wire daemon-to-GS update calls (registration controller interactions with `d_.reg` / `d_.engine` plus any `global_store` client hooks) to send identity kind and TTL metadata; update GS ingestion accordingly.
- Add Bazel targets/deps exposing the new identity helper header (`core/common/BUILD.bazel`, `core/store/BUILD.bazel`) for reuse in store engine and tests.
- Create migration script for DuckDB deployments applying schema alterations with reversible DDL.
- Refresh documentation indexes listing CGID support and cross-link plan/design.
- Remove transitional compatibility toggles so CGID code paths are always available; callers supply CGID identifiers without any gating.

# Test / Rollout / Backout

- Unit: `uv run pytest tests/python/test_register_cgid.py` (executed).
- Integration: `uv run pytest tests/python/global_store/test_grpc_service.py::TestGRPCService::test_register_replica_with_cgid` (executed) and extend broader suites as needed.
- C++: `bazel test //daemon:session_lifecycle_test //core/store:store_engine_test --define=use_fake_cuda=true` (executed).
- Proto/Buf validation: `bazel test //proto/... --test_output=streamed`.
- Migration dry-run: apply DuckDB schema patch on staging snapshot, verify mixed mi2/CGID records.
- Rollout: deploy the CGID-capable daemon/GS builds directly; monitor `register_latency{identity_kind="cgid"}` and replica residency metrics to confirm healthy adoption.
- Backout: block CGID registrations at the daemon (reject `client_artifact_id`) and revert the schema migration using `tensorcast/global_store/migrations/0017_cgid.py::downgrade`, restoring pre-change artifacts if necessary.

# Risks & Tracking

- Collision or misuse of CGID values: require tenancy-scoped prefixes and add monitoring for duplicate registrations per namespace.
- Mixed-client compatibility: ensure SDK version gating—older clients must default to mi2; publish upgrade note.
- Schema migration fallout: mitigate with migration dry-run, automated backups, and revert script.
- Observability drift: coordinate metric name changes with dashboards; flag in release checklist.
