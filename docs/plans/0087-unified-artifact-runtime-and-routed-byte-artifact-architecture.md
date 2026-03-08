---
slug: unified-artifact-runtime-and-routed-byte-artifact-architecture
title: Plan - Unified Artifact Runtime and Routed Byte Artifact Architecture
status: completed
areas: ["daemon", "sdk", "global_store", "proto", "docs"]
created: 2026-03-08
last_updated: 2026-03-08
related_code:
  - docs/designs/0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
  - daemon/service/grpc_service_impl.cc
  - daemon/service/grpc_service_impl_rpc_delegates.cc
  - daemon/service/controllers/byte_artifact_controller.cc
  - daemon/service/controllers/external_target_access_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - daemon/service/controllers/target_publish_service.cc
  - daemon/service/payload_transport_broker.cc
  - daemon/state/daemon_kernel.h
  - docs/README.md
links:
  design: ../designs/0087-unified-artifact-runtime-and-routed-byte-artifact-architecture.md
---

# Objective

Track the completion work for the consolidated artifact-runtime design line.

The large semantic and architectural cutover was already in place when this plan was opened. The remaining work was
intentionally narrow:

- remove dead legacy runtime code that no longer owns live paths,
- finish doc and reference cleanup around the new canonical `0087` design,
- confirm that selection-normalization and artifact-profile coverage remain sufficient after consolidation.

# Current State & Grounding

Current baseline as of 2026-03-08:

- `StoreDaemonServiceImpl` live batch paths already delegate through `ByteArtifactController` and `TransportController`.
- `DaemonKernel` already owns routed byte-artifact body, route, payload-transport, and worker-directory state.
- `ExternalTargetAccessService` is already reused by both materialization flows and byte-artifact ingress flows.
- publication naming is already cut over to `target_publication_token`, `TargetPublicationScope`, and
  `TargetPublicationRegistry`.
- `PayloadRefScope` is already the typed signed-only payload transport capability.

Completion summary as of 2026-03-08:

- the legacy pre-controller batch implementation block is no longer present in `daemon/service/grpc_service_impl.cc`,
- active docs now treat `0087` as the single canonical artifact-runtime design and use
  `target_publication_token` / `TargetPublicationScope` naming,
- repository-wide searches no longer find `0084`, `0085`, or `0086` design-line filenames,
- targeted Python validation passed, and no extra byte-artifact selection-normalization parity tests were required.

# Phases & Milestones

- [x] Phase 1: Remove stale runtime leftovers
  - [x] Milestone 1.1: Delete the dead pre-controller batch implementation block under `#if 0` in
    `daemon/service/grpc_service_impl.cc`.
  - [x] Milestone 1.2: Remove any now-unreferenced helper code that existed only for the superseded batch path.

- [x] Phase 2: Consolidate docs on the canonical design
  - [x] Milestone 2.1: Point active docs and plans at `0087` as the single canonical artifact-runtime design.
  - [x] Milestone 2.2: Remove superseded `0084`, `0085`, and `0086` design and plan documents.
  - [x] Milestone 2.3: Finish wording cleanup for old split-line language and stale pre-publication terminology in
    active docs.

- [x] Phase 3: Final verification and optional parity hardening
  - [x] Milestone 3.1: Confirm no repository docs still reference the removed `0084`, `0085`, or `0086` files.
  - [x] Milestone 3.2: Re-evaluate byte-artifact selection-normalization parity coverage and add targeted tests only if
    a real gap remains.
  - [x] Milestone 3.3: Re-run targeted validation if Phase 1 changes touch live daemon code.

# Tasks

- Runtime cleanup:
  - remove dead batch code from `grpc_service_impl.cc`
  - keep live controller delegates unchanged

- Documentation cleanup:
  - update `docs/README.md`
  - update dependent designs and plans that still point to the old design line
  - make `0087` the only active design entry for this runtime area

- Verification:
  - use repository-wide searches to confirm stale design references are gone
  - only run daemon builds or tests if cleanup touches non-doc code

# Acceptance Checks

- repository-wide search for the removed 0084/0085/0086 design-line filenames returns no remaining hits
- `rg` finds no stale active-doc references to superseded publication-token terminology in the consolidated runtime docs
- targeted Python validation passed:
  - `pytest tests/python/test_byte_artifact_identity.py tests/python/test_selection_identity_vectors.py tests/python/test_kvcache_adapter.py tests/python/test_binding.py tests/python/test_inplace_slot.py`
  - `pytest tests/python/global_store/test_shard_home_lease_rpc.py tests/python/global_store/test_grpc_service.py tests/python/node_agent/test_plan_execution.py tests/python/api/test_plan_spec.py tests/python/api/test_mapped_binding.py`
- if `grpc_service_impl.cc` is edited beyond dead-code deletion, run:
  - `bazel build //daemon:grpc_service_impl`
  - `bazel build //daemon:grpc_service_impl_batch_runtime_test`
  - `bazel build //daemon:grpc_service_impl_batch_redirect_e2e_test`

# Test / Rollout / Backout

This plan is primarily documentation consolidation plus small cleanup.

Rollout:

- treat `0087` as the single canonical design for this runtime area,
- keep runtime behavior unchanged while removing stale docs and dead code.

Backout:

- if consolidation uncovers missing context, restore specific explanatory text in `0087` rather than reviving the old
  split design line.

# Risks & Tracking

- [x] Dead-code deletion in `grpc_service_impl.cc` must not accidentally remove symbols still used by tests or private
  helpers.
- [x] Doc cleanup must preserve the current implementation facts rather than reintroducing historical split semantics.
- [x] Additional selection-normalization tests should be added only when they cover a real remaining gap.
