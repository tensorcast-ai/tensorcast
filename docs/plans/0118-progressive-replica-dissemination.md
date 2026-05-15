---
slug: progressive-replica-dissemination
title: Progressive Replica Dissemination Plan
status: in-progress
areas: ["global_store", "daemon", "core", "sdk", "proto", "docs", "tests", "benchmarks"]
created: 2026-05-11
last_updated: 2026-05-15
related_code:
  - docs/designs/0118-progressive-replica-dissemination.md
  - docs/designs/0117-group-realization-transaction.md
  - tensorcast/schema.sql
  - proto/tensorcast/global_store/v1/global_store.proto
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - proto/tensorcast/config/v1/global_store_config.proto
  - proto/tensorcast/config/v1/daemon_config.proto
  - tensorcast/global_store/services/transport_service.py
  - tensorcast/global_store/repositories/replica_repository.py
  - tensorcast/global_store/repositories/transport_repository.py
  - core/store/materialization/control/materialize_orchestrator.cc
  - core/store/materialization/contracts/loading_spec.h
  - core/store/components/global_store_client.h
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/service/controllers/target_materialization_service.cc
links:
  design: ../designs/0118-progressive-replica-dissemination.md
  dependencies:
    - ../designs/0117-group-realization-transaction.md
---

# Objective

Implement progressive partial-source dissemination as a separate follow-on to
group realization:

- in-progress receivers report verified prefix coverage;
- Global Store exposes that coverage only through a progressive atomic
  assignment path;
- targets read safe prefix segments and repeat until complete;
- ordinary complete-replica availability remains unchanged.

# Current State And Grounding

The current repository already contains:

- complete replica availability in `artifact_replicas`;
- source selection in `ReplicaRepository.find_available_for_transport`;
- group dispatch and source spread metrics from `0083`;
- transport completion outcomes and request idempotency;
- export metadata and worker heartbeat checks.
- Global Store control-plane saturation evidence from current code and
  benchmark notes: repository transactions are serialized under the global DB
  execution lock, ordinary transport dispatch already has a centralized
  dispatch lock, and request-rate/write-amplification pressure can saturate the
  server.

Important gaps:

- closed in the 2026-05-15 control-plane pass: progressive coverage tables,
  coverage identity fields, durable assignment/counter tables, Global Store
  repository/service/RPC paths, source-claim fencing, and default cross-domain
  TCP seed skipping;
- closed in the 2026-05-15 daemon/benchmark pass: target progressive segment
  read loop, typed requester/source-domain derivation, source read failure
  retirement, and cross-host WeightPublisher runner switches for the progressive
  lane;
- closed in the 2026-05-15 deep review continuation: complete target
  publications preserve grouped progressive identity and report terminal
  byte-prefix coverage only after ordinary registration plus state-sync barrier
  success;
- still open: repeated multi-host benchmark results and safe non-terminal
  in-flight source publication. The Global Store control plane accepts and
  schedules non-terminal verified-prefix coverage, but the daemon currently
  reports only terminal verified coverage for completed materializations and
  complete target publications because incomplete target/replica memory has no
  progressive-only remote-export contract. Publishing those buffers through
  ordinary `artifact_replicas` would violate the ordinary availability boundary.

# Latest Progress

## 2026-05-15 Control-Plane Pass

Completed:

- Added `replica_progress_coverage`, `progressive_source_assignments`, and
  `progressive_source_counters` to the canonical root `schema.sql`.
- Added progressive coverage identity, coverage report, source claim,
  assignment completion, retirement, and expiration RPCs to
  `ClusterRuntimeService`.
- Added Global Store progressive config and daemon progressive reporting config
  proto fields. Daemon `verify_before_report` uses proto presence so an omitted
  value keeps the safe default.
- Implemented `ProgressiveCoverageRepository`,
  `ProgressiveReplicationService`, and `ProgressiveRpcHandler`.
- Wired the progressive repository/service/RPC handler into
  `GlobalStoreServicer`, the ClusterRuntime mixin, maintenance expiration, and
  Prometheus metrics.
- Added C++ `GlobalStoreClient` wrappers for report, claim, assignment
  completion, coverage retirement, and expiration.
- Added `StoreEngineOptions` and `DaemonOptions` progressive config hooks, and
  mapped daemon config into both option structs.
- Added Python tests for schema presence, ordinary source-selection isolation,
  monotonic coverage updates, report throttling, durable replay,
  source-cap/counter separation, heartbeat/export-generation fencing,
  assignment/coverage expiration, cross-domain TCP seed skipping, and RPC
  report/claim wiring.

Validation run:

- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py`
  passed: 9 tests.
- `source .venv/bin/activate && pytest tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py`
  passed: 113 tests.
- `source .venv/bin/activate && ruff check tensorcast/global_store tests/python/global_store/test_progressive_replication.py`
  passed.
- `source .venv/bin/activate && ruff format --check tensorcast/global_store tests/python/global_store/test_progressive_replication.py`
  passed.
- `bazel build //core/store:global_store_client --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //proto/... --test_output=streamed --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.

Known validation caveat:

- `bash tools/build_proto_python.sh` still fails because the Buf remote plugin
  server reports unavailable. Python stubs were regenerated with the local
  `grpc_tools.protoc` fallback using the same proto input set.

## 2026-05-15 Daemon Terminal Coverage Pass

Completed:

- Wired daemon progressive replication config into the materialization
  controller and replica materialization service.
- Added daemon-side terminal coverage reporting for canonical byte-prefix
  materialization. The daemon reports only after materialization succeeds,
  verification is known when `verify_before_report=true`, Global Store is
  connected, the replica has a Global Store replica id, and remote export is
  active with a non-zero export generation.
- Derived `source_domain` from typed daemon/worker topology (`node_id`, falling
  back to `daemon_id`) rather than request-provided free-form strings for daemon
  coverage reports.
- Populated Global Store memory-replica registration transport metadata with
  the actual export generation. Progressive source claims fence against
  `artifact_replicas.export_generation`, so C++ registration must preserve this
  value for source eligibility.
- Added a C++ metadata gateway regression test that verifies memory-replica
  registration preserves the runtime export generation used by progressive
  fencing.
- Extended progressive source assignments to return source `MemoryInfo`
  metadata. This gives the future target read loop the exact source endpoint,
  memory keys, buffer sizes, byte space, and export generation from the
  progressive claim instead of re-entering ordinary complete-replica source
  selection.

Validation run:

- `bazel build //core/store:global_store_client --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //core/store/runtime/metadata:metadata_gateway_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //proto/... --test_output=streamed --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py`
  passed: 9 tests.
- `source .venv/bin/activate && ruff check tensorcast/global_store tests/python/global_store/test_progressive_replication.py`
  passed.
- `source .venv/bin/activate && ruff format --check tensorcast/global_store tests/python/global_store/test_progressive_replication.py`
  passed.
- `git diff --check` passed.

## 2026-05-15 Target Loop And Benchmark Lane Pass

Completed:

- Added the `MaterializeIntoTarget` progressive byte-prefix read loop. When
  daemon progressive replication is enabled, the target path constructs the same
  canonical coverage identity as daemon coverage reports, claims the next safe
  prefix from Global Store, copies only the assigned byte range through
  `P2PLoader` and StoreEngine mapped target writes, completes durable
  assignments, and repeats until the target is complete.
- Kept progressive source selection separate from ordinary complete-replica
  selection. If no progressive source is available before any progressive write,
  the path falls back only to the existing ordinary policy; once progressive has
  written target bytes, failures return a hard error and poison the target
  lease.
- Enforced identity and range invariants at the target loop: canonical
  byte-prefix only, no view transform/subset, matching selection/layout/hash
  identity, monotonic prefix cursor, and terminal-tail assignment semantics.
- Retired failed source coverage after progressive source read/setup failures
  and completed assignment counters for both success and failure paths.
- Derived requester `source_domain` from daemon worker topology
  (`node_id`, falling back to `daemon_id`) for target claims.
- Added explicit progressive benchmark lane switches to
  `examples/cross_host/cross_host_weight_publisher_runner.py` and
  `examples/cross_host/run_multihost_weight_publisher_suite.sh`, plus default
  disabled progressive config stanzas in the cross-host Global Store and daemon
  configs.
- Updated WeightPublisher deployment docs with the progressive benchmark lane
  controls.
- Updated Global Store and daemon READMEs with default-off progressive
  replication config, enablement requirements, and source/target behavior
  boundaries.
- Confirmed the remaining non-terminal in-flight coverage item is blocked on a
  separate progressive-only publication/export contract, not on the Global Store
  assignment path. The current implementation does not mark incomplete targets
  as ordinary available replicas.

Validation run:

- `bazel build //daemon:target_materialization_service_lib --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed: 148 tests.
- `bazel test //proto/... --test_output=streamed --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //core/store/runtime/metadata:metadata_gateway_test //core/store:store_engine_test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `source .venv/bin/activate && ruff check examples/cross_host/cross_host_weight_publisher_runner.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tensorcast/global_store tests/python/global_store/test_progressive_replication.py`
  passed.
- `source .venv/bin/activate && ruff format --check examples/cross_host/cross_host_weight_publisher_runner.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tensorcast/global_store tests/python/global_store/test_progressive_replication.py`
  passed.
- `bash -n examples/cross_host/run_multihost_weight_publisher_suite.sh`
  passed.
- `git diff --check` passed.

Known validation caveat:

- `source .venv/bin/activate && bash tools/build_proto_python.sh` still fails
  with `Failure: the server hosted at that remote is unavailable.` The local
  `grpc_tools.protoc` fallback was rerun successfully after the failed official
  script attempt.

## 2026-05-15 Failure-Injection Benchmark Lane Pass

Completed:

- Added an explicit failure-injection plan to the cross-host WeightPublisher
  runner. `--failure-injection-mode stop-daemon` currently supports
  `receiver:<zero-based-index>` targets, waits a configured delay after
  publisher start, stops the selected receiver daemon, records injection
  timing/status in the case summary, and excludes the expected interrupted
  receiver from ordinary completion/probe gates while keeping the remaining
  receivers strict.
- Added suite controls for an optional standalone failure-injection case:
  `TC_WP_FAILURE_INJECTION_ENABLE`,
  `TC_WP_FAILURE_INJECTION_MODE`,
  `TC_WP_FAILURE_INJECTION_TARGET`,
  `TC_WP_FAILURE_INJECTION_DELAY_S`, and
  `TC_WP_FAILURE_INJECTION_RECEIVER_COUNT`. The lane is skipped with an
  explicit manifest entry unless enabled and at least two receivers are
  available.
- Updated WeightPublisher deployment docs with failure-injection controls and
  delay-tuning guidance.
- Added runner unit tests for failure-injection target resolution and daemon
  stop execution.

Validation run:

- `source .venv/bin/activate && pytest tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed: 29 tests.
- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed: 38 tests.
- `source .venv/bin/activate && pytest tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py`
  passed: 113 tests.
- `source .venv/bin/activate && ruff check examples/cross_host/cross_host_weight_publisher_runner.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed.
- `source .venv/bin/activate && ruff format --check examples/cross_host/cross_host_weight_publisher_runner.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed.
- `bash -n examples/cross_host/run_multihost_weight_publisher_suite.sh`
  passed.
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //core/store/runtime/metadata:metadata_gateway_test //core/store:store_engine_test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //proto/... --test_output=streamed --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `source .venv/bin/activate && bash tools/build_proto_python.sh` still failed
  with `Failure: the server hosted at that remote is unavailable.`
- `source .venv/bin/activate && python -m grpc_tools.protoc ...` local fallback
  regeneration succeeded.
- `git diff --check` passed.

## 2026-05-15 Benchmark Report Tooling Pass

Completed:

- Added progressive/failure-injection metadata to cross-host WeightPublisher
  case JSON summaries and params so benchmark artifacts are self-describing.
- Extended `examples/cross_host/summarize_scaleout_suite.py` to classify TP
  cases as baseline, progressive, failure-injection, or
  progressive-failure-injection lanes and to emit per-lane aggregate metrics for
  source concentration, publish-to-apply p95, and peak active throughput.
- Added a summarizer unit test with synthetic baseline/progressive/failure
  case JSON.
- Updated WeightPublisher deployment docs to point at the summarizer for
  progressive on/off reports.

Validation run:

- `source .venv/bin/activate && pytest tests/python/tools/test_summarize_scaleout_suite.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed: 30 tests.
- `source .venv/bin/activate && ruff check examples/cross_host/summarize_scaleout_suite.py examples/cross_host/cross_host_weight_publisher_runner.py tests/python/tools/test_summarize_scaleout_suite.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed.
- `source .venv/bin/activate && ruff format --check examples/cross_host/summarize_scaleout_suite.py examples/cross_host/cross_host_weight_publisher_runner.py tests/python/tools/test_summarize_scaleout_suite.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py`
  passed.
- `git diff --check` passed.

## 2026-05-15 Review Hardening Pass

Completed:

- Strengthened Global Store progressive report validation so same-`coverage_id`
  updates cannot change replica identity, source export generation,
  materialization attempt identity, or total coverage bounds even when a report
  would otherwise be coalesced by throttling.
- Changed report throttling to always persist export-state transitions; a source
  moving from in-progress exportable to complete exportable is no longer hidden by
  byte-delta/time-interval coalescing.
- Enforced typed `source_domain` at the service boundary. Source reports and
  requester claims now require the supplied domain to match the corresponding
  worker directory entry (`node_id`, falling back to `daemon_id`), so the
  cross-domain policy cannot be bypassed with request-provided strings. Disabled
  source claims still return the cheap `progressive_disabled` result without
  requiring a requester directory row.
- Extended default cross-domain seed protection to the daemon-reported `p2p`
  seed kind as well as explicit `tcp`, because terminal daemon coverage currently
  records `MaterializationSource::kP2P` rather than a lower-level communicator
  protocol.
- Rechecked source worker directory state at assignment time. Candidate coverage
  is no longer eligible if the source worker's current daemon/domain or replica
  owner no longer matches the coverage row.
- Tightened request-fingerprint replay so a durable assignment is replayed only
  for the same coverage identity, requester, materialization attempt, and cursor;
  mismatched reuse returns `request_fingerprint_conflict`.
- Fixed expiration cleanup for active assignments whose coverage expires first.
  Coverage expiration now cancels those assignments and releases
  `progressive_source_counters` in the same sweep.
- Removed the physical DuckDB foreign key from
  `progressive_source_assignments.coverage_id`. DuckDB rejects updates to a
  referenced coverage row even for non-key state changes, so coverage retirement
  keeps the relationship as an indexed application-level invariant.
- Hardened the target progressive read loop against idempotent terminal
  assignment replays. `CLAIMED` and `READING` assignments remain readable;
  terminal assignment states are not treated as new source segments.
- Added tests for throttling-vs-identity updates, export-state transition
  persistence, worker-directory source-domain validation, `tcp`/`p2p`
  cross-domain seed skipping, disabled-claim fast path, directory-drift
  ineligibility, fingerprint replay conflicts, coverage-expiration assignment
  cleanup, and terminal assignment replay after completion.

Validation run:

- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py`
  passed: 15 tests.
- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tests/python/tools/test_summarize_scaleout_suite.py`
  passed: 158 tests.
- `source .venv/bin/activate && ruff check tensorcast/global_store examples/cross_host tests/python/global_store/test_progressive_replication.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tests/python/tools/test_summarize_scaleout_suite.py`
  passed.
- `source .venv/bin/activate && ruff format --check tensorcast/global_store examples/cross_host tests/python/global_store/test_progressive_replication.py tests/python/tools/test_cross_host_weight_publisher_runner_transport_probe.py tests/python/tools/test_summarize_scaleout_suite.py`
  passed.
- `bazel build //daemon:target_materialization_service_lib --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `git diff --check` passed.

## 2026-05-15 Compatibility Cleanup Pass

Completed:

- Standardized progressive assignment payloads on
  `MemoryInfo.transport.remote_memory_keys`, `buffer_sizes`, and
  `verification_json`. Removed the legacy flat `MemoryInfo` transport fields
  from the common proto and reserved their field numbers/names.
- Removed the daemon progressive P2P source fallback that accepted flat
  `MemoryInfo.remote_memory_keys`/`buffer_sizes`. Progressive reads now require
  explicit `transport` metadata and validate key/size cardinality, non-zero
  buffer sizes, and total buffer bytes against `memory_size`.
- Standardized replica memory serialization on the same transport metadata path.
  `replica_to_memory_info` now writes export metadata only under
  `MemoryInfo.transport`, and `GlobalStoreClient::convert_from_proto_memory_info`
  reads communicator keys only from `transport`.
- Removed the `preserve_transport` update path from replica registration. Ordinary
  C++ `register_replica` requests now send an explicit presence-only transport,
  and Global Store updates transport state from each request through one path.
- Simplified recovery inventory fingerprinting and reconcile updates so
  transport metadata is compared and rewritten through the canonical
  `MemoryInfo.transport` path instead of mirroring legacy flat fields.
- Replaced the internal flattened `ProgressiveAssignment.source_*` source-memory
  fields with `source_memory` and `source_memory.transport`, matching the RPC
  payload shape and removing another source of parallel naming.
- Updated tests to assert transport metadata through `MemoryInfo.transport`.
- Attempted the official `bash tools/build_proto_python.sh` generation path; Buf
  remote plugins were still unavailable. Regenerated the affected Python common
  proto locally with `grpc_tools.protoc`.

Validation run:

- `source .venv/bin/activate && pytest tests/python/global_store/test_replica_memory_codec.py tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py`
  passed: 133 tests.
- `source .venv/bin/activate && pytest tests/python/global_store`
  passed: 349 tests.
- `source .venv/bin/activate && pytest tests/python/global_store tests/python/test_transport.py`
  passed: 353 tests.
- `source .venv/bin/activate && python -m grpc_tools.protoc -I proto --python_out=proto/gen/python --grpc_python_out=proto/gen/python --pyi_out=proto/gen/python proto/tensorcast/common/v1/common.proto`
  passed after the official Buf remote generation path returned unavailable.
- `source .venv/bin/activate && ruff check tensorcast/global_store tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_replica_memory_codec.py`
  passed.
- `source .venv/bin/activate && ruff format --check tensorcast/global_store tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_replica_memory_codec.py`
  passed.
- `source .venv/bin/activate && ruff check tensorcast/global_store tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_replica_memory_codec.py tests/python/test_transport.py`
  passed.
- `source .venv/bin/activate && ruff format --check tensorcast/global_store tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_replica_memory_codec.py tests/python/test_transport.py`
  passed.
- `bazel build //core/store:global_store_client //daemon:target_materialization_service_lib --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `git diff --check` passed.

## 2026-05-15 Deep Review Hardening Pass

Completed:

- Rejected progressive coverage reports when Global Store progressive
  replication is disabled, while keeping disabled source claims as cheap
  no-source responses.
- Required explicit progressive visibility state and export state at the RPC
  boundary. The handler no longer infers verified/exportable state from byte
  counts when enum fields are unspecified.
- Capped client-supplied coverage and assignment deadlines by the configured
  Global Store TTLs so request budgets cannot extend source visibility or keep
  progressive counters held beyond policy.
- Made same-epoch coverage report retries idempotent only for exact replayed
  content, and made `daemon_id`, `worker_id`, and `source_domain` immutable for
  an existing coverage row.
- Changed invalid `source_domain_policy` values to fail closed before candidate
  scanning instead of behaving like an implicit allow mode.
- Added a claim-path identity index covering the hot progressive candidate
  predicates used by `find_progressive_source`.
- Changed the daemon byte-prefix coverage order marker to a SHA-256 digest and
  changed target request fingerprints to length-prefixed binary fields so
  binary selection/layout hashes cannot collide through delimiter ambiguity.
- Added terminal progressive coverage reporting for completed target
  publications. Target publication records now preserve grouped coverage
  identity, and the publish path reports coverage only for canonical full
  byte-space publications after ordinary registration and state-sync barrier
  success.

Validation and profiling run:

- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py -q`
  passed: 20 tests.
- `source .venv/bin/activate && ruff check tensorcast/global_store/services/progressive_service.py tensorcast/global_store/rpc/progressive_rpc_handler.py tests/python/global_store/test_progressive_replication.py`
  passed.
- `source .venv/bin/activate && ruff format --check tensorcast/global_store/services/progressive_service.py tensorcast/global_store/rpc/progressive_rpc_handler.py tests/python/global_store/test_progressive_replication.py`
  passed.
- `source .venv/bin/activate && pytest tests/python/global_store/test_progressive_replication.py tests/python/global_store/test_services.py tests/python/global_store/test_grpc_service.py -q`
  passed: 133 tests.
- `bazel test //proto/... --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel build //daemon:replica_materialization_service_lib --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //daemon:materialize_into_target_validation_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel test //daemon:grpc_service_impl_publish_target_replica_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
  passed.
- `git diff --check` passed.
- In-memory DuckDB microbenchmark with 512 coverage rows and 256
  claim/complete cycles: report mean/p95 `4.96/5.70 ms`, claim mean/p95
  `17.60/19.90 ms`, complete mean/p95 `3.78/3.92 ms`. This keeps the expected
  control-plane budget acceptable for coalesced byte-prefix segments, but
  confirms the design should not report at chunk, tensor, or CUDA-event
  granularity.

# Phases And Milestones

- [ ] Phase 0: Dependency and baseline closure
  - [x] Milestone 0.1: Confirm `0117` staged publish barrier and
        `GroupVersionSet` part-selection identity are stable before grouped
        serving benchmark lanes enter rollout.
  - [x] Milestone 0.2: Re-run the `0117` drift audit and require no active
        `tp_version`, `#tcg:`, operation-id group metadata, or relaxed
        transport-group artifact/view semantics in code, tests, runners, proto,
        or schema. Rechecked on 2026-05-15 with repository search excluding
        docs.
  - [ ] Milestone 0.3: Record old fanout baseline for source HHI, top-1 share,
        p95/p99 apply latency, timeout count, and grouped transport ratio.
  - [x] Milestone 0.4: Confirm v1.0 is byte-prefix only. Tensor-prefix waits for
        canonical tensor order hash owners and tests.
  - [x] Milestone 0.5: Freeze the progressive control-plane budget:
        coverage reports are coalesced by time/byte delta, source claims are
        indexed bounded atomic claims, target loops do not poll while reads are
        in flight, and no Global Store operation is proportional to copy chunks,
        RDMA completions, tensor count, or bytes inside an assigned segment.

- [x] Phase 1: Schema, proto, and config foundation
  - [x] Milestone 1.1: Add `replica_progress_coverage` to
        `tensorcast/schema.sql` with one active row per materialization attempt,
        replica, and source export generation.
  - [x] Milestone 1.2: Add `progressive_source_assignments` to
        `tensorcast/schema.sql`, plus `progressive_source_counters` for source
        cap admission separate from ordinary `replica_counters`.
  - [x] Milestone 1.3: Add schema smoke tests for coverage identity fields,
        active-row constraints, assignment idempotency, indexes, and insert
        constraints.
  - [x] Milestone 1.4: Add proto messages for coverage identity, coverage
        reporting, atomic source claim, assignment completion, retirement, and
        failure.
  - [x] Milestone 1.4a: Include `group_version_set_id` and `group_part_id` in
        coverage identity. They are required for grouped flows and empty only for
        ungrouped single-artifact fanout.
  - [x] Milestone 1.5: Add Global Store and daemon config proto fields for
        progressive enablement, TTLs, report cadence, source caps, and
        cross-domain seed policy, including minimum/maximum assignment bytes,
        maximum assignments per materialization, candidate scan limit, claim
        QPS cap per daemon, and cleanup batch size.
  - [x] Milestone 1.6: Run `bash tools/build_proto_python.sh`.
        Attempted on 2026-05-15; blocked by Buf remote plugin server
        unavailability. Local `grpc_tools.protoc` fallback succeeded.

- [x] Phase 2: Global Store progressive coverage observation
  - [x] Milestone 2.1: Implement `ProgressiveCoverageRepository`.
  - [x] Milestone 2.2: Implement `report_progressive_coverage` with monotonic
        coverage epoch and verified-prefix validation.
  - [x] Milestone 2.3: Define verified prefix as completed writes plus
        sink-side commit plus known verification state.
        The Global Store service rejects reports where verified units/bytes
        exceed completed units/bytes. Daemon terminal reports enforce known
        verification before reporting when `verify_before_report=true`;
        non-terminal daemon publication still requires a separate
        progressive-only remote-export contract.
  - [x] Milestone 2.4: Keep progressive coverage out of ordinary replica source
        selection and ordinary availability.
  - [x] Milestone 2.5: Implement failure, retirement, and expiration cleanup.
  - [x] Milestone 2.6: Implement report coalescing and throttling. The daemon
        and Global Store accept terminal transitions, but non-terminal coverage
        updates must satisfy configured byte-delta or time-interval thresholds.
        Export-state transitions always bypass throttling.
  - [x] Milestone 2.7: Add metrics for throttled reports, report QPS, DB
        conflict retries, and cleanup batch sizes.

- [x] Phase 3: Daemon coverage reporting
  - [x] Milestone 3.1: Add materialization attempt identity and coverage identity
        construction in daemon/core materialization paths. Current support is
        terminal canonical byte-prefix coverage for completed materializations.
  - [x] Milestone 3.2: Report verified byte-prefix coverage after commit and
        verification. The daemon skips reports when verification is not known
        and `verify_before_report=true`.
  - [x] Milestone 3.3: Stop reporting coverage on source export generation
        changes or daemon shutdown. Reports are one-shot terminal reports and
        are skipped during daemon shutdown or when no current remote export
        generation is active.
  - [x] Milestone 3.4: Ensure `ReportProgressiveCoverage(completed=true)` does
        not publish ordinary replica availability. The RPC path is isolated;
        ordinary availability still comes only from the existing full-replica
        registration path.

- [x] Phase 4: Progressive assignment claim
  - [x] Milestone 4.1: Add C++ Global Store client wrappers for progressive
        report/claim/complete/failure RPCs.
  - [x] Milestone 4.2: Implement atomic `find_progressive_source` claim that
        creates or replays a durable assignment. Target reads accept only active
        `claimed`/`reading` assignments and do not treat terminal replay states as
        readable segments. Replays are accepted only when the request fingerprint
        also matches the same coverage identity, requester, materialization
        attempt, and cursor.
  - [x] Milestone 4.3: Enforce source heartbeat, current exportability,
        `artifact_replicas.export_generation`, source caps, and source-domain
        policy at assignment time.
  - [x] Milestone 4.4: Maintain progressive outgoing caps separately from
        ordinary complete-replica transport counters through
        `progressive_source_counters`.
  - [x] Milestone 4.5: Implement assignment timeout, completion, failure, and
        expiration cleanup. Coverage expiration also cancels still-active
        assignments and releases progressive counters.
  - [x] Milestone 4.6: Bound source claim scans by config and indexes. The
        progressive path must not add another global dispatcher, reuse
        `_dispatch_loop_lock`, or scan all coverage rows under request load.
  - [x] Milestone 4.7: Enforce minimum assignment bytes unless returning the
        terminal tail segment, so tiny-segment workloads cannot create a Global
        Store write storm.

- [x] Phase 5: Target progressive read loop
  - [x] Milestone 5.1: Add target loop that requests the next uncovered prefix,
        reads only assigned safe coverage, completes the assignment, verifies it,
        and re-queries.
  - [x] Milestone 5.2: Ensure the loop refuses identity changes across artifact,
        byte space, selection hash, layout hash, hash-space, order hash,
        `group_version_set_id`, and `group_part_id`.
  - [x] Milestone 5.3: Publish ordinary replica availability only through the
        existing full materialization path.
  - [x] Milestone 5.4: Invalidate or retire source coverage on failed source
        read, stale heartbeat, export generation change, and visibility-fence
        violation.

- [x] Phase 6: Failure and topology policy
  - [x] Milestone 6.1: Implement cross-domain TCP seed smart skipping.
  - [x] Milestone 6.2: Source domain must come from typed daemon/worker topology
        metadata or config, not request-provided free-form strings. Daemon
        terminal coverage reports and target claims now derive source domain
        from worker topology (`node_id`, falling back to `daemon_id`), and Global
        Store validates reported/requested domains against the worker directory
        and rechecks current source directory state during assignment.
  - [x] Milestone 6.3: Add diagnostics for skipped source reasons and recovery
        path.

- [ ] Phase 7: Benchmarks and rollout
  - [x] Milestone 7.1: Enable progressive coverage reporting in observation-only
        benchmark lanes.
  - [x] Milestone 7.2: Enable progressive assignment claim for canonical full
        byte-prefix WeightPublisher fanout.
  - [ ] Milestone 7.3: Compare source HHI, top-1 share, p95/p99 latency, timeout
        count, correctness failures, and Global Store control-plane telemetry
        against complete-replica transport.
  - [ ] Milestone 7.4: Enable grouped serving prefetch/staged publication lanes
        only after `0117` publish-barrier gates pass.

- [ ] Phase 8: Cleanup and documentation
  - [ ] Milestone 8.1: Update benchmark reports with progressive on/off results.
        Report tooling now emits baseline/progressive/failure-injection lane
        summaries; concrete result tables still require repeated multi-host
        runs.
  - [x] Milestone 8.2: Update Global Store and daemon READMEs when operator
        behavior becomes visible.
  - [ ] Milestone 8.3: Keep progressive disabled by default until acceptance
        gates pass in repeated multi-host runs.
  - [x] Milestone 8.4: Remove 0118 compatibility transport duplication and keep
        replica/progressive memory metadata on the canonical
        `MemoryInfo.transport` path.

# Test Plan

Python tests:

- `source .venv/bin/activate && pytest tests/python/global_store/test_services.py`
- `source .venv/bin/activate && pytest tests/python/global_store/test_grpc_service.py`
- Add new tests under `tests/python/global_store/` for:
  - coverage identity exact-match source eligibility;
  - progressive coverage ignored by ordinary source selection;
  - monotonic coverage epoch and prefix growth;
  - one active source-eligible coverage row per materialization attempt,
    replica, and source export generation;
  - atomic assignment claim and idempotent replay by request fingerprint;
  - assignment timeout and completion cleanup;
  - export generation invalidation;
  - assignment-time export generation mismatch rejection;
  - heartbeat stale filtering;
  - progressive source cap enforcement independent from ordinary transport caps;
  - progressive counter increment/decrement exactly once across success,
    failure, expiration, cancellation, and replay;
  - candidate scan limit prevents broad coverage-table scans under request load;
  - non-terminal coverage reports are throttled until byte-delta or
    time-interval thresholds are met;
  - cross-domain TCP seed skipping.

C++ tests:

- `bazel test //core/store:store_engine_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- Add or extend daemon/core tests for:
  - coverage identity construction;
  - daemon report cadence and minimum delta;
  - target loop segment bounds;
  - identity mismatch fail-fast;
  - verified prefix report requires completed writes, sink commit, and known
    verification state;
  - source read failure and retry.

Proto/schema checks:

- `bash tools/build_proto_python.sh`
- `bazel test //proto/... --test_output=streamed`

Benchmarks:

- replay WeightPublisher fanout matrix with progressive disabled and enabled;
- include at least one cross-domain seed case;
- compare source top-1 share, HHI, p95/p99 publish-to-apply, timeout count, and
  transport outcome distribution;
- record Global Store control-plane telemetry and progressive metrics:
  report QPS, throttled reports, assignment claim latency, claim candidate
  rows, DB conflict retries, active progressive counters, executor queue depth,
  and inflight requests;
- assert progressive control writes scale with coverage-report cadence and
  assignment count, not copy chunks, RDMA completions, tensor count, or bytes
  inside assigned segments;
- run the failure-injection lane with progressive enabled and tune the selected
  receiver/delay so an in-progress source fails mid-read.

Cross-document consistency tests:

- after a group transaction freezes a key target, all parts continue using the
  frozen `version_set_id` even if the key is swapped before later members join;
- after a progressive source export generation bump, an old coverage assignment
  fails instead of rebinding to the same source's new generation.

# Acceptance Gates

- Progressive coverage is never returned by ordinary complete-replica selection.
- Progressive assignments are durable atomic claims and never cross verified
  prefix boundaries.
- Coverage has one active source-eligible row per materialization attempt,
  replica, and source export generation.
- Target loop fails fast on artifact, byte-space, selection-hash, layout-hash,
  hash-space, or order-hash mismatch.
- Source export generation changes retire older coverage and make old
  assignments fail rather than silently rebinding to the new generation.
- Assignment-time eligibility rechecks heartbeat, exportability, ordinary
  replica export generation, source cap, and typed source domain.
- Sources used by progressive assignment obey the shared source visibility fence.
- V1.0 supports byte-prefix only.
- Source failure mid-read recovers from another eligible source or returns a
  clear unavailable error.
- Slow cross-domain TCP seed sources are skipped by default.
- Progressive fanout improves at least one high-fanout concentration metric
  without increasing correctness failures or timeout rate.
- Coverage reporting is coalesced; non-terminal report QPS is bounded by
  configured byte-delta and interval thresholds.
- Source claims are indexed bounded atomic claims using progressive counters,
  not a second global dispatcher or ordinary `replica_counters`.
- Progressive dissemination introduces no Global Store writes proportional to
  copy chunks, RDMA completions, tensor count, CUDA events, or bytes inside an
  assigned segment.

# Rollout And Backout

Rollout:

1. Land schema/proto/config with feature flags default off.
2. Enable daemon coverage reporting in observation-only tests.
3. Enable Global Store assignment claim in fake-CUDA integration tests.
4. Enable target loop in benchmark-only lanes.
5. Enable WeightPublisher canonical full byte-prefix artifact fanout lanes.
6. Enable grouped serving prefetch/staged publication lanes after `0117`
   publish-barrier gates pass.
7. Promote after repeated correctness and concentration gates pass.

Backout:

- disable progressive source eligibility in config;
- disable daemon coverage reporting if needed;
- keep additive schema/proto fields;
- expire or retire coverage rows and assignments;
- route materialization through ordinary complete-replica transport.

# Risks And Tracking

- Incorrect prefix identity could expose corrupt data.
  Mitigation: v1 prefix-only, strict identity fields, verified-before-report.
- Coverage rows can become hot under large fanout.
  Mitigation: report deltas, interval throttling, TTLs, bounded indexes, and
  source caps.
- Assignment claims can turn Global Store into a segment scheduler bottleneck.
  Mitigation: minimum assignment bytes, candidate scan limits, claim QPS caps,
  progressive source counters, and stress gates that fail rollout when
  control-plane saturation appears before data-path saturation.
- Progressive source caps can underutilize bandwidth.
  Mitigation: start conservative, then tune with benchmark evidence.
- Cross-domain policy can accidentally serialize local fanout behind TCP seed.
  Mitigation: smart skipping default and explicit skipped-source diagnostics.

# Owner Checklist

- [x] `schema.sql` updated.
- [x] Schema smoke tests updated.
- [x] Global Store proto updated; generated Python stubs regenerated with local
      `grpc_tools.protoc` fallback because Buf remote plugin was unavailable.
- [x] daemon proto updated; generated Python stubs regenerated with local
      `grpc_tools.protoc` fallback because Buf remote plugin was unavailable.
- [x] Global Store and daemon config protos updated.
- [x] Progressive coverage repository/service added.
- [x] Progressive assignment repository/service added.
- [x] Python Global Store tests added.
- [x] C++ daemon/core tests added.
- [x] WeightPublisher progressive benchmark lane added.
- [x] Failure-injection benchmark lane added.
- [x] Public/operator docs updated when behavior becomes user-visible.
