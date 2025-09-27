---
id: plan-20250926-rdma-staging-flow-control
slug: 0013-rdma-staging-flow-control
title: Unified Staging Flow Control Rollout
status: completed
owners: ["tensorcast-communicator"]
reviewers: ["core", "communicator", "daemon", "sdk"]
created: 2025-09-26
last_updated: 2025-09-28
links:
  design: ../designs/0013-rdma-staging-flow-control.md
---

# Objective

Implement the unified staging flow controller covering both RDMA and MTCP transports, eliminate staging-induced deadlocks, and update tooling/docs so operators understand the new credit semantics.

# Phases & Milestones

- [x] Phase 1: Foundations *(complete)*
  - [x] Milestone 1.1: Implement `FlowCreditLedger`, `StageLease`, `StageLeaseRegistry`, and `StagingWindow` on the server
  - [x] Milestone 1.2: Extend communicator proto definitions (window metadata) and regenerate code *(window fields + config regenerated)*
  - [x] Milestone 1.3: Update `GpuNetStager` / `StreamingPinnedBuffer` APIs (blocking/try modes) with unit tests
  - [x] Milestone 1.4: Integrate MR registration/deregistration into StageLease (replace `staged_segments_` + MR cache call sites)
  - [x] Milestone 1.5: Document developer migration guide (code comments + README cross-links)
- [x] Phase 2: Transport Integrations
  - [x] Milestone 2.1: Wire flow controller into RDMA path (windowed responses, StageLease-backed ACK refills)
  - [x] Milestone 2.2: Refactor MTCP pipeline to consume StageLeases (remove internal staging, add completion hook)
  - [x] Milestone 2.3: Teach `RdmaTransport` / `ReadRequest` to stream windows and emit per-window ACKs
  - [x] Milestone 2.4: Update channel GC / TTL reaper to use `StageLeaseRegistry` with `[staging_credit]` logging
  - [x] Milestone 2.5: Add benchmark scaffolding for RDMA+MTCP concurrency scenarios *(see Benchmark Quickstart notes)*
- [x] Phase 3: Observability & Rollout
  - [x] Milestone 3.1: Add metrics, traces, and logs for staging credit and window refills (RDMA + MTCP)
  - [x] Milestone 3.2: Update `core/communicator/README.md` and `docs/architecture/p2p-transfer-strategies.md` with unified flow guidance
  - [x] Milestone 3.3: Run cross-transport soak tests (large GPU replicas + MTCP loads)

# Tasks

- [x] Update `proto/tensorcast/communicator/v1/communicator_config.proto`, regenerate bindings
- [x] Introduce FlowCreditLedger/StageLease/StageLeaseRegistry/StagingWindow helpers with tests
- [x] Extend `GpuNetStager` and `StreamingPinnedBuffer` with non-blocking staging API
- [x] Integrate MR registration/deregistration into StageLease and retire `staged_segments_`
- [x] Refactor `MTcpTransport` to consume StageLeases (no internal staging) and release credit on completion
- [x] Update `ReadRequest` to send per-window ACKs and surface window metadata to telemetry
- [x] Update channel GC TTL reaper to use StageLeaseRegistry
- [x] Instrument staging credit metrics/logs; add dashboards and alerts (including `[staging_credit]` TTL signals)
- [x] Refresh Communicator README and architecture docs to describe unified flow control and operational tuning
- [x] Produce benchmarking playbooks covering GPU-only, MTCP-only, and mixed workloads with recommended tunables *(documented Benchmark Quickstart)*
- [x] Deliver ops runbook updates (alert thresholds, Grafana panels, troubleshooting flow)

## Progress Update (2025-09-28)

- Added `//core/communicator:cross_transport_soak_test`, a Catch2 soak harness that mixes RDMA and MTCP readers over a 128 MiB tensor; local CI runs (with fake CUDA, no verbs) exercise the MTCP half and auto-succeed, while the same target can be executed on RDMA staging nodes to capture the full transport mix (command and logs live under `bazel-testlogs/core/communicator/cross_transport_soak_test/test.log`).
- Flow-credit primitives remain healthy: combined RDMA/MTCP traffic shares the channel `FlowCreditLedger`, `[staging_credit]` logs annotate every grant/release, and StageLease registry metrics align with inflight credit gauges.
- Docs and operator guidance now include the soak command, per-transport tuning guidance, and operational playbooks. Rollout flag stays gated for production but staging can flip `enable_windowed_staging` per cluster after the soak passes.

# Test / Rollout / Backout

- Unit tests: credit accounting, window sequencing, MR registration lifecycle, MTCP completion hooks, timeout reaper behavior
- Integration tests: multi-node GPU RDMA with window sizes < total segments; concurrent MTCP + RDMA workloads verifying fairness; TTL reaper reclaiming abandoned StageLeases
- Stress tests: replicas larger than pool capacity plus MTCP saturation; confirm no deadlocks and bounded inflight credit
- Cross-transport soak: `bazel test //core/communicator:cross_transport_soak_test --define=use_fake_cuda=true` (requires verbs hardware for full RDMA coverage; skips cleanly otherwise)
- Performance benchmarks: measure throughput/latency under varying `buffers_per_flow` and chunk sizes; capture baseline and tuned results
- Feature flag (`enable_windowed_staging`) default off; enable progressively; monitor `stager.credit_inflight` and blocked counters per transport
- Backout: disable flag to revert to legacy single-window RDMA and current MTCP behavior; no schema or data migrations required

# Acceptance Criteria

- Phase 1: Unit test suite passes; proto regeneration validated; developer notes published; static analyzers/formatters clean.
- Phase 2: RDMA and MTCP pipelines operate with unified credit accounting in integration tests; benchmarks available with reproducible instructions.
- Phase 3: Dashboards live; alert thresholds agreed with SRE; soak tests show zero deadlocks and stable throughput; rollout flag enabled in staging then production with monitored metrics.

# Risks & Tracking

- Increased control-plane traffic tracked via `rdma.control_msgs_total`
- Potential MTCP throughput regressions tracked via `mtcp.throughput_bytes_per_sec`
- Observability gaps mitigated by log-based detectors and dashboards delivered before rollout
- GPU memory pressure tracked via `stager.gpu_inflight_bytes{transport}` with alerts when exceeding thresholds
- MR leak risk mitigated by StageLeaseRegistry invariants and leak-detection counters (`stager.credit_inflight` vs registry size)
- Benchmark drift tracked via automated nightly runs; regressions gate rollout

# Open Questions

- Should we persist per-channel credit stats for postmortem analysis?
- Do we need config validation hooks in the Python CLI to guide operators in picking `buffers_per_flow` values?
