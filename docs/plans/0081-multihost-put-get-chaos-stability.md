---
slug: multihost-put-get-chaos-stability
title: Multi-Host Put/Get Chaos Stability Plan
status: in_progress
areas: ["core", "daemon", "global_store", "sdk", "infra", "tests", "docs"]
created: 2026-02-22
last_updated: 2026-02-22
related_code:
  - docs/designs/0081-multihost-put-get-chaos-stability.md
  - docs/benchmarks/20260221-multi-host-p2p-fanout-benchmark-report.md
  - docs/internals/chaos-report-template.md
  - docs/internals/chaos-gate-checklist.md
  - examples/cross_host/cross_host_fanout_runner.py
  - examples/cross_host/cross_host_chaos_runner.py
  - examples/cross_host/cross_host_get_once.py
  - examples/cross_host/case_schemas/chaos_suite_example.json
  - examples/cross_host/run_0081_multihost_chaos.sh
  - examples/cross_host/run_multihost_benchmark_suite.sh
  - examples/cross_host/run_multihost_chaos_suite.sh
  - examples/config/global_store_config_cross_host_bench.yaml
  - examples/config/global_store_config_cross_host_bench_fast_failover.yaml
  - examples/config/global_store_config_cross_host_bench_slow_cleanup.yaml
  - tensorcast/global_store/services/worker_service.py
  - tests/python/global_store/test_services.py
  - tensorcast/api/store/materialization.py
  - tensorcast/api/store/retry.py
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/control/materialize_orchestrator.cc
  - core/store/runtime/ingestion/materialization_facade.cc
  - daemon/service/controllers/replica_materialization_service.cc
  - daemon/service/controllers/target_materialization_service.cc
  - tensorcast/global_store/services/transport_service.py
links:
  design: ../designs/0081-multihost-put-get-chaos-stability.md
---

# Objective

Execute full multi-host chaos verification on top of the current implemented baseline, using strict hard gates and one mode-compatible runner framework.

Success means:

- required positive cases pass with `all_get_complete=true`
- required negative cases pass only via `expected_failure_pass=true`
- recovery SLO and correctness gates are enforced as hard pass/fail
- execution remains in a single extensible runner stack (`fanout_runner` + `chaos_runner`)

# Current State and Grounding

Current baseline already includes prerequisite capabilities and is treated as done, not planned work:

- fanout/cascade summary fields and event timeline are available
- single-call helper path exports retry/budget diagnostics
- request budget and transport wait timeout are propagated to daemon/core
- deadline-aware reselection is in place
- GS transport create-failure rollback and idempotent complete are implemented
- chaos runner and chaos suite entry script exist
- fast-failover and slow-cleanup GS profiles exist

Current execution gaps for `0081` are owner-level closure gaps:

- hard gate evaluator must be enforced at case/run level
- diffusion identity fallback must be upgraded beyond source-type labels
- request-level event correlation needs stronger causality (`request_id` lineage)
- mixed steady-state mode and reusable network-fault templates need standardization
- per-fault recovery thresholds need cluster evidence signoff

# Execution Decisions

1. `0081` is the active execution plan for chaos stability closure.
2. Required gates are hard gates; no soft bypass for required scenarios.
3. Extend existing runner stack; do not create another independent data-plane runner.
4. Keep mode compatibility inside one framework:
   - `phase`: `small|medium|large`
   - `mode`: `fanout|cascade`
   - `chaos_profile`: `none|single_fault|combo_fault`
   - `expected_outcome`: `success|failure`
5. Preserve architecture boundary: SDK code does not connect directly to Global Store.

# Phases and Milestones

- [x] Phase 0: Baseline Freeze on Implemented Prerequisites
  - [x] Milestone 0.1: Baseline capability set treated as current state.
  - [x] Milestone 0.2: Design and plan updated to remove stale “not implemented” assumptions.

- [ ] Phase 1: Hard Gate Engine Codification
  - [ ] Milestone 1.1: Add explicit gate spec to case schema (required gates + thresholds).
  - [ ] Milestone 1.2: Evaluate gates per case and per run; gate violation returns non-zero.
  - [ ] Milestone 1.3: Persist gate evaluation artifacts (`gate_review.json`, `gate_review.md`).

- [ ] Phase 2: Existing Runner Extension and Mode Compatibility
  - [ ] Milestone 2.1: Keep `cross_host_fanout_runner.py` as traffic engine for `fanout|cascade`.
  - [ ] Milestone 2.2: Extend traffic engine with `mixed_steady_state` mode.
  - [ ] Milestone 2.3: Keep `cross_host_chaos_runner.py` as orchestration wrapper across all modes.
  - [ ] Milestone 2.4: Unify suite inputs so benchmark and chaos scripts share phase/case contracts.

- [ ] Phase 3: Diffusion and Causality Observability Hardening
  - [ ] Milestone 3.1: Diffusion identity uses `ticket_replica_uuid` as primary source key.
  - [ ] Milestone 3.2: DB probe fallback is marked degraded and never treated equivalent.
  - [ ] Milestone 3.3: Propagate and emit request-level correlation fields in events.

- [ ] Phase 4: Fault Profile and Control-Plane Hardening
  - [ ] Milestone 4.1: Standard netem fault templates available in case schema.
  - [ ] Milestone 4.2: Restart fault templates standardized by role (`seed/relay/getter`).
  - [ ] Milestone 4.3: GS selection policy and cleanup interaction reviewed and signed off.

- [ ] Phase 5: Scale Execution with Hard Gates
  - [x] Milestone 5.1: `small` phase passes full hard gates.
  - [x] Milestone 5.2: `medium` phase passes full hard gates.
  - [x] Milestone 5.3: `large` phase passes full hard gates.
  - [ ] Milestone 5.4: Recovery thresholds are confirmed with archived evidence.

- [ ] Phase 6: Regressionization and Ownership Closure
  - [ ] Milestone 6.1: Add automated tests for chaos schema parsing and gate evaluation.
  - [ ] Milestone 6.2: Add deterministic scheduler replay tests.
  - [ ] Milestone 6.3: Add periodic canary run and owner signoff package.

# Task Breakdown

- [ ] Hard gate implementation
  - [ ] Extend `examples/cross_host/case_schemas/chaos_suite_example.json` with required gate thresholds.
  - [ ] Update `examples/cross_host/cross_host_chaos_runner.py` to enforce case/run hard gates.
  - [ ] Update `examples/cross_host/chaos_gate_review.py` and `examples/cross_host/chaos_phase_gate_review.py` outputs for strict pass/fail.

- [ ] Existing runner extension
  - [ ] Extend `examples/cross_host/cross_host_fanout_runner.py` with `mixed_steady_state` mode.
  - [ ] Keep backward mode compatibility for `fanout` and `cascade` in same entrypoint.
  - [ ] Ensure `examples/cross_host/run_multihost_benchmark_suite.sh` and `examples/cross_host/run_multihost_chaos_suite.sh` consume aligned phase/case semantics.

- [ ] Diffusion/observability
  - [ ] In `examples/cross_host/cross_host_get_once.py`, ensure source identity output includes stable source key fields for diffusion.
  - [ ] In `examples/cross_host/cross_host_fanout_runner.py`, compute diffusion from source identity rather than source-type labels.
  - [ ] Add request correlation fields to timeline outputs.

- [ ] Control-plane review
  - [ ] Validate `tensorcast/global_store/repositories/replica_repository.py` selection strategy under chaos load.
  - [ ] Validate `tensorcast/global_store/services/transport_service.py` cleanup behavior under fast/slow profiles.

- [ ] Test coverage
  - [ ] Add Python tests for chaos runner gate logic and status transitions.
  - [ ] Extend integration checks for diffusion identity and expected-failure semantics.

- [ ] Documentation
  - [ ] Keep `docs/designs/0081-multihost-put-get-chaos-stability.md` and this plan synchronized with shipped behavior.
  - [ ] Keep `examples/cross_host/README.md`, `docs/internals/chaos-report-template.md`, and `docs/internals/chaos-gate-checklist.md` aligned with final gate contract.

# Execution-Ready TODO Backlog (Handoff Grade)

This section is the canonical implementation backlog for the remaining open work.
Each TODO is written so a different engineer can execute directly from this plan.

| ID | Priority | Necessity | Scope | Primary files | Required deliverables | Exit criteria | Effort |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1 | P0 | Must | Hard gate schema contract | `examples/cross_host/case_schemas/chaos_suite_example.json`, `examples/cross_host/cross_host_chaos_runner.py` | Per-case gate spec in schema (`required_gates`, thresholds) with parser validation and clear errors | Invalid schema fails fast; parsed gates persisted into case result payload | 1 day |
| T2 | P0 | Must | Strict gate enforcement | `examples/cross_host/cross_host_chaos_runner.py`, `examples/cross_host/chaos_gate_review.py`, `examples/cross_host/chaos_phase_gate_review.py` | Case-level and run-level gate verdicts; non-zero exit on required gate failure | Any required gate failure returns non-zero; no warning-only bypass path | 1-2 days |
| T3 | P0 | Must | Fault profile templates | `examples/cross_host/case_schemas/chaos_suite_example.json`, new `examples/cross_host/case_schemas/chaos_suite_0081_fault_matrix.json`, `examples/cross_host/cross_host_chaos_runner.py` | Reusable templates for `daemon_stop`, `daemon_kill`, restart, `tc netem` (delay/loss/reorder), resource pressure | Templates run without ad-hoc script edits; each template has expected impact and role targeting | 2-3 days |
| T4 | P0 | Must | Recovery SLO evidence closure | `examples/cross_host/run_0081_multihost_chaos.sh`, `examples/cross_host/run_multihost_chaos_suite.sh`, report docs under `docs/benchmarks/` | Archived runs for each required fault class with recovery measurements | Recovery thresholds in this plan are all evidence-backed and reproducible | 1-2 days (plus cluster queue time) |
| T5 | P0 | Must | Diffusion identity hardening | `examples/cross_host/cross_host_get_once.py`, `examples/cross_host/cross_host_fanout_runner.py`, `examples/cross_host/chaos_gate_review.py` | Diffusion computed from stable source identity (`ticket_replica_uuid` primary, fallback explicitly degraded) | Diffusion gate checks real diversity/rebuild, not only timeline presence | 1-2 days |
| T6 | P1 | Should | Request-level causality fields | `examples/cross_host/cross_host_get_once.py`, `examples/cross_host/cross_host_fanout_runner.py`, `examples/cross_host/cross_host_chaos_runner.py` | Request correlation fields in events (`request_id`, per-attempt ids, timeline linkage) | Single request can be traced end-to-end across run/case/events | 1 day |
| T7 | P0 | Must | Control-plane policy validation | `tensorcast/global_store/repositories/replica_repository.py`, `tensorcast/global_store/services/transport_service.py`, new tests in `tests/python/global_store/` | Policy-focused tests/review notes for selection bias and cleanup behavior under chaos profiles | No unresolved selection/cleanup regressions in fast/slow profiles | 2-3 days |
| T8 | P0 | Must | Automated gate logic tests | new tests in `tests/python/` for chaos schema parsing and gate transitions | Unit tests for schema parsing, gate pass/fail transitions, expected-failure semantics | CI-style local test command passes and covers gate regressions | 1-2 days |
| T9 | P1 | Should | Deterministic replay tests | new deterministic test harness in `tests/python/` | Replay tests for scheduler/event ordering critical paths | Reproducible failure/recovery behavior under fixed seed/order | 2 days |
| T10 | P1 | Should | Periodic canary automation | new canary runner script + documentation | Scheduled `small` or `small+medium` periodic run with artifact retention policy | Canary outputs stable gate artifacts and alertable pass/fail status | 1-2 days |
| T11 | P2 | Could | Mixed steady-state mode | `examples/cross_host/cross_host_fanout_runner.py`, schema files, docs | `mixed_steady_state` workload mode with long-window metrics | New mode is compatible with existing fanout/cascade entrypoint | 2-3 days |
| T12 | P1 | Should | Documentation/signoff closure | `docs/designs/0081-multihost-put-get-chaos-stability.md`, this plan, `examples/cross_host/README.md`, `docs/internals/chaos-report-template.md`, `docs/internals/chaos-gate-checklist.md` | Final aligned docs + owner signoff pack | Docs match shipped behavior and owner checklist is complete | 0.5-1 day |

# Priority and Necessity Analysis

## Must-complete for claiming “0081 robustness validated”

- `T1`, `T2`: without hard gate contract and strict enforcement, pass/fail is not auditable.
- `T3`, `T4`: without real fault matrix evidence, recovery SLO claims are unsupported.
- `T5`: without stable diffusion identity, diversity/rebuild gate is weak and can be misleading.
- `T7`, `T8`: without control-plane validation and automated tests, regressions are likely.

## Can be deferred after robustness baseline

- `T6`, `T9`, `T10`: observability/replay/operational hardening; high value but not blocking initial closure.
- `T11`: coverage expansion, not a blocking prerequisite for baseline robustness signoff.

# Recommended Execution Waves (for assignee)

1. Wave A (P0 foundation):
   - implement `T1`, `T2`, `T8`
   - objective: lock gate contract and enforcement semantics before adding more scenarios
2. Wave B (P0 signal quality):
   - implement `T5`, `T6`
   - objective: make diffusion and request causality trustworthy for debugging and audit
3. Wave C (P0 robustness evidence):
   - implement `T3`, `T4`, `T7`
   - objective: run and archive real fault evidence across required classes
4. Wave D (P1/P2 productionization):
   - implement `T9`, `T10`, `T11`, `T12`
   - objective: long-term regression prevention and ownership handoff closure

# Detailed TODO Specs (Implementation + Verification)

Use this section as the step-by-step handoff checklist for implementation owners.

## T1: Hard gate schema contract

- Implementation steps:
  - add explicit gate schema fields to `examples/cross_host/case_schemas/chaos_suite_example.json` (for required gates and thresholds).
  - update schema parsing in `examples/cross_host/cross_host_chaos_runner.py` with strict validation and actionable error messages.
  - persist normalized gate config into case result artifacts.
- Verification commands:
  - `source .venv/bin/activate && python examples/cross_host/cross_host_chaos_runner.py --case-schema <invalid_schema.json> --out-dir /tmp/tc_cross_20260222/dryrun --run-id dryrun`
  - `source .venv/bin/activate && python examples/cross_host/cross_host_chaos_runner.py --case-schema examples/cross_host/case_schemas/chaos_suite_example.json --out-dir /tmp/tc_cross_20260222/dryrun --run-id dryrun-valid`
- Required artifacts:
  - parser validation evidence (invalid schema fails).
  - valid run artifact showing parsed gate config persisted.

## T2: Strict gate enforcement

- Implementation steps:
  - enforce required gates at case level and run level in `examples/cross_host/cross_host_chaos_runner.py`.
  - keep `examples/cross_host/chaos_gate_review.py` and `examples/cross_host/chaos_phase_gate_review.py` semantics aligned (required gate failure => non-zero exit).
  - ensure no required gate can silently downgrade to warning.
- Verification commands:
  - `source .venv/bin/activate && python examples/cross_host/chaos_gate_review.py --run-dir <run_dir_with_required_gate_failure>`
  - `source .venv/bin/activate && python examples/cross_host/chaos_phase_gate_review.py --small-run-dir <small> --medium-run-dir <medium> --large-run-dir <large>`
- Required artifacts:
  - one forced-failure run proving non-zero exit.
  - one pass run proving exit 0 and complete gate payload.

## T3: Fault profile templates

- Implementation steps:
  - create a fault-matrix schema file (recommended path: `examples/cross_host/case_schemas/chaos_suite_0081_fault_matrix.json`).
  - standardize templates for:
    - daemon stop/kill/restart by role (`seed`, `relay`, `getter`)
    - network perturbation (`tc netem` delay/loss/reorder)
    - resource pressure (bounded CPU/memory stress profile)
  - document template fields in `examples/cross_host/README.md`.
- Verification commands:
  - `source .venv/bin/activate && bash examples/cross_host/run_multihost_chaos_suite.sh` with the new fault-matrix schema.
- Required artifacts:
  - committed schema templates.
  - at least one successful execution log per template class.

## T4: Recovery SLO evidence closure

- Implementation steps:
  - execute required fault classes at target phases and archive run outputs.
  - compute and report `recover_time_sec` against thresholds in this plan.
  - add an evidence report under `docs/benchmarks/` with threshold pass/fail table.
- Verification commands:
  - `source .venv/bin/activate && python examples/cross_host/chaos_gate_review.py --run-dir <run_dir> --max-recover-time-sec <threshold>`
  - `source .venv/bin/activate && python examples/cross_host/chaos_phase_gate_review.py --small-run-dir <small> --medium-run-dir <medium> --large-run-dir <large>`
- Required artifacts:
  - archived run directories per fault class.
  - benchmark report with per-fault threshold evidence.

## T5: Diffusion identity hardening

- Implementation steps:
  - treat `ticket_replica_uuid` as primary source identity in diffusion calculation path.
  - mark DB-probe-based source inference as degraded fallback only.
  - upgrade gate checks from “timeline present” to diversity/rebuild criteria.
- Verification commands:
  - run fault and non-fault fanout cases and verify `source_cardinality_timeline` plus identity fields in outputs.
  - run gate review to verify diffusion-related required gates.
- Required artifacts:
  - run outputs with explicit identity source fields.
  - gate evidence proving diffusion gate behavior at scale.

## T6: Request-level causality fields

- Implementation steps:
  - add request correlation fields in helper/runner outputs (`request_id`, attempt lineage).
  - emit correlation fields into case/run `events.jsonl`.
- Verification commands:
  - inspect one successful and one transient-failure request path in `events.jsonl`.
- Required artifacts:
  - sample traces showing end-to-end request linkage.

## T7: Control-plane policy validation

- Implementation steps:
  - add targeted tests/review for selection strategy under chaos load in `tensorcast/global_store/repositories/replica_repository.py`.
  - add targeted tests/review for cleanup under fast/slow profiles in `tensorcast/global_store/services/transport_service.py`.
- Verification commands:
  - `source .venv/bin/activate && pytest tests/python/global_store/... -q`
- Required artifacts:
  - tests and evidence notes for selection/cleanup behavior.

## T8: Automated gate logic tests

- Implementation steps:
  - add tests for schema parsing, case gate transitions, expected-failure semantics, and phase aggregation.
  - ensure tests are deterministic and independent of remote infra.
- Verification commands:
  - `source .venv/bin/activate && pytest tests/python/... -q`
- Required artifacts:
  - new test files and passing local test output.

# Handoff Runbook (Operator-Executable)

This runbook is mandatory for anyone executing the remaining items.

## Preconditions

- Global Store session is running and reachable (`tensorcast-cli global status --json`).
- Quota group can launch required workers.
- Use `--private-machine group` for all worker launches.
- Always activate virtualenv before Python commands:
  - `source .venv/bin/activate`

## Baseline fixed launcher

- Primary entrypoint:
  - `bash examples/cross_host/run_0081_multihost_chaos.sh --phase all`
- Keep workers for debugging:
  - add `--keep-workers`
- Default behavior (without `--keep-workers`):
  - workers are cleaned up automatically at script exit.

## Fault-matrix execution requirement

- For `T3/T4`, run with fault-template schemas and archive each run:
  - output root should include run label and schema identifier
  - each run must include `summary.json`, `events.jsonl`, case artifacts, `gate_review.json`, and phase review when applicable

## Mandatory artifacts per execution wave

- Wave A:
  - parser and gate tests
  - strict gate failure example with non-zero exit proof
- Wave B:
  - diffusion identity sample outputs including `ticket_replica_uuid`
  - request correlation fields in timeline/event outputs
- Wave C:
  - per-fault-class recovery evidence bundle with threshold comparison table
  - control-plane validation test outputs
- Wave D:
  - canary config/runbook
  - owner signoff package

## Cleanup policy (required)

- After every run, verify temporary workers are deleted.
- Verification command:
  - `brainctl get process -n shai-core | rg "ws-<workspace-id>-worker-|<owner-name>"`
- If stale workers exist, delete explicitly:
  - `brainctl delete process <worker_id> -n shai-core`

## Failure-handling policy

- Stop rollout on first required gate failure.
- Root-cause first; do not add outer retry wrappers to hide first-start failures.
- Allowed rerun scope:
  - case-level rerun only for classified infra noise with recorded reason.

# Completion Definition for 0081 (Signoff Criteria)

`0081` can be marked complete only when all conditions below are true:

1. `T1`-`T8` are complete and validated.
2. Recovery thresholds in this plan are evidence-backed for each required fault class.
3. Diffusion identity gates pass at target scales using stable identity semantics.
4. No unexpected case statuses in archived required runs.
5. Gate artifacts exist for all required runs/phases and are reproducible.
6. Owner checklist in this document is fully signed off.

# Hard Gate Matrix

Required positive cases:

- `all_get_complete == true`
- `put_success_rate == 1.0`
- `get_success_rate == 1.0`
- `comm_errors_delta == 0`
- `comm_bytes_delta` matches transferred payload bytes

Required negative cases:

- `expected_failure_pass == true`
- no `unexpected_failure`
- no `unexpected_success`

Required diffusion gates:

- for fanout with getter count >= 2: `max(unique_source_count) >= 2`
- post-fault fanout: source diversity re-established within recovery SLO

Recovery hard thresholds (seconds):

- daemon stop: `<= 120`
- daemon kill: `<= 180`
- daemon restart: `<= 180`
- GS short outage: `<= 240`
- network perturbation: `<= 300`
- resource pressure: `<= 300`

Hard-gate policy:

- any required gate failure fails run exit code
- required gates are never downgraded to warning

# Test, Rollout, and Backout

## Test Plan

Python:

- `source .venv/bin/activate && pytest tests/python/...`

C++:

- `bazel test //core/store/... --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
- `bazel test //daemon/... --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`

Chaos suite:

- `source .venv/bin/activate && bash examples/cross_host/run_multihost_chaos_suite.sh`
- `source .venv/bin/activate && python examples/cross_host/chaos_gate_review.py --run-dir <run_dir> --max-recover-time-sec 180`
- `source .venv/bin/activate && python examples/cross_host/chaos_phase_gate_review.py --small-run-dir <small> --medium-run-dir <medium> --large-run-dir <large>`

## Rollout

- Execute strictly in order: `small -> medium -> large`.
- Stop immediately on any hard-gate failure.
- Promote thresholds only after archived evidence for each fault class.

## Backout

- Revert latest isolated phase change when regression is confirmed.
- Do not reintroduce request-level runner replay or required-gate soft bypass.

# Acceptance Gates

- [x] Required positive cases all pass hard gates.
- [x] Required negative cases all pass expected-failure hard gates.
- [ ] Recovery thresholds satisfied for all required fault classes.
- [ ] Diffusion identity gates pass at each target scale.
- [x] No unexpected case statuses.
- [x] Gate review artifacts are generated and archived for `small/medium/large`.

# Latest Execution Snapshot (2026-02-22)

## Blocking failure encountered and fixed

- Initial `large` run failed with `all_get_complete=false` on `get3/get4/get5`.
- Root cause:
  - getter stderr reported `ArtifactError: GlobalStoreClient requires a non-zero P2P port`
  - daemon logs reported `Address/port already registered by another worker`
  - GS worker registration hit a stale inactive endpoint conflict after medium-to-large relaunch.
- Fundamental fix shipped in `tensorcast/global_store/services/worker_service.py`:
  - reclaim inactive endpoint row when daemon-id rebind collides with an inactive worker row on the same address/port
  - mark stale worker replicas unavailable before row cleanup/update
  - keep hard conflict behavior for active endpoint collisions
- Added regression test:
  - `tests/python/global_store/test_services.py::test_worker_service_registration_reclaims_inactive_endpoint_conflict`

## Validation evidence

- Global Store service tests passed:
  - `pytest tests/python/global_store/test_maintenance_coordinator.py tests/python/global_store/test_repositories.py tests/python/global_store/test_services.py -q`
- Revalidation runs after fix (strict order `small -> medium -> large`):
  - `/tmp/tc_cross_20260222/results_chaos_phase_relaunch/phase-small-20260222e` (`passed=true`)
  - `/tmp/tc_cross_20260222/results_chaos_phase_relaunch/phase-medium-20260222e` (`passed=true`)
  - `/tmp/tc_cross_20260222/results_chaos_phase_relaunch/phase-large-20260222f` (`passed=true`)
- Phase aggregate gate review:
  - `/tmp/tc_cross_20260222/results_chaos_phase_relaunch/phase_gate_review_fixed.json` (`passed=true`)
  - `/tmp/tc_cross_20260222/results_chaos_phase_relaunch/phase_gate_review_fixed.md`

## Resource lifecycle

- Launched workers with required placement constraint `--private-machine group`.
- Post-validation cleanup completed: all temporary `ws-7681b3683947089e-worker-*` workers deleted.

## Re-execution snapshot (2026-02-22, private-machine group)

- Re-launched 8 workers with `brainctl launch --private-machine group`.
- Re-generated phase schemas using current worker ids/pod IPs:
  - `/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/small.json`
  - `/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/medium.json`
  - `/tmp/tc_cross_20260222/chaos_phase_schemas_0081_exec/large.json`
- Re-executed strict order `small -> medium -> large`:
  - `/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase-small-20260222g` (`passed=true`)
  - `/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase-medium-20260222f` (`passed=true`)
  - `/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase-large-20260222g` (`passed=true`)
- Re-generated phase aggregate gate review:
  - `/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase_gate_review_0081_exec.json` (`passed=true`)
  - `/tmp/tc_cross_20260222/results_chaos_phase_0081_exec/phase_gate_review_0081_exec.md`
- Post-run cleanup completed: all temporary `ws-7681b3683947089e-worker-{hhnxx,7dtq7,dm7sk,r4sh7,5l5p6,994zx,sq7d2,hldhs}` deleted.
- Fixed launcher entrypoint added for repeatable 0081 execution:
  - `examples/cross_host/run_0081_multihost_chaos.sh`

# Risks and Tracking

- [ ] Risk: infra volatility inflates false failures.
  - Mitigation: strict classification + case-level rerun policy.
- [ ] Risk: mode growth increases complexity in one runner stack.
  - Mitigation: single traffic engine + explicit mode contract + tests.
- [ ] Risk: gate strictness blocks progress during initial rollout.
  - Mitigation: stage by phase, but keep required gates hard.
- [ ] Risk: GS selection bias under scale.
  - Mitigation: dedicated selection-policy review and benchmark evidence.

# Owner Checklist

- [ ] Runner owner signs off hard-gate implementation and mode compatibility.
- [ ] Core owner signs off budget/reselection behavior under required faults.
- [ ] Daemon owner signs off request-budget propagation semantics.
- [ ] GS owner signs off transport consistency and selection behavior.
- [ ] QA owner signs off `small/medium/large` hard-gate evidence.
- [ ] Docs owner signs off report/checklist/template alignment.
