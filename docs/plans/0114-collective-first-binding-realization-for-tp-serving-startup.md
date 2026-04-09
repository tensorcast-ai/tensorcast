---
slug: collective-first-binding-realization-for-tp-serving-startup
title: Collective-First Binding Realization for TP Serving Startup Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests", "benchmarks", "serving"]
related_code:
  - docs/designs/0084-binding-unified-model-and-contract.md
  - docs/designs/0112-binding-native-serving-realization-and-publication.md
  - docs/designs/0113-step3p5-closure-and-sot-convergence.md
  - docs/designs/0114-collective-first-binding-realization-for-tp-serving-startup.md
  - docs/plans/0113-step3p5-closure-and-sot-convergence.md
  - proto/tensorcast/daemon/v2/store_daemon.proto
  - tensorcast/types.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/owned_binding_slot.py
  - daemon/service/controllers/representation_transform_builder.h
  - daemon/service/controllers/representation_transform_builder.cc
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/materialization_target_plan_utils_test.cc
  - daemon/service/controllers/materialization_policy_utils.cc
  - daemon/service/controllers/assembly_operation_service.cc
  - daemon/service/controllers/owned_binding_service.cc
  - daemon/service/controllers/status_controller.h
  - daemon/state/lip_manager.cc
  - core/store/materialization/contracts/representation_contract.h
  - core/store/materialization/contracts/representation_contract.cc
  - core/store/materialization/contracts/representation_contract_test.cc
  - core/store/runtime/ingestion/materialization_strategy_types.h
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_facade_test.cc
  - core/store/replica/collective_disk_loader.h
  - core/store/replica/collective_disk_loader.cc
  - core/store/replica/collective_disk_loader_test.cc
  - daemon/service/owned_binding_service_test.cc
  - tests/python/test_binding.py
links:
  design: ../designs/0114-collective-first-binding-realization-for-tp-serving-startup.md
---

# Objective

Implement `0114` against the current repository as it actually exists today:

- keep `Binding.realize_from(...)` on the shared TensorCast runtime trunk,
- make the same-binding TP startup path `collective-first` by default,
- stop treating compatibility byte ranges as the primary execution graph,
- split planner truth from actual execution truth in the source-bound
  diagnostics surface,
- converge same-binding `canonical_full` publication to exactly one
  identity-forming hash over the final post-finalize serving bytes,
- require that surviving identity-forming hash to execute on GPU for the
  audited same-binding `canonical_full` startup path rather than silently
  downgrading to D2H/CPU hashing,
- and use this document as the single active total execution plan for the
  remaining source-bound TP startup convergence work.

Plan ownership rule:

- `0113` remains the accepted closure design for capability and version handoff,
  single-mint identity constraints, and delete-gate invariants;
- active execution tracking is centralized here in `0114` plan;
- `docs/plans/0113-step3p5-closure-and-sot-convergence.md` remains a historical
  closure-handoff record rather than the active total checklist.

# Implementation Snapshot

As of 2026-04-02, `0114` is partially landed but not complete in the
repository.

Implemented so far:

- explicit execution-core `pad fill` exists and is treated as local typed work
  rather than as hidden residual-pad coverage;
- source-bound mapped plan assembly no longer overwrites
  `RepresentationWorkPlan.residual_fallback_map` with the compatibility map;
- mapped collective execution now consumes a lane-local planner lowering rather
  than reading `residual_fallback_map` as the primary request graph;
- additive `SourceBoundPlanDiagnostics` and actual-backend byte counters are
  exposed through the daemon proto and Python SDK;
- execution diagnostics now also expose `collective_skip_reason`, so mounted
  generic fallback can report why collective was not used instead of surfacing
  only `collective_handled=0`;
- `GetServerConfig.source_bound_contract_version` advertises `3`.

Still not complete:

- the mounted Step3p5 closure path must be judged on the real operator
  entrypoint, not on an explicit-address lab variant:
  - `tensorcast-cli daemon start --blocking --config vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml --global-store-mode start`
    in `internal-vllm` owns the daemon config and local Global Store startup;
  - `vllm serve ... --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_show_daemon_logs":true,"tensorcast_enable_runtime_binding":true}'`
    only attaches to the current local daemon session and leaves daemon
    strategy/config unchanged;
- mounted TP-startup evidence is still incomplete:
  - 2026-04-02 Step3p5 reruns reached same-binding startup and serving
    readiness on 8xH800, but daemon execution remained generic-dominant;
  - after fixing the real packaged daemon config and rerunning with a rebuilt
    daemon binary, the mounted sample still reported
    `collective_handled=0`,
    `dominant_executor=SourceOrderedMappedTargetExecutor`, and
    `collective_skip_reason=planner_typed_work_not_collective_admitted`;
  - that mounted sample was taken while the real `internal-vllm` packaged
    daemon config still had `enable_owner_file_collective=false`; the operator
    config is now corrected to `true`, but the mounted rerun on the corrected
    config has now been captured and narrowed the remaining blocker to planner
    admission rather than operator config;
  - a second representative `BINDING_FINALIZE` family such as Mixtral is not
    currently available through a documented local mounted runbook in this
    environment;
- same-binding identity and hash convergence is still incomplete on the real
  Step3p5 path:
  - the 2026-04-02 mounted/profile evidence still shows one explicit
    post-finalize `binding.seal_current()` hash at about `20.08s/rank` and one
    explicit `seal_from_cut.compute_data_multihash` at about `15.43s/rank`;
  - the source-bound `realize_from(...)` path also still enters
    `local_only_ready` and precomputes a sealed commit result before
    framework-side finalize, so the measured path still has at least two
    model-scale hash rounds and likely three on Step3p5;
  - `binding-subject` closeout already captures `sealed_commit_result` and can
    feed it into `commit_lease_in_place(..., identity_override=...)`, but that
    reuse currently happens only after `engine.seal_assembly_from_cut(...)`
    has already paid a second full-data hash;
  - `seal_current()` and `build_commit_lease_result(...)` still hash through
    `LipSeekableSource` D2H reads plus host hashing instead of a GPU hash path;
- public API compatibility still constrains how fast the builder path can cut
  over:
  - `Binding.realize_from(...)` and `Store.realize_into_binding(...)`
    currently require a returned `current_value` / `SealedBindingValue`;
  - removing the pre-finalize seal from the public default would therefore be a
    public API behavior change rather than a local daemon-only refactor.
- deletion of transitional bridge logic only after the mounted evidence and
  migration gates pass;
- executor evidence capture beyond unit tests still needs mounted validation for
  collective unique-source bytes, peer-transfer bytes, peak temporary bytes,
  and batch counts.

Implemented on 2026-04-02 and verified through targeted repo-level tests:

- strict preflight rejection preserves existing binding state instead of
  dirtying `READY_ARTIFACT` or `READY_LOCAL`;
- strict `require_collective` now propagates an explicit runtime fail-fast flag
  and returns before generic fallback when the collective path remains
  unhandled;
- mapped collective lane maps are finalized at runtime against the selected
  source byte space and source layout before collective launch;
- collective success now continues into sparse generic residual execution and
  then local typed overlay, with split actual byte accounting for collective,
  generic backend, and local typed lanes;
- planner summary now treats `collective_lane_eligible` as lane executability,
  while `strict_pure_collective_eligible` remains the whole-request
  pure-collective gate.
- the real Step3p5 packaged daemon config under `internal-vllm` now enables
  owner-file collective by default, matching the intended collective-first
  operator path instead of pinning the mounted rerun to generic-only behavior.

Still open after behavior closes:

- mounted TP-startup evidence remains open:
  - Step3p5 reruns are captured, but they are still generic-dominant rather
    than collective-dominant; the latest rebuilt-daemon sample now identifies
    the blocker as `planner_typed_work_not_collective_admitted`;
  - the second-family mounted sample is still environment-blocked;
- deletion of transitional bridge logic only after the mounted evidence and
  migration gates pass.

## Verification Record

Historical verification completed on 2026-04-01 for the landed first cut:

- `bash tools/build_proto_python.sh`
- `bazel test //core/store/materialization/contracts:representation_contract_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //daemon:materialization_target_plan_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //core/store/replica:collective_disk_loader_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source .venv/bin/activate && pytest tests/python/test_daemon_ctl_resolve_rpc_config.py`
- `source .venv/bin/activate && ruff check tensorcast/types.py tensorcast/api/store/owned_binding_slot.py tensorcast/api/store/binding.py tensorcast/api/store/__init__.py tests/python/test_binding.py tests/python/test_daemon_ctl_resolve_rpc_config.py`

Targeted verification completed on 2026-04-02 for the strict/runtime closure:

- `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --verbose_failures --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --verbose_failures --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source /data/workspace/internal-vllm/.venv/bin/activate && pytest /data/workspace/internal-vllm/tests/model_executor/model_loader/test_tensorcast_loader_config.py -q`
- `source /data/workspace/internal-vllm/.venv/bin/activate && pytest /data/workspace/internal-vllm/tests/entrypoints/openai/test_weight_version_endpoints.py -q`
- `source /data/workspace/internal-vllm/.venv/bin/activate && pytest /data/workspace/internal-vllm/tests/v1/worker/test_gpu_model_runner_tensorcast_reload_guards.py -q`
- `source /data/workspace/internal-vllm/.venv/bin/activate && pytest /data/workspace/internal-vllm/tests/v1/engine/test_weight_reload_overrides.py -q`

Targeted verification completed on 2026-04-02 for the Step3p5 operator-path
fix and mounted diagnostics clarification:

- `bash tools/build_proto_python.sh`
- `bazel test //core/store/runtime/ingestion:materialization_facade_test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --verbose_failures --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source /data/workspace/internal-vllm/.venv/bin/activate && pytest /data/workspace/internal-vllm/tests/model_executor/model_loader/test_tensorcast_loader_config.py -q`
- `bazel build //daemon:tensorcast_daemon --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

Mounted validation captured on 2026-04-02 and still left open as a closure gate:

- Step3p5 mounted rerun used the real operator runbook from
  `/data/workspace/internal-vllm`:
  - `tensorcast-cli daemon start --blocking --config vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml --global-store-mode start`
  - `LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 NCCL_DEBUG=WARN VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0 vllm serve /mnt/step3-alignment/checkpoints/step3p5_flash_release_hf_mtp3_fp8 --tensor-parallel-size 8 --disable-cascade-attn --reasoning-parser=step3p5 --tool-call-parser=step3p5 --enable-expert-parallel --enable-auto-tool-choice --max-model-len 4096 --load-format tensorcast --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_show_daemon_logs":true,"tensorcast_enable_runtime_binding":true}' --gpu-memory-utilization 0.82`
- Runtime attach semantics for that rerun:
  - `tensorcast_daemon_address` was intentionally omitted;
  - `TensorcastModelLoader` validated the extra-config, then called
    `tc.init(mode="connect", show_daemon_logs=True)`;
  - `tensorcast.startup.init(mode="connect")` resolved the current local
    daemon session through `runtime.status()` / session metadata written by
    the CLI-launched daemon;
  - `tensorcast_collective_policy` was omitted, so the mounted run used the
    default `allow_not_eligible_fallback` policy, which is the intended
    collective-first mixed-execution closure target for `0114`.
- Step3p5 mounted evidence observed:
  - daemon loaded `/mnt/step3-alignment/checkpoints/step3p5_flash_release_hf_mtp3_fp8` and reported `DiskLoader initialized: 44 partitions, total size: 208561446144 bytes`;
  - all eight TP ranks logged `Tensorcast bootstrap realized SAME_BINDING_FAST_PATH`;
  - `vllm serve` reached readiness with `Starting vLLM API server 0 on http://0.0.0.0:8000`; `/health` returned `200`; `/v1/models` exposed the Step3p5 root path;
  - the first operator-path rerun still used a stale daemon binary and only
    proved that the packaged config fix had not yet changed the generic-only
    result:
    `materialize_mapped_into_target execution_commit ... committed_bytes=25550557848 fallback_bytes=25550556928 actual_local_typed_bytes=920 actual_generic_backend_bytes=25550556928 collective_handled=0 dominant_executor=SourceOrderedMappedTargetExecutor`;
  - follow-up root-cause review after that rerun found the real operator config
    issue directly: the packaged daemon config at
    `internal-vllm/vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml`
    still had `enable_owner_file_collective=false`, so the mounted path was
    pinned to `planner_collective_strategy_disabled` before execution;
  - this repo now fixes that packaged config to
    `enable_owner_file_collective=true` and exposes
    `ExecutionDiagnostics.collective_skip_reason` so future mounted reruns can
    distinguish planner/config rejection from runtime executor failure without
    relying only on daemon log folklore;
  - a second rerun on the rebuilt daemon binary kept the same startup/same-binding
    success result but also surfaced the current precise blocker:
    `materialize_mapped_into_target execution_commit ... committed_bytes=25550557848 fallback_bytes=25550556928 actual_local_typed_bytes=920 actual_generic_backend_bytes=25550556928 collective_handled=0 collective_skip_reason=planner_typed_work_not_collective_admitted dominant_executor=SourceOrderedMappedTargetExecutor`;
  - mounted prod serving did not expose `/weight_version` because `VLLM_SERVER_DEV_MODE` was not enabled for this run, so mounted evidence was taken from daemon/vLLM startup logs plus `/health` and `/v1/models` rather than the dev-only route.
- Second-family availability check remains open:
  - `/mnt/step3-alignment/checkpoints` currently contains only `step3p5_flash_release_hf_mtp3_fp8` and `step3p5_flash_v1_release_hf_mtp3_fp8`;
  - broader mounted paths contain other model directories, but no documented local mounted runbook or known-ready representative second `BINDING_FINALIZE` family such as Mixtral was found during this pass.

## Mounted Closure Semantics

The Step3p5 closure gate must be judged on the exact operator entrypoint above.
That entrypoint has three properties that the plan must keep explicit:

- daemon lifecycle and daemon strategy/config are owned by
  `tensorcast-cli daemon start ... --config ... --global-store-mode start`;
- vLLM bootstrap is attach-only in this mode:
  `tensorcast_init_mode=connect` without `tensorcast_daemon_address` means the
  loader binds to the current local daemon session rather than reconfiguring
  the daemon from inside `vllm serve`;
- and the mounted success target is the default
  `allow_not_eligible_fallback` policy, not a strict
  `require_collective` diagnostic override.

Closure evidence for Step3p5 therefore has to come from the surfaces this real
path actually exercises:

- vLLM runtime-init logging showing `mode=connect` against the resolved local
  daemon address;
- `Tensorcast bootstrap realized SAME_BINDING_FAST_PATH`;
- daemon `materialize_mapped_into_target execution_commit ...` evidence;
- additive runtime diagnostics such as `collective_skip_reason` when collective
  is requested but not used;
- and serving readiness from `/health` plus `/v1/models`.

`/weight_version` is not a required mounted closure signal unless
`VLLM_SERVER_DEV_MODE` is enabled, because the route is dev-only on the current
downstream server.

## Single-Hash Closure Semantics

The current mounted Step3p5 evidence is sufficient to treat identity/hash
closure as a first-class remaining gate in this active plan, not as a closed
historical detail.

Grounding from the 2026-04-02 Step3p5 cold-start profile:

- visible hash work on the current path is already about `35.5s/rank`:
  - `bootstrap.binding_finalize.seal_current ~= 20.08s/rank`
  - `seal_from_cut.compute_data_multihash ~= 15.43s/rank`
- the source-bound `tensorcast_realize` scope also still contains a hidden
  `local_only_ready` seal mint, so realistic total hash cost is closer to
  `50.9s ~ 55.6s/rank` on the current path rather than `35.5s/rank`;
- the visible hash cost alone is already about `14.4%` of the full
  `vllm serve -> /health ready` time and about `19.9%` of the worker
  model-load window;
- the same-binding path therefore is not yet in the `0113` steady state,
  because the runtime still depends on more than one model-scale identity hash
  over effectively the same finalized serving byte image.

Normative closure target for this plan:

- the audited same-binding `canonical_full` startup path must have exactly one
  identity-forming full-data hash;
- that hash must occur after framework finalize has produced the final serving
  byte image, not before finalize;
- binding-subject closeout and promotion may validate, publish, attach layout,
  and register metadata, but they must reuse the seal-produced identity rather
  than reminting from the same immutable bytes;
- for the audited same-binding `canonical_full` startup path, the surviving
  identity-forming hash must execute on GPU;
- in that audited path, silent D2H/CPU fallback is forbidden:
  if TensorCast cannot prove a GPU hash over the final canonical serving byte
  space, the operation must fail closed instead of quietly downgrading.

Scope of the mandatory GPU-hash rule in this plan:

- same-binding builder path,
- `canonical_full` publication subject derived from a daemon-owned binding
  current value,
- final serving bytes hosted on one local GPU-backed binding layout,
- and canonical coverage provable from `bound_canonical_spans` / final binding
  layout metadata.

This rule is intentionally narrower than "all possible future publish shapes".
The closure target here is the audited startup path that `0112` and `0113`
already treat as the preferred same-binding steady shape.

## Remaining Closure Gates

The highest-priority behavior gaps from `plan.md` are now implemented in code
and have targeted unit verification. The remaining open work is the
rollout/evidence layer:

- close same-binding identity and hash convergence on the real startup path
  before treating `seal_reuse` as achieved in operator reality;
- close mounted TP-serving evidence for at least two representative
  shared-source startup cases;
- convert the current Step3p5 rerun from "startup succeeded but execution is
  still generic-dominant" into evidence that shows collective-lane committed
  bytes and a non-generic dominant executor;
  current blocker: `planner_typed_work_not_collective_admitted`;
- delete transitional bridge logic only after those evidence and migration
  gates pass.

# Open Work

The completed behavior closures above are intentionally not repeated below.
The remaining work is only the still-open execution, evidence, and delete-gate
surface.

## Open Milestones

- [ ] Milestone A: close same-binding `canonical_full` identity reuse before the
  seal worker
  - keep `representation_publish` on the existing assembly attempt trunk, but
    stop paying `seal_from_cut.compute_data_multihash` when the canonical slot
    is a daemon-owned sealed binding current value that already carries
    `sealed_commit_result`;
  - route that path through one subject-first closeout step that reuses the
    seal-produced descriptor and artifact id before `engine.seal_assembly_from_cut(...)`
    would otherwise mint a second identity;
  - preserve readiness cut capture, closeout prevalidation, layout attachment,
    workspace `artifact_binding`, serving-manifest validation, and attempt
    lifecycle semantics on the same trunk;
  - make the serving execution diagnostics for this path report
    `hash_rounds=0`, `hash_location=seal`, and
    `identity_mint_strategy=seal_reuse`.
- [ ] Milestone B: hard-cut the builder source-bound path to execution-only
  `realize_from(...)`
  - remove the implicit pre-finalize `local_only_ready` seal from the
    realization-plan path instead of preserving it as a public default;
  - cut `Binding.realize_from(...)` / `Store.realize_into_binding(...)` to the
    target-state semantics where realization writes bytes but does not seal or
    return a `SealedBindingValue`;
  - make the audited builder path use the explicit binding update window:
    `begin_update(...) -> realize_from(...) -> post-bind finalize ->
    seal_current()`;
  - update the stable source-bound readiness/version surface for the direct
    contract change rather than hiding it behind an additive compatibility
    mode.
- [ ] Milestone C: require GPU hash for the surviving identity-forming hash on
  the audited same-binding `canonical_full` path
  - add a GPU hash path to the seal flow used by `seal_current()` /
    `build_commit_lease_result(...)` so final serving identity can be minted
    without `LipSeekableSource` D2H reads plus host hashing;
  - define the audited same-binding `canonical_full` startup path as GPU-hash
    mandatory:
    if TensorCast cannot hash the final canonical serving byte space on GPU for
    that path, the operation must fail closed rather than silently downgrade to
    D2H/CPU hashing;
  - extend typed diagnostics so mounted evidence can prove not only hash
    location and round count, but also hash backend, bytes hashed, wall time,
    and whether the hash was identity-forming.
- [ ] Milestone D: close mounted evidence for Step3p5
  - capture a mounted rerun using the exact operator path above:
    CLI-started daemon from `internal-vllm` packaged config plus
    `vllm serve` attach via `tensorcast_init_mode=connect` with no explicit
    `tensorcast_daemon_address`;
  - confirm the mounted run still reaches same-binding startup and serving
    readiness on that path;
  - additionally show collective-lane committed bytes and a non-generic
    dominant executor, rather than the current
    `collective_handled=0` /
    `dominant_executor=SourceOrderedMappedTargetExecutor` sample;
  - additionally confirm that the mounted startup path now reports one
    identity-forming GPU hash rather than multiple hash rounds or any
    D2H/CPU hash fallback.
- [ ] Milestone E: capture a second representative `BINDING_FINALIZE` family
  - use a documented local mounted runbook for a second family such as Mixtral;
  - if the environment still does not provide a known-ready local sample, keep
    this gate open rather than substituting an undocumented family ad hoc.
- [ ] Milestone F: emit mounted executor and hash evidence beyond current unit
  tests
  - surface actual collective bytes,
    unique source bytes,
    peer-transfer bytes,
    peak temporary bytes,
    and batch count in mounted validation;
  - surface actual hash backend,
    identity-forming hash bytes,
    hash wall time,
    and any non-identity-forming residual hash work in mounted validation.
- [ ] Milestone G: close delete gates only after behavior and mounted evidence
  both pass
  - remove transitional bridge fields and residual-map-based collective
    gating only after Milestones A through F are complete;
  - remove the binding-subject second-stage hash path only after Milestones A
    through F prove true `seal_reuse` plus mandatory GPU hash on the audited
    path;
  - retire stale active references to
    `docs/plans/0113-step3p5-closure-and-sot-convergence.md`;
  - clean up old policy aliases only after the mounted evidence and downstream
    migration gates pass.

## Concrete Tasks

- [ ] Task A1: restructure binding-subject `canonical_full` closeout so
  `assembly_operation_service.cc` reaches a true reuse path before
  `engine.seal_assembly_from_cut(...)` would run a second full-data hash.
- [ ] Task A2: keep `artifact_binding(workspace_assembly_id -> artifact_id)`,
  layout attachment, serving-manifest validation, and attempt finalization on
  the same typed closeout trunk even when Milestone A bypasses the seal worker
  hash.
- [ ] Task A3: make the closeout result and Python decode path expose
  `seal_reuse` only when the second-stage full-data hash is actually absent on
  that request.
- [ ] Task B1: remove the daemon-side `local_only_ready` default from the
  realization-plan source-bound path and stop computing a sealed commit result
  inside `RefillOwnedBinding` for the audited builder flow.
- [ ] Task B2: update `RefillOwnedBindingResponse`, `OwnedBindingSlot`,
  `Binding.realize_from(...)`, and `Store.realize_into_binding(...)` to the
  direct target-state semantics where realization is execution-only and no
  `current_value` is returned from that ingress.
- [ ] Task B3: switch the `internal-vllm` same-binding builder path to the
  explicit update window so `binding.seal_current()` after finalize becomes
  the only seal point for Step3p5 / Mixtral.
- [ ] Task C1: implement GPU hashing for the seal path in
  `daemon/state/lip_manager.cc` and the supporting lower layers rather than
  relying on `LipSeekableSource` D2H reads.
- [ ] Task C2: make the audited same-binding `canonical_full` path fail closed
  if Milestone C cannot prove GPU hashing over the final canonical serving byte
  space.
- [ ] Task C3: extend typed execution/hash diagnostics with enough information
  to prove GPU hash vs D2H/CPU hash in repo tests and mounted validation, and
  advance `source_bound_contract_version` for the direct typed hash contract
  change.

## Remaining Acceptance Checks

- [x] `bazel test //core/store/materialization/contracts:representation_contract_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //daemon:materialization_target_plan_utils_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //core/store/replica:collective_disk_loader_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `bazel test //daemon:owned_binding_service_test --test_env=TENSORCAST_CUDA_BACKEND=fake`
- [x] `source .venv/bin/activate && pytest tests/python/test_binding.py`
- [ ] `bazel test //daemon:owned_binding_service_test //core/store/runtime/ingestion:materialization_facade_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors`
  covering:
  - binding-subject `seal_reuse` with no preceding `seal_from_cut.compute_data_multihash`
  - fail-closed behavior when the audited path cannot satisfy the mandatory
    GPU-hash rule
- [ ] `source .venv/bin/activate && pytest tests/python/test_binding.py tests/python/test_serving_publication_types.py tests/python/test_assembly_attempt.py`
  covering:
  - direct execution-only `realize_from(...)` behavior
  - binding-subject closeout result decoding
  - typed hash diagnostics surfacing
- [ ] `source /data/workspace/internal-vllm/.venv/bin/activate && pytest /data/workspace/internal-vllm/tests/model_executor/model_loader/test_tensorcast_loader_config.py -q`
  covering:
  - builder migration to the direct execution-only `realize_from(...)` contract
  - bootstrap summary showing one identity-forming hash and GPU hash backend
- [ ] mounted TP-serving evidence for at least two representative shared-source
  startup cases
  Current state: Step3p5 rerun completed and reached serving readiness, but it
  did not satisfy the collective-dominance close criteria on the real
  CLI-daemon plus `connect` attach path, still paid multiple model-scale hash
  rounds, and no second representative family was available locally.

## Backout

- If a regression is found before the delete gates pass, back out by reverting
  the `0114` implementation commits rather than leaving a new permanent runtime
  toggle behind.
- A short-lived bring-up guard is acceptable during development, but it must be
  deleted before this plan is closed.

# Risks & Tracking

- [ ] Risk: mounted evidence remains generic-dominant and creates false
  confidence that `collective-first` is closed.
  Mitigation: require mounted evidence to show lane bytes and a dominant
  executor shift away from generic-only behavior before closing the plan.
  Current evidence: the 2026-04-02 Step3p5 rerun still reported
  `collective_handled=0` and
  `dominant_executor=SourceOrderedMappedTargetExecutor`, so this risk remains
  open and still blocks delete gates.
- [ ] Risk: the second representative family remains environment-blocked and
  stalls closure even if Step3p5 improves.
  Mitigation: keep the gate explicit and require a documented local mounted
  sample rather than silently substituting an unrelated family.
- [ ] Risk: executor evidence remains too coarse for mounted validation.
  Mitigation: do not close the plan until mounted validation reports actual
  collective bytes, unique source bytes, peer-transfer bytes, peak temporary
  bytes, batch count, hash backend, hash bytes, and hash wall time.
- [ ] Risk: the direct public-contract cutover for `Binding.realize_from(...)`
  and `Store.realize_into_binding(...)` fans out into more repo surfaces than
  expected.
  Mitigation: treat Milestone B as an intentional hard cut tied to the revised
  design authority, update the SDK/proto/tests in one sequence, and avoid
  preserving the old implicit-seal semantics in parallel.
- [ ] Risk: the audited path silently falls back to D2H/CPU hashing and still
  claims `single-mint` in high-level summaries.
  Mitigation: make GPU hash mandatory for the audited same-binding
  `canonical_full` startup path and fail closed when the GPU-hash preconditions
  are not met.
