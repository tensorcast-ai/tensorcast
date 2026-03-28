---
slug: tensor-aware-materialization-strategy-plane
title: Tensor-Aware Materialization Strategy Plane Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "proto", "docs", "tests"]
created: 2026-03-23
last_updated: 2026-03-28
related_code:
  - core/store/runtime/ingestion/materialization_facade.cc
  - core/store/runtime/ingestion/materialization_service.cc
  - core/store/materialization/contracts/loading_spec.h
  - core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc
  - core/store/materialization/dataplane/sources/byte_range_map_builder.cc
  - core/store/materialization/dataplane/sources/byte_range_mapped_source.cc
  - core/store/materialization/dataplane/view/view_plan_source.cc
  - core/store/replica/collective_disk_loader.cc
  - daemon/service/controllers/materialization_target_plan_utils.cc
  - daemon/service/controllers/materialization_mapped_copy_plan_utils.cc
  - daemon/service/controllers/target_materialization_service.cc
  - proto/tensorcast/config/v1/daemon_config.proto
  - tests/python/test_store_view_api.py
links:
  design: ../designs/0108-tensor-aware-materialization-strategy-plane.md
  related:
    - ../plans/0107-retrieval-policy-plane-cleanup.md
    - ../plans/0108-01-pre-109-strategy-plane-convergence.md
    - ../plans/0109-batched-owner-file-collective-executor.md
---

# Objective

Add a tensor-aware strategy plane to common runtime materialization so
TensorCast can:

- keep one selection-first public retrieval model,
- keep copy contract separate from execution strategy,
- choose the best executor only after semantic truth and source capabilities are
  known,
- reach host-local performance parity with `fastsafetensors`,
- preserve current correctness, external-target safety, and JFS advantages,
- reabsorb the current mapped-target and replica-layer prototypes into one
  internal planner and executor architecture.

# Execution Ownership Note

This document remains the historical landing ledger for the already-completed
`0108` work through mapped-target strategy extraction, semantic-contract
lowering, and typed rollout config.

It is no longer the primary execution queue for remaining strategy-plane work:

- `docs/plans/0107-retrieval-policy-plane-cleanup.md` owns the request
  normalizer and transport-boundary prerequisite.
- `docs/plans/0108-01-pre-109-strategy-plane-convergence.md` owns the remaining
  ordinary-replica strategy convergence, coordinator extraction, and diagnostics
  work.
- `docs/plans/0109-batched-owner-file-collective-executor.md` owns the new
  owner-file batched executor after those prerequisites land.

The unchecked items below should be read as historical scope decomposition, not
as the active ordered queue.

# Latest Status

As of `2026-03-24`, this branch has the following implementation status:

- [x] Controller-side mapped copy-plan analysis now produces tensor-job and
  concat-job candidates plus compatibility diagnostics in
  `daemon/service/controllers/materialization_mapped_copy_plan_utils.cc`.
- [x] Mapped-target controller wiring now forwards candidate executor metadata
  and source-selection context into the common runtime entrypoints.
- [x] `MaterializationFacade` mapped-target execution now includes the staged
  strategy seam for local-canonical, disk, and P2P sources plus the collective
  mapped-target handoff and residual generic fallback.
- [x] Mapped-target runtime now consumes `ResolvedMaterializationPlan` as the
  authoritative semantic contract and rejects hint/plan artifact drift instead
  of accepting duplicate semantic inputs.
- [x] Replica-side local-batched and collective mapped-target loader paths are
  still wired for the remaining prototype-owned workloads.
- [x] SDK subset selection plumbing now carries the required view-selection
  context without changing public APIs, and scalar-only subsets now fall back to
  `tensor_names` requests instead of forcing a synthetic `view_id`.
- [x] Python regression coverage was added for scalar subset fallback in
  `tests/python/test_store_view_api.py`.
- [x] Mapped-target daemon regression coverage is aligned with the stronger
  local target-access boundary and currently passes on
  `//daemon:materialize_into_mapped_target_test`.
- [x] Core-owned semantic contract extraction is landed in
  `core/store/runtime/ingestion/materialization_strategy_types.h` and is now
  threaded through the mapped-target controller/runtime boundary.
- [x] Typed rollout config is landed under
  `engine.materialization_strategy` in
  `proto/tensorcast/config/v1/daemon_config.proto` and mapped into
  `StoreEngineOptions::MaterializationStrategyConfig`.
- [x] Partial `engine.materialization_strategy` blocks now preserve the 0108
  defaults for local batched disk load and owner-file collective toggles.
- [x] Controller-produced mapped executor metadata no longer travels through
  `MaterializeHints`; mapped-target execution now consumes the internal
  resolved-plan contract directly.
- [x] Runtime env-gated mapped/local-batched/owner-file rollout controls are
  removed from the common hot path and replaced with typed daemon config.
- [x] Step3p5 real target-layout subset gate now passes on host-local SSD:
  layout remains stable for all `8` ranks, rank-by-rank digests match the
  generic baseline, and the batched daemon log now emits
  `local_batched_disk_load timings`.
- [x] The full real target-layout harness now passes on host-local SSD for the
  current probe set: `auto` and `tensorcast` keep stable target layout and
  produce digest-identical results for all `8` ranks.
- [x] Full `vllm serve` tensorcast correctness now passes for the planned
  completion request on:
  - host-local SSD model root
  - JFS model root
- [x] The subset gate uncovered and closed two real hot-path regressions:
  `AllocationStage` was not threading
  `StoreEngineOptions::MaterializationStrategyConfig` into `ReplicaConfig`, and
  `Replica::ensure_loaded_async()` was incorrectly trying to discover
  `DiskLoader::shared_context()` from `SeekableSource` instead of the owning
  `DiskLoader`.
- [ ] The dedicated common-runtime local tensor-aware executor for mapped-target
  requests is still follow-up; `executor_preference=TENSOR_AWARE_LOCAL` is not
  yet backed by a separate local executor implementation.
- [ ] Dedicated `tensorcast_param_digest_compare.py` serve-level digest checks
  and strict performance sign-off versus `fastsafetensors` / current JFS best
  remain operational benchmark follow-up.

Verification status for this update:

- [x] `bash tools/build_proto_python.sh`
- [x] `source .venv/bin/activate && ruff check tensorcast/api/store/artifact.py tensorcast/api/store/view_composer.py tests/python/test_store_view_api.py`
- [x] `source .venv/bin/activate && pytest tests/python/test_store_view_api.py`
- [x] `bazel build //daemon:materialization_mapped_copy_plan_utils_lib //core/store/runtime/ingestion:materialization_facade`
- [x] `bazel build //core/store/replica:collective_disk_loader //core/store/runtime/ingestion:materialization_facade //daemon:materialize_into_mapped_target_test //core/common:daemon_config_io_test --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/common:daemon_config_io_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/store/runtime/ingestion:materialization_service_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/store/runtime/ingestion:materialization_facade_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/store/runtime/ingestion:ingestion_runtime_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //daemon:materialize_into_mapped_target_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] `bazel test //core/store/materialization/runtime/pipeline/tests:p2p_ingestion_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`
- [x] Step3p5 real target-layout subset baseline:
  `torchrun --nproc_per_node=8 tools/tensorcast_real_target_subset_bench.py --model /mnt/host0/tc-subset-cold-a ... > /tmp/tc_real_target_subset_copyA_baseline.json`
- [x] Step3p5 real target-layout subset batched validation after the hot-path
  fixes:
  `torchrun --nproc_per_node=8 tools/tensorcast_real_target_subset_bench.py --model /mnt/host0/tc-subset-cold-b ... > /tmp/tc_real_target_subset_copyB_batched_fixed.json`
- [x] Batched daemon evidence:
  `rg "local_batched_disk_load timings" /tmp/tc_daemon_0108_batched.log`
- [x] Full real target-layout host-local validation:
  `torchrun --nproc_per_node=8 tools/tensorcast_real_target_layout_bench.py --model /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8 --load-format auto ... > /tmp/tc_full_layout_auto.json`
- [x] Full real target-layout host-local tensorcast validation:
  `torchrun --nproc_per_node=8 tools/tensorcast_real_target_layout_bench.py --model /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8 --load-format tensorcast ... > /tmp/tc_full_layout_tensorcast.json`
- [x] Host-local SSD `vllm serve` tensorcast correctness request:
  `curl http://127.0.0.1:8000/v1/completions ...` returned a response beginning
  with `北京`
- [x] JFS `vllm serve` tensorcast correctness request:
  `curl http://127.0.0.1:8000/v1/completions ...` returned a response beginning
  with `北京`

# Current State & Grounding

Current execution is split across the wrong abstraction boundary:

- controller and SDK are already selection-first:
  - `tensorcast/api/store/artifact.py`
  - `daemon/service/controllers/materialization_target_plan_utils.cc`
- replica materialization already has a source-acquisition chain:
  - `core/store/runtime/ingestion/materialization_service.cc`
- but common runtime still lowers too early into:
  - `ByteRangeMap`
  - `ByteRangeCompiler`
  - `ByteRangeMappedSource`
  - in `core/store/runtime/ingestion/materialization_facade.cc`

Current prototype risks that must be corrected during implementation:

- executor-private mapped-target jobs are currently shaped in controller code and
  propagated through generic runtime hints:
  - `daemon/service/controllers/materialization_target_plan_utils.cc`
  - `daemon/service/controllers/target_materialization_service.cc`
  - `core/store/materialization/contracts/loading_spec.h`
- `build_copy_plan(...)` currently mixes semantic copy-contract truth and
  executor compatibility analysis:
  - `daemon/service/controllers/materialization_mapped_copy_plan_utils.cc`
- backend-specific coverage subtraction is currently performed in late runtime or
  replica-layer helpers:
  - `core/store/replica/collective_disk_loader.cc`
- strategy rollout is currently too close to ad-hoc environment gating in common
  runtime code:
  - `core/store/runtime/ingestion/materialization_facade.cc`
  - `core/store/replica/collective_disk_loader.cc`

Grounded observations from actual experiments:

- exact `879` workload, host-local SSD, current common collective path:
  - about `47s`
- exact `879` benchmark batched-optimal tensor-aware model:
  - about `7s` average ready time
- exact `879` single-rank common path:
  - about `7.1s`, already near `fastsafetensors`
- exact `614` sliced tensor subset:
  - TensorCast non-collective about `10.0s`
  - `fastsafetensors` about `8.7s`
- real target-layout subset:
  - apply is milliseconds
  - materialization dominates
  - current runtime still goes through generic `disk_fallback ->
    ByteRangeMappedSource -> pump_ranges`

Architecture constraints for this plan:

- `ArtifactSelection` remains the only public selection contract.
- copy contract remains distinct from selection and placement.
- external-target safety, region poison, and publication-token semantics remain
  controller-owned.
- source acquisition remains distinct from execution strategy.
- planner-generated executor ops remain internal to common runtime.
- disabling one executor path must expand residual fallback work, never suppress
  requested bytes.
- production rollout controls must converge on typed config, not ambient
  environment variables.
- because the project does not yet need historical compatibility, prototype
  coexistence should be actively removed after the strategy plane is proven.

# Phases & Milestones

- [x] Phase 1: Freeze The Layered Boundary
  - [x] Milestone 1.1: Define one explicit semantic-resolution boundary before
    strategy lowering:
    - `ResolvedMaterializationPlan`
    - `MappedCopyContract`
    - `ResolvedSourceBinding`
  - [x] Milestone 1.2: Keep controller ownership of request validation,
    external-target safety, and publication policy.
  - [x] Milestone 1.3: Document that source acquisition remains distinct from
    execution strategy for both replica and into-target flows.

- [x] Phase 2: Extract Semantic Contracts From Prototypes
  - [x] Milestone 2.1: Move target and mapped-target semantic plan ownership out
    of controller-only utilities into a core-owned internal contract library.
  - [x] Milestone 2.2: Split `build_copy_plan(...)` outputs into:
    - semantic mapped copy contract
    - executor compatibility analysis
  - [x] Milestone 2.3: Keep public SDK and proto surfaces unchanged while
    removing executor-private request hints from shared runtime contracts only
    after the replacement internal contracts exist.

- [ ] Phase 3: Residual Correctness And Observability Baseline
  - [x] Milestone 3.1: Add explicit residual accounting and
    `ExecutionCommitReport` semantics.
  - [ ] Milestone 3.2: Promote exact trace workload tooling into stable
    benchmark inputs for host-local SSD and JFS comparisons.
  - [ ] Milestone 3.3: Add planner diagnostics that explain source acquisition,
    executor choice, op mix, and residual fallback bytes.

- [x] Phase 4: Strategy Plane Lowering
  - [x] Milestone 4.1: Add the strategy-selection seam to
    `MaterializationFacade` after semantic truth and source capabilities are
    known.
  - [x] Milestone 4.2: Allow mixed execution within one request rather than
    forcing one executor for all ranges.
    Current landing covers collective plus residual generic fallback; the
    local tensor-aware mixed-execution path remains Phase 5 work.
  - [x] Milestone 4.3: Keep `ByteRangeMap` generation as the exact residual
    fallback path, not the only planner IR.

- [ ] Phase 5: Local Tensor-Aware Executor
  - [ ] Milestone 5.1: Implement host-local tensor-aware execution for
    contiguous, dim1-pack, and dedup-copy dominant workloads.
  - [ ] Milestone 5.2: Route irregular or low-confidence slices back to
    residual fallback without semantic drift.
  - [ ] Milestone 5.3: Match or beat current exact `879` benchmark and real
    target-layout subset baselines.

- [ ] Phase 6: Owner-File Collective Executor
  - [ ] Milestone 6.1: Add owner-file or owner-segment collective planning only
    where it reduces duplicate source reads.
  - [ ] Milestone 6.2: Keep host-local SSD on the faster local executor when
    collective would amplify reads or synchronization.
  - [ ] Milestone 6.3: Preserve or improve current JFS behavior.

- [ ] Phase 7: Surface Adoption And Prototype Reabsorption
  - [ ] Milestone 7.1: Make `tensor_dict` use the strategy plane by default.
  - [x] Milestone 7.2: Extend the same planner to target-backed retrieval and
    mapped-target materialization through the shared semantic plan boundary.
  - [x] Milestone 7.3: Remove controller-produced mapped execution hints and
    re-express the useful parts of the current prototype as internal planner
    analysis.
  - [ ] Milestone 7.4: Retire replica-layer local-batched late hooks for the
    workloads now owned by the strategy plane.
  - [ ] Milestone 7.5: Retire naive owner-file preload prototypes once
    owner-file collective is represented as a real planner plus executor path.

- [ ] Phase 8: Typed Config And Final Cleanup
  - [x] Milestone 8.1: Replace common-runtime env gating with typed rollout
    config and diagnostics controls.
  - [ ] Milestone 8.2: Remove obsolete late-hook and prototype-only paths that
    no longer serve the common runtime hot path.
  - [x] Milestone 8.3: Remove mapped fast-path env policy switches after mapped
    lowering is fully reabsorbed into the strategy plane.
  - [x] Milestone 8.4: Update architecture and internals docs to reflect the
    final layered boundary.

# Remaining Work Breakdown

- [ ] Land the common-runtime local tensor-aware executor for mapped-target
  requests.
  - Reuse benchmark-proven planning logic from
    `core/store/materialization/benchmarks/safetensors_load_strategy_benchmark_main.cc`.
  - Implement direct contiguous reads to final layout.
  - Implement staged row-block reads plus GPU 2D pack for dim1 patterns.
  - Implement source-slice dedup and D2D copy reuse.
  - Route irregular or low-confidence slices back to exact residual fallback.

- [ ] Retire remaining replica-side and prototype-only hooks once the common
  runtime owns those workloads.
  - Remove replica-layer local-batched late hooks after Phase 5 lands.
  - Re-express useful owner-file prototype behavior as internal executor logic.
  - Delete naive owner-file preload-only behavior once replaced.

- [ ] Continue owner-file collective evolution under the shared strategy plane.
  - Keep collective execution compatible with local external-target write
    boundaries.
  - Preserve or improve current JFS behavior.
  - Track the executor-specific deepening work alongside
    `docs/designs/0109-batched-owner-file-collective-executor.md`.

- [ ] Close the remaining regression and benchmark gaps.
  - Add planner-level C++ tests for deterministic lowering.
  - Add mapped-target regression coverage for executor disablement, partial
    eligibility, byte-identical residual fallback, region poison on `DataLoss`,
    and publication-token gating after success only.
  - Add exact-trace benchmark checks for host-local SSD and JFS.
  - Complete dedicated serve-level digest checks and strict performance sign-off
    versus `fastsafetensors` / current JFS best.

- [ ] Fold the final executor landing and prototype retirements back into the
  design/docs set once Phase 5 and Phase 6 are complete.

# Test / Rollout / Backout

## Test Plan

Microbench / exact-workload:

- [ ] Keep exact `879` and exact `614` workload tools passing.
- [ ] Add one planner-vs-generic benchmark for local-disk exact workload.
- [ ] Add one planner-vs-collective benchmark for JFS exact workload.

Correctness:

- [x] Real target-layout subset digests match generic baseline.
- [x] Full real target-layout host-local probe digests match the `auto`
  baseline.
- [ ] Planner residual coverage equals requested destination coverage for:
  - [ ] full tensor-aware local execution
  - [ ] mixed tensor-aware plus fallback execution
  - [ ] executor-disabled execution
  - [ ] executor-failed execution with residual retry
- [ ] Full `vllm serve` completion outputs remain correct on:
  - [x] host-local SSD
  - [x] JFS

Safety:

- [ ] Remote or home daemons never write directly into caller-owned CUDA
  regions.
- [ ] `DataLoss` still poisons external target regions.
- [ ] `target_publication_token` is never minted for failed or unverified target
  writes.

Performance:

- [ ] Host-local SSD end-to-end TensorCast matches or beats
  `fastsafetensors` on the same workload.
- [ ] JFS end-to-end TensorCast does not regress from current best common-path
  behavior.

### Step3p5 validation matrix

`0108` must be validated against the current Step3p5 workload before any
executor becomes the default for common runtime. The validation target is:

- host-local SSD model root:
  - `/mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8`
- JFS model root:
  - `/mnt/step3-alignment/checkpoints/step3p5_flash_release_hf_mtp3_fp8`
- vLLM workspace:
  - `/data/workspace/internal-vllm`
- TensorCast workspace:
  - `/data/workspace/tensorcast-280`
- default daemon config:
  - `/data/workspace/internal-vllm/vllm/model_executor/model_loader/configs/tensorcast/store_daemon_config.yaml`

Use the following shared environment for all Step3p5 validation unless a
specific experiment says otherwise:

```bash
export LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64:$LD_LIBRARY_PATH
export NCCL_DEBUG=WARN
export VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0
```

Important execution rule:

- typed executor/runtime config must be applied on the daemon process, not only
  on the vLLM or microbench client process.

#### 1. Exact-workload strategy proof

Use the exact Step3p5 trace workload as the source-side proof that the planner
is choosing the right execution model before any end-to-end claim.

- Trace directory:
  - `/tmp/tc_bind_debug/trace_plan`
- Convert the existing TP trace into benchmark load plans using:
  - `/data/workspace/internal-vllm/tools/tensorcast_trace_to_benchmark_plan.py`
- Compare:
  - current common/collective runtime baseline
  - benchmark `Strategy C / batched optimal`
  - exact-subset `fastsafetensors` microbench

Required commands:

```bash
cd /data/workspace/internal-vllm
python tools/tensorcast_trace_to_benchmark_plan.py \
  --trace-plan-dir /tmp/tc_bind_debug/trace_plan \
  --output /tmp/tc_exact879_load_plan.json
```

The exact-workload proof is only accepted if:

- `Strategy C` materially beats the current common/collective runtime on the
  same exact workload,
- `Strategy C` is at or below `fastsafetensors` on host-local SSD, or the gap
  is fully explained by executor scope not yet implemented in common runtime.

#### 2. Real target-layout subset gate

Use a real target-layout subset gate before full `vllm serve`.

This gate exists to prove:

- planner/executor changes preserve true vLLM target tensor layout,
- target `shape/stride/data_ptr/storage_offset` remain stable,
- the loaded bytes are digest-identical to the generic baseline,
- the new executor is actually intercepting the real TensorDict/common path.

Use:

- `/data/workspace/internal-vllm/tools/tensorcast_real_target_subset_bench.py`

The current representative Step3p5 destination subset is:

```text
model.embed_tokens.weight
lm_head.weight
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.30.moe.experts.w13_weight
model.layers.30.moe.experts.w2_weight
model.layers.30.moe.experts.w13_weight_scale_inv
model.layers.30.moe.experts.w2_weight_scale_inv
model.layers.31.moe.experts.w13_weight
model.layers.31.moe.experts.w2_weight
model.layers.31.moe.experts.w13_weight_scale_inv
model.layers.31.moe.experts.w2_weight_scale_inv
```

For the current Step3p5 trace, this subset resolves to:

- `18` source tensors
- `438` copy entries
- `4` safetensors shard files:
  - `model-00001.safetensors`
  - `model-00002.safetensors`
  - `model-00030.safetensors`
  - `model-00031.safetensors`

Create the subset name file:

```bash
cat > /tmp/tc_real_target_subset_names.txt <<'EOF'
model.embed_tokens.weight
lm_head.weight
model.layers.0.self_attn.qkv_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.30.moe.experts.w13_weight
model.layers.30.moe.experts.w2_weight
model.layers.30.moe.experts.w13_weight_scale_inv
model.layers.30.moe.experts.w2_weight_scale_inv
model.layers.31.moe.experts.w13_weight
model.layers.31.moe.experts.w2_weight
model.layers.31.moe.experts.w13_weight_scale_inv
model.layers.31.moe.experts.w2_weight_scale_inv
EOF
```

To reduce page-cache coupling for cold A/B, create two fresh subset copies on
host-local SSD using direct I/O:

```bash
for dst in /mnt/host0/tc-subset-cold-a /mnt/host0/tc-subset-cold-b; do
  sudo rm -rf "$dst"
  sudo mkdir -p "$dst"
  sudo chown "$USER":"$USER" "$dst"
  cp /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8/config.json "$dst/"
  cp /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8/tokenizer_config.json "$dst/" 2>/dev/null || true
  cp /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8/chat_template.jinja "$dst/" 2>/dev/null || true
  for f in model-00001.safetensors model-00002.safetensors model-00030.safetensors model-00031.safetensors; do
    dd if="/mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8/$f" \
       of="$dst/$f" bs=16M iflag=direct oflag=direct status=none
  done
done
```

Baseline daemon on port `50063`:

```bash
cd /data/workspace/tensorcast-280
./bazel-bin/daemon/tensorcast_daemon \
  --config /tmp/tc_daemon_0108_base.yaml \
  > /tmp/tc_daemon_cold_base_copyA.log 2>&1
```

Batched daemon on port `50064`:

```bash
cd /data/workspace/tensorcast-280
./bazel-bin/daemon/tensorcast_daemon \
  --config /tmp/tc_daemon_0108_batched.yaml \
  > /tmp/tc_daemon_cold_batch_v2.log 2>&1
```

Current-path subset run:

```bash
cd /data/workspace/internal-vllm
torchrun --nproc_per_node=8 tools/tensorcast_real_target_subset_bench.py \
  --model /mnt/host0/tc-subset-cold-a \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --enforce-eager \
  --skip-tokenizer-init \
  --disable-collective \
  --dst-names-file /tmp/tc_real_target_subset_names.txt \
  --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_daemon_address":"127.0.0.1:50063"}' \
  > /tmp/tc_real_target_subset_copyA_baseline.json
```

Batched-path subset run:

```bash
cd /data/workspace/internal-vllm
torchrun --nproc_per_node=8 tools/tensorcast_real_target_subset_bench.py \
  --model /mnt/host0/tc-subset-cold-b \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --enable-expert-parallel \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.82 \
  --enforce-eager \
  --skip-tokenizer-init \
  --disable-collective \
  --dst-names-file /tmp/tc_real_target_subset_names.txt \
  --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_daemon_address":"127.0.0.1:50064"}' \
  > /tmp/tc_real_target_subset_copyB_batched.json
```

Required interpretation rules:

- The result is valid only if all of the following hold:
  - `layout_stable == true` for every rank
  - digests match byte-for-byte against the generic baseline
  - the daemon log for the batched run contains
    `local_batched_disk_load timings`
- If the batched run does **not** emit
  `local_batched_disk_load timings`, then the strategy-plane path has not
  intercepted the real hot path yet; the performance number must not be
  credited to the new executor.
- If the daemon log still shows only:
  - `materialize_view.disk_fallback`
  - `pump_ranges staged path`
  - `ingest_from_disk_internal`
  then the request is still executing through the generic fallback path.

Observed result on `2026-03-23` after the final hot-path fixes:

- Baseline output:
  - `/tmp/tc_real_target_subset_copyA_baseline.json`
- Batched output:
  - `/tmp/tc_real_target_subset_copyB_batched_fixed.json`
- Batched daemon log:
  - `/tmp/tc_daemon_0108_batched.log`
- All `8` ranks reported `layout_stable == true`.
- All `8` ranks reported digest equality versus the generic baseline.
- Batched daemon log emitted `local_batched_disk_load timings` for every rank.
- Materialization wall time improved from about `1.074s` average baseline to
  about `0.624s` average on the fixed batched run for this subset gate.

#### 3. Full vLLM validation

After the subset gate passes, validate the full Step3p5 workload in vLLM.

Use the exact serving environment and request shape below.

Host-local SSD `auto` reference:

```bash
LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
NCCL_DEBUG=WARN \
VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0 \
vllm serve /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8 \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --reasoning-parser=step3p5 \
  --tool-call-parser=step3p5 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --max-model-len 4096 \
  --load-format auto \
  --gpu-memory-utilization 0.82 \
  --enforce-eager
```

Host-local SSD `fastsafetensors` reference:

```bash
LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
NCCL_DEBUG=WARN \
VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0 \
vllm serve /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8 \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --reasoning-parser=step3p5 \
  --tool-call-parser=step3p5 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --max-model-len 4096 \
  --load-format fastsafetensors \
  --gpu-memory-utilization 0.82 \
  --enforce-eager
```

Host-local SSD `tensorcast` candidate:

```bash
LD_LIBRARY_PATH=/data/cuda/compat:/data/cuda/cuda-12.8/lib64:/usr/local/nvidia/lib64 \
NCCL_DEBUG=WARN \
VLLM_USE_OPTIMUS_GEMM_AR_MULMEM=0 \
vllm serve /mnt/host0/vllm-loader-bench/step3p5_flash_release_hf_mtp3_fp8 \
  --tensor-parallel-size 8 \
  --disable-cascade-attn \
  --reasoning-parser=step3p5 \
  --tool-call-parser=step3p5 \
  --enable-expert-parallel \
  --enable-auto-tool-choice \
  --max-model-len 4096 \
  --load-format tensorcast \
  --model-loader-extra-config '{"tensorcast_init_mode":"connect","tensorcast_show_daemon_logs":true}' \
  --gpu-memory-utilization 0.82 \
  --enforce-eager
```

JFS `tensorcast` candidate is identical except for the model root:

```bash
/mnt/step3-alignment/checkpoints/step3p5_flash_release_hf_mtp3_fp8
```

Correctness request:

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"","prompt":"中国的首都是：","max_tokens":16,"temperature":0}'
```

Acceptance criteria for full vLLM:

- response must be non-empty and semantically correct on:
  - host-local SSD
  - JFS
- selected tensor digests must match the `auto` baseline using:
  - `/data/workspace/internal-vllm/tools/tensorcast_param_digest_compare.py`
- host-local SSD end-to-end TensorCast must match or beat
  `fastsafetensors`
- JFS end-to-end TensorCast must not regress from current best common-path
  behavior

#### 4. Full real-target-layout harness

When GPU state is clean enough for full model allocation, also run:

- `/data/workspace/internal-vllm/tools/tensorcast_real_target_layout_bench.py`

This is the full-model, real-target-layout harness. It should be used to
confirm that:

- target layout stays stable on a full model load,
- selected target digests match the `auto` baseline,
- runtime planner changes preserve real target-layout behavior beyond the
  reduced subset gate.

If the machine is in a dirty-GPU state with invisible residual contexts, the
subset gate remains required, but final sign-off is blocked until the full
real-target-layout harness also passes on a clean run.

Observed result on `2026-03-23` for the full host-local harness:

- `auto` output:
  - `/tmp/tc_full_layout_auto.json`
- `tensorcast` output:
  - `/tmp/tc_full_layout_tensorcast.json`
- All `8` ranks reported `layout_stable == true`.
- All `8` ranks reported digest equality versus `auto` for the planned probe
  tensor set.
- Host-local full-layout load averaged about `9.488s` on `auto` and about
  `9.604s` on `tensorcast` for this run.

## Rollout

- land semantic boundary extraction first, with current generic fallback still
  authoritative
- land typed rollout config for planner and executor preference before default
  enablement
- validate exact workloads, residual coverage, target safety, and real
  target-layout subset
- enable local tensor-aware executor first for host-local `tensor_dict`
  workloads
- enable owner-file collective only after JFS or remote-backed wins are proven
- extend to target-backed and mapped-target retrieval only after common planner
  correctness is stable
- then delete prototype coexistence paths in order:
  - replica-layer local-batched late hook
  - naive owner-file preload
  - mapped fast-path env policy gates
  - executor-shaped `MaterializeHints` fields
  - remaining mapped fallback override knobs

## Backout

- keep generic byte-range execution intact as the fallback path
- gate new executors behind typed runtime config until parity is proven
- backout by forcing planner choice to residual generic fallback, not by
  reverting selection identity, copy-contract semantics, or API changes
- do not preserve prototype-only compatibility code once the new path is proven
  and adopted

# DoD Checks

- [x] `rg "mapped_tensor_jobs|mapped_concat_jobs" core/store/materialization/contracts/loading_spec.h daemon/service/controllers core/store/runtime/ingestion`
- [x] `rg "TENSORCAST_.*MAPPED|ENABLE_LOCAL_BATCHED_DISK_LOAD|ENABLE_COLLECTIVE_OWNER_FILE_STRATEGY" core/store/runtime/ingestion core/store/replica`
- [x] `rg "ResolvedMaterializationPlan|MappedCopyContract|ResolvedSourceBinding|ExecutionCommitReport" core/store`
- [x] `rg "MaterializationStrategy|allow_mixed_execution|enable_owner_file_collective" proto/tensorcast/config/v1`
- [x] planner diagnostics explain source acquisition, dominant executor, op mix,
  and residual fallback bytes on mapped-target runtime paths

# Risks & Tracking

- [ ] Risk: planner is added too late in the pipeline and never intercepts the
  real hot path.
  - Mitigation: keep the seam in `MaterializationFacade`, after semantic truth
    and source capabilities are known.

- [ ] Risk: semantic truth remains split across controller helpers, runtime
  hints, and executor code.
  - Mitigation: extract core-owned semantic contracts before hint cleanup.

- [ ] Risk: executor disablement or rollout gating drops bytes instead of
  widening residual fallback.
  - Mitigation: make residual accounting and commit reporting first-class
    planner contracts, and add explicit regression tests for disabled and mixed
    paths.

- [ ] Risk: collective ownership reduces source bytes but still loses on latency
  due to synchronization or staging.
  - Mitigation: require exact-workload benchmark wins before enabling any
    ownership executor by default.

- [ ] Risk: planner logic drifts into model-specific heuristics.
  - Mitigation: keep all decisions based on metadata, copy-contract, and
    topology facts only.

- [ ] Risk: target-backed and mapped-target flows diverge from TensorDict.
  - Mitigation: extend the same common semantic plan and strategy plane rather
    than adding a second executor stack.

- [ ] Risk: external-target safety regresses during collective or mixed
  execution adoption.
  - Mitigation: keep controller-owned target safety boundaries explicit and
    add poison plus publication-token regression tests.

# Owner Checklist

- [ ] Core owner sign-off on semantic contract extraction and the planner seam
  in `MaterializationFacade`.
- [ ] Daemon owner sign-off on controller safety ownership and diagnostics.
- [ ] SDK owner sign-off that public retrieval APIs remain unchanged.
- [ ] Integration owner sign-off on host-local and JFS end-to-end correctness.
- [ ] Docs owner sign-off on updated runtime and materialization documentation.
