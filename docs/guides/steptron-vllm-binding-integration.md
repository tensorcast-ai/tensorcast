# Steptron and vLLM Binding Integration

## Goal

Describe the concrete landing path for the binding work from:

- `docs/designs/0084-binding-unified-model-and-contract.md`
- `docs/designs/0085-distributed-binding-assembly-and-coordinator.md`

This guide is intentionally grounded in the current `steptron` training code and
the current `internal-vllm` TensorCast runtime so the design can be mapped to
real integration points instead of staying abstract.

Repository roots referenced below:

- `steptron`: `/data/workspace/steptron`
- `internal-vllm`: `/data/workspace/internal-vllm`

## Current Steptron Training Path

The current `steptron` path for the referenced PPO experiment is:

1. experiment setup resolves the HF bootstrap path in
   `playground/users/ys/step4air/rl/step4air_toy_fully_async_dual8_h800.py`
   (`_resolve_init_hf_path` around line 248, `Exp.__init__` around line 693).
2. `PPOTrainer.before_train()` calls `self.load_checkpoint()` and then creates
   `PackedModel(...)` in `steptron/core/ppo_trainer.py` around lines 147-162.
3. `PackedModel.setup_model()` constructs model chunks through
   `model_config.build_model()` and then moves them onto CUDA in
   `steptron/core/multi_model.py` around lines 122-167.
4. `load_model_checkpoint()` dispatches to `model.load_hf_state_dict(...)` in
   `steptron/core/utils.py` around lines 247-269.
5. `load_hf_state_dict()` performs `reshaper.forward(...)` and then
   `load_state_dict(...)` in `steptron/model/module.py` around lines 290-305.

Important current properties:

- model construction happens before HF weights are loaded
- `@init_weight_callback` recursively runs `init_model_weight()` immediately
  after model construction in `steptron/model/utils.py` around lines 130-146
- tensor-parallel layers allocate real parameter storage in their constructors
  unless told otherwise
- TP layers already expose `weight_memory_loc` hooks in
  `steptron/core/tensor_parallel/layers.py` around lines 1205-1277 and
  1469-1542

## Current internal-vLLM TensorCast Path

The current `internal-vllm` TensorCast runtime already has the serving-side
consumer model we want to preserve:

- `tensorcast_loader.py` can:
  - load directly from a serving artifact with `artifact.bind(...)`
  - or bootstrap a local/source input into a serving artifact and then bind it
  - or acquire a retained serving binding through explicit retained authority
  - and later reload through `binding.swap(...)`
- `gpu_model_runner.py` consumes the dedicated TensorCast serving reload path
  using `ServingArtifactLocator` and manifest runtime policy
- `api_server.py` gates `/reload_serving_artifact` with drain + serialized
  reload, while `/set_model_weight` is rejected for `load_format="tensorcast"`

Concrete current anchors:

- direct serving-artifact startup path in
  `internal-vllm/vllm/model_executor/model_loader/tensorcast_loader.py`
  (`TensorcastModelLoader.load_model(...)`)
- retained binding acquire path in
  `internal-vllm/vllm/tensorcast/retained_binding.py`
- local bootstrap path in
  `internal-vllm/vllm/model_executor/model_loader/tensorcast_builder/local_dir_prepare.py`
- reload-by-swap path in
  `internal-vllm/vllm/model_executor/model_loader/tensorcast_loader.py`
- HTTP drain + serialized serving reload in
  `internal-vllm/vllm/entrypoints/openai/api_server.py`

This means the integration goal is not to redesign vLLM reload semantics. It is
to make training publish the right versions in the right representation.

## TensorCast-Side Readiness And Diagnostics Contract

The current TensorCast-side source-bound readiness surface for this integration
has one active cut point:

- `source_bound_contract_version >= 4`
  - same-binding `Binding.realize_from(...)` /
    `Store.realize_into_binding(...)` are execution-only ingress points that
    return `BindingUpdateEpoch` rather than a sealed current value;
  - typed hash diagnostics on the surviving seal path are part of the stable
    downstream contract;
  - downstream builder code must gate on this version before depending on the
    `update_epoch` response shape.

The public SDK surfaces to consume for those cut points are:

- `binding.last_execution_diagnostics`
  - actual execution facts such as `actual_collective_committed_bytes`,
    `actual_local_typed_bytes`, `actual_generic_backend_bytes`,
    `collective_failure_class`, `dominant_executor`, and
    `direct_write_supported`
- `binding.last_source_bound_plan_diagnostics`
  - planner facts such as `execution_plan_kind`,
    `planned_collective_candidate_bytes`,
    `planned_collective_admitted_bytes`,
    `planned_local_typed_bytes`,
    `planned_non_admitted_typed_bytes`,
    `planned_generic_residual_bytes`,
    `planner_reject_reason_buckets`, `planner_version`, and `plan_hash`

Downstream integration summaries should prefer those typed surfaces rather than
daemon log parsing or repo-version heuristics.

## Integration Principle

The binding work should land around one simple separation:

- `steptron` owns framework construction, optimizer execution, and local weight
  mutation
- TensorCast owns stable weight location lifecycle, sealing, publishability, and
  distributed assembly
- `internal-vllm` continues to consume published serving versions through the
  existing runtime binding path

## Local Binding Landing in Steptron

### Recommended Initialization Shape

Use a layout-first initialization path.

The target shape is:

1. do a planning build that determines the full local weight layout
2. create one local contiguous binding slab for all local model weights
3. attach model parameters to views of that slab
4. load HF weights directly into that final storage

This is preferable to:

- allocating one normal model weight graph first and adopting it later
- or forcing a framework-side gather/dump path before publish

### Why Planning Is Needed

The binding design assumes one coalesced local slab. That requires knowing the
full local parameter layout before final allocation.

In `steptron`, a naive online storage provider is not enough because:

- `PackedModel.setup_model()` loops over all local model chunks
- `init_weight_callback` runs immediately after construction
- TP layers allocate storage in constructors

The provider therefore must be driven by a full local layout plan, not by
incremental “give me the next tensor” allocation.

### Practical Hook Points

The practical hooks in `steptron` today are:

- `PackedModel.setup_model()` in `steptron/core/multi_model.py`
  - around lines 122-167
  - natural place to add a planning mode and a binding-backed build mode
- `model_config.build_model()`
  - `Step4Model` builder is in
    `playground/pretrain/step4air_v4_f.py` around lines 593-605
  - `Step4RewardModel` builder is in the RL recipe file around lines 517-530
  - natural framework entry to thread a planning or weight-provider object
- TP layer constructors in `steptron/core/tensor_parallel/layers.py`
  - already accept `weight_memory_loc`
- `init_weight_callback` in `steptron/model/utils.py`
  - must be disabled or made binding-aware in planning mode
- `load_hf_state_dict()` in `steptron/model/module.py`
  - around lines 290-305
  - should continue to own HF -> steptron parameter mapping
  - should write directly into binding-backed tensors

### Recommended Sequence

1. Add a `plan` build mode for `steptron` model construction.
2. Use the planning build to collect the full local tensor set, shapes, dtypes,
   alias/tie relations, and persistent buffers.
3. Convert that result into `BindingLayout`.
4. Create one local `Binding` with one contiguous slab.
5. Rebuild or attach the real model to binding-backed tensors.
6. Run `load_hf_state_dict()` so reshaped HF weights land directly in the final
   slab.
7. Call `seal_current(...)` for the initial local sealed value once load
   completes.
8. Start an assembly attempt on the existing assembly/layout trunk and
   contribute that sealed value into the required contribution contract, even
   for the single-rank case.
   - For single-rank landing, use a legal contract on the trunk:
     `canonical_full` or deterministic multi-view coverage.
   - Do not use a binding-local full-coverage piece shortcut.

## Distributed Publish Landing for Steptron

### Phase-1 Scope

The first production landing should stay within:

- PP / VPP / EP / CP
- `TP1`
- subset-only coverage

This matches the target experiment class better than a general `TP > 1`
transform-aware design.

### What Each Rank Owns

For the current production direction:

- one local process may own multiple local model chunks because `PackedModel`
  loops over `vp_size`
- EP ownership is a tensor-subset problem once global expert ids are fixed
- CP does not change weight-byte ownership materially

The correct distributed publish shape is therefore:

- one local binding per process
- one global `LayoutSpec` for the full model family, including the expected
  contribution view set for layout-derived disjoint-piece contracts
- one planner-derived `BindingContributionPlan` per local binding
- one explicit snapped contribution contract per publish attempt
- many local sealed-value `contribute_to_assembly(...)` operations
- one coordinator-triggered source `seal_assembly(...)`
- one published model-version result carrying source lineage now
- and a serving-lineage extension only after typed child closeout contracts
  exist

### What TensorCast Must Do

TensorCast must own:

- `LayoutSpec` capture, validation, and expected-view completeness enforcement
- explicit contribution-contract capture and snapshot
- local coverage capture
- binding-backed contribution that compiles to the same structural
  view-or-piece registration trunk used by other assembly frontends
  - phase 1 does not require a LIP-backed piece path
- distributed completion tracking
- contributor liveness through the existing lease/guard/finalizer runtime
- final source `seal_assembly(...)`
- source immutable version-key publication in the current dependency-ready wave
- optional source -> serving builder or publisher only in the successor wave
  after typed child closeout contracts exist
- final serving-key or serving-manifest publication only in that successor wave

`steptron` should not:

- gather the full model
- dump per-rank fragments for later reassembly outside TensorCast
- restate tensor subsets every update

## Serving-Artifact Hand-off to internal-vLLM

### Preferred End State

The best consumer contract remains:

- `steptron` publishes serving-compatible versions
- `internal-vllm` receives those versions as a serving selector
  (`selector.kind="version_key"` or `selector.kind="artifact_ref"`)
- `internal-vllm` reloads through `/reload_serving_artifact` and the current
  runtime binding path

This keeps `internal-vllm` close to its current design:

- startup may still use direct serving-artifact bind or bootstrap bind-into
- reload stays `binding.swap(...)`
- selector durability and unhealthy handling stay in place

### Representation Boundary

If the training-local weight layout is not already the final serving
representation, then the integration must preserve the existing source vs
serving split:

- training local binding first seals a local value; TensorCast then promotes it
  through the assembly trunk into a source artifact
- the current dependency-ready wave stops at source published lineage and
  returns `PublishedModelVersion` with serving fields unset
- a later typed closeout wave may extend that same lineage with a serving
  artifact
- `internal-vllm` consumes the serving artifact

Do not force `internal-vllm` back into consuming arbitrary training-local
layouts at reload time.

## Minimal Change Matrix

### Steptron

- add planning build mode
- add binding-backed build mode
- thread weight-provider or attach path through `PackedModel.setup_model()`
- keep `load_hf_state_dict()` as the mapping/fill step
- add calls to `begin_update()`, `seal_current()`, and sealed-value
  `contribute_to_assembly()`

### TensorCast

- implement layout-first local binding from `0084`
- implement binding-backed contribution on top of the existing
  assembly/layout/view-registration trunk from `0085`
- expose a published model-version result with source or serving lineage instead
  of a bare assembly artifact outcome
- keep publish/retire/activation daemon-mediated

### internal-vLLM

- keep current runtime binding and reload path as-is
- optionally add only the minimal integration needed to resolve the new
  published version keys or manifests

## Practical Rollout Order

1. Land local binding in `steptron` and prove one-slab initialization works.
2. Land single-rank `seal_current(...)` plus legal same-trunk publish
   (`canonical_full` or deterministic multi-view contract).
3. Land distributed sealed-value contribution + source assembly for `TP1`.
4. Land typed child closeout contracts for source -> serving publication.
5. Land serving publication and immutable serving-key output on that same
   lineage.
6. Point `internal-vllm` at the published serving selector and validate
   `/reload_serving_artifact`.
7. Only after that, expand toward `TP > 1` range coverage and
   transform-aware assembly.

## Common Failure Modes To Watch

- planning build and real build diverge in tensor set or alias structure
- `init_weight_callback` initializes tensors in planning mode accidentally
- `load_hf_state_dict()` writes into a staging allocation instead of the final
  binding-backed storage
- EP global-id mapping drifts between initialization and publish
- attempt-time contribution contract drifts from what the planner produced
- distributed assembly succeeds on incomplete or stale contributor data
- source assembly succeeds but serving-artifact publication or immutable
  serving-key publication does not
- `internal-vllm` is handed a source-layout version when it expects a serving
  artifact

## When To Revisit The Design

Revisit `0084` and `0085` if any of the following become true:

- `steptron` needs `TP > 1` for the production path
- local weight layouts require transform-aware distributed assembly
- daemon-owned training slabs become necessary instead of client-owned slabs
- `internal-vllm` can consume the training-local layout directly without a
  source -> serving conversion boundary
