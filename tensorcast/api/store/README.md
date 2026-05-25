# TensorCast Store SDK

The `tensorcast.api.store` package exposes registration, retrieval, and lazy
artifact handle surfaces. This module builds on the process-wide `Store`
singleton (`tensorcast.store()`) so callers can reuse daemon sessions without
managing clients manually.

## Artifact Handles

- `tensorcast.artifact(...)` and `Store.artifact(...)` return a lazy `Artifact`
  bound to the process `Store`. `tc.artifact("my-key")` is shorthand for
  `tc.artifact(key="my-key")`; reserved prefixes (`mi2:`/`cgid:`/`msa1:`) map to
  explicit identifier kinds. `disk:` is reserved and rejected; use
  `from_disk(...)` or `import_from_disk(...)` for local disk-backed sources.
  Handles support metadata accessors
  (`tensor_names`, `tensor_meta`, `describe`), existence checks (`exists`), and
  selective materialization via `subset(...).tensor_dict(...)` and `tensor(name, ...)`.
- `artifact.tensor_dict_with_diagnostics(...)` returns
  `TensorDictMaterializationResult(tensors, diagnostics)` so callers can collect
  source/path and timing metadata (`source`, `total_bytes`, `replica_uuid`,
  `materialize_sec`, `total_sec`) alongside tensors for benchmarking and tuning.
- `artifact.tensor_into(name, target_tensor, device=None)` materializes a single
  tensor directly into the provided buffer. Only the requested tensor must be
  present in the target mapping, so multi-tensor artifacts no longer require
  pre-allocating placeholders for every entry when using the helper.
- `tensor_dict_into` / `tensor_into` region-backed paths stream into registered
  CUDA regions via `MaterializeIntoTarget`, supporting view specs, packed
  subset selection (`tensor_names`), and ordered multi-storage layouts. For
  non-identity views the SDK resolves and sends a deterministic `view_id` in
  the `TargetLayout`. `region_backed_mode` (`auto`/`require`/`disable`) controls
  fallback behavior.
- Convenience materialization APIs remain the ergonomic SDK surface:
  `tensor_dict(...)`, `tensor_dict_with_diagnostics(...)`,
  `tensor_dict_into(...)`, `tensor_into(...)`, `bind(...)`, and
  `bind_into(...)` lower through `ArtifactRealizationSpec` and the shared
  `Artifact.realize(...)` handle facade. Use `Artifact.realize(...)` directly
  when callers need the `ArtifactRealizationReport`, projection owner, or
  lifecycle handle; use the convenience methods for normal retrieval and
  binding workflows.
- Handles retain whichever identifiers are available (`artifact_id`, `key`).
  At least one identifier is required when instantiating or
  rehydrating a handle, but resolved handles may keep both `artifact_id` and
  `key` while serialization (`to_dict`/`from_dict`) remains identity-only.
- `tensorcast.from_disk(path)` / `Store.from_disk(path)` now default to the
  metadata-first mounted-source path for same-daemon loading. Successful calls
  return a lazy `Artifact` seeded from `ResolvePublicDiskSource` metadata,
  usually with primary `artifact_id = msa1:...`, without hashing payload bytes
  during metadata resolution. The returned mounted-source artifact keeps the
  daemon-attested source handle for direct
  `Artifact.realize(ArtifactRealizationSpec.model_runtime(...),
  runtime_host=...)` startup through framework runtime host capabilities. Use
  `show_progress=True` or call `import_from_disk(...)` explicitly when you need
  streamed daemon import.
- `tensorcast.import_from_disk(path)` / `Store.import_from_disk(path)` keep the
  explicit daemon import contract via `ImportArtifactFromPath` /
  `ImportArtifactFromPathStream`. This path returns `mi2:` and remains the
  explicit reference-only registration flow for payload bytes: no payload
  copy/link/reflink, but the daemon may hash payload bytes and may perform
  bounded metadata backfill such as `artifact_descriptor.json` (and
  `tensor_index.json` for safetensors directories) so later explicit imports
  can reuse trusted metadata and skip full data hashing.
  For one-off backfill on root-owned directories, use
  `bash tools/backfill_from_disk_import.sh`, which starts an isolated temporary
  daemon and can auto-escalate to `sudo`.
  Stream events are the canonical explicit-import progress contract (`phase`,
  bytes, `percent`, terminal `done`, machine-readable `error_code`).
  Same-host collective disk loading is now an explicit per-call contract:
  prefer `GetArtifactOptions(execution_topology=ExecutionTopologyContext(...))`
  as the daemon-owned source-bound contract.
  `ctx=CallContext(collective=CollectiveLoadGroup(...))` is no longer accepted
  by `Binding.swap(...)` / `Binding.realize_from(...)` on daemon-owned
  bindings; keep `ctx` there for timeout/tags only.
  TensorCast no longer auto-enables collective mode from ambient GPU
  environment variables, and `replica_uuid` remains a pure operation/session id.
  Set `verify_checksums=False` on `from_disk(...)` / `import_from_disk(...)` to
  relax descriptor mismatch checks for local development.
- `tensorcast.resolve_public_disk_source(path)` /
  `Store.resolve_public_disk_source(path)` expose the metadata-first disk
  ingress needed by binding-native realization. The returned
  `PublicDiskSourceHandle` now carries a non-empty session-local
  `artifact_id` (`msa1:...`) together with the normalized disk locator,
  `canonical_index_bytes`, optional `trusted_content_artifact_id`, and
  mounted-source attestation metadata. `Binding.realize_from(...)` /
  `Store.realize_into_binding(...)` can consume it directly without first
  registering an artifact through `from_disk(...)`.
- `tensorcast.promote_mounted_source(artifact)` /
  `Store.promote_mounted_source(artifact)` provide the explicit
  `msa1 -> mi2` promotion path when callers want a content-verified identity
  without re-specifying the mounted source path. The daemon revalidates the
  mounted snapshot, hashes payload bytes through the import path, records the
  session-local promotion hint, and returns an `Artifact` backed by `mi2:...`.
  By default the client inherits the same timeout/retry budget as explicit
  import. Pass `timeout_s=...` when promoting unusually large mounted
  directories and you want an explicit per-call override.
- For rollout/evidence collection on real mounted sources, use
  `python tools/measure_mounted_source_evidence.py --help` to capture JSON
  timing packets for `from_disk`, `import_from_disk`,
  `promote_mounted_source`, and optional materialization. The helper now also
  accepts `--timeout-s` for long-running promotion evidence runs.
- Handles are tied to the originating `Store` lifecycle. After `Store.close()`
  (or `release()` on the handle), materialization raises
  `ArtifactError(status_code="FAILED_PRECONDITION")` while cached metadata
  remains readable for debugging.
- Retrieval policy is execution-scoped via `GetArtifactOptions`; handles no
  longer clone or serialize source-selection hints.

Design and execution details: `../../../docs/designs/0077-unified-reference-only-disk-import.md`,
`../../../docs/plans/0077-unified-reference-only-disk-import.md`.

## View Composition

- `artifact.view(...)`, `.subset(...)`, and `.slice(...)` return derived
  handles with composed view specs. Parent/child handles keep independent locks
  and metadata caches so repeated calls avoid daemon lookups.
- `artifact.view_builder()` exposes a fluent builder for chaining multiple view
  operations before building a child handle.
- Nested slice operations are collapsed into a single narrow so storage offsets
  are computed exactly once in the derived view (no double-application of the
  parent slice).

## Piece Registration and Sealing

- `Store.register_piece(...)` registers dense view pieces under an assembly id
  (`cgid:`). Pieces are selection-only (narrow only), do not allow transpose,
  and require server placement.
- Pass `canonical_index_bytes` to bootstrap a new assembly without needing
  Global Store state. `register_view(..., registration_kind="piece")` is
  equivalent.
- `Store.start_assembly_attempt(layout_id=..., requirements=...)` creates a
  fresh assembly attempt bound to a `LayoutSpec` and returns an
  `AssemblyAttemptRef` containing durable attempt scope plus the workspace
  assembly id. Requirement truth must be explicit in the request; use
  `AssemblyRequirementSetRef.pp_from_structural_views(...)`,
  `AssemblyRequirementSetRef.ep_from_structural_views(...)`, or
  `AssemblyRequirementSetRef.canonical_full()` for the current dependency-ready
  carriers.
- The dependency-ready requirement mapping is exact and deterministic:
  - `PP`: dedupe + sort the structural view ids, then emit one requirement per
    view with `slot_id = view_id`, `target = structural_view(view_id)`, and
    `coverage_contract = "pp_structural_view"`.
  - `EP`: dedupe + sort the structural view ids, then emit one requirement per
    view with `slot_id = view_id`, `target = structural_view(view_id)`, and
    `coverage_contract = "ep_structural_view"`.
  - `canonical_full`: emit one requirement with
    `slot_id = "__canonical_full__"`, `target = canonical_layout`, and
    `coverage_contract = "canonical_full"`.
  - `TP > 1` is not part of the current dependency-ready carrier set.
- `Store.seal_assembly_attempt(attempt)` explicitly transitions an open attempt
  onto its sealing workflow and returns `Operation[PublishedModelVersion]`.
  - The daemon captures one durable readiness cut, persists full structural
    evidence into that cut, and seals from the cut rather than rereading live
    workspace view state.
- `Store.wait_assembly_attempt(attempt)` observes an existing attempt workflow
  and decodes the dependency-ready source publish lineage into a
  `PublishedModelVersion`.
  - `source_publish_only` attempts carry real source lineage and optional
    `source_version_key`.
  - `representation_publish` attempts now require a typed
    `RepresentationPublishContract` child contract on
    `AssemblyCloseoutContract`.
  - When the serving artifact exists, the manifest carrier is readable, and the
    serving manifest agrees with the typed child contract, the returned
    `PublishedModelVersion` also carries:
    `serving_artifact_id`, `serving_descriptor`, `serving_version_key`,
    `representation_contract_hash`, `serving_build_digest`, and
    `serving_manifest_ref`.
  - Phase 1 currently supports the reserved manifest-tensor carrier
    `tensor:__tensorcast_meta__.manifest_json`.
  - `RuntimeArtifactManifest` now self-describes its phase-1 carrier through
    `serving_manifest_ref`, and the typed serving-lineage models can derive a
    strict runtime gate:
    `RepresentationPublishContract.to_runtime_policy()`,
    `RuntimeArtifactManifest.to_runtime_policy()`, and
    `PublishedModelVersion.require_runtime_artifact_policy()`.
  - The repo-owned serving-lineage carriers now also expose explicit phase-1
    build identity fields:
    `RuntimeArtifactManifest.serving_build_digest_version` and
    `RepresentationPublishContract.serving_build_digest_version`.
    Runtime policy gates on `serving_manifest_ref`,
    `representation_contract_hash`, and `serving_build_digest`.
  - For integrations that already have a transformed serving artifact in hand,
    `build_pure_transform_publication_bundle_from_registered_artifact(...)`
    assembles a typed `RepresentationPublishSpec` containing the repo-owned
    phase-1 serving manifest bytes,
    `RepresentationPublishContract`, and
    `AssemblyCloseoutContract(kind="representation_publish", ...)` for
    `PURE_TRANSFORM` publication.
  - `Store.start_representation_publish_attempt(...)` can now consume that
    spec directly and forward it into `start_assembly_attempt(...)` through the
    typed `representation_publish_spec` daemon ingress instead of re-authoring
    the generic closeout shell at each call site.
    When the spec carries optional `ServingAdmissionFacts`, TensorCast validates
    the supplied finalize classification, same-binding proof, and support level
    for consistency without inferring missing integration-private rollout state.
  - `BINDING_FINALIZE` publication is same-binding-only. Use
    `Store.complete_binding_finalize_publication_from_binding(...)` after the
    runtime binding current value has been realized, finalized, and sealed.
    The resulting spec must carry a binding-value publication subject and
    `same_binding_fast_path_validated=True`.
  - Tensor-entry `BINDING_FINALIZE` publication helpers have been removed.
    Finalized tensors in ordinary memory are not an admitted publication path
    for representation-changing families; they must first be realized through
    the binding-native path.
  - For same-binding pure transforms, use
    `Store.complete_pure_transform_publication_from_binding(...)`. Pure
    transform publication may still use registered-artifact helpers because it
    does not represent a framework finalize that changes serving bytes.
  - `Store.complete_representation_publish_attempt(...)` runs the same repo-owned
    spec path through `start -> seal -> wait` and returns the final
    `PublishedModelVersion`.
  - `Store.list_artifact_layouts(artifact_id)` exposes the daemon-owned
    artifact-layout attachment query path.
  - If `layout_id` is omitted for the representation-publish helpers, TensorCast
    now tries to infer a unique attached layout from the bundle's source or
    serving artifact lineage. `requirements` still stay explicit and are not
    re-derived from layout metadata.
  - For the current single-rank canonical publish shape, use
    `Store.start_canonical_representation_publish_attempt(...)` or
    `Store.complete_canonical_representation_publish_attempt(...)` to bind the
    same bundle path to `AssemblyRequirementSetRef.canonical_full()`.
  - For structural publish shapes, use
    `build_representation_publish_requirements(...)`,
    `Store.start_structural_representation_publish_attempt(...)`, or
    `Store.complete_structural_representation_publish_attempt(...)` with an
    explicit `contract_family`. `pp`/`ep` lowering can consume deterministic
    source `view_id` lineage when the source artifact already carries it.
  - If the bundle already carries `contract_family`, use
    `Store.start_repo_owned_representation_publish_attempt(...)` or
    `Store.complete_repo_owned_representation_publish_attempt(...)` to route
    between canonical and structural lowering without selecting a second helper
    at the call site.
  - If the publication came back through `PlanResult`, use
    `PlanResult.require_representation_publish_spec(...)` to extract the typed
    publish spec, or call
    `Store.start_plan_repo_owned_representation_publish_attempt(...)` /
    `Store.complete_plan_repo_owned_representation_publish_attempt(...)`
    directly to route `transform_register_pure_transform(...)` into the same
    repo-owned publish path without manual `artifact_result` inspection.
  - For offline or pipeline-style `PURE_TRANSFORM` builders that already have
    finalized tensors in memory, use
    `Store.register_pure_transform_publication(...)` to inject the reserved
    manifest tensor and register a durable serving artifact plus typed
    publication bundle, or
    `Store.complete_pure_transform_publication(...)` to run the same
    repo-owned register + `representation_publish` closeout path in one call.
    When the publish attempt also needs a canonical source contribution, pass
    `source_contribution_device=...` and TensorCast will bind the source
    artifact, seal the current value, and contribute it into the attempt before
    sealing. For structural `pp` / `ep` shapes, pass
    `source_contribution_artifacts=(view_a, view_b, ...)` and TensorCast will
    derive structural view ids from those handles and contribute them one by
    one before sealing.
  - If that publication runs through `transform_register`, prefer
    `build_pure_transform_publication_spec(...)` or
    `build_pure_transform_transform_spec(...)`. These helpers now attach typed
    publish intent on `TransformSpec.publication_spec`.
    `representation_contract_hash` can still be provided explicitly, but the
    repo-owned `PURE_TRANSFORM` path can auto-derive it from source and serving
    canonical indexes when the source artifact metadata is available. The
    default identity
    `transform_register` path now also prepares the reserved manifest tensor
    before registration, so the resulting serving artifact can already carry
    `tensor:__tensorcast_meta__.manifest_json`.
  - For steady-state runtime bind or swap, pass
    `runtime_artifact_policy=...` to `artifact.bind(...)`,
    `artifact.bind_into(...)`, or `binding.swap(...)`.
    This keeps generic artifact load permissive while giving serving runtime an
    explicit strict gate. When the policy is present, the daemon requires a
    serving manifest and validates `serving_manifest_ref`,
    `representation_contract_hash`, and `serving_build_digest` before the
    artifact is accepted into the serving path.
    If you pass a full `RepresentationPublishSpec` instead of a plain runtime
    policy, TensorCast also requires
    `ServingSupportLevel.RUNTIME_BIND_SWAP_READY` when caller-supplied
    admission facts are present.
  - The same runtime-ready gate now also applies to serving-key activation on
    typed `representation_publish` specs: a spec carrying
    `serving_version_key` must be admitted at
    `ServingSupportLevel.RUNTIME_BIND_SWAP_READY`.
- `Store.seal_assembly(assembly_id, publish_canonical=True)` seals an assembly
  into a stable MI2 identity and returns the bound descriptor.

## Binding (Preferred Inplace Updates)

Canonical binding design: `../../../docs/designs/0084-binding-unified-model-and-contract.md`.

- `artifact.bind(device=..., packing=\"byte_space\", publish=False)` allocates a
  daemon-owned CUDA target layout, maps shared tensor views into the client
  process, and returns a `Binding` ready for pointer-stable swaps.
- `artifact.bind(device=..., mapping=copy_plan, packing=\"byte_space\")` keeps
  the same owner semantics while inferring the destination tensor layout from
  the mapped copy plan.
- `artifact.bind_into({name: tensor, ...}, packing=\"byte_space\", publish=False)`
  adopts **user-owned** CUDA tensors (already allocated in the current process),
  fills them once, and returns a `Binding`.
- `artifact.subset(...).view(...).bind(...)` captures a rank-local selection once;
  later `binding.swap("model:v2")` reapplies that same selection to the new
  artifact version instead of materializing the whole model.
- `artifact.bind_into(..., mapping=copy_plan, packing=\"byte_space\", publish=False)`
  executes the same traced copy plan (`CopyPlanEntry`/`Range`) against
  user-owned CUDA tensors; the mapping is stored and reused on `swap(...)`.
- `Store.create_binding(layout, ownership=\"daemon\", device=\"cuda:0\")` creates a
  layout-seeded binding before any artifact is installed. The binding starts with
  `current_value is None`.
- Builder-side serving realization may use that same layout-seeded binding as the
  host of the future serving representation: attach framework tensor views onto
  the binding-backed storage, call `binding.realize_from(...)` to execute the
  source-bound load and open one explicit mutable update window, perform the
  builder-owned finalize work inside that returned `BindingUpdateEpoch`, then
  `seal_current(...)` and route the sealed value through `representation_publish`
  closeout. In that shape, realization is execution-only, the binding remains
  unsealed until `seal_current(...)`, and the serving artifact identity still
  arrives only after closeout.
- `binding.publish_replica(ctx=...)` publishes the current bound layout without
  performing a swap. Use this when bind/swap should stay `publish=False` but you
  still want routable replicas after a successful apply.
- `binding.publish_replica_operation(ctx=...)` exposes the same publish path as
  `Operation[T]`, so callers can attach, wait, and inspect status through the
  unified public continuation surface.
- This publish path is for ordinary artifact-backed replica routing only. It is
  not the source-to-serving `representation_publish` closeout path for new
  serving-artifact lineage.
- `binding.current_value` is the authoritative sealed value handle for the local
  binding. `binding.artifact_id` / `binding.selection` are convenience mirrors
  and become `None` when the current value is absent or local-only.
- `binding.begin_update(...)` requires the binding to be unpublished, clears the
  current sealed value, and returns a `BindingUpdateEpoch`. Call
  `binding.retire(...)` explicitly first if a published replica is still live.
- `binding.realize_from(...)` on daemon-owned bindings now returns a
  `BindingUpdateEpoch` instead of a `SealedBindingValue`: it writes bytes into
  the binding-backed storage and leaves the binding unsealed for subsequent
  builder-side finalize work.
- `binding.seal_current(update_epoch=...)` closes the mutable window and produces
  a local-only `SealedBindingValue`. Local seal does not mint a routable
  artifact id; publish/key activation remain valid only for artifact-backed
  current values created by bind/swap flows.
- `sealed_value.contribute_to_assembly(attempt=...)` compiles the sealed binding
  onto the existing assembly trunk:
  piece/view-backed bindings satisfy structural-view slots, while
  full-canonical bindings satisfy the canonical-layout slot.
- Mapped binding v1 requires contiguous CUDA tensors with `storage_offset=0`,
  enforces full dst coverage with no overlaps, and is local-only for materialization RPC.
- Packed/subset selections become publishable when TensorCast can derive a
  stable `view_id`; publish routing is then scoped to that derived byte-space
  instead of the canonical artifact id.
- Mapped binding publish on bind/swap (`publish=True`) now uses a
  `binding_current_value_publication_token` minted from the artifact-backed
  daemon-owned binding current value. `MaterializeIntoTarget` and
  `MaterializeIntoMappedTarget` do not mint standalone publish authority.
- Because mapped binding retains the copy plan for future `swap(...)`, it is not
  the preferred steady-state host object when mapped source semantics are needed
  only for one bootstrap fill. In that case, prefer a layout-seeded serving
  binding as the long-lived local slot and treat the source-to-target mapped
  write as a builder-side realization step rather than as the binding's
  persistent overwrite contract.
- View compatibility for mapped binding is narrow-only: transpose/permutation views
  are rejected and copy-plan ranges are expressed in canonical coordinates.
- `binding.swap(artifact_or_ref, publish=False, activate_key=None, ...)` performs
  safe retire → overwrite → optional publish, reusing the original selection
  (including view slices) without restating them.
- `binding.last_execution_diagnostics` exposes the daemon-reported typed
  execution facts for the most recent source-bound refill or promote, including
  collective policy/use, dominant executor, and hash/identity observability for
  closeout-driven publication paths.

Example (vLLM-style split weight):

```python
from tensorcast.api.store import CopyPlanEntry, Range

copy_plan = [
    CopyPlanEntry(
        ckpt_name="layers.0.mlp.gate_up_proj.weight",
        ckpt_range=Range(dim=0, start=0, end=4096),
        dst_name="layers.0.mlp.gate_proj.weight",
        dst_range=Range(dim=0, start=0, end=4096),
    ),
    CopyPlanEntry(
        ckpt_name="layers.0.mlp.gate_up_proj.weight",
        ckpt_range=Range(dim=0, start=4096, end=8192),
        dst_name="layers.0.mlp.up_proj.weight",
        dst_range=Range(dim=0, start=0, end=4096),
    ),
]

binding = artifact.bind_into(dst_tensors, mapping=copy_plan, packing="byte_space")
binding.swap("model:v2")
```

## Batching, Async, and Prefetch

- Use `with artifact.batch(device="cuda:0")` to coalesce multiple tensor fetches
  into a single RPC while keeping sync semantics.
- Async consumers can `await artifact.tensor_async(...)` or
  `await artifact.tensor_dict_async(...)`; calls are coalesced by the process
  `MaterializationBatcher` (1ms window) on the store event loop.
- `tensor_dict_into_async` / `tensor_into_async` cancellation is best-effort:
  once a region-backed RPC is in-flight, `cancel()` may return `False`; streaming
  materialization remains cancellable before the RPC boundary.
- `artifact.prefetch(device=..., ctx=..., options=...) -> Operation[PrefetchedReplica]` issues background
  materialization (`wait_for_completion=False`) and returns an operation handle. Use `op.result(timeout_s=...)` (or
  `op.wait(...)`) to block and `op.cancel()` to best-effort release the operation record. Prefetch defaults to
  `lease_mode=NO_LEASE` so it does not create PID-bound UseLeases and does not mint IPC handle leases. Prefetch is
  supported for both GPU VRAM (`"cuda:0"`/`0`) and daemon-owned host DRAM (`"cpu"`/`"dram"`/`-1`). Handle-exporting APIs
  remain PID/lease-bound and are separate from daemon-owned warm replicas.
- Retained realization prefetch lowers through the unified realization facade:
  `ArtifactRealizationSpec.retained_binding(...)` for one retained binding and
  `ArtifactRealizationSpec.target_set(...)` for TP/group target sets.
  `artifact.prefetch(target=RealizationTarget(...))` and
  `artifact.prefetch(target=RealizationTargetSet(...))` remain ergonomic
  wrappers, but target sets must use the target-set realization path so group
  admission, strategy, lifecycle, resource-envelope, and report state all carry
  `target_kind="target_set"`. Ordinary `device=` prefetch behavior is
  unchanged. Runtime targets require runtime-provided resolved layout/index
  metadata before daemon allocation; unresolved layouts fail closed before GPU
  memory is reserved. The daemon keeps serving prefetch behind
  `daemon_config.serving_prefetch.enabled` and returns a typed
  `PrefetchHandoff` / `PrefetchHandoffSet` result once the
  retained binding materialization path is enabled.
- Prefetch idempotency derives a stable action fingerprint from selection identity (`artifact_id`,
  `logical_layout_hash`, `selection_hash`) and target placement (daemon + device/tier). `selection_hash` is computed via
  `tensorcast.common.selection_identity` (stable `view_id` + `view_subset_hash`), matching Plan selection identity
  semantics.
- `artifact.pin_device_residency(device=..., ttl_ms=..., ctx=...) -> Operation[PlacementPin]` creates a placement pin
  (process-independent device residency intent) backed by a daemon-scoped capability token; the returned `PlacementPin`
  supports `renew()` / `release()`.
- `ctx.deadline_ms` clamps retry and polling budgets for control-plane actions so waits do not exceed the call budget.

## Retrieval Preferences

`GetArtifactOptions.source` controls retrieval preferences:

- `source="auto"` — daemon chooses the optimal source
- `source="local_only"` — disallow P2P and disk
- `source="disk_first"` — prioritize disk while still allowing P2P fallback
- `source="disk_only"` — require disk and disallow P2P

Use a structured `RetrievalPolicy` when you need explicit `allow_p2p` /
`allow_disk` gating or `prefer_p2p`.
Set `GetArtifactOptions.replica_uuid` to hint daemon-side reuse of a prefetched
replica.

## Feature Toggles

- `TENSORCAST_STORE_ENABLE_BATCHER` (default: enabled) — disable to bypass the
  async batcher and route `tensor_async` through direct fetches.
- `TENSORCAST_STORE_ENABLE_PREFETCH` (default: enabled) — disable to prevent
  `Artifact.prefetch()` from issuing background materialization.

## Metadata Cache

- The process runtime owns an `ArtifactCache` that stores canonical index bytes,
  parsed indices, and generation metadata keyed by `artifact_id`. Cache entries
  expire by TTL and obey an LRU bound.
- Environment defaults:
  - `TENSORCAST_STORE_INDEX_CACHE_TTL_SECONDS=600` (set `<=0` to disable)
  - `TENSORCAST_STORE_CACHE_MAX_ENTRIES=1000` (set `<=0` to disable)
- Cache metrics are emitted as:
  - `tc_store_artifact_cache_hits_total`
  - `tc_store_artifact_cache_misses_total`
  - `tc_store_artifact_cache_evictions_total` (dimensions: `reason=ttl|lru`)
  - `tc_store_artifact_cache_invalidations_total` (dimension: `reason`)
- Invalidation hooks run after registration, deregistration, and materialization
  errors (`NOT_FOUND`/`FAILED_PRECONDITION`) to keep key→artifact mappings and
  cached indices consistent.
- Key→artifact-id lookups are cached with TTL (see
  `TENSORCAST_STORE_KEY_CACHE_TTL_SECONDS`) to avoid repeated resolver RPCs.
- Disk-backed ingress seeds the cache with `canonical_index_bytes` and
  `generation` so repeated `from_disk` / `import_from_disk` calls can reuse
  local metadata and preserve generation context.
- Metadata hydration (`_ensure_metadata`) applies `_set_metadata` while holding
  the artifact’s reentrant lock so concurrent callers never observe partially
  populated canonical metadata.
