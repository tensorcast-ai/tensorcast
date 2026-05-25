---
slug: artifact-centered-model-runtime-realization
title: Artifact-Centered Model Runtime Realization Plan
status: draft
areas: ["sdk", "artifact_runtime", "daemon", "core", "integrations", "docs", "tests"]
created: 2026-05-23
last_updated: 2026-05-26
related_code:
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - docs/designs/0121-unified-artifact-realization-kernel.md
  - docs/plans/0121-unified-artifact-realization-kernel.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/realization_kernel.py
  - tensorcast/api/store/publication_builder.py
  - tensorcast/api/store/runtime_realization_reference_consumer.py
  - tensorcast/api/store/runtime_realization_spec_cache.py
  - tensorcast/artifact_runtime/
  - tensorcast/retained_realization.py
  - tensorcast/retained_realization_authority.py
  - tensorcast/types.py
links:
  design: ../designs/0120-artifact-centered-model-runtime-realization.md
  dependencies:
    - ../designs/0121-unified-artifact-realization-kernel.md
---

# Objective

Move TensorCast model-runtime loading, retained prefetch/acquire, binding, and
TensorDict retrieval onto one artifact-centered realization path.

The final state must not preserve old serving-rooted compatibility facades.
Serving remains a valid term only for protocol fields or representation payloads
that are truly model-serving ABI concepts.

# Current State

- Public `tensorcast.serving` has been removed. Runtime implementation now lives
  under `tensorcast.artifact_runtime`, organized by artifact, binding,
  publication, recipe, config, policy, lifecycle, diagnostics, readiness, state,
  and view responsibilities.
- `Artifact.tensor_dict(...)`, `tensor_dict_into(...)`, `bind(...)`,
  `bind_into(...)`, `prefetch(...)`, and direct model-runtime realization lower
  through `Artifact.realize(...)` / `Artifact.realize_async(...)`.
- Runtime config uses the `runtime_artifact` section. The old top-level
  `serving` config key and serving-named settings aliases are rejected rather
  than translated.
- Retained handoff parsing uses `tensorcast.retained_realization` and
  `tensorcast.retained_realization_authority`. The retained authority models are
  typed; no `Any` authority bypass is part of the target path.
- Runtime target and handoff DTOs use neutral names:
  `RealizationTarget`, `RealizationTargetSet`, `PrefetchHandoff`,
  `PrefetchHandoffSet`, `RuntimeBinding*`, `RuntimeTopologyRef`, and
  `RuntimeRealizationSpecCacheEntry`.
- Runtime realization DTOs and retained claim payloads use
  `runtime_build_digest`. Serialization still maps to existing daemon protobuf
  fields such as `serving_build_digest` where that is the current wire ABI.
- Direct model-runtime realization defaults to a resolver bound to the current
  `Artifact` store when the caller omits `runtime_resolver`. This preserves the
  artifact's daemon/session authority instead of reopening through the
  process-global store.
- Model-runtime request facts are resolved fail-closed. `spec.device`,
  `runtime_context.target_device`, host target device, topology, member,
  adapter version, and runtime ABI version are normalized/compared when more
  than one source provides them; missing facts are filled only after agreement.
- `ArtifactRealizationSpec.model_runtime(options=...)` carries materialization
  options. `runtime_artifact_policy` is a separate manifest/build/preflight
  policy and is passed independently through direct runtime realization.
- Store representation-publication helpers live in
  `tensorcast.api.store.publication_builder`. Some publication DTO fields and
  helper names still contain `serving`; these are the remaining
  serving-publication ABI classification boundary. Serving-named publication
  helpers and `ServingSupportLevel` are no longer exported from the root
  `tensorcast` package; callers that intentionally operate on this ABI boundary
  must import them from `tensorcast.api.store`, `tensorcast.types`, or the owning
  builder module.
- Tests have been moved toward `tests/python/artifact_runtime/` and
  runtime-realization API test names. Old serving module tests have been deleted
  or renamed unless they are still testing a daemon/proto/publication ABI term.
- internal-vLLM has been migrated to the final TensorCast boundary for active
  runtime paths: it imports `tensorcast.artifact_runtime.*`,
  `tensorcast.retained_realization*`, and `Artifact.realize(... model_runtime ...)`
  instead of `tensorcast.serving`, root-level `tensorcast.runtime_*` modules, or
  serving-named retained authority helpers. The external
  `/reload_serving_artifact` method remains a vLLM control-plane ABI name.

# Target State

- Artifact is the durable identity and lifecycle root for model-runtime startup,
  reload, retained acquire, publication, and diagnostics.
- Public APIs are rooted in `tensorcast`, `tensorcast.api.store`, and the
  structured `tensorcast.artifact_runtime` implementation package. There is no
  public `tensorcast.serving` facade and no compatibility alias for removed
  serving names.
- Core call paths are unified:
  `Artifact.realize(...)` and `Artifact.realize_async(...)` are the common entry
  points for TensorDict materialization, binding, retained prefetch/acquire,
  target-set realization, mounted-source startup, and model-runtime attach.
- Runtime implementation names use artifact/runtime vocabulary. A serving name
  may remain only when it is an explicit daemon proto field, persisted
  representation metadata field, or model-serving ABI payload.
- Tests assert the final contract directly. They should not keep old tests whose
  only purpose is serving-surface compatibility.
- File layout should stay organized by responsibility. Avoid reintroducing flat
  `runtime_xx.py` modules at the package root for model-runtime behavior.

# Migration Work

## 1. Keep The Runtime Call Path Unified

- Ensure `ArtifactRealizationSpec` remains the only public realization request
  shape for generic SDK operations and model-runtime attach.
- Keep TensorDict, binding, retained prefetch/acquire, mounted-source, and
  target-set paths routed through the realization kernel.
- Keep runtime artifact policy, locator, request context, diagnostics, and
  publication actions under `tensorcast.artifact_runtime`.
- When a new capability appears adjacent to loading or publication, model it as
  artifact metadata, artifact lifecycle, or artifact replica semantics first.

## 2. Finish Serving-Publication ABI Classification

- Classify every remaining serving-named Python symbol as one of:
  daemon/proto wire ABI, persisted serving-manifest ABI, or accidental runtime
  naming.
- Rename accidental runtime names to artifact/runtime names without aliases.
- Leave true wire and persisted ABI fields in place until a deliberate proto or
  metadata migration is planned.
- Document any intentionally retained serving-publication names in the owning
  module, not as broad compatibility policy.

## 3. Keep Tests Aligned With The Final Surface

- Prefer tests under `tests/python/artifact_runtime/` for runtime internals and
  `tests/python/api/test_runtime_realization_*.py` for SDK/runtime-realization
  APIs.
- Delete serving-surface compatibility tests once the old surface is removed.
- Negative public-surface tests should assert removed names are absent, not that
  old names resolve to new classes.
- Daemon/proto tests may keep serving names where the proto or RPC still uses
  them.

## 4. Validate internal-vLLM Boundary

- internal-vLLM should consume artifact selection, realization specs/handles,
  retained claims, runtime host capabilities, runtime reload, and publication
  actions directly.
- It should not construct `ServingRuntimeSession`, import `tensorcast.serving`,
  or use serving-named retained authority helpers.
- Active internal-vLLM docs and tests should use
  `runtime_artifact.artifact_locator`, retained realization claims, and
  runtime-artifact wording for implementation boundaries. Keep serving names
  only where they are vLLM endpoint/wire names or serving-manifest ABI fields.

# Remaining Work

- Complete the serving-publication field-map cleanup in
  `tensorcast/types.py`, `tensorcast/api/store/publication_builder.py`, and
  related tests. Keep true serving-manifest ABI fields; rename accidental
  runtime fields.
- Audit docs and tests for stale serving module paths, old test filenames, and
  accidental runtime aliases.
- Run the broad Python validation slice after each cleanup batch.
- Run the available internal-vLLM TensorCast focused suite after cross-repo API
  or boundary changes.
- Decide whether daemon protobuf serving-prefetch message names need a later
  wire-ABI migration. Do not silently rename proto fields without a separate
  migration plan.

# Validation

Use these as the recurring local gates for this plan:

```bash
source .venv/bin/activate
ruff check .
git diff --check
TENSORCAST_SKIP_TORCH_ABI_CHECK=1 pytest \
  tests/python/api/test_runtime_realization_target.py \
  tests/python/api/test_prefetch_operation.py \
  tests/python/api/test_realization_kernel.py \
  tests/python/api/test_artifact_handle.py \
  tests/python/api/test_runtime_realization_spec_cache.py \
  tests/python/api/test_runtime_realization_reference_consumer.py \
  tests/python/artifact_runtime \
  tests/python/test_serving_publication_types.py \
  tests/python/api/test_public_surface.py \
  tests/python/examples/test_runtime_reference_framework.py -q
```

Focused gates for recent runtime DTO and retained-claim cleanup:

```bash
source .venv/bin/activate
TENSORCAST_SKIP_TORCH_ABI_CHECK=1 pytest \
  tests/python/api/test_runtime_realization_target.py \
  tests/python/api/test_runtime_realization_spec_cache.py \
  tests/python/api/test_prefetch_operation.py \
  tests/python/api/test_realization_kernel.py \
  tests/python/api/test_plan_spec.py \
  tests/python/artifact_runtime/test_config.py \
  tests/python/artifact_runtime/test_fake_framework_boundary.py \
  tests/python/artifact_runtime/binding/test_retained.py -q
```

Current local evidence after the latest cleanup batch:

- `ruff check .` passed.
- `git diff HEAD --check` passed.
- `TENSORCAST_SKIP_TORCH_ABI_CHECK=1 pytest tests/python/api/test_realization_kernel.py tests/python/artifact_runtime/test_fake_framework_boundary.py tests/python/artifact_runtime/test_lifecycle.py -q` passed.
- The broad 0120 Python validation slice listed above passed.
- In `/data/workspace/internal-vllm`, the TensorCast focused suite passed:
  `136 passed, 1 warning`, plus compile and diff checks.

# Audit Commands

Use these checks to keep future cleanup honest:

```bash
rg "tensorcast\.serving|ServingRuntimeSession|ServingConfig|ServingRuntimePolicy" tensorcast tests/python examples
rg "ServingBinding|ServingTopology|plan_serving_binding_source_reuse" tensorcast tests/python examples
rg "serving_build_digest|serving_manifest" tensorcast/artifact_runtime tensorcast/api/store/runtime_realization_*.py
rg "tensorcast\.serving|tensorcast\.runtime_|ServingIntegration|ServingArtifactResolver|serving\.selector" \
  /data/workspace/internal-vllm/vllm /data/workspace/internal-vllm/tests/model_executor \
  /data/workspace/internal-vllm/docs/tensorcast/README.md \
  /data/workspace/internal-vllm/docs/tensorcast/tensorcast_vllm_architecture.md
```

Expected results:

- The first command should be empty except for negative tests or documentation.
- The second command should be empty outside daemon protobuf type references and
  negative public-surface assertions.
- The third command should identify only true daemon wire or serving-manifest
  ABI boundaries, not generic runtime DTO fields.

# Rollout Rule

Every cleanup batch must move toward the target state. Do not add aliases such
as `ServingConfig = TensorCastRuntimeConfig` or broad `Any` authority fields to
bridge old callers. If a caller still depends on an old serving name, migrate
that caller or delete the stale path.
