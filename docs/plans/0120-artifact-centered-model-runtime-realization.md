---
slug: artifact-centered-model-runtime-realization
title: Artifact-Centered Model Runtime Realization Follow-up Plan
status: active
areas: ["sdk", "artifact_runtime", "integrations", "docs", "tests"]
created: 2026-05-27
last_updated: 2026-05-27
related_code:
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/realization_kernel.py
  - tensorcast/artifact_runtime/lifecycle.py
  - tensorcast/artifact_runtime/request_facts.py
  - tensorcast/retained_realization.py
  - tests/python/api/test_realization_kernel.py
  - tests/python/artifact_runtime/test_fake_framework_boundary.py
  - tests/python/artifact_runtime/test_lifecycle.py
links:
  design: ../designs/0120-artifact-centered-model-runtime-realization.md
---

# Objective

Keep the implementation aligned with the artifact-centered model-runtime design.
This plan intentionally tracks only remaining work. Historical serving-session
migration detail should stay in design/evidence history, not in this active
todo list.

# Current State

- Direct model-runtime startup enters through
  `Artifact.realize(ArtifactRealizationSpec.model_runtime(...),
  runtime_host=..., runtime_context=...)`.
- Retained startup enters through
  `RetainedRealizationClaim.realize_model_runtime(...)`.
- Runtime artifact policy and materialization options are separate fields on
  `ArtifactRealizationSpec`.
- Materialization defaults to the direct path
  (`materialization.collective=disabled`). Collective is an explicit strategy,
  not the implicit runtime default.
- Runtime request facts are resolved once and shared by direct and retained
  realization paths.
- Direct realization uses the current `Artifact` store when no custom resolver
  is supplied.
- Framework/process-local placement facts are fail-closed: device, topology,
  member, adapter version, and runtime ABI mismatches are rejected instead of
  silently overriding one source with another.
- `tensorcast.serving` is not a framework-facing runtime namespace. Serving
  names may remain only for wire, ABI, profile, or existing endpoint fields.

# Todo

- [ ] Keep public APIs artifact-rooted. Do not reintroduce
      `ServingRuntimeSession`, `ServingConfig`, or `tensorcast.serving` as a
      framework-facing root.
- [ ] Continue deleting compatibility aliases, shallow wrappers, and duplicate
      tests once a public artifact-runtime test covers the behavior.
- [ ] Keep `serving.policy` and `materialization.collective` semantics
      separate: manifest/admission policy is not materialization strategy, and
      collective stays opt-in until a topology/model has measured benefit.
- [ ] Keep placement and framework facts fail-closed across direct,
      retained, mounted-source, reload, and publication paths.
- [ ] Keep direct model-runtime realization off TensorDict/Python state-dict
      intermediates; profile evidence should report selected source kind,
      copy bytes, temporary bytes, and attach timing.
- [ ] Keep framework integrations thin: they supply runtime host capabilities,
      placement/source facts, and model hooks; TensorCast owns artifact
      identity, binding lifecycle, runtime view, reload, and publication
      actions.
