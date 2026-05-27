---
slug: prefetch-serving-binding-target
title: Prefetch Serving Binding Target Follow-up Plan
status: retired
areas: ["artifact_runtime", "sdk", "daemon", "tests"]
related_code:
  - docs/designs/0116-prefetch-serving-binding-target.md
  - docs/designs/0120-artifact-centered-model-runtime-realization.md
  - docs/plans/0120-artifact-centered-model-runtime-realization.md
links:
  design: ../designs/0116-prefetch-serving-binding-target.md
  superseded_by:
    - ../designs/0120-artifact-centered-model-runtime-realization.md
    - ../plans/0120-artifact-centered-model-runtime-realization.md
---

# Status

This plan is retired. The retained-prefetch and acquire semantics from `0116`
are now part of the artifact-centered model-runtime direction tracked by `0120`.

Do not execute this as a standalone serving-target plan, and do not reintroduce
serving-rooted public APIs from the old terminology.

# Preserved Semantics

- A prefetched value is a daemon-owned retained realization, not durable
  artifact identity.
- Worker handoff is a retained realization claim/capability scoped to member,
  device, layout, schema, reservation, and expiry.
- Acquire/restore/attach must fail closed on identity or placement mismatch.
- Promotion to a durable runtime artifact remains explicit publication work.

# Remaining Work

Any still-open work should be tracked in
`docs/plans/0120-artifact-centered-model-runtime-realization.md` under the
artifact-runtime vocabulary.
