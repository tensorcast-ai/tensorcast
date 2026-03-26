---
slug: artifact-representation-contract-and-transform-unification
title: Artifact Representation Semantic Core and Transform Unification Plan
status: in_progress
areas: ["core", "daemon", "sdk", "integrations", "docs", "tests"]
created: 2026-03-24
last_updated: 2026-03-27
related_code:
  - docs/designs/0055-programmable-framework.md
  - docs/designs/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
  - docs/designs/0108-tensor-aware-materialization-strategy-plane.md
  - docs/designs/0109-batched-owner-file-collective-executor.md
  - docs/designs/0110-artifact-representation-contract-and-transform-unification.md
  - docs/designs/0111-source-to-serving-builder-and-representation-publication.md
  - docs/internals/model-loading.md
  - tensorcast/api/store/README.md
  - tensorcast/types.py
links:
  design: ../designs/0110-artifact-representation-contract-and-transform-unification.md
---

# Objective

Track only the remaining follow-up after the semantic-core hard cut landed.
Completed rollout work has been folded back into the design, SDK docs, and
tests. This plan now covers the open integration boundary and the follow-on
design extraction that still sit outside the landed cut.

# Current State & Grounding

- `RepresentationTransformContract` is the resolved semantic truth for mapped
  binding, mapped-target lowering, and shared semantic-to-work planning.
- Executor-local semantic recovery and mapped-only shared contracts have already
  been removed from the primary runtime path.
- The remaining open work is integration convergence around
  `/data/workspace/internal-vllm` and extraction of separate follow-on designs
  for topology-scoped execution and durable publication metadata.

# Phases & Milestones

- [ ] Phase 1: Integration Producer Convergence
  - [ ] Milestone 1.1: Define the TensorCast-side producer boundary expected by
    `internal-vllm` so `TracePlan` lowers into TensorCast representation
    contracts instead of keeping a private runtime truth.
  - [ ] Milestone 1.2: Move builder-side
    `representation_contract_hash` ownership onto TensorCast-defined normalized
    contract inputs.
  - [ ] Milestone 1.3: Keep source-bind startup documented and implemented only
    as a producer or bootstrap path, not as steady-state semantic truth.
- [ ] Phase 2: Follow-On Design Extraction
  - [ ] Milestone 2.1: Write a dedicated design for topology-scoped
    representation transform execution.
  - [ ] Milestone 2.2: Write a dedicated design for durable representation
    publication or catalog metadata if persistence beyond
    `PublishedModelVersion` is still needed.

# Tasks

- [ ] Coordinate `/data/workspace/internal-vllm` follow-up so
  `TracePlan.copy_plan` lowers into TensorCast representation contracts.
- [ ] Align integration-owned `representation_contract_hash` computation to the
  TensorCast-defined normalized inputs.
- [ ] Keep steady-state runtime on serving-artifact bind or swap and avoid
  reintroducing a second private semantic runtime path.
- [ ] Extract the topology-scoped transform executor work into its own design.
- [ ] Extract any durable publication or catalog expansion into its own design
  instead of widening `0110`.

# Test / Rollout / Backout

## Test plan

- [ ] Re-run the focused regression suites that cover the producer-boundary
  integration once the remaining `internal-vllm` follow-up lands.
- [ ] Capture benchmark or regression evidence if the producer-boundary changes
  alter lowering cost or execution behavior.

## Rollout

- [ ] Keep the remaining work scoped to producer-boundary convergence and
  follow-on design extraction.
- [ ] Do not restart topology-scoped execution or durable catalog expansion
  inside this follow-up plan.

## Backout

- [ ] If the remaining producer-boundary work proves under-specified, revise the
  `0110` design before widening scope.
- [ ] Do not add compatibility toggles or revive dual semantic stacks as a
  fallback.

# Risks & Tracking

- [ ] Risk: integration follow-up quietly reintroduces a private semantic truth.
  - Mitigation: keep `0110` as the only owner of normalized transform-contract
    inputs and review any new producer carrier against that boundary.
- [ ] Risk: topology-scoped execution gets pulled back into this plan too early.
  - Mitigation: stop and write the dedicated follow-on design instead of growing
    the scope here.
