---
slug: binding-unified-model-and-contract-plan
title: Binding Unified Model and Contract Plan
links:
  design: ../designs/0084-binding-unified-model-and-contract.md
areas: ["sdk", "daemon", "core", "proto"]
related_code:
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/inplace_slot.py
  - tensorcast/api/store/owned_binding_slot.py
  - daemon/service/controllers/owned_binding_service.cc
  - tests/python/test_binding.py
  - docs/guides/steptron-vllm-binding-integration.md
  - docs/designs/0085-distributed-binding-assembly-and-coordinator.md
  - docs/plans/0105-assembly-attempt-hard-cut-spec-runtime-slot-closeout.md
---

# Objective

Track only the remaining follow-up work after the `0084` local binding model
landed.

The shipped `0084` surface already provides:

- `Binding` as the stable local location
- `SealedBindingValue` as the authoritative local immutable value
- `binding_layout_id` in place of local `layout_id` overload
- artifact-backed inference through `swap(...)`
- training-style local mutation through `begin_update(...)` and
  `seal_current(...)`
- daemon-authored `binding_value_id` / `seal_generation`

This plan now covers only the remaining work needed to align overwrite control
with the assembly trunk from `0085` and to clean up the downstream integration
guide.

# TODO

- [ ] Add a pre-overwrite contribution fence for client-owned unmapped
  `swap(...)` so live slot occupancy is checked before
  `materialize_into_target_v2(...)`.
- [ ] Add a pre-overwrite contribution fence for client-owned mapped
  `swap(...)` so live slot occupancy is checked before
  `materialize_into_mapped_target(...)`.
- [ ] Keep overwrite fencing anchored on the same slot-occupancy plus liveness
  authority already used by daemon-owned paths.
- [ ] Eliminate any remaining overwrite path where SDK-local state can imply a
  more permissive contract than daemon authority.
- [ ] Add Python regression coverage for client-owned unmapped `swap(...)`
  rejecting live-contribution conflicts before mutation.
- [ ] Add Python regression coverage for client-owned mapped `swap(...)`
  rejecting live-contribution conflicts before mutation.
- [ ] Add C++ validation coverage for overwrite fencing on refill and
  commit/preflight paths.
- [ ] Update
  [docs/guides/steptron-vllm-binding-integration.md](/data/workspace/tensorcast-1/docs/guides/steptron-vllm-binding-integration.md)
  after the mutation-fence contract is stable.

# Remaining Work

## 1. Finish Contribution-Fence Enforcement On Overwrite Paths

Goal:

- all byte-overwriting entry points must reject mutation while the current
  `binding_value_id` still has live assembly slot occupancy
- this must hold for both daemon-owned and client-owned bindings
- overwrite must fail before bytes are mutated, not only when post-overwrite
  commit metadata is written back

Current gap:

- daemon-owned paths already gate overwrite on live occupancy
- `Binding.begin_update(...)` is already fenced
- client-owned `InplaceSlot.swap(...)` still overwrites local bytes before the
  daemon-side `commit_binding_artifact(...)` precondition check runs

Required changes:

- add a pre-overwrite fence for client-owned `swap(...)` so live slot occupancy
  is checked before `materialize_into_target_v2(...)` /
  `materialize_into_mapped_target(...)`
- keep the same authoritative source of truth as daemon-owned paths:
  assembly slot occupancy plus lifecycle-backed liveness
- avoid introducing a second SDK-local fence dialect

Acceptance:

- `begin_update(...)` fails with `FAILED_PRECONDITION` while the current value
  has live occupancy
- daemon-owned `swap(...)` / refill continues to fail before overwrite
- client-owned unmapped `swap(...)` fails before overwrite
- client-owned mapped `swap(...)` fails before overwrite
- no path relies on “overwrite first, reject at commit time” semantics

Tests to add or tighten:

- Python: client-owned unmapped `swap(...)` rejects live-contribution conflict
  before mutation
- Python: client-owned mapped `swap(...)` rejects live-contribution conflict
  before mutation
- C++: daemon validation for overwrite fencing on both refill and
  commit/preflight paths

## 2. Normalize Remaining Overwrite-Control Semantics

Goal:

- remove the last split-brain cases where overwrite behavior depends on a mix of
  daemon proto state and SDK-local booleans rather than one coherent transition
  model

Focus:

- preserve binding-scoped update tokens
- preserve the strict dirty rule
- remove any remaining path where SDK-local state can temporarily imply a more
  permissive overwrite contract than daemon authority allows

Acceptance:

- one overwrite contract applies across `begin_update(...)`, daemon-owned
  artifact refill, and client-owned `swap(...)`
- SDK mirrors daemon authority; it does not widen or weaken overwrite legality

## 3. Refresh The Steptron / vLLM Integration Guide

Update [docs/guides/steptron-vllm-binding-integration.md](../../docs/guides/steptron-vllm-binding-integration.md)
once the mutation-fence shape is stable so the guide reflects the final
contract:

- local `seal_current(...)` creates a binding-local sealed value only
- promotion to artifact visibility happens through the assembly trunk
- overwrite while contributed is fenced by live slot occupancy
- serving hand-off remains artifact- or published-version-based rather than
  binding-local

## 4. Sequencing With `0085` / `0105`

Do not wait for all of `0085` to finish.

The dependency is narrower:

- `0084` follow-up work should wait only for the mutation-fence authority from
  `0085` / `0105` to be stable enough to reuse
- once slot occupancy and liveness semantics are stable, `0084` can wire the
  remaining overwrite entry points without waiting for the entire distributed
  assembly program to close out
- doc cleanup should happen after that wiring lands so the guide describes the
  final fence semantics only once

Practical order:

1. Stabilize the slot-occupancy-backed mutation-fence contract in `0085` /
   `0105`.
2. Apply that fence to client-owned `swap(...)` pre-overwrite paths in `0084`.
3. Add regression tests covering both mapped and unmapped swap.
4. Update the Steptron / vLLM guide.

# Validation

Run the focused checks for the remaining work:

- `source .venv/bin/activate && pytest tests/python/test_binding.py`
- `source .venv/bin/activate && pytest tests/python/test_inplace_slot.py`
- `source .venv/bin/activate && pytest tests/python/api/test_mapped_binding.py`
- `bazel test //daemon:session_lifecycle_test --test_env=TENSORCAST_CUDA_BACKEND=fake --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error`

# Exit Criteria

This follow-up is done when:

- every overwrite entry point observes the same live-contribution fence before
  mutation
- client-owned `swap(...)` no longer mutates bytes before occupancy checks pass
- SDK-local state mirrors daemon authority instead of widening overwrite
  semantics
- the Steptron / vLLM guide reflects the final local-seal vs assembly-publish
  contract
