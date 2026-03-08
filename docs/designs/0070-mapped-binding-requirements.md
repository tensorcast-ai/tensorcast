---
slug: 0070-mapped-binding-requirements
title: Mapped Binding (Map Abiding) Requirements
areas: ["sdk", "materialization", "binding", "inplace"]
status: draft
created: 2026-02-03
last_updated: 2026-03-04
related_code:
  - docs/designs/0063-binding-first-inplace-updates.md
  - docs/designs/0061-slot-based-inplace-binding-and-swap.md
  - docs/designs/0039-artifact-first-sdk.md
  - tensorcast/api/store/artifact.py
  - tensorcast/api/store/binding.py
  - tensorcast/api/store/materialization.py
links:
  predecessors:
    - ./0063-binding-first-inplace-updates.md
    - ./0061-slot-based-inplace-binding-and-swap.md
    - ./0039-artifact-first-sdk.md
  plan: ../plans/0070-mapped-binding-requirements.md
---

# Summary

Mapped Binding (Map Abiding) allows Tensorcast to fill user owned CUDA tensors
according to an explicit copy plan (source name + slice -> destination name + slice),
and to swap to new artifact versions using the same plan without reallocating model
parameters. This enables vLLM reloads without Python level copy loops.

# Background and Motivation

vLLM loads weights through model.load_weights(...), which can rename tensors, split
or merge fused weights, apply TP slices, and use custom weight_loader logic. These
semantics are discovered by tracing the load once. The trace yields a copy plan that
is the single source of truth for correct loading. Tensorcast already supports Binding
for safe in place swap, but only for name aligned materialization. Mapped Binding
extends Binding so Tensorcast can consume the copy plan directly.

# Terminology

- Artifact: immutable value handle (content address or key).
- Binding: mutable location handle that owns or adopts target buffers and supports
  safe swap.
- Copy plan: list of mapping entries describing src tensor slices -> dst tensor slices.
- Selection: view or slice metadata captured by artifact.view(...).

# Goals

1) Allow adopt and fill into existing CUDA tensors using an explicit mapping.
2) Allow swap to a new artifact using the same mapping and buffers.
3) Preserve pointer stability and Binding safety semantics (retire -> overwrite -> optional publish).
4) Make integration with vLLM zero logic duplication: vLLM supplies the plan, Tensorcast executes it.
5) Keep disk as a fallback, not the default; prefer artifact resolution via daemon or GS.

# Non Goals

- Support arbitrary view transforms (transpose or permute) in mapping.
- Support multi dimensional slicing (beyond single dim ranges) in v1.
- Make partial mappings publishable (full coverage only for publish).
- Replace or eliminate vLLM tracing (trace remains the semantic authority).

# Functional Requirements

## 1) Mapping model and types

### 1.1 Python types

```python
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

@dataclass(frozen=True)
class Range:
    dim: int        # 0 or 1 (v1 constraint)
    start: int      # inclusive
    end: int        # exclusive

@dataclass(frozen=True)
class CopyPlanEntry:
    ckpt_name: str
    ckpt_range: Range | None      # None = full tensor
    dst_name: str
    dst_range: Range | None       # None = full tensor

CopyPlan = Sequence[CopyPlanEntry]
TargetTensors = Mapping[str, torch.Tensor]
```

### 1.2 Coordinate system (MUST be explicit)

- Mapping ranges are expressed in **canonical artifact coordinates**, not view
  coordinates. This matches vLLM trace output.
- If the Binding is created from an artifact view (artifact.view(...)), the
  Binding MUST internally translate canonical ranges into the view materialization
  coordinates (for example by subtracting view slice offsets).

Rationale: vLLM trace is stable across swaps; callers should not rewrite the plan
for each view or swap.

### 1.3 Constraints (v1)

- Only one slicing dimension per entry; dim in {0, 1}.
- start < end for any Range.
- None means full tensor.
- Must support rename (ckpt_name != dst_name).
- Must support multi entry split or merge (many entries can target same dst or same src).
- Must support scalar fill: 0 d source -> 1 element destination.
- Require contiguous src/dst tensors; dst storage_offset must be 0.
- Require full dst coverage with no overlaps or gaps (v1 determinism).

### 1.4 Serialization

- Copy plans serialize to stable JSON with a version tag (v1) and deterministic ordering.
- The on-wire schema mirrors the same structure via `CopyPlan` / `CopyPlanEntry` in the daemon proto.

## 2) API surface (precise signatures)

**Decision: choose Option B (extend `bind_into`) as the primary API.**

Rationale: `bind_into(...)` is already the “adopt + fill” construction surface; adding `mapping=...` keeps the
user-facing abstraction minimal and avoids proliferating near-duplicate methods.

**API (required):**

```python
class Artifact:
    def bind_into(
        self,
        target_tensors: TargetTensors,
        *,
        mapping: CopyPlan | None = None,
        packing: str = "byte_space",
        publish: bool = False,
        ctx: CallContext | None = None,
    ) -> "Binding":
        ...
```

Binding swap (unchanged API, but mapping aware):

```python
class Binding:
    def swap(
        self,
        artifact_or_ref: "Artifact | str",
        *,
        publish: bool = False,
        activate_key: str | None = None,
        expected_active_artifact_id: str | None = None,
        expected_active_generation: int | None = None,
        wait: bool = True,
        drain_timeout_s: float | None = None,
        ctx: CallContext | None = None,
    ) -> None:
        ...
```

### Naming compliance (Python)

- Methods: `bind_into`, `swap` are `snake_case`.
- Data classes: `Range`, `CopyPlanEntry` are `PascalCase`.
- Type aliases: `CopyPlan`, `TargetTensors` are `PascalCase`.

Notes:
- artifact_or_ref may be an Artifact handle or a string key or artifact_id.
- ctx is `tensorcast.api.context.CallContext` (optional per-call deadlines/metadata).
- The Binding created by mapped bind MUST store the mapping and reuse it for swap.

## 3) Validation rules (MUST)

At creation time:
- All target_tensors must be CUDA, writable, and user owned (no CUDA IPC imports).
- For each mapping entry, validate dtype and shape compatibility.
- ckpt_range and dst_range must be in bounds.
- No empty slice; end > start.
- All dst tensors referenced in mapping must exist in target_tensors.
- Disallow non contiguous dst tensors (v1) unless explicitly supported.
- Require full dst coverage (no overlaps or gaps) before any bytes are written.
- For view-narrowed tensors, require an explicit range and translate to view coordinates.
- Reject transpose/permutation view ops for mapped binding in v1.

At swap time:
- Validate new artifact selection against mapping (shape or dtype mismatches).
- On mismatch, fail swap with a clear error and do not partially overwrite.

## 4) Error semantics (MUST)

- If overwrite begins and materialization fails, Binding becomes Dirty.
- If publish fails after overwrite, bytes are correct but Binding remains local only.
- Errors must include enough context: entry index, src and dst name, and ranges.

## 5) Performance requirements

- Must avoid materializing full tensor_dict just to then copy in Python.
- Should execute inside Tensorcast materialization pipeline.
- Should allow reuse of existing region backed write paths.

## 6) Observability

- Each bind or swap should have a single operation_id propagated to internal RPCs.
- Basic metrics: bytes copied, tensors mapped, time to swap.

# Expected Effects

- vLLM can reload weights via binding.swap without re tracing on every reload.
- Eliminates Python level per entry copy loops.
- Enables stable pointer updates across weight versions with minimal peak memory.

# Concrete Example (vLLM TP reload)

## 1) Model and checkpoint shapes

- ckpt tensor: "layers.0.mlp.gate_up_proj.weight" shape (8192, 4096)
- dst tensor:  "layers.0.mlp.gate_proj.weight" shape (4096, 4096)
- dst tensor:  "layers.0.mlp.up_proj.weight" shape (4096, 4096)

## 2) Copy plan produced by trace

```python
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
```

## 3) View to minimize materialization

```python
src_hull = {
    "layers.0.mlp.gate_up_proj.weight": [(0, slice(0, 8192))],
}
materialize_names = ["layers.0.mlp.gate_up_proj.weight"]

artifact = tc.artifact(
    key="model:v1",
    fallback=tc.FallbackOptions.for_disk(hf_dir),
)
artifact_tp = artifact.view(slices=src_hull, names=materialize_names)
```

## 4) Binding creation and swap

```python
binding = artifact_tp.bind_into(
    target_tensors=dst_by_name,
    mapping=copy_plan,
    packing="byte_space",
)

# Later reload to a new version
binding.swap("model:v2")
```

Outcome: parameters update in place, pointers are stable, no Python copy loop.

# JSON serialization example (for caching)

```json
{
  "copy_plan": [
    {
      "ckpt_name": "layers.0.mlp.gate_up_proj.weight",
      "ckpt_range": {"dim": 0, "start": 0, "end": 4096},
      "dst_name": "layers.0.mlp.gate_proj.weight",
      "dst_range": {"dim": 0, "start": 0, "end": 4096}
    },
    {
      "ckpt_name": "layers.0.mlp.gate_up_proj.weight",
      "ckpt_range": {"dim": 0, "start": 4096, "end": 8192},
      "dst_name": "layers.0.mlp.up_proj.weight",
      "dst_range": {"dim": 0, "start": 0, "end": 4096}
    }
  ]
}
```

# Compatibility and Interactions

- Fully compatible with Binding design (0063); mapped binding is a specialization.
- Mapped binding supports publish in v1 when full destination coverage validation passes and the daemon mints a
  `target_publication_token`; otherwise publish fails with a clear precondition error.
- Works with artifact.view(...): selection is captured and reused on swap.
- If a new artifact has incompatible shapes or dtypes, swap must fail or trigger a rebind (caller decision).

# Test Plan (Acceptance Criteria)

1) Pointer stability: data_ptr unchanged after swap.
2) Correctness vs baseline: mapped binding output equals trace + Python copy baseline.
3) Split or fused mapping: multiple entries per src or dst behave correctly.
4) Scalar fill: 0 d source fills 1 element dst.
5) Mismatch detection: wrong dtype or shape causes clean failure without partial overwrite.
6) Swap failure semantics: dirty state on overwrite failure.
7) Publish gating: mapped bindings publish only under the documented v1 constraints (full coverage + valid
   `target_publication_token`), otherwise fail without partial publish state.
