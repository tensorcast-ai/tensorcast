---
slug: process-scoped-region-backed-artifact-session-plan
title: Process-Scoped Region-Backed Artifact Session Implementation Plan
links:
  design: ../designs/0122-process-scoped-region-backed-artifact-session.md
areas: ["sdk", "tests", "docs"]
related_code:
  - tensorcast/api/store/__init__.py
  - tensorcast/api/store/README.md
  - tensorcast/api/store/region_backed_artifact_session.py
  - tensorcast/common/identity.py
  - tensorcast/common/selection_contract.py
  - tensorcast/daemon_ctl.py
  - tests/python/api/test_region_backed_artifact_session.py
  - tests/python/test_region_backed_artifact_session_e2e.py
  - tests/python/api/test_public_surface.py
  - tests/python/utils/daemon.py
last_updated: 2026-09-04
---

# Objective

Implement design `0122` as an additive Python SDK surface that lets one caller
process attach to one ready, node-local StoreDaemon and use process-pinned
host-shared regions for batched byte-artifact `exists`, `get_into`, and
`put_from` operations.

The implementation must deliver both initial transfer modes:

- fixed-capacity, direction-specific scratch arenas; and
- allocator-backed caller-side direct transfer across one or more live regions.

The SDK owns artifact identity lowering, region registration and attachment,
multi-storage layouts, slot wire tokens, message-size admission, response
validation, sticky RPC health, and process-session lifecycle. External callers
must not construct generated protobuf messages or use `DaemonCtl` directly.

This plan is execution-oriented. Every phase adds focused unit coverage and has
an explicit validation gate. A phase is not complete until its gate passes.

# Starting State & Grounding

The repository already provides the daemon and wire primitives required by
`0122`:

- `tensorcast/daemon_ctl.py` provides:
  - StoreDaemon channel construction and refresh;
  - `BatchExists`, `BatchGetIntoRegion`, and
    `BatchPutIfAbsentFromRegion` wrappers;
  - host-shared region registration, attachment, release, and unregister
    helpers; and
  - validated client gRPC maximum send/receive environment settings.
- `tensorcast/common/identity.py` provides
  `build_byte_artifact_cgid()`.
- `tensorcast/common/selection_contract.py` provides canonical byte-artifact
  selection construction.
- the current daemon protobuf already carries multi-storage `TargetLayout`,
  region offsets, slot index/generation tokens, per-item outcomes, and
  layout-and-size-only put invariants.
- daemon tests already cover host-shared region layouts, stable backing, and
  batch-region operations.

The missing layer is the public process-scoped SDK session and its internal
compilation and lifecycle logic.

Current `DaemonCtl` region get/put wrappers use one SDK retry by default.
`0122` must preserve existing behavior for existing callers while giving the
new Session an explicit zero-retry path. Current channel message limits are
resolved while building channel options but are not retained as one explicit
per-client immutable snapshot.

# Scope And Constraints

## In scope

- Python SDK implementation of every public type and method in design `0122`.
- A concrete public attach-time exception, `RegionSessionAttachError`, so
  pre-Session failures never leak raw gRPC exceptions.
- Minimal internal changes to `DaemonCtl` for:
  - immutable effective message-limit snapshots;
  - reuse of those limits across channel refresh; and
  - explicit zero-retry region get/put invocation without changing existing
    default behavior.
- Unit tests based on fake clients, generated protobuf fixtures, temporary
  memfds, and fake CUDA mode.
- End-to-end Python acceptance tests against a prebuilt StoreDaemon binary.
- Public SDK exports and documentation.

## Out of scope

- StoreDaemon C++ implementation changes.
- Protobuf or persistent-schema changes.
- Building the C++ daemon as part of this implementation plan. The validation
  environment supplies a compatible prebuilt daemon.
- Building a Python native extension specifically for this feature. The Session
  uses existing Python SDK and generated-protobuf facilities.
- Framework-specific cache, page, rank, model, or error-policy logic.
- Recovery of a failed or terminated Session in the same process.
- Runtime release of process-pinned regions before owner-process exit.
- Solving shared-channel refresh interference, process-ID reuse in diagnostic
  region names, or post-attach fork/spawn behavior.

## Implementation rules

- Treat the `0122` design as normative. If implementation reveals a necessary
  semantic change, update the design before changing the contract.
- Keep generated protobuf and `DaemonCtl` objects inside the SDK boundary.
- Prefer one new implementation module:
  `tensorcast/api/store/region_backed_artifact_session.py`.
  Extract a private helper module only if the implementation becomes materially
  clearer; do not split public resource ownership across several Sessions.
- Preserve existing `DaemonCtl` defaults for all old call sites.
- Direct region get and put always invoke the underlying RPC with `retries=0`.
- Unit tests must not require a daemon process. Only the final end-to-end lane
  may require `TENSORCAST_DAEMON_BIN` or the default
  `bazel-bin/daemon/tensorcast_daemon`.
- Test-only registry reset/injection may be private and may operate only on fake
  Sessions with no process-pinned production mapping. Do not add a public reset
  API.
- Every Python command activates `.venv` and uses `uv`.
- Python tests select fake CUDA with `TENSORCAST_CUDA_BACKEND=fake`. Bazel uses
  the equivalent `--test_env=TENSORCAST_CUDA_BACKEND=fake` spelling.

# Proposed Code And Test Layout

```text
tensorcast/
├── daemon_ctl.py
└── api/store/
    ├── __init__.py
    ├── README.md
    └── region_backed_artifact_session.py

tests/python/
├── api/
│   ├── test_public_surface.py
│   └── test_region_backed_artifact_session.py
└── test_region_backed_artifact_session_e2e.py
```

The primary unit-test file is organized into stable test classes so each phase
can run a narrow gate without proliferating small files:

```text
TestPublicContracts
TestGrpcMessageLimits
TestAttachRegistry
TestSessionState
TestHostMemorySpan
TestRegionAllocation
TestArtifactLowering
TestWireBudget
TestOutcomeValidation
TestScratchTransfer
TestAllocatorDirectTransfer
TestConcurrentAdmission
TestFailureContract
TestTerminationContract
```

# Execution Order

Execute phases in this order. Do not begin a later phase while an earlier
validation gate is red.

- [x] Phase 1: Public contracts and RPC primitives
- [x] Phase 2: Process attach, registry, and state admission
- [x] Phase 3: Host spans, region allocation, and pinned ownership
- [x] Phase 4: Artifact lowering, layouts, wire budget, and outcomes
- [x] Phase 5: Scratch transfer mode
- [x] Phase 6: Allocator-backed direct transfer mode
- [x] Phase 7: Failure hardening, daemon acceptance, and documentation closure

# Phases & Milestones

- [x] Phase 1
  - [x] Public value/resource contracts are defined and validated in the new
        leaf module.
  - [x] `DaemonCtl` has one immutable effective message-limit snapshot.
  - [x] direct region RPC wrappers can be called with zero SDK retries without
        changing existing defaults.
- [x] Phase 2
  - [x] one process attaches to at most one canonical daemon endpoint;
  - [x] the simple registry mutex prevents duplicate concurrent construction;
  - [x] sticky health and terminal lifecycle admission order is executable.
- [x] Phase 3
  - [x] owned spans and allocator tensors keep backing memory alive;
  - [x] multiple host-shared region records coexist and resolve safely;
  - [x] only exact unexposed `Building` rollback releases a region.
- [x] Phase 4
  - [x] callers' keyspaces and engine keys lower to canonical artifacts;
  - [x] scratch and multi-region direct layouts are compiled internally;
  - [x] payload limits, geometry, slot tokens, and outcomes are validated.
- [x] Phase 5
  - [x] scratch put packs caller bytes and scratch get copies only successful
        outcomes;
  - [x] get and put arenas have independent direction locks.
- [x] Phase 6
  - [x] direct get/put uses caller allocations without Session scratch copies;
  - [x] one batch may span multiple allocator regions;
  - [x] direct RPC retry and timeout contracts match `0122`.
- [x] Phase 7
  - [x] every fatal class latches exactly one stable first failure;
  - [x] termination and process-pinned retention are covered;
  - [x] prebuilt-daemon acceptance and relevant daemon contract tests pass;
  - [x] public docs and API-surface tests are complete.

# Detailed Plan

## Phase 1: Public Contracts And RPC Primitives

Purpose:

- establish the caller-visible vocabulary before stateful implementation;
- make RPC retry and payload-size behavior observable to the Session without
  exposing transport details publicly.

### Implementation tasks

- [x] Add `tensorcast/api/store/region_backed_artifact_session.py`.
- [x] Implement the frozen Pydantic and typed resource/result declarations from
      design `0122`:
  - [x] `RegionTransferMode`;
  - [x] `RegionSessionHealth`;
  - [x] `RegionSessionLifecycleState`;
  - [x] `RegionSessionFailureCode`;
  - [x] `RegionSessionOperationKind`;
  - [x] `ByteArtifactKeyspace` and `ByteArtifactSpec`;
  - [x] scratch/allocator transfer options and Session options;
  - [x] `RegionArtifactTransfer`;
  - [x] exists/transfer results; and
  - [x] input, attach, failed, and terminated exceptions.
- [x] Give public Pydantic models `frozen=True` and `extra="forbid"`.
- [x] Reject empty identity fields, non-positive byte lengths, invalid timeout
      values, and mode-incompatible configuration locally.
- [x] Keep canonical artifact IDs, protobufs, region handles, and wire enums out
      of all public models.
- [x] Add a private frozen `_GrpcMessageLimits` value in `daemon_ctl.py`.
- [x] Resolve maximum send and receive bytes once in `DaemonCtl.__init__()`
      through the existing validated environment/default functions.
- [x] Pass the snapshot into channel-option construction and reuse it in
      `_refresh_channel()` rather than rereading environment variables.
- [x] Add a private read-only way for the Session implementation to obtain the
      same snapshot.
- [x] Add a backwards-compatible internal retry parameter to region get/put
      wrappers:
  - [x] existing callers retain the current default;
  - [x] a Session call can pass `retries=0` explicitly; and
  - [x] the original exception remains reachable through `__cause__` for stable
        Session failure classification.
- [x] Keep the incomplete Session out of `tensorcast.api.store` re-exports until
      Phase 7 closes both transfer modes and the final public-surface gate.

### Unit tests

- [x] `TestPublicContracts`
  - [x] validates frozen/forbid-extra models and the transfer discriminator;
  - [x] rejects malformed keyspaces, lengths, capacities, and timeouts;
  - [x] verifies result tuple types and nullable empty-transfer operation ID;
  - [x] verifies public exception inheritance and failure schema; and
  - [x] verifies caller inputs expose no artifact-id or protobuf field.
- [x] `TestGrpcMessageLimits`
  - [x] verifies defaults and environment overrides are normalized once;
  - [x] mutates the environment after client construction and proves refresh
        retains the original snapshot;
  - [x] proves channel options and Session access observe identical values;
  - [x] proves invalid environment values follow existing fallback behavior;
        and
  - [x] spies on region get/put wrappers and proves `retries=0` reaches
        `_unary_call()` when requested while the old default remains unchanged.

### Validation gate

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "PublicContracts or GrpcMessageLimits"
uv run ruff check \
  tensorcast/daemon_ctl.py \
  tensorcast/api/store/region_backed_artifact_session.py \
  tensorcast/api/store/__init__.py \
  tests/python/api/test_region_backed_artifact_session.py
uv run ruff format --check \
  tensorcast/daemon_ctl.py \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
```

Phase 1 exit criteria:

- all contract declarations and effective-limit tests pass in the leaf module;
- old `DaemonCtl` callers keep their previous retry default;
- no public signature contains a generated protobuf or daemon-private type.

## Phase 2: Process Attach, Registry, And State Admission

Purpose:

- establish one process-scoped Session and one sticky failure domain;
- make lifecycle and admission behavior deterministic before adding memory and
  transfer logic.

### Implementation tasks

- [x] Implement `RegionBackedArtifactSession.attach()` with a private client
      factory seam for unit tests.
- [x] Normalize options and canonicalize the daemon endpoint before registry
      lookup.
- [x] Capture owner PID internally; never accept it from the caller.
- [x] Implement the process registry keyed by owner PID and canonical endpoint.
- [x] Use one ordinary mutex held across lookup, endpoint validation, basic
      handshake, and final publication.
- [x] Enforce one daemon endpoint per process and compare normalized operational
      fingerprints for repeated attach.
- [x] Make equal attach return the same attached Session; reject conflicting
      options, a second endpoint, and reattach after terminal termination.
- [x] Perform the minimal attach handshake:
  - [x] daemon response is reachable;
  - [x] `startup_phase == READY`;
  - [x] CPU shared memory is enabled;
  - [x] configured endpoint and local-handle facts are node-local; and
  - [x] required basic fields are present.
- [x] Map all pre-publication connection/readiness/capability failures to
      `RegionSessionAttachError` with the original cause retained.
- [x] Implement independent `READY | FAILED` health and
      `ATTACHED | TERMINATED` lifecycle state.
- [x] Implement the two-stage state gate:
  - [x] take a strong-reference input snapshot;
  - [x] check `FAILED` before `TERMINATED`;
  - [x] validate outside the lock;
  - [x] recheck before final RPC admission; and
  - [x] recheck state before returning a local validation error.
- [x] Implement atomic first-failure retention and diagnostic in-flight count.
- [x] Implement empty-batch behavior with the preliminary state gate, no RPC,
      no in-flight increment, empty tuples, zero timings, and
      `operation_id=None` for empty transfer results.
- [x] Implement idempotent `terminate_process_session()` admission closure.
- [x] Do not call `release_daemon_client()` from termination.
- [x] Add a private unit-test fixture that isolates registry state without
      creating a public reset API.

### Unit tests

- [x] `TestAttachRegistry`
  - [x] same normalized options return one object;
  - [x] different endpoint or operational fingerprint is rejected;
  - [x] session name and region prefix are first-attach-wins diagnostics;
  - [x] concurrent attach performs exactly one handshake/publication;
  - [x] non-ready daemon and disabled CPU shared memory raise attach error;
  - [x] attach error publishes no Session; and
  - [x] registry lock is not acquired by data-plane method stubs.
- [x] `TestSessionState`
  - [x] failed state wins over terminated state and invalid input;
  - [x] terminated state wins over invalid input on a healthy Session;
  - [x] a concurrent failure between preliminary and final gates prevents RPC;
  - [x] empty methods issue no RPC and return exact empty result schemas;
  - [x] empty methods still raise for failed or terminated state;
  - [x] termination is idempotent; and
  - [x] termination never releases the shared daemon client.

### Validation gate

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "AttachRegistry or SessionState"
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/test_daemon_ctl_retry.py \
  tests/python/api/test_startup_client_config.py
uv run ruff check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
uv run ruff format --check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
```

Phase 2 exit criteria:

- concurrent attach cannot create two Sessions;
- terminal state ordering and empty-call behavior are fully unit tested;
- no attach failure leaks a raw gRPC/protobuf exception through the public API.

## Phase 3: Host Spans, Region Allocation, And Pinned Ownership

Purpose:

- make the process-local memory and region lifecycle correct before any
  artifact RPC references those addresses.

### Implementation tasks

- [x] Implement `HostMemorySpan.from_tensor()`:
  - [x] require CPU-accessible, dense contiguous storage;
  - [x] validate byte offset, byte length, bounds, and overflow;
  - [x] resolve address and length once per call snapshot; and
  - [x] strongly retain the tensor/storage owner.
- [x] Implement `HostMemorySpan.from_address()` with mandatory explicit owner,
      positive address/length checks, and overflow validation.
- [x] Enforce `artifact.byte_length == span.byte_length` when constructing or
      validating `RegionArtifactTransfer`.
- [x] Implement a private per-allocation region record containing handle,
      attachment, FD/mmap roots, tensor root, address interval, capacity,
      lifecycle, and optional slot geometry.
- [x] Implement daemon-managed, non-expiring `HOST_SHARED/ALLOCATOR` region
      creation for `allocate_host_tensor()`.
- [x] Attach the local FD, mmap the complete region, and construct a dense
      contiguous CPU `torch.Tensor` with the requested shape and dtype.
- [x] Validate non-negative shape dimensions, supported element size,
      multiplication overflow, non-zero allocation size, and CPU-only device
      semantics.
- [x] Retain every allocation; never overwrite the previous allocation record.
- [x] Implement interval-based address containment with exactly-one-record
      resolution and cross-region rejection.
- [x] Implement region lifecycle transitions:
  - [x] `Building -> RolledBack` only when no view escaped, no data RPC used the
        region, and cleanup is exact;
  - [x] `Building -> ProcessPinned` when a tensor or scratch arena becomes
        usable; and
  - [x] no SDK-driven reclamation after `ProcessPinned`.
- [x] On exact `Building` rollback, close local resources and invoke release and
      unregister exactly once.
- [x] On ambiguous setup failure, latch Session failure and retain every
      possibly live daemon/local resource.
- [x] Ensure Session failure and termination retain all mappings and tensor
      roots until process teardown.

### Unit tests

- [x] `TestHostMemorySpan`
  - [x] tensor view address/offset/length is correct;
  - [x] non-contiguous, non-CPU, zero, negative, out-of-bounds, and overflowing
        spans are rejected;
  - [x] explicit owner is mandatory for raw address spans;
  - [x] caller sequence mutation does not change the captured span facts; and
  - [x] weak-reference tests prove tensor and explicit owner retention through
        return or raise.
- [x] `TestRegionAllocation`
  - [x] fake host-shared registration plus a temporary memfd produces a dense,
        page-aligned CPU tensor with stable address and exact size;
  - [x] two or more allocations remain live and independently resolvable;
  - [x] ranges crossing region boundaries or matching no region fail locally;
  - [x] exact unexposed rollback releases/unregisters once;
  - [x] ambiguous registration/FD/mmap outcomes latch and do not clean up;
  - [x] returned allocation, failure, and termination never release a
        process-pinned region; and
  - [x] endpoint/local-handle reachability checks allocate no probe region.

### Validation gate

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "HostMemorySpan or RegionAllocation"
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/test_store_region_registration.py
uv run ruff check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
uv run ruff format --check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
```

Phase 3 exit criteria:

- allocator tensors are real mapped CPU tensors, not copies;
- multiple region mappings remain strongly rooted and address-resolvable;
- no process-pinned cleanup RPC is reachable from failure or termination.

## Phase 4: Artifact Lowering, Layouts, Wire Budget, And Outcomes

Purpose:

- centralize all generated-protobuf construction and response interpretation;
- prove pure compilation rules before executing transfers.

### Implementation tasks

- [x] Derive canonical artifact IDs exclusively with
      `build_byte_artifact_cgid()` from each `ByteArtifactSpec` keyspace and
      engine key.
- [x] Reuse canonical byte-artifact selection helpers; do not maintain an
      identity or selection cache.
- [x] Reject duplicate derived artifact IDs in one batch before RPC admission.
- [x] Build put invariants with layout ID, byte length, and
      `LAYOUT_AND_SIZE_ONLY` verification; do not hash caller bytes.
- [x] Implement scratch layout compilation with one storage and packed offsets.
- [x] Implement allocator layout compilation:
  - [x] resolve every span to exactly one allocation record;
  - [x] preserve caller order;
  - [x] deduplicate regions in first-appearance order;
  - [x] assign request-local storage IDs;
  - [x] compute checked logical storage bases and offsets; and
  - [x] emit one storage entry per touched region.
- [x] Implement per-region candidate geometry validation.
- [x] Commit unseen geometry atomically and all-or-nothing only during final
      admission after all local and wire-budget validation passes.
- [x] Establish lock order: Session state before geometry; hold neither across
      RPC.
- [x] Implement non-zero monotonic RPC generation and derive slot index from
      region-local offset and frozen slot size.
- [x] Attach one request generation to all direct offsets and validate exact
      echoed slot tokens.
- [x] Build completed request protobufs before applying `ByteSize()`.
- [x] Estimate ordinary response size from artifact IDs, statuses, direct slot
      tokens, protobuf overhead, and fixed headroom.
- [x] Reject oversized transfer batches before final admission.
- [x] Partition exists input into maximal contiguous sub-batches that fit both
      client send and receive limits.
- [x] Give every exists partition one SDK correlation/operation ID; retries of
      that partition reuse it, while the next partition gets another ID.
- [x] Implement artifact-based outcome correlation and restore caller order.
- [x] Allow only operation-specific `OK` and `MISS` statuses from design
      `0122`; reject missing, duplicate, unknown, or malformed outcomes.
- [x] Map `REGION_LOST` only from structured machine-readable region evidence;
      never parse free-form error text.

### Unit tests

- [x] `TestArtifactLowering`
  - [x] same keyspace/engine key produces the canonical existing byte-artifact
        identity;
  - [x] multiple keyspaces coexist in one batch;
  - [x] duplicates are rejected;
  - [x] put invariant is layout-and-size-only; and
  - [x] no public result exposes artifact IDs or selections.
- [x] `TestWireBudget`
  - [x] uses the exact `DaemonCtl` limit snapshot;
  - [x] measures completed request protobufs;
  - [x] rejects get/put before RPC when either direction exceeds its limit;
  - [x] partitions exists at exact boundary conditions and preserves order;
  - [x] uses one operation ID per partition and reuses it for retries; and
  - [x] reports total exists RPC elapsed time across partitions and attempts.
- [x] `TestOutcomeValidation`
  - [x] accepts operation-specific OK/MISS combinations;
  - [x] rejects missing, duplicate, unknown, reordered-with-bad-correlation,
        and non-allowlisted outcomes;
  - [x] validates echoed direct slot tokens;
  - [x] maps generic `FAILED_PRECONDITION` to `DAEMON_STATUS`;
  - [x] maps only typed region evidence to `REGION_LOST`; and
  - [x] proves error-message text does not affect classification.
- [x] Geometry tests
  - [x] compatible first use freezes one slot size;
  - [x] locally invalid or wire-oversized input freezes nothing;
  - [x] a multi-region install is all-or-nothing; and
  - [x] concurrent conflicting first use admits at most one geometry and sends
        no RPC for the loser.

### Validation gate

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "ArtifactLowering or WireBudget or OutcomeValidation or geometry"
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/test_byte_artifact_identity.py \
  tests/python/api/test_materialization_token_guards.py
uv run ruff check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
uv run ruff format --check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
```

Phase 4 exit criteria:

- pure lowering produces valid scratch and multi-region direct requests;
- an invalid request cannot mutate frozen geometry or issue an RPC;
- all outcome and failure-code decisions are deterministic and message-text
  independent.

## Phase 5: Scratch Transfer Mode

Purpose:

- complete the staged compatibility path using ordinary caller host memory;
- isolate late writes from caller targets after a fatal get.

### Implementation tasks

- [x] Lazily create one fixed-capacity daemon-managed `HOST_SHARED/SCRATCH`
      region for get and one for put.
- [x] Move each arena to `ProcessPinned` before its first admitted transfer.
- [x] Retain both mappings until process teardown; never resize, replace,
      expire, release, or unregister them after pinning.
- [x] Use independent get and put locks so one get and one put may run
      concurrently while same-direction calls serialize.
- [x] Implement scratch put:
  - [x] validate and snapshot all source spans;
  - [x] reject total packed bytes above capacity;
  - [x] pack in caller order;
  - [x] issue one region-backed put with `retries=0`; and
  - [x] validate all outcomes before returning the success mask.
- [x] Implement scratch get:
  - [x] build packed target offsets;
  - [x] issue one region-backed get with `retries=0`;
  - [x] validate the complete response before copying;
  - [x] copy only successful items to caller spans; and
  - [x] leave every false target unconsumable and unchanged where practical.
- [x] On fatal scratch get, copy no scratch bytes to caller targets and never
      reuse the failed arena.
- [x] Record pack, copy, RPC, bytes, and direction metrics without artifact IDs
      as unbounded labels.

### Unit tests

- [x] `TestScratchTransfer`
  - [x] regions are lazy, direction-specific, fixed-capacity, and created once;
  - [x] put packs exact bytes and preserves caller order;
  - [x] get copies only OK items after whole-response validation;
  - [x] MISS/False target is not consumed;
  - [x] malformed/fatal get copies no target bytes and latches Session;
  - [x] overflow rejects before arena mutation and RPC;
  - [x] get/get and put/put serialize, while get/put can overlap;
  - [x] get and put each call the raw RPC with `retries=0`;
  - [x] empty calls allocate no arena; and
  - [x] termination/failure sends no arena cleanup RPC.

### Validation gate

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "ScratchTransfer"
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/test_store_region_registration.py \
  tests/python/test_daemon_ctl_retry.py
uv run ruff check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
uv run ruff format --check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
```

Phase 5 exit criteria:

- scratch mode can complete mixed-hit batches without exposing scratch storage;
- no failed scratch get can publish partially validated bytes to caller targets;
- same-direction arena reuse is race-free.

## Phase 6: Allocator-Backed Direct Transfer Mode

Purpose:

- complete caller-side zero-copy transfer for Session-allocated host memory;
- support asymmetric or otherwise independently allocated regions in one batch.

### Implementation tasks

- [x] Accept direct spans only when wholly contained in allocations owned by the
      same Session.
- [x] Compile every direct batch into one multi-storage request, regardless of
      how many owned regions it touches.
- [x] Submit caller-region addresses directly; do not allocate or copy through
      a Session scratch arena.
- [x] Keep all referenced spans, tensor roots, mmap roots, and explicit owners
      alive until return or raise.
- [x] Enforce pairwise non-overlap within one batch.
- [x] Preserve caller-owned cross-call exclusivity without adding an in-flight
      interval allocator to the SDK.
- [x] Use no client deadline when `transfer_timeout_s=None`.
- [x] Issue direct get and put with exactly one SDK attempt (`retries=0`).
- [x] Return one transfer operation ID for the one direct RPC.
- [x] Treat only `True` direct-get targets as consumable; after fatal direct get,
      mark every submitted target untrusted through the raised Session error.
- [x] Allow concurrent direct get/put calls without a global data-plane lock;
      retain only short state, region, geometry, and generation locks.
- [x] After successful put returns, release the source borrow and ensure later
      caller mutation cannot change the stored artifact.
- [x] Never call explicit stable-backing activation; rely on daemon layout
      validation and the frozen region geometry.

### Unit tests

- [x] `TestAllocatorDirectTransfer`
  - [x] one-region get/put compiles exact offsets with no scratch allocation;
  - [x] two-region and three-region batches emit one storage entry per region
        and one RPC total;
  - [x] first-appearance storage ordering and logical bases are deterministic;
  - [x] a batch can mix keyspaces while preserving input-order results;
  - [x] foreign, crossing, overlapping, and out-of-range spans fail before RPC;
  - [x] `transfer_timeout_s=None` is passed as no deadline;
  - [x] configured finite timeout is passed through but retry remains zero;
  - [x] get and put never call scratch-copy helpers;
  - [x] success/MISS consumption rules match the public contract; and
  - [x] failure and termination never release allocator mappings.
- [x] `TestConcurrentAdmission`
  - [x] direct get and put may overlap in time;
  - [x] generation values remain unique and non-zero under concurrency;
  - [x] Session/geometry locks are released before blocking RPC;
  - [x] a failure latched by one call prevents later admission;
  - [x] an already completed concurrent success is discarded if health failed
        before exposure; and
  - [x] counter wrap fails closed.

### Validation gate

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "AllocatorDirectTransfer or ConcurrentAdmission"
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py \
  -k "geometry or OutcomeValidation"
uv run ruff check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
uv run ruff format --check \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py
```

Phase 6 exit criteria:

- allocator mode is caller-side zero-copy and supports multiple live regions;
- direct region RPCs have no SDK retry path;
- concurrency never holds a Session/geometry/registry lock across blocking RPC.

## Phase 7: Failure Hardening, Daemon Acceptance, And Documentation Closure

Purpose:

- prove the complete public contract against fault injection and a real prebuilt
  daemon;
- close public exports, docs, and compatibility validation.

### Implementation tasks

- [x] Complete stable first-failure classification:
  - [x] transport/deadline/cancellation/availability -> `TRANSPORT`;
  - [x] non-allowlisted/unknown item status -> `DAEMON_STATUS`;
  - [x] ambiguous region setup -> `REGION_SETUP`;
  - [x] typed already-pinned region loss -> `REGION_LOST`;
  - [x] malformed response/token mismatch -> `MALFORMED_RESPONSE`; and
  - [x] otherwise unexpected admitted SDK failure -> `INTERNAL`.
- [x] Atomically retain only the first failure with UTC timestamp, operation
      kind, and failing operation ID.
- [x] Ensure later allocation, exists, get, and put calls raise that same
      failure without issuing RPC.
- [x] For partitioned exists, stop before the next partition after failure or
      termination and expose no partial masks.
- [x] Ensure first fatal logging contains traceback once and derivative logs are
      rate-limited.
- [x] Finish termination behavior:
  - [x] close new admission;
  - [x] retain control resources until admitted work exits;
  - [x] stop only Session-owned helper resources;
  - [x] do not close/release the process-shared `DaemonCtl`; and
  - [x] never unmap or release process-pinned regions.
- [x] Add generic observability required by `0122` without high-cardinality
      metric labels.
- [x] Add end-to-end Python tests that launch the supplied prebuilt daemon using
      `tests/python/utils/daemon.py`.
- [x] Update `tensorcast/api/store/README.md` with scratch and allocator
      examples and explicit ownership/failure warnings.
- [x] Finalize `tensorcast.api.store` exports and update API documentation.
- [x] Synchronize design `0122` if implementation closes any naming-only gap;
      do not change its accepted ownership or failure semantics silently.

### Unit tests

- [x] `TestFailureContract`
  - [x] each stable failure-code mapping is covered;
  - [x] simultaneous fatal calls retain exactly one first failure;
  - [x] every later method raises the same retained failure object/data;
  - [x] no RPC occurs after the latch;
  - [x] a fatal exists partition records that partition's operation ID;
  - [x] generic status text cannot manufacture `REGION_LOST`; and
  - [x] successful concurrent results are suppressed after a latch.
- [x] `TestTerminationContract`
  - [x] healthy and failed termination are idempotent;
  - [x] failed error continues to win after termination;
  - [x] admitted synchronous work may reach its terminal path;
  - [x] new work and new exists partitions are rejected;
  - [x] helper cleanup is deferred until in-flight reaches zero; and
  - [x] no daemon client, FD attachment, region, mmap, or tensor root is
        released by termination.
- [x] Public documentation examples execute as unit tests or doctest-equivalent
      snippets with a fake client.

### Prebuilt-daemon acceptance tests

- [x] Scratch put -> exists -> get round trip with multiple artifacts.
- [x] Scratch partial-hit get copies only successful targets.
- [x] Allocator tensor put -> exists -> direct get round trip.
- [x] One allocator RPC spans at least two independent host-shared regions.
- [x] Direct put return is quiescent enough that mutating/reusing the caller
      source afterward does not change later retrieved bytes.
- [x] Daemon rejection causes sticky Session failure and later SDK calls issue
      no RPC.
- [x] Session termination sends no region release/unregister request.
- [x] A subprocess owner exit allows the daemon's existing PID cleanup path to
      reclaim process-pinned regions.

### Validation gate

The Python lane uses the environment-prefix form documented by the TensorCast
testing guide. If the daemon is not at the default Bazel output path, set
`TENSORCAST_DAEMON_BIN` to its absolute executable path.

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_public_surface.py \
  tests/python/test_daemon_ctl_retry.py \
  tests/python/test_store_region_registration.py \
  tests/python/test_byte_artifact_identity.py
TENSORCAST_CUDA_BACKEND=fake \
  TENSORCAST_DAEMON_BIN=/absolute/path/to/tensorcast_daemon \
  uv run pytest tests/python/test_region_backed_artifact_session_e2e.py
uv run ruff check \
  tensorcast/daemon_ctl.py \
  tensorcast/api/store/region_backed_artifact_session.py \
  tensorcast/api/store/__init__.py \
  tests/python/api/test_region_backed_artifact_session.py \
  tests/python/test_region_backed_artifact_session_e2e.py
uv run ruff format --check \
  tensorcast/daemon_ctl.py \
  tensorcast/api/store/region_backed_artifact_session.py \
  tests/python/api/test_region_backed_artifact_session.py \
  tests/python/test_region_backed_artifact_session_e2e.py
```

The existing daemon contract lane is validation of the supplied binary/source
contract, not a request to change or build daemon code for `0122`:

```bash
bazel test \
  //daemon:grpc_service_impl_batch_runtime_test \
  //daemon:byte_artifact_region_layout_host_shared_test \
  --test_env=TENSORCAST_CUDA_BACKEND=fake \
  --test_output=errors
```

Phase 7 exit criteria:

- all focused and cumulative Python tests pass in fake CUDA mode;
- both transfer modes pass against the prebuilt daemon;
- existing daemon host-shared/batch contract tests remain green;
- public docs require no raw protobuf, daemon-control, or region-handle use.

# Test Plan Summary

## Fast unit lane

Run after every SDK change:

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_region_backed_artifact_session.py
```

This lane must use fake clients and temporary local memory only. It must not
look for or launch `tensorcast_daemon`.

## SDK regression lane

Run before each phase closes:

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake uv run pytest \
  tests/python/api/test_public_surface.py \
  tests/python/test_daemon_ctl_retry.py \
  tests/python/test_store_region_registration.py \
  tests/python/test_byte_artifact_identity.py
```

## Prebuilt-daemon lane

Run at Phase 7 and before merge:

```bash
source .venv/bin/activate
TENSORCAST_CUDA_BACKEND=fake \
  TENSORCAST_DAEMON_BIN=/absolute/path/to/tensorcast_daemon \
  uv run pytest tests/python/test_region_backed_artifact_session_e2e.py
```

No `setup.py build_ext`, `bazel build`, proto generation, or C++ source change
is part of this plan. The optional Bazel test invocation above is a compatibility
gate when the prepared build environment is available.

# Rollout And Backout

## Rollout

- [ ] Land phases in execution order or keep later phases hidden until all prior
      gates are green.
- [x] Keep the new Session opt-in; do not alter ordinary `Artifact` or `Store`
      behavior.
- [x] Preserve old `DaemonCtl` retry defaults and make zero retry explicit only
      for the new direct/scratch region transfer path.
- [x] Publish public docs only when both scratch and allocator prebuilt-daemon
      tests pass.
- [ ] Integrations should first enable scratch mode, then allocator mode, while
      retaining their own policy for converting sticky L3 failure into local
      fallback.

## Backout

- [ ] Remove the public exports and Session module without changing existing
      Store APIs.
- [ ] Revert the private effective-limit accessor while preserving existing
      environment parsing if Session code is backed out.
- [ ] Preserve backwards-compatible retry defaults in every partial rollback.
- [ ] Never back out by making a process-pinned region releasable on RPC failure
      or Session termination.
- [ ] Do not add a framework-specific raw protobuf path as a temporary fallback.

# Risks And Tracking

- [x] Risk: a Session data-path call accidentally inherits the existing region
      RPC retry default.
  - mitigation: spy-based tests assert the exact `retries=0` value for scratch
    and direct get/put.
- [x] Risk: unit-test cleanup masks illegal production region cleanup.
  - mitigation: fake registry reset is test-only; process-pinned lifecycle tests
    assert no release/unregister call, and real reclamation uses a subprocess
    owner exit.
- [x] Risk: Python owner objects are collected while an RPC still uses an
      address.
  - mitigation: immutable call snapshots and strong-reference/weak-reference
    tests cover every owner path.
- [x] Risk: multi-region offset arithmetic aliases two storages or overflows.
  - mitigation: checked arithmetic, deterministic first-appearance ordering,
    and two/three-region layout tests.
- [x] Risk: an invalid first batch poisons region slot geometry.
  - mitigation: all-or-nothing commit after validation with concurrent
    conflicting-first-use tests.
- [x] Risk: response size exceeds the client receive limit despite conservative
      ordinary-outcome estimation.
  - mitigation: keep fixed headroom; classify exceptional transport overflow as
    fatal; require compatible daemon/client limits operationally.
- [x] Risk: direct no-deadline RPC hangs a test or shutdown path.
  - mitigation: unit tests use bounded fakes; end-to-end tests use an outer
    subprocess/test timeout without changing the Session's no-deadline contract.
- [x] Risk: sticky failure hides an artifact-scoped daemon rejection.
  - mitigation: retain the intentionally small success allowlist and test every
    non-allowlisted status as Session-fatal.
- [x] Risk: fake CUDA proves CPU/shared-memory correctness but not accelerator
      host registration behavior.
  - mitigation: accelerator registration remains explicitly caller-owned and
    outside `0122` acceptance.

# Owner Checklist

- [x] Public caller supplies only keyspace, engine key, byte length, and owned
      host spans.
- [x] Public types expose no canonical artifact ID, protobuf, daemon client,
      region handle, layout enum, storage ID, or slot token.
- [x] One process has one attached endpoint and one sticky Session failure
      domain.
- [x] Attach is serialized with one ordinary registry mutex.
- [x] Effective client send/receive limits are one immutable per-client
      snapshot reused by channel refresh and Session sizing.
- [x] Scratch and direct get/put use zero SDK retries.
- [x] Exists partitions transparently, uses one operation ID per partition, and
      exposes one ordered all-or-error result.
- [x] Multiple allocator regions coexist and one direct RPC can reference all
      of them.
- [x] Caller memory is strongly retained for every synchronous borrow.
- [x] Only successful get targets are consumable.
- [x] Fatal RPC, malformed response, and typed region loss latch one stable first
      failure and admit no later RPC.
- [x] Empty calls check Session state but issue no RPC.
- [x] Geometry commits only after complete local and wire-budget validation.
- [x] Failure and termination never release process-pinned regions or close the
      process-shared daemon client.
- [x] Unit, SDK regression, prebuilt-daemon, daemon-contract, and Ruff gates are
      green.
- [x] Design, implementation, tests, and public SDK documentation describe the
      same contract.
