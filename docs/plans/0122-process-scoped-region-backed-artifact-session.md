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
last_updated: 2026-09-03
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
- daemon tests already cover host-shared region layouts, stable backing,
  batch-region operations, and CPU memfd FD exchange.

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

- [ ] Phase 1: Public contracts and RPC primitives
- [ ] Phase 2: Process attach, registry, and state admission
- [ ] Phase 3: Host spans, region allocation, and pinned ownership
- [ ] Phase 4: Artifact lowering, layouts, wire budget, and outcomes
- [ ] Phase 5: Scratch transfer mode
- [ ] Phase 6: Allocator-backed direct transfer mode
- [ ] Phase 7: Failure hardening, daemon acceptance, and documentation closure

# Phases & Milestones

- [ ] Phase 1
  - [ ] Public value/resource contracts are defined and validated in the new
        leaf module.
  - [ ] `DaemonCtl` has one immutable effective message-limit snapshot.
  - [ ] direct region RPC wrappers can be called with zero SDK retries without
        changing existing defaults.
- [ ] Phase 2
  - [ ] one process attaches to at most one canonical daemon endpoint;
  - [ ] the simple registry mutex prevents duplicate concurrent construction;
  - [ ] sticky health and terminal lifecycle admission order is executable.
- [ ] Phase 3
  - [ ] owned spans and allocator tensors keep backing memory alive;
  - [ ] multiple host-shared region records coexist and resolve safely;
  - [ ] only exact unexposed `Building` rollback releases a region.
- [ ] Phase 4
  - [ ] callers' keyspaces and engine keys lower to canonical artifacts;
  - [ ] scratch and multi-region direct layouts are compiled internally;
  - [ ] payload limits, geometry, slot tokens, and outcomes are validated.
- [ ] Phase 5
  - [ ] scratch put packs caller bytes and scratch get copies only successful
        outcomes;
  - [ ] get and put arenas have independent direction locks.
- [ ] Phase 6
  - [ ] direct get/put uses caller allocations without Session scratch copies;
  - [ ] one batch may span multiple allocator regions;
  - [ ] direct RPC retry and timeout contracts match `0122`.
- [ ] Phase 7
  - [ ] every fatal class latches exactly one stable first failure;
  - [ ] termination and process-pinned retention are covered;
  - [ ] prebuilt-daemon acceptance and relevant daemon contract tests pass;
  - [ ] public docs and API-surface tests are complete.

# Detailed Plan

## Phase 1: Public Contracts And RPC Primitives

Purpose:

- establish the caller-visible vocabulary before stateful implementation;
- make RPC retry and payload-size behavior observable to the Session without
  exposing transport details publicly.

### Implementation tasks

- [ ] Add `tensorcast/api/store/region_backed_artifact_session.py`.
- [ ] Implement the frozen Pydantic and typed resource/result declarations from
      design `0122`:
  - [ ] `RegionTransferMode`;
  - [ ] `RegionSessionHealth`;
  - [ ] `RegionSessionLifecycleState`;
  - [ ] `RegionSessionFailureCode`;
  - [ ] `RegionSessionOperationKind`;
  - [ ] `ByteArtifactKeyspace` and `ByteArtifactSpec`;
  - [ ] scratch/allocator transfer options and Session options;
  - [ ] `RegionArtifactTransfer`;
  - [ ] exists/transfer results; and
  - [ ] input, attach, failed, and terminated exceptions.
- [ ] Give public Pydantic models `frozen=True` and `extra="forbid"`.
- [ ] Reject empty identity fields, non-positive byte lengths, invalid timeout
      values, and mode-incompatible configuration locally.
- [ ] Keep canonical artifact IDs, protobufs, region handles, and wire enums out
      of all public models.
- [ ] Add a private frozen `_GrpcMessageLimits` value in `daemon_ctl.py`.
- [ ] Resolve maximum send and receive bytes once in `DaemonCtl.__init__()`
      through the existing validated environment/default functions.
- [ ] Pass the snapshot into channel-option construction and reuse it in
      `_refresh_channel()` rather than rereading environment variables.
- [ ] Add a private read-only way for the Session implementation to obtain the
      same snapshot.
- [ ] Add a backwards-compatible internal retry parameter to region get/put
      wrappers:
  - [ ] existing callers retain the current default;
  - [ ] a Session call can pass `retries=0` explicitly; and
  - [ ] the original exception remains reachable through `__cause__` for stable
        Session failure classification.
- [ ] Keep the incomplete Session out of `tensorcast.api.store` re-exports until
      Phase 7 closes both transfer modes and the final public-surface gate.

### Unit tests

- [ ] `TestPublicContracts`
  - [ ] validates frozen/forbid-extra models and the transfer discriminator;
  - [ ] rejects malformed keyspaces, lengths, capacities, and timeouts;
  - [ ] verifies result tuple types and nullable empty-transfer operation ID;
  - [ ] verifies public exception inheritance and failure schema; and
  - [ ] verifies caller inputs expose no artifact-id or protobuf field.
- [ ] `TestGrpcMessageLimits`
  - [ ] verifies defaults and environment overrides are normalized once;
  - [ ] mutates the environment after client construction and proves refresh
        retains the original snapshot;
  - [ ] proves channel options and Session access observe identical values;
  - [ ] proves invalid environment values follow existing fallback behavior;
        and
  - [ ] spies on region get/put wrappers and proves `retries=0` reaches
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

- [ ] Implement `RegionBackedArtifactSession.attach()` with a private client
      factory seam for unit tests.
- [ ] Normalize options and canonicalize the daemon endpoint before registry
      lookup.
- [ ] Capture owner PID internally; never accept it from the caller.
- [ ] Implement the process registry keyed by owner PID and canonical endpoint.
- [ ] Use one ordinary mutex held across lookup, endpoint validation, basic
      handshake, and final publication.
- [ ] Enforce one daemon endpoint per process and compare normalized operational
      fingerprints for repeated attach.
- [ ] Make equal attach return the same attached Session; reject conflicting
      options, a second endpoint, and reattach after terminal termination.
- [ ] Perform the minimal attach handshake:
  - [ ] daemon response is reachable;
  - [ ] `startup_phase == READY`;
  - [ ] CPU shared memory is enabled;
  - [ ] configured endpoint and local-handle facts are node-local; and
  - [ ] required basic fields are present.
- [ ] Map all pre-publication connection/readiness/capability failures to
      `RegionSessionAttachError` with the original cause retained.
- [ ] Implement independent `READY | FAILED` health and
      `ATTACHED | TERMINATED` lifecycle state.
- [ ] Implement the two-stage state gate:
  - [ ] take a strong-reference input snapshot;
  - [ ] check `FAILED` before `TERMINATED`;
  - [ ] validate outside the lock;
  - [ ] recheck before final RPC admission; and
  - [ ] recheck state before returning a local validation error.
- [ ] Implement atomic first-failure retention and diagnostic in-flight count.
- [ ] Implement empty-batch behavior with the preliminary state gate, no RPC,
      no in-flight increment, empty tuples, zero timings, and
      `operation_id=None` for empty transfer results.
- [ ] Implement idempotent `terminate_process_session()` admission closure.
- [ ] Do not call `release_daemon_client()` from termination.
- [ ] Add a private unit-test fixture that isolates registry state without
      creating a public reset API.

### Unit tests

- [ ] `TestAttachRegistry`
  - [ ] same normalized options return one object;
  - [ ] different endpoint or operational fingerprint is rejected;
  - [ ] session name and region prefix are first-attach-wins diagnostics;
  - [ ] concurrent attach performs exactly one handshake/publication;
  - [ ] non-ready daemon and disabled CPU shared memory raise attach error;
  - [ ] attach error publishes no Session; and
  - [ ] registry lock is not acquired by data-plane method stubs.
- [ ] `TestSessionState`
  - [ ] failed state wins over terminated state and invalid input;
  - [ ] terminated state wins over invalid input on a healthy Session;
  - [ ] a concurrent failure between preliminary and final gates prevents RPC;
  - [ ] empty methods issue no RPC and return exact empty result schemas;
  - [ ] empty methods still raise for failed or terminated state;
  - [ ] termination is idempotent; and
  - [ ] termination never releases the shared daemon client.

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

- [ ] Implement `HostMemorySpan.from_tensor()`:
  - [ ] require CPU-accessible, dense contiguous storage;
  - [ ] validate byte offset, byte length, bounds, and overflow;
  - [ ] resolve address and length once per call snapshot; and
  - [ ] strongly retain the tensor/storage owner.
- [ ] Implement `HostMemorySpan.from_address()` with mandatory explicit owner,
      positive address/length checks, and overflow validation.
- [ ] Enforce `artifact.byte_length == span.byte_length` when constructing or
      validating `RegionArtifactTransfer`.
- [ ] Implement a private per-allocation region record containing handle,
      attachment, FD/mmap roots, tensor root, address interval, capacity,
      lifecycle, and optional slot geometry.
- [ ] Implement daemon-managed, non-expiring `HOST_SHARED/ALLOCATOR` region
      creation for `allocate_host_tensor()`.
- [ ] Attach the local FD, mmap the complete region, and construct a dense
      contiguous CPU `torch.Tensor` with the requested shape and dtype.
- [ ] Validate non-negative shape dimensions, supported element size,
      multiplication overflow, non-zero allocation size, and CPU-only device
      semantics.
- [ ] Retain every allocation; never overwrite the previous allocation record.
- [ ] Implement interval-based address containment with exactly-one-record
      resolution and cross-region rejection.
- [ ] Implement region lifecycle transitions:
  - [ ] `Building -> RolledBack` only when no view escaped, no data RPC used the
        region, and cleanup is exact;
  - [ ] `Building -> ProcessPinned` when a tensor or scratch arena becomes
        usable; and
  - [ ] no SDK-driven reclamation after `ProcessPinned`.
- [ ] On exact `Building` rollback, close local resources and invoke release and
      unregister exactly once.
- [ ] On ambiguous setup failure, latch Session failure and retain every
      possibly live daemon/local resource.
- [ ] Ensure Session failure and termination retain all mappings and tensor
      roots until process teardown.

### Unit tests

- [ ] `TestHostMemorySpan`
  - [ ] tensor view address/offset/length is correct;
  - [ ] non-contiguous, non-CPU, zero, negative, out-of-bounds, and overflowing
        spans are rejected;
  - [ ] explicit owner is mandatory for raw address spans;
  - [ ] caller sequence mutation does not change the captured span facts; and
  - [ ] weak-reference tests prove tensor and explicit owner retention through
        return or raise.
- [ ] `TestRegionAllocation`
  - [ ] fake host-shared registration plus a temporary memfd produces a dense,
        page-aligned CPU tensor with stable address and exact size;
  - [ ] two or more allocations remain live and independently resolvable;
  - [ ] ranges crossing region boundaries or matching no region fail locally;
  - [ ] exact unexposed rollback releases/unregisters once;
  - [ ] ambiguous registration/FD/mmap outcomes latch and do not clean up;
  - [ ] returned allocation, failure, and termination never release a
        process-pinned region; and
  - [ ] endpoint/local-handle reachability checks allocate no probe region.

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

- [ ] Derive canonical artifact IDs exclusively with
      `build_byte_artifact_cgid()` from each `ByteArtifactSpec` keyspace and
      engine key.
- [ ] Reuse canonical byte-artifact selection helpers; do not maintain an
      identity or selection cache.
- [ ] Reject duplicate derived artifact IDs in one batch before RPC admission.
- [ ] Build put invariants with layout ID, byte length, and
      `LAYOUT_AND_SIZE_ONLY` verification; do not hash caller bytes.
- [ ] Implement scratch layout compilation with one storage and packed offsets.
- [ ] Implement allocator layout compilation:
  - [ ] resolve every span to exactly one allocation record;
  - [ ] preserve caller order;
  - [ ] deduplicate regions in first-appearance order;
  - [ ] assign request-local storage IDs;
  - [ ] compute checked logical storage bases and offsets; and
  - [ ] emit one storage entry per touched region.
- [ ] Implement per-region candidate geometry validation.
- [ ] Commit unseen geometry atomically and all-or-nothing only during final
      admission after all local and wire-budget validation passes.
- [ ] Establish lock order: Session state before geometry; hold neither across
      RPC.
- [ ] Implement non-zero monotonic RPC generation and derive slot index from
      region-local offset and frozen slot size.
- [ ] Attach one request generation to all direct offsets and validate exact
      echoed slot tokens.
- [ ] Build completed request protobufs before applying `ByteSize()`.
- [ ] Estimate ordinary response size from artifact IDs, statuses, direct slot
      tokens, protobuf overhead, and fixed headroom.
- [ ] Reject oversized transfer batches before final admission.
- [ ] Partition exists input into maximal contiguous sub-batches that fit both
      client send and receive limits.
- [ ] Give every exists partition one SDK correlation/operation ID; retries of
      that partition reuse it, while the next partition gets another ID.
- [ ] Implement artifact-based outcome correlation and restore caller order.
- [ ] Allow only operation-specific `OK` and `MISS` statuses from design
      `0122`; reject missing, duplicate, unknown, or malformed outcomes.
- [ ] Map `REGION_LOST` only from structured machine-readable region evidence;
      never parse free-form error text.

### Unit tests

- [ ] `TestArtifactLowering`
  - [ ] same keyspace/engine key produces the canonical existing byte-artifact
        identity;
  - [ ] multiple keyspaces coexist in one batch;
  - [ ] duplicates are rejected;
  - [ ] put invariant is layout-and-size-only; and
  - [ ] no public result exposes artifact IDs or selections.
- [ ] `TestWireBudget`
  - [ ] uses the exact `DaemonCtl` limit snapshot;
  - [ ] measures completed request protobufs;
  - [ ] rejects get/put before RPC when either direction exceeds its limit;
  - [ ] partitions exists at exact boundary conditions and preserves order;
  - [ ] uses one operation ID per partition and reuses it for retries; and
  - [ ] reports total exists RPC elapsed time across partitions and attempts.
- [ ] `TestOutcomeValidation`
  - [ ] accepts operation-specific OK/MISS combinations;
  - [ ] rejects missing, duplicate, unknown, reordered-with-bad-correlation,
        and non-allowlisted outcomes;
  - [ ] validates echoed direct slot tokens;
  - [ ] maps generic `FAILED_PRECONDITION` to `DAEMON_STATUS`;
  - [ ] maps only typed region evidence to `REGION_LOST`; and
  - [ ] proves error-message text does not affect classification.
- [ ] Geometry tests
  - [ ] compatible first use freezes one slot size;
  - [ ] locally invalid or wire-oversized input freezes nothing;
  - [ ] a multi-region install is all-or-nothing; and
  - [ ] concurrent conflicting first use admits at most one geometry and sends
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

- [ ] Lazily create one fixed-capacity daemon-managed `HOST_SHARED/SCRATCH`
      region for get and one for put.
- [ ] Move each arena to `ProcessPinned` before its first admitted transfer.
- [ ] Retain both mappings until process teardown; never resize, replace,
      expire, release, or unregister them after pinning.
- [ ] Use independent get and put locks so one get and one put may run
      concurrently while same-direction calls serialize.
- [ ] Implement scratch put:
  - [ ] validate and snapshot all source spans;
  - [ ] reject total packed bytes above capacity;
  - [ ] pack in caller order;
  - [ ] issue one region-backed put with `retries=0`; and
  - [ ] validate all outcomes before returning the success mask.
- [ ] Implement scratch get:
  - [ ] build packed target offsets;
  - [ ] issue one region-backed get with `retries=0`;
  - [ ] validate the complete response before copying;
  - [ ] copy only successful items to caller spans; and
  - [ ] leave every false target unconsumable and unchanged where practical.
- [ ] On fatal scratch get, copy no scratch bytes to caller targets and never
      reuse the failed arena.
- [ ] Record pack, copy, RPC, bytes, and direction metrics without artifact IDs
      as unbounded labels.

### Unit tests

- [ ] `TestScratchTransfer`
  - [ ] regions are lazy, direction-specific, fixed-capacity, and created once;
  - [ ] put packs exact bytes and preserves caller order;
  - [ ] get copies only OK items after whole-response validation;
  - [ ] MISS/False target is not consumed;
  - [ ] malformed/fatal get copies no target bytes and latches Session;
  - [ ] overflow rejects before arena mutation and RPC;
  - [ ] get/get and put/put serialize, while get/put can overlap;
  - [ ] get and put each call the raw RPC with `retries=0`;
  - [ ] empty calls allocate no arena; and
  - [ ] termination/failure sends no arena cleanup RPC.

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

- [ ] Accept direct spans only when wholly contained in allocations owned by the
      same Session.
- [ ] Compile every direct batch into one multi-storage request, regardless of
      how many owned regions it touches.
- [ ] Submit caller-region addresses directly; do not allocate or copy through
      a Session scratch arena.
- [ ] Keep all referenced spans, tensor roots, mmap roots, and explicit owners
      alive until return or raise.
- [ ] Enforce pairwise non-overlap within one batch.
- [ ] Preserve caller-owned cross-call exclusivity without adding an in-flight
      interval allocator to the SDK.
- [ ] Use no client deadline when `transfer_timeout_s=None`.
- [ ] Issue direct get and put with exactly one SDK attempt (`retries=0`).
- [ ] Return one transfer operation ID for the one direct RPC.
- [ ] Treat only `True` direct-get targets as consumable; after fatal direct get,
      mark every submitted target untrusted through the raised Session error.
- [ ] Allow concurrent direct get/put calls without a global data-plane lock;
      retain only short state, region, geometry, and generation locks.
- [ ] After successful put returns, release the source borrow and ensure later
      caller mutation cannot change the stored artifact.
- [ ] Never call explicit stable-backing activation; rely on daemon layout
      validation and the frozen region geometry.

### Unit tests

- [ ] `TestAllocatorDirectTransfer`
  - [ ] one-region get/put compiles exact offsets with no scratch allocation;
  - [ ] two-region and three-region batches emit one storage entry per region
        and one RPC total;
  - [ ] first-appearance storage ordering and logical bases are deterministic;
  - [ ] a batch can mix keyspaces while preserving input-order results;
  - [ ] foreign, crossing, overlapping, and out-of-range spans fail before RPC;
  - [ ] `transfer_timeout_s=None` is passed as no deadline;
  - [ ] configured finite timeout is passed through but retry remains zero;
  - [ ] get and put never call scratch-copy helpers;
  - [ ] success/MISS consumption rules match the public contract; and
  - [ ] failure and termination never release allocator mappings.
- [ ] `TestConcurrentAdmission`
  - [ ] direct get and put may overlap in time;
  - [ ] generation values remain unique and non-zero under concurrency;
  - [ ] Session/geometry locks are released before blocking RPC;
  - [ ] a failure latched by one call prevents later admission;
  - [ ] an already completed concurrent success is discarded if health failed
        before exposure; and
  - [ ] counter wrap fails closed.

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

- [ ] Complete stable first-failure classification:
  - [ ] transport/deadline/cancellation/availability -> `TRANSPORT`;
  - [ ] non-allowlisted/unknown item status -> `DAEMON_STATUS`;
  - [ ] ambiguous region setup -> `REGION_SETUP`;
  - [ ] typed already-pinned region loss -> `REGION_LOST`;
  - [ ] malformed response/token mismatch -> `MALFORMED_RESPONSE`; and
  - [ ] otherwise unexpected admitted SDK failure -> `INTERNAL`.
- [ ] Atomically retain only the first failure with UTC timestamp, operation
      kind, and failing operation ID.
- [ ] Ensure later allocation, exists, get, and put calls raise that same
      failure without issuing RPC.
- [ ] For partitioned exists, stop before the next partition after failure or
      termination and expose no partial masks.
- [ ] Ensure first fatal logging contains traceback once and derivative logs are
      rate-limited.
- [ ] Finish termination behavior:
  - [ ] close new admission;
  - [ ] retain control resources until admitted work exits;
  - [ ] stop only Session-owned helper resources;
  - [ ] do not close/release the process-shared `DaemonCtl`; and
  - [ ] never unmap or release process-pinned regions.
- [ ] Add generic observability required by `0122` without high-cardinality
      metric labels.
- [ ] Add end-to-end Python tests that launch the supplied prebuilt daemon using
      `tests/python/utils/daemon.py`.
- [ ] Update `tensorcast/api/store/README.md` with scratch and allocator
      examples and explicit ownership/failure warnings.
- [ ] Finalize `tensorcast.api.store` exports and update API documentation.
- [ ] Synchronize design `0122` if implementation closes any naming-only gap;
      do not change its accepted ownership or failure semantics silently.

### Unit tests

- [ ] `TestFailureContract`
  - [ ] each stable failure-code mapping is covered;
  - [ ] simultaneous fatal calls retain exactly one first failure;
  - [ ] every later method raises the same retained failure object/data;
  - [ ] no RPC occurs after the latch;
  - [ ] a fatal exists partition records that partition's operation ID;
  - [ ] generic status text cannot manufacture `REGION_LOST`; and
  - [ ] successful concurrent results are suppressed after a latch.
- [ ] `TestTerminationContract`
  - [ ] healthy and failed termination are idempotent;
  - [ ] failed error continues to win after termination;
  - [ ] admitted synchronous work may reach its terminal path;
  - [ ] new work and new exists partitions are rejected;
  - [ ] helper cleanup is deferred until in-flight reaches zero; and
  - [ ] no daemon client, FD attachment, region, mmap, or tensor root is
        released by termination.
- [ ] Public documentation examples execute as unit tests or doctest-equivalent
      snippets with a fake client.

### Prebuilt-daemon acceptance tests

- [ ] Scratch put -> exists -> get round trip with multiple artifacts.
- [ ] Scratch partial-hit get copies only successful targets.
- [ ] Allocator tensor put -> exists -> direct get round trip.
- [ ] One allocator RPC spans at least two independent host-shared regions.
- [ ] Direct put return is quiescent enough that mutating/reusing the caller
      source afterward does not change later retrieved bytes.
- [ ] Daemon rejection causes sticky Session failure and later SDK calls issue
      no RPC.
- [ ] Session termination sends no region release/unregister request.
- [ ] A subprocess owner exit allows the daemon's existing PID cleanup path to
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
  //daemon:grpc_service_impl_cpu_memfd_e2e_test \
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
- [ ] Keep the new Session opt-in; do not alter ordinary `Artifact` or `Store`
      behavior.
- [ ] Preserve old `DaemonCtl` retry defaults and make zero retry explicit only
      for the new direct/scratch region transfer path.
- [ ] Publish public docs only when both scratch and allocator prebuilt-daemon
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

- [ ] Risk: a Session data-path call accidentally inherits the existing region
      RPC retry default.
  - mitigation: spy-based tests assert the exact `retries=0` value for scratch
    and direct get/put.
- [ ] Risk: unit-test cleanup masks illegal production region cleanup.
  - mitigation: fake registry reset is test-only; process-pinned lifecycle tests
    assert no release/unregister call, and real reclamation uses a subprocess
    owner exit.
- [ ] Risk: Python owner objects are collected while an RPC still uses an
      address.
  - mitigation: immutable call snapshots and strong-reference/weak-reference
    tests cover every owner path.
- [ ] Risk: multi-region offset arithmetic aliases two storages or overflows.
  - mitigation: checked arithmetic, deterministic first-appearance ordering,
    and two/three-region layout tests.
- [ ] Risk: an invalid first batch poisons region slot geometry.
  - mitigation: all-or-nothing commit after validation with concurrent
    conflicting-first-use tests.
- [ ] Risk: response size exceeds the client receive limit despite conservative
      ordinary-outcome estimation.
  - mitigation: keep fixed headroom; classify exceptional transport overflow as
    fatal; require compatible daemon/client limits operationally.
- [ ] Risk: direct no-deadline RPC hangs a test or shutdown path.
  - mitigation: unit tests use bounded fakes; end-to-end tests use an outer
    subprocess/test timeout without changing the Session's no-deadline contract.
- [ ] Risk: sticky failure hides an artifact-scoped daemon rejection.
  - mitigation: retain the intentionally small success allowlist and test every
    non-allowlisted status as Session-fatal.
- [ ] Risk: fake CUDA proves CPU/shared-memory correctness but not accelerator
      host registration behavior.
  - mitigation: accelerator registration remains explicitly caller-owned and
    outside `0122` acceptance.

# Owner Checklist

- [ ] Public caller supplies only keyspace, engine key, byte length, and owned
      host spans.
- [ ] Public types expose no canonical artifact ID, protobuf, daemon client,
      region handle, layout enum, storage ID, or slot token.
- [ ] One process has one attached endpoint and one sticky Session failure
      domain.
- [ ] Attach is serialized with one ordinary registry mutex.
- [ ] Effective client send/receive limits are one immutable per-client
      snapshot reused by channel refresh and Session sizing.
- [ ] Scratch and direct get/put use zero SDK retries.
- [ ] Exists partitions transparently, uses one operation ID per partition, and
      exposes one ordered all-or-error result.
- [ ] Multiple allocator regions coexist and one direct RPC can reference all
      of them.
- [ ] Caller memory is strongly retained for every synchronous borrow.
- [ ] Only successful get targets are consumable.
- [ ] Fatal RPC, malformed response, and typed region loss latch one stable first
      failure and admit no later RPC.
- [ ] Empty calls check Session state but issue no RPC.
- [ ] Geometry commits only after complete local and wire-budget validation.
- [ ] Failure and termination never release process-pinned regions or close the
      process-shared daemon client.
- [ ] Unit, SDK regression, prebuilt-daemon, daemon-contract, and Ruff gates are
      green.
- [ ] Design, implementation, tests, and public SDK documentation describe the
      same contract.
