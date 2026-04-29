---
slug: control-plane-coordinated-weight-broadcast
title: Control-Plane Coordinated Weight Broadcast Implementation Plan
links:
  design: ../designs/0116-control-plane-coordinated-weight-broadcast.md
---

# Control-Plane Coordinated Weight Broadcast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread model-weight prefetch transport request ids and group hints from the Python SDK through Store Daemon into existing Global Store group dispatch.

**Architecture:** Add typed SDK transport scheduling context, additive daemon proto fields, DaemonCtl forwarding, and C++ daemon mapping into existing `MaterializeHints`. The Global Store scheduler and P2P data plane remain unchanged; Phase 1 only makes daemon-owned `Artifact.prefetch()` visible to the existing group dispatcher.

**Tech Stack:** Python SDK, pydantic options, protobuf/buf generation, C++ Store Daemon controllers, StoreEngine `MaterializeHints`, pytest, Bazel/Catch2.

---

# Current State & Grounding

- Branch: `runze/broadcast-weight`.
- Design: `docs/designs/0116-control-plane-coordinated-weight-broadcast.md`.
- Existing Global Store fields: `proto/tensorcast/global_store/v1/global_store.proto` already defines `RequestReplicaTransportRequest.request_id` and `TransportSchedulingGroup`.
- Existing core hints: `core/store/materialization/contracts/loading_spec.h` already defines `TransportSchedulingGroupHint` and `MaterializeHints.transport_request_id`.
- Existing C++ transport client: `core/store/components/global_store_client.cc` already copies hints into `RequestReplicaTransportRequest`.
- Current gap: `proto/tensorcast/daemon/v2/store_daemon.proto::MaterializeReplicaRequest` has no transport hint fields, so `Artifact.prefetch()` cannot send group hints to Store Daemon.
- Current prefetch entrypoint: `tensorcast/api/store/artifact.py::Artifact.prefetch()`.
- Current SDK materialization path: `tensorcast/api/_materialize.py::materialize_artifact_v2()` to `tensorcast/daemon_ctl.py::DaemonCtl.materialize_by_artifact_id_v2()`.
- Current daemon materialization path: `daemon/service/controllers/replica_materialization_service.cc::materialize_replica()`.
- Existing dirty worktree before this plan includes generated proto files and `pyproject.toml`; implementation must not revert or stage unrelated pre-existing changes.

# Files

- Modify: `proto/tensorcast/daemon/v2/store_daemon.proto`
- Generate locally: `proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2.py` (ignored by git)
- Generate locally: `proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2.pyi` (ignored by git)
- Generate locally: `proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2_grpc.py` (ignored by git)
- Modify: `tensorcast/api/context.py`
- Modify: `tensorcast/api/__init__.py`
- Modify: `tensorcast/__init__.py`
- Modify: `tensorcast/api/store/artifact.py`
- Modify: `tensorcast/api/_materialize.py`
- Modify: `tensorcast/daemon_ctl.py`
- Modify: `daemon/service/controllers/materialization_policy_utils.h`
- Modify: `daemon/service/controllers/materialization_policy_utils.cc`
- Modify: `daemon/service/controllers/replica_materialization_service.cc`
- Modify: `daemon/service/materialization_policy_utils_test.cc`
- Test: `tests/python/api/test_prefetch_operation.py`
- Test: add `tests/python/api/test_daemon_ctl_transport_hints.py`
- Modify: `docs/designs/0116-control-plane-coordinated-weight-broadcast.md`

# Phases & Milestones

- [ ] Phase 1: Add public SDK transport group context and deterministic prefetch hint resolution.
- [ ] Phase 2: Add daemon proto fields and regenerate Python stubs.
- [ ] Phase 3: Forward transport hints through DaemonCtl and Store Daemon.
- [ ] Phase 4: Verify SDK, daemon, and Global Store regressions.

### Task 1: SDK Transport Group Context

**Files:**
- Modify: `tensorcast/api/context.py`
- Modify: `tensorcast/api/__init__.py`
- Modify: `tensorcast/__init__.py`
- Test: `tests/python/api/test_prefetch_operation.py`

- [ ] **Step 1: Write failing validation and export tests**

Add these imports near the top of `tests/python/api/test_prefetch_operation.py`:

```python
from tensorcast.api.context import TransportSchedulingGroup
```

Append these tests:

```python
def test_transport_scheduling_group_rejects_invalid_values() -> None:
    invalid_cases = [
        {"group_kind": "", "group_id": "model:v1", "total_parts": 2, "part_id": "d0"},
        {"group_kind": "weight_broadcast", "group_id": "", "total_parts": 2, "part_id": "d0"},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 0, "part_id": "d0"},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 2, "part_id": ""},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 2, "part_id": "d0", "priority": -1},
        {"group_kind": "weight_broadcast", "group_id": "model:v1", "total_parts": 2, "part_id": "d0", "epoch": -1},
    ]

    for kwargs in invalid_cases:
        try:
            TransportSchedulingGroup(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected invalid transport group: {kwargs}")


def test_context_accepts_typed_transport_group() -> None:
    group = tc.TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        epoch=42,
        total_parts=8,
        part_id="daemon-3",
    )

    ctx = tc.context(request_id="req-1", transport_group=group)

    assert ctx.transport_group == group
    assert tc.TransportSchedulingGroup is TransportSchedulingGroup
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py::test_transport_scheduling_group_rejects_invalid_values tests/python/api/test_prefetch_operation.py::test_context_accepts_typed_transport_group -v
```

Expected: FAIL because `TransportSchedulingGroup` is not defined/exported.

- [ ] **Step 3: Implement `TransportSchedulingGroup`**

In `tensorcast/api/context.py`, add this dataclass after `CollectiveLoadGroup`:

```python
@dataclass(frozen=True, slots=True)
class TransportSchedulingGroup:
    """Control-plane transport scheduling group for coordinated P2P source selection."""

    group_id: str
    group_kind: str
    total_parts: int
    part_id: str
    priority: int = 0
    epoch: int = 0
    request_id: str | None = None

    def __post_init__(self) -> None:
        group_kind = str(self.group_kind).strip()
        group_id = str(self.group_id).strip()
        part_id = str(self.part_id).strip()
        total_parts = int(self.total_parts)
        priority = int(self.priority)
        epoch = int(self.epoch)
        request_id = None if self.request_id is None else str(self.request_id).strip()
        if not group_kind:
            raise ValueError("TransportSchedulingGroup.group_kind must be non-empty")
        if not group_id:
            raise ValueError("TransportSchedulingGroup.group_id must be non-empty")
        if total_parts <= 0:
            raise ValueError("TransportSchedulingGroup.total_parts must be positive")
        if not part_id:
            raise ValueError("TransportSchedulingGroup.part_id must be non-empty")
        if priority < 0:
            raise ValueError("TransportSchedulingGroup.priority must be non-negative")
        if epoch < 0:
            raise ValueError("TransportSchedulingGroup.epoch must be non-negative")
        object.__setattr__(self, "group_kind", group_kind)
        object.__setattr__(self, "group_id", group_id)
        object.__setattr__(self, "total_parts", total_parts)
        object.__setattr__(self, "part_id", part_id)
        object.__setattr__(self, "priority", priority)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "request_id", request_id or None)
```

Add `transport_group` to `CallContext`:

```python
transport_group: TransportSchedulingGroup | None = None
```

Add `transport_group` to `context(...)` and pass it into `CallContext(...)`:

```python
transport_group: TransportSchedulingGroup | None = None,
```

```python
transport_group=transport_group,
```

Add `"TransportSchedulingGroup"` to `__all__`.

- [ ] **Step 4: Export the new type**

In `tensorcast/api/__init__.py`, import and export `TransportSchedulingGroup` from `tensorcast.api.context`.

In `tensorcast/__init__.py`, add:

```python
"TransportSchedulingGroup": ("tensorcast.api", "TransportSchedulingGroup"),
```

Add `TransportSchedulingGroup` to the `TYPE_CHECKING` import list and `__all__`.

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py::test_transport_scheduling_group_rejects_invalid_values tests/python/api/test_prefetch_operation.py::test_context_accepts_typed_transport_group -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tensorcast/api/context.py tensorcast/api/__init__.py tensorcast/__init__.py tests/python/api/test_prefetch_operation.py
git commit -m "feat(sdk): add transport scheduling group context"
```

### Task 2: SDK Prefetch Hint Resolution

**Files:**
- Modify: `tensorcast/api/store/artifact.py`
- Test: `tests/python/api/test_prefetch_operation.py`

- [ ] **Step 1: Write failing prefetch forwarding tests**

Append these tests to `tests/python/api/test_prefetch_operation.py`:

```python
def test_prefetch_forwards_typed_transport_group_hint() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")
    group = tc.TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        epoch=42,
        total_parts=16,
        part_id="daemon-1",
        priority=7,
        request_id="explicit-transport-req",
    )

    artifact.prefetch(device="cuda:0", ctx=tc.context(transport_group=group))

    call = store._materialization.calls[0]
    assert call["transport_request_id"] == "explicit-transport-req"
    forwarded = call["transport_scheduling_group"]
    assert forwarded.group_kind == "weight_broadcast"
    assert forwarded.group_id == "model-a:v42"
    assert forwarded.epoch == 42
    assert forwarded.total_parts == 16
    assert forwarded.part_id == "daemon-1"
    assert forwarded.priority == 7


def test_prefetch_derives_stable_transport_request_id_for_group() -> None:
    group = tc.TransportSchedulingGroup(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        epoch=42,
        total_parts=16,
        part_id="daemon-1",
    )
    ctx = tc.context(transport_group=group)
    first_store = _Store()
    second_store = _Store()
    first = Artifact(store_ref=weakref.ref(first_store), artifact_id="aid")
    second = Artifact(store_ref=weakref.ref(second_store), artifact_id="aid")

    first.prefetch(device="cuda:0", ctx=ctx)
    second.prefetch(device="cuda:0", ctx=ctx)

    first_request_id = first_store._materialization.calls[0]["transport_request_id"]
    second_request_id = second_store._materialization.calls[0]["transport_request_id"]
    assert first_request_id
    assert first_request_id == second_request_id
    assert first_request_id.startswith("prefetch:")


def test_prefetch_without_group_sends_no_transport_hint() -> None:
    store = _Store()
    artifact = Artifact(store_ref=weakref.ref(store), artifact_id="aid")

    artifact.prefetch(device="cuda:0")

    call = store._materialization.calls[0]
    assert call["transport_request_id"] is None
    assert call["transport_scheduling_group"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py::test_prefetch_forwards_typed_transport_group_hint tests/python/api/test_prefetch_operation.py::test_prefetch_derives_stable_transport_request_id_for_group tests/python/api/test_prefetch_operation.py::test_prefetch_without_group_sends_no_transport_hint -v
```

Expected: FAIL because `Artifact.prefetch()` does not pass `transport_request_id` or `transport_scheduling_group`.

- [ ] **Step 3: Add transport hint helper**

In `tensorcast/api/store/artifact.py`, add `TransportSchedulingGroup` to the context import:

```python
from tensorcast.api.context import CallContext, TransportSchedulingGroup
```

Add this helper near `_build_transport_operation_id(...)`:

```python
def _transport_group_from_ctx_tags(ctx: CallContext | None) -> TransportSchedulingGroup | None:
    if ctx is None or not ctx.tags:
        return None
    group_kind = _read_context_tag_str(ctx.tags, _TRANSPORT_GROUP_KIND_TAG)
    group_id = _read_context_tag_str(ctx.tags, _TRANSPORT_GROUP_ID_TAG)
    part_id = _read_context_tag_str(ctx.tags, _TRANSPORT_GROUP_PART_ID_TAG)
    total_parts = _read_context_tag_int(ctx.tags, _TRANSPORT_GROUP_TOTAL_PARTS_TAG, default=0)
    if not (group_kind and group_id and part_id and total_parts > 0):
        return None
    return TransportSchedulingGroup(
        group_kind=group_kind,
        group_id=group_id,
        total_parts=total_parts,
        part_id=part_id,
        priority=_read_context_tag_int(ctx.tags, _TRANSPORT_GROUP_PRIORITY_TAG, default=0),
        epoch=_read_context_tag_int(ctx.tags, _TRANSPORT_GROUP_EPOCH_TAG, default=0),
        request_id=_read_context_tag_str(ctx.tags, _TRANSPORT_REQUEST_ID_TAG) or None,
    )


def _resolve_prefetch_transport_hints(
    *,
    ctx: CallContext | None,
    daemon_id: str,
    artifact_id: str,
    selection_hash: str,
    logical_layout_hash: str,
    device_id: int,
    device_uuid: str,
) -> tuple[str | None, TransportSchedulingGroup | None]:
    group = (ctx.transport_group if ctx is not None else None) or _transport_group_from_ctx_tags(ctx)
    if group is None:
        return None, None
    if group.request_id:
        return group.request_id, group
    digest = hashlib.sha256()
    digest.update(b"tensorcast.prefetch.transport.v1")
    for value in (
        daemon_id,
        artifact_id,
        logical_layout_hash,
        selection_hash,
        str(device_id),
        device_uuid,
        group.group_kind,
        group.group_id,
        str(group.epoch),
        str(group.total_parts),
        group.part_id,
    ):
        digest.update(b"|")
        digest.update(str(value).encode("utf-8"))
    return f"prefetch:{digest.hexdigest()}", group
```

- [ ] **Step 4: Thread hints from `Artifact.prefetch()`**

In `Artifact.prefetch()`, after `device_uuid = device_uuid_for(device_id)` is available for deterministic operation id generation, keep a local `device_uuid_value` for both deterministic replica id and transport id:

```python
device_uuid_value = device_uuid_for(device_id)
```

Use `device_uuid_value` in the existing action fingerprint instead of recomputing `device_uuid`.

Before calling `pipeline.materialize_subset(...)`, add:

```python
transport_request_id, transport_scheduling_group = _resolve_prefetch_transport_hints(
    ctx=ctx,
    daemon_id=daemon_id,
    artifact_id=artifact_id,
    selection_hash=selection_hash,
    logical_layout_hash=bytes(selection.logical_layout_hash).hex(),
    device_id=device_id,
    device_uuid=device_uuid_value,
)
```

Pass these kwargs into `pipeline.materialize_subset(...)`:

```python
transport_request_id=transport_request_id,
transport_scheduling_group=transport_scheduling_group,
```

- [ ] **Step 5: Run tests to verify they pass**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tensorcast/api/store/artifact.py tests/python/api/test_prefetch_operation.py
git commit -m "feat(sdk): derive prefetch transport group hints"
```

### Task 3: Daemon Proto and Python Generation

**Files:**
- Modify: `proto/tensorcast/daemon/v2/store_daemon.proto`
- Generate locally: `proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2.py` (ignored by git)
- Generate locally: `proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2.pyi` (ignored by git)
- Generate locally: `proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2_grpc.py` (ignored by git)

- [ ] **Step 1: Add daemon transport group proto fields**

In `proto/tensorcast/daemon/v2/store_daemon.proto`, add this message before `MaterializeReplicaRequest`:

```proto
message TransportSchedulingGroupHint {
  string group_id = 1;
  string group_kind = 2;
  uint32 total_parts = 3;
  string part_id = 4;
  uint32 priority = 5;
  uint64 epoch = 6;
}
```

Add these fields to `MaterializeReplicaRequest` after `serving_artifact_policy = 20;`:

```proto
  string transport_request_id = 21;
  TransportSchedulingGroupHint transport_scheduling_group = 22;
```

- [ ] **Step 2: Format and regenerate protos**

Run:

```bash
bazel run @rules_buf_toolchains//:buf -- format ./proto -w
bash tools/build_proto_python.sh
```

Expected: local generated `store_daemon_pb2.py` and `store_daemon_pb2.pyi` include `TransportSchedulingGroupHint`, `transport_request_id`, and `transport_scheduling_group`. These generated files are ignored by `.gitignore`; they are validation artifacts, not files to force-add.

- [ ] **Step 3: Inspect generated changes carefully**

Run:

```bash
git status --short proto/tensorcast/daemon/v2/store_daemon.proto proto/gen/python/tensorcast/daemon/v2
git diff -- proto/tensorcast/daemon/v2/store_daemon.proto proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2.py proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2.pyi proto/gen/python/tensorcast/daemon/v2/store_daemon_pb2_grpc.py
```

Expected: tracked diffs are limited to `proto/tensorcast/daemon/v2/store_daemon.proto`. Generated daemon files may appear under ignored or untracked paths for local validation. Do not stage unrelated pre-existing generated proto directories.

- [ ] **Step 4: Commit**

Run:

```bash
git add proto/tensorcast/daemon/v2/store_daemon.proto
git commit -m "feat(proto): add materialize replica transport hints"
```

### Task 4: DaemonCtl and Materialization Pipeline Forwarding

**Files:**
- Modify: `tensorcast/api/_materialize.py`
- Modify: `tensorcast/daemon_ctl.py`
- Test: add `tests/python/api/test_daemon_ctl_transport_hints.py`
- Test: `tests/python/api/test_prefetch_operation.py`

- [ ] **Step 1: Write failing DaemonCtl request construction test**

Create `tests/python/api/test_daemon_ctl_transport_hints.py`:

```python
#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2


class _FakeUnary:
    _method = b"/tensorcast.daemon.v2.StoreDaemonService/MaterializeReplica"

    def __init__(self) -> None:
        self.requests: list[store_daemon_pb2.MaterializeReplicaRequest] = []

    def __call__(self, request, timeout=None):  # noqa: ANN001, ANN204
        del timeout
        self.requests.append(request)
        response = store_daemon_pb2.MaterializeReplicaResponse()
        response.status = (
            store_daemon_pb2.MaterializeReplicaStatus.MATERIALIZE_REPLICA_STATUS_ALLOCATED
        )
        response.ticket.replica_uuid = request.replica_uuid
        return response


class _FakeStub:
    def __init__(self) -> None:
        self.MaterializeReplica = _FakeUnary()


def test_daemon_ctl_forwards_materialize_transport_hints(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setattr(DaemonCtl, "_create_channel", lambda self, addr: None)
    ctl = DaemonCtl("fake-daemon")
    fake_stub = _FakeStub()
    ctl.stub_v2 = fake_stub
    ctl.stub = fake_stub
    monkeypatch.setattr(ctl, "_get_effective_pid", lambda: 123)
    monkeypatch.setattr(ctl, "_unary_call", lambda method, request, **kwargs: method(request, timeout=kwargs.get("timeout")))
    selection = common_pb2.ArtifactSelection(artifact_id="aid")
    group = store_daemon_pb2.TransportSchedulingGroupHint(
        group_kind="weight_broadcast",
        group_id="model-a:v42",
        total_parts=16,
        part_id="daemon-1",
        priority=7,
        epoch=42,
    )

    ctl.materialize_by_artifact_id_v2(
        selection=selection,
        replica_uuid="replica-1",
        device_uuid="device-uuid",
        wait_for_completion=False,
        return_response=True,
        transport_request_id="transport-req-1",
        transport_scheduling_group=group,
    )

    request = fake_stub.MaterializeReplica.requests[0]
    assert request.transport_request_id == "transport-req-1"
    assert request.transport_scheduling_group.group_kind == "weight_broadcast"
    assert request.transport_scheduling_group.group_id == "model-a:v42"
    assert request.transport_scheduling_group.total_parts == 16
    assert request.transport_scheduling_group.part_id == "daemon-1"
    assert request.transport_scheduling_group.priority == 7
    assert request.transport_scheduling_group.epoch == 42
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
pytest tests/python/api/test_daemon_ctl_transport_hints.py -v
```

Expected: FAIL because `DaemonCtl.materialize_by_artifact_id_v2()` does not accept transport hint kwargs.

- [ ] **Step 3: Add DaemonCtl kwargs and request copying**

In all three overloads and the implementation of `DaemonCtl.materialize_by_artifact_id_v2(...)`, add:

```python
transport_request_id: str | None = None,
transport_scheduling_group: store_daemon_pb2.TransportSchedulingGroupHint | None = None,
```

After constructing `MaterializeReplicaRequest`, add:

```python
if transport_request_id:
    request.transport_request_id = str(transport_request_id)
if transport_scheduling_group is not None:
    request.transport_scheduling_group.CopyFrom(transport_scheduling_group)
```

- [ ] **Step 4: Forward hints from `materialize_artifact_v2()`**

In `tensorcast/api/_materialize.py`, import `TransportSchedulingGroup`:

```python
from tensorcast.api.context import CallContext, CollectiveLoadGroup, TransportSchedulingGroup
```

Add optional parameters to `materialize_artifact_v2(...)`:

```python
transport_request_id: str | None = None,
transport_scheduling_group: TransportSchedulingGroup | None = None,
```

Before the `client.materialize_by_artifact_id_v2(...)` call, convert the SDK group:

```python
transport_group_proto = None
if transport_scheduling_group is not None:
    transport_group_proto = store_daemon_pb2.TransportSchedulingGroupHint(
        group_id=transport_scheduling_group.group_id,
        group_kind=transport_scheduling_group.group_kind,
        total_parts=int(transport_scheduling_group.total_parts),
        part_id=transport_scheduling_group.part_id,
        priority=int(transport_scheduling_group.priority),
        epoch=int(transport_scheduling_group.epoch),
    )
```

Pass:

```python
transport_request_id=transport_request_id,
transport_scheduling_group=transport_group_proto,
```

- [ ] **Step 5: Run SDK forwarding tests**

Run:

```bash
pytest tests/python/api/test_daemon_ctl_transport_hints.py tests/python/api/test_prefetch_operation.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add tensorcast/api/_materialize.py tensorcast/daemon_ctl.py tests/python/api/test_daemon_ctl_transport_hints.py tests/python/api/test_prefetch_operation.py
git commit -m "feat(sdk): forward prefetch transport hints to daemon"
```

### Task 5: C++ Daemon Hint Mapping

**Files:**
- Modify: `daemon/service/controllers/materialization_policy_utils.h`
- Modify: `daemon/service/controllers/materialization_policy_utils.cc`
- Modify: `daemon/service/controllers/replica_materialization_service.cc`
- Modify: `daemon/service/materialization_policy_utils_test.cc`

- [ ] **Step 1: Write failing C++ mapping tests**

In `daemon/service/materialization_policy_utils_test.cc`, add this using declaration:

```cpp
using tensorcast::daemon::materialization_policy::resolve_transport_scheduling_group_hint;
```

Append these test cases:

```cpp
TEST_CASE(
    "MaterializeReplica transport scheduling group maps to loading hint",
    "[daemon][materialization][policy]") {
  v2::TransportSchedulingGroupHint proto;
  proto.set_group_kind("weight_broadcast");
  proto.set_group_id("model-a:v42");
  proto.set_total_parts(16);
  proto.set_part_id("daemon-1");
  proto.set_priority(7);
  proto.set_epoch(42);

  auto hint_or = resolve_transport_scheduling_group_hint(proto);

  REQUIRE(hint_or.has_value());
  CHECK(hint_or->group_kind == "weight_broadcast");
  CHECK(hint_or->group_id == "model-a:v42");
  CHECK(hint_or->total_parts == 16);
  CHECK(hint_or->part_id == "daemon-1");
  CHECK(hint_or->priority == 7);
  CHECK(hint_or->epoch == 42);
}

TEST_CASE(
    "MaterializeReplica transport scheduling group rejects incomplete values",
    "[daemon][materialization][policy]") {
  v2::TransportSchedulingGroupHint proto;
  proto.set_group_kind("weight_broadcast");
  proto.set_group_id("model-a:v42");
  proto.set_total_parts(0);
  proto.set_part_id("daemon-1");

  auto hint_or = resolve_transport_scheduling_group_hint(proto);

  CHECK(!hint_or.has_value());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: FAIL because `resolve_transport_scheduling_group_hint` does not exist.

- [ ] **Step 3: Add C++ mapping helper**

In `daemon/service/controllers/materialization_policy_utils.h`, declare:

```cpp
std::optional<store::loading::TransportSchedulingGroupHint> resolve_transport_scheduling_group_hint(
    const v2::TransportSchedulingGroupHint& group);
```

In `daemon/service/controllers/materialization_policy_utils.cc`, implement:

```cpp
std::optional<store::loading::TransportSchedulingGroupHint> resolve_transport_scheduling_group_hint(
    const v2::TransportSchedulingGroupHint& group) {
  if (group.group_kind().empty() || group.group_id().empty() || group.part_id().empty() || group.total_parts() == 0) {
    return std::nullopt;
  }
  store::loading::TransportSchedulingGroupHint out;
  out.group_kind = group.group_kind();
  out.group_id = group.group_id();
  out.total_parts = group.total_parts();
  out.part_id = group.part_id();
  out.priority = group.priority();
  out.epoch = group.epoch();
  return out;
}
```

- [ ] **Step 4: Apply request fields in `materialize_replica()`**

In `daemon/service/controllers/replica_materialization_service.cc`, add this using declaration near the existing materialization policy aliases:

```cpp
using materialization_policy::resolve_transport_scheduling_group_hint;
```

After `apply_request_context_to_hints(request_context, &hints);`, add:

```cpp
if (!req.transport_request_id().empty()) {
  hints.transport_request_id = req.transport_request_id();
}
if (req.has_transport_scheduling_group()) {
  hints.transport_scheduling_group =
      resolve_transport_scheduling_group_hint(req.transport_scheduling_group());
}
```

- [ ] **Step 5: Run C++ mapping test**

Run:

```bash
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add daemon/service/controllers/materialization_policy_utils.h daemon/service/controllers/materialization_policy_utils.cc daemon/service/controllers/replica_materialization_service.cc daemon/service/materialization_policy_utils_test.cc
git commit -m "feat(daemon): map materialize transport hints"
```

### Task 6: Plan Context and Node-Agent Prefetch Propagation

**Files:**
- Modify: `proto/tensorcast/plan/v1/plan.proto`
- Generate locally: `proto/gen/python/tensorcast/plan/v1/plan_pb2.py` (ignored by git)
- Generate locally: `proto/gen/python/tensorcast/plan/v1/plan_pb2.pyi` (ignored by git)
- Modify: `tensorcast/api/plan/plan.py`
- Modify: `tensorcast/node_agent/executor.py`
- Test: `tests/python/api/test_plan_spec.py`
- Test: `tests/python/node_agent/test_plan_execution.py`

- [ ] **Step 1: Write failing plan serialization test**

In `tests/python/api/test_plan_spec.py`, add a test:

```python
def test_plan_context_serializes_transport_group() -> None:
    ctx = CallContext(
        request_id="req-weight-broadcast",
        idempotency_key="idem-weight-broadcast",
        transport_group=TransportSchedulingGroup(
            group_kind="weight_broadcast",
            group_id="model-a:v42",
            epoch=42,
            total_parts=16,
            part_id="daemon-1",
            priority=3,
            request_id="transport-req-1",
        ),
    )

    spec = Plan(ctx).to_spec()

    assert spec.context.transport_group.group_kind == "weight_broadcast"
    assert spec.context.transport_group.group_id == "model-a:v42"
    assert spec.context.transport_group.epoch == 42
    assert spec.context.transport_group.total_parts == 16
    assert spec.context.transport_group.part_id == "daemon-1"
    assert spec.context.transport_group.priority == 3
    assert spec.context.transport_group.request_id == "transport-req-1"
```

Add imports if missing:

```python
from tensorcast.api.context import TransportSchedulingGroup
```

- [ ] **Step 2: Write failing node-agent propagation test**

In `tests/python/node_agent/test_plan_execution.py`, extend `_DaemonStub.__init__`:

```python
self.transport_request_ids: list[str | None] = []
self.transport_groups: list[object] = []
```

Extend `_DaemonStub.materialize_by_artifact_id_v2(...)`:

```python
self.transport_request_ids.append(kwargs.get("transport_request_id"))
self.transport_groups.append(kwargs.get("transport_scheduling_group"))
```

Add:

```python
def test_node_agent_prefetch_forwards_transport_group_from_plan_context() -> None:
    daemon = _DaemonStub()
    executor = NodeAgentExecutor(client=daemon)
    plan = plan_pb2.PlanSpec()
    plan.context.request_id = "req-weight-broadcast"
    plan.context.idempotency_key = "idem-weight-broadcast"
    plan.context.transport_group.group_kind = "weight_broadcast"
    plan.context.transport_group.group_id = "model-a:v42"
    plan.context.transport_group.epoch = 42
    plan.context.transport_group.total_parts = 16
    plan.context.transport_group.part_id = "daemon-1"
    plan.context.transport_group.priority = 3
    plan.context.transport_group.request_id = "transport-req-1"
    step = plan.steps.add()
    step.step_id = "prefetch-1"
    step.target.target_type = plan_pb2.TARGET_TYPE_WORKER
    step.target.target_id = "daemon-1"
    step.action.prefetch.selection.CopyFrom(_selection())
    step.action.prefetch.device_id = 0

    result = executor.execute(plan)

    assert result.steps[0].status.state == node_agent_pb2.OPERATION_STATE_SUCCESS
    assert daemon.transport_request_ids == ["transport-req-1"]
    group = daemon.transport_groups[0]
    assert group.group_kind == "weight_broadcast"
    assert group.group_id == "model-a:v42"
    assert group.total_parts == 16
    assert group.part_id == "daemon-1"
    assert group.priority == 3
    assert group.epoch == 42
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
pytest tests/python/api/test_plan_spec.py::test_plan_context_serializes_transport_group tests/python/node_agent/test_plan_execution.py::test_node_agent_prefetch_forwards_transport_group_from_plan_context -v
```

Expected: FAIL because `plan.v1.CallContext` does not have `transport_group`.

- [ ] **Step 4: Add plan proto fields**

In `proto/tensorcast/plan/v1/plan.proto`, add:

```proto
message TransportSchedulingGroup {
  string group_id = 1;
  string group_kind = 2;
  uint32 total_parts = 3;
  string part_id = 4;
  uint32 priority = 5;
  uint64 epoch = 6;
  string request_id = 7;
}
```

Add to `message CallContext`:

```proto
  TransportSchedulingGroup transport_group = 6;
```

Run:

```bash
bazel run @rules_buf_toolchains//:buf -- format ./proto -w
bash tools/build_proto_python.sh
```

- [ ] **Step 5: Serialize plan context transport group**

In `tensorcast/api/plan/plan.py`, import `TransportSchedulingGroup` where `CallContext` is imported if needed. In `_call_context_proto()`, add:

```python
if self._ctx.transport_group is not None:
    group = self._ctx.transport_group
    proto.transport_group.group_id = group.group_id
    proto.transport_group.group_kind = group.group_kind
    proto.transport_group.total_parts = int(group.total_parts)
    proto.transport_group.part_id = group.part_id
    proto.transport_group.priority = int(group.priority)
    proto.transport_group.epoch = int(group.epoch)
    if group.request_id:
        proto.transport_group.request_id = group.request_id
```

In `_action_context(...)`, preserve the plan-level group:

```python
transport_group=self._ctx.transport_group,
```

- [ ] **Step 6: Node-agent converts plan group to daemon group**

In `tensorcast/node_agent/executor.py`, add helper:

```python
def _transport_group_from_plan_context(
    call_ctx: CallContext,
) -> store_daemon_pb2.TransportSchedulingGroupHint | None:
    group = call_ctx.transport_group
    if group is None:
        return None
    return store_daemon_pb2.TransportSchedulingGroupHint(
        group_id=group.group_id,
        group_kind=group.group_kind,
        total_parts=int(group.total_parts),
        part_id=group.part_id,
        priority=int(group.priority),
        epoch=int(group.epoch),
    )
```

In `tensorcast/node_agent/executor.py::_call_context_from_proto(...)`, include `transport_group` in the returned `CallContext`:

```python
transport_group = None
if plan.context.HasField("transport_group") and plan.context.transport_group.group_kind:
    proto_group = plan.context.transport_group
    transport_group = TransportSchedulingGroup(
        group_id=proto_group.group_id,
        group_kind=proto_group.group_kind,
        total_parts=int(proto_group.total_parts),
        part_id=proto_group.part_id,
        priority=int(proto_group.priority),
        epoch=int(proto_group.epoch),
        request_id=proto_group.request_id or None,
    )
```

Pass transport hints in `_materialize_selection(...)` by adding parameters:

```python
transport_request_id: str | None = None,
transport_scheduling_group: store_daemon_pb2.TransportSchedulingGroupHint | None = None,
```

Forward to `self._client.materialize_by_artifact_id_v2(...)`:

```python
transport_request_id=transport_request_id,
transport_scheduling_group=transport_scheduling_group,
```

In `_prefetch(...)`, call `_materialize_selection(...)` with:

```python
transport_request_id=call_ctx.transport_group.request_id if call_ctx.transport_group else None,
transport_scheduling_group=_transport_group_from_plan_context(call_ctx),
```

In `_action_context(...)` and `_instance_action_context(...)`, preserve the existing group when creating derived action contexts:

```python
transport_group=call_ctx.transport_group,
```

- [ ] **Step 7: Run plan and node-agent tests**

Run:

```bash
pytest tests/python/api/test_plan_spec.py::test_plan_context_serializes_transport_group tests/python/node_agent/test_plan_execution.py::test_node_agent_prefetch_forwards_transport_group_from_plan_context -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add proto/tensorcast/plan/v1/plan.proto tensorcast/api/plan/plan.py tensorcast/node_agent/executor.py tests/python/api/test_plan_spec.py tests/python/node_agent/test_plan_execution.py
git commit -m "feat(plan): propagate prefetch transport groups"
```

### Task 7: Documentation, Verification, and Cleanup

**Files:**
- Modify: `docs/designs/0116-control-plane-coordinated-weight-broadcast.md`
- Modify: `docs/plans/0116-control-plane-coordinated-weight-broadcast.md`
- Modify: `tensorcast/api/README.md`
- Modify: `tensorcast/api/store/README.md`

- [ ] **Step 1: Link design to plan**

In `docs/designs/0116-control-plane-coordinated-weight-broadcast.md`, add under `links:`:

```yaml
  plan: ../plans/0116-control-plane-coordinated-weight-broadcast.md
```

- [ ] **Step 2: Update API docs for the public typed context**

In `tensorcast/api/README.md`, add a short bullet under "Programmable Control-Plane Primitives":

```markdown
- `CallContext.transport_group=TransportSchedulingGroup(...)` carries explicit transport scheduling metadata for coordinated prefetch fanout. Use `group_kind="weight_broadcast"` and a stable `group_id` such as a model version to let Global Store group dispatch spread source selection.
```

In `tensorcast/api/store/README.md`, extend the prefetch section with:

```markdown
Grouped model-weight prefetch can pass `ctx=CallContext(transport_group=TransportSchedulingGroup(...))`; the daemon forwards the group to Global Store transport scheduling while keeping `replica_uuid` as a pure daemon replica/session id.
```

- [ ] **Step 3: Run focused Python tests**

Run:

```bash
pytest tests/python/api/test_prefetch_operation.py tests/python/api/test_daemon_ctl_transport_hints.py tests/python/api/test_plan_spec.py::test_plan_context_serializes_transport_group tests/python/node_agent/test_plan_execution.py::test_node_agent_prefetch_forwards_transport_group_from_plan_context -v
```

Expected: PASS.

- [ ] **Step 4: Run focused C++ test**

Run:

```bash
bazel test //daemon:materialization_policy_utils_test --test_output=errors --noshow_progress --noshow_loading_progress --ui_event_filters=warning,error
```

Expected: PASS.

- [ ] **Step 5: Run Global Store scheduler regression tests**

Run:

```bash
pytest tests/python/global_store/test_services.py::test_group_dispatch_refreshes_group_source_counts_per_pending tests/python/global_store/test_services.py::test_group_dispatch_rejects_duplicate_group_part_id tests/python/global_store/test_services.py::test_transport_service_group_progress_counts_success_only -v
```

If a test name has drifted, run:

```bash
pytest tests/python/global_store/test_services.py -k "group_dispatch or group_progress or group_source" -v
```

Expected: PASS.

- [ ] **Step 6: Check generated and unrelated dirty files**

Run:

```bash
git status --short
git diff --check
```

Expected: only this feature's tracked files are modified or staged. Pre-existing unrelated `pyproject.toml` and generated proto dirt must remain unstaged unless this implementation changed the same file for this feature.

- [ ] **Step 7: Commit docs and verification updates**

Run:

```bash
git add docs/designs/0116-control-plane-coordinated-weight-broadcast.md docs/plans/0116-control-plane-coordinated-weight-broadcast.md tensorcast/api/README.md tensorcast/api/store/README.md
git commit -m "docs: document weight broadcast prefetch hints"
```

# Acceptance Checks

- [ ] `Artifact.prefetch()` without `CallContext.transport_group` sends no transport hint.
- [ ] `Artifact.prefetch()` with typed transport group sends stable `transport_request_id` and complete group metadata.
- [ ] `DaemonCtl.materialize_by_artifact_id_v2()` copies transport hints into `MaterializeReplicaRequest`.
- [ ] `ReplicaMaterializationService::materialize_replica()` copies transport hints into `MaterializeHints`.
- [ ] Existing Global Store group dispatch tests still pass.
- [ ] `replica_uuid` is not overloaded with `#tcg:` metadata for prefetch.
- [ ] No unrelated pre-existing dirty files are reverted or staged.
