#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import tensorcast as tc
from tensorcast.serving import (
    ExternalPreloadExpectedDigests,
    acquire_preload_lease,
    external_preload_extra_from_prefetched_binding,
    external_preload_extra_json,
    external_preload_mode,
    external_preload_trusted_reservation_bytes,
    parse_external_preload_authority,
    promote_current_value_and_wait,
)
from tensorcast.serving.preload import ParsedExternalPreloadAuthority
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    BindingValueVerificationState,
    GroupRealizationAcquireRef,
    ServingBindingMemberRef,
    ServingBindingResolvedLayout,
    ServingBindingSourceRef,
    ServingBindingSourceReuseDecision,
    ServingBindingTarget,
    ServingTopologyRef,
)


def _authority(*, reservation_bytes: int = 4096) -> ParsedExternalPreloadAuthority:
    member = ServingBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=reservation_bytes,
        scope_digest="scope-1",
    )
    return ParsedExternalPreloadAuthority(
        group_id="group-1",
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=binding_ref,
        reservation_capability=capability,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="gpu-0",
        member=member,
        reservation_bytes=reservation_bytes,
        expected=ExternalPreloadExpectedDigests(
            target_layout_hash="layout-hash",
            tensor_schema_hash="schema-hash",
            serving_build_digest="build-digest",
            resolved_spec_digest="spec-digest",
        ),
        readiness="serving_local_ready",
        verification_state="local_only",
    )


def _response(*, reservation_bytes: int = 4096, lease_token: bytes = b"lease"):
    return SimpleNamespace(
        reservation_bytes=reservation_bytes,
        mem_handle=SimpleNamespace(lease_token=lease_token),
        current_value=SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
        ),
    )


def _topology() -> ServingTopologyRef:
    return ServingTopologyRef(
        schema_topology_digest="topology-schema",
        admission_topology_digest="topology-admission",
    )


def _source() -> ServingBindingSourceRef:
    return ServingBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:checkpoint",
        source_schema_hash="source-schema",
    )


def _target(
    member: ServingBindingMemberRef,
    *,
    topology: ServingTopologyRef | None = None,
) -> ServingBindingTarget:
    resolved_topology = topology or _topology()
    source = _source()
    source_reuse = ServingBindingSourceReuseDecision(
        mode="checkpoint_to_serving",
        representation_contract_hash="repr-contract",
    )
    resolved_layout = ServingBindingResolvedLayout(
        binding_layout_id="layout-1",
        source=source,
        source_reuse=source_reuse,
        topology=resolved_topology,
        member=member,
        target_layout=b"target-layout",
        target_index_bytes=b"target-index",
        target_layout_hash="target-layout-hash",
        tensor_schema_hash="tensor-schema",
        spec_digest="spec-digest",
        source_schema_hash="source-schema",
    )
    return ServingBindingTarget(
        runtime="vllm",
        device="cuda:0",
        device_uuid="GPU-0",
        source=source,
        topology=resolved_topology,
        member=member,
        model_config_digest="model-config",
        serving_build_digest="serving-build",
        resolved_layout=resolved_layout,
    )


def _prefetched(
    member: ServingBindingMemberRef,
    *,
    reservation_bytes: int = 4096,
) -> tc.PrefetchedServingBinding:
    binding_ref = BindingValueRef(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id="capability-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=reservation_bytes,
        scope_digest="scope-1",
    )
    return tc.PrefetchedServingBinding(
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=reservation_bytes,
        reservation_capability=capability,
        readiness="serving_local_ready",
        verification_state=BindingValueVerificationState.LOCAL_ONLY,
    )


class _Client:
    def __init__(self, response) -> None:
        self.response = response
        self.acquire_calls: list[dict[str, object]] = []
        self.released_tokens: list[bytes] = []

    def acquire_binding_value(self, **kwargs):
        self.acquire_calls.append(kwargs)
        return self.response

    def release_placement_lease(self, *, lease_token: bytes, **_kwargs):
        self.released_tokens.append(bytes(lease_token))


class _Runtime:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def ensure_client(self):
        return self.client


def test_acquire_preload_lease_releases_unrestored_lease_on_context_exit():
    authority = _authority()
    client = _Client(_response())

    with acquire_preload_lease(
        authority,
        runtime=_Runtime(client),
        caller_pid=123,
    ) as lease:
        assert lease.binding_value_ref == authority.binding_value_ref
        assert lease.member_ref == authority.member
        assert lease.reservation_bytes == 4096

    assert client.acquire_calls[0]["caller_pid"] == 123
    assert client.acquire_calls[0]["expected_tensor_schema_hash"] == "schema-hash"
    assert client.released_tokens == [b"lease"]


def test_restore_failure_releases_acquired_lease():
    authority = _authority()
    client = _Client(_response())

    def fail_restore(**_kwargs):
        raise RuntimeError("restore failed")

    with (
        pytest.raises(RuntimeError, match="restore failed"),
        acquire_preload_lease(authority, runtime=_Runtime(client)) as lease,
    ):
        lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=fail_restore,
        )

    assert client.released_tokens == [b"lease"]


def test_attached_close_releases_once_after_successful_restore():
    authority = _authority()
    client = _Client(_response())

    with acquire_preload_lease(authority, runtime=_Runtime(client)) as lease:
        attached = lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        attached.close()
        attached.close()

    assert client.released_tokens == [b"lease"]


def test_transfer_to_runtime_moves_close_ownership():
    authority = _authority()
    client = _Client(_response())

    with acquire_preload_lease(authority, runtime=_Runtime(client)) as lease:
        attached = lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        runtime_handle = attached.transfer_to_runtime()
        attached.close()
        assert runtime_handle.binding_layout_id == "layout-1"
        runtime_handle.close()
        runtime_handle.close()

    assert client.released_tokens == [b"lease"]


def test_restored_lease_releases_on_context_exit_when_not_transferred():
    authority = _authority()
    client = _Client(_response())

    with (
        pytest.raises(RuntimeError, match="attach failed"),
        acquire_preload_lease(authority, runtime=_Runtime(client)) as lease,
    ):
        lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        raise RuntimeError("attach failed")

    assert client.released_tokens == [b"lease"]


def test_preload_lifecycle_rejects_invalid_transitions():
    authority = _authority()
    client = _Client(_response())

    with acquire_preload_lease(authority, runtime=_Runtime(client)) as lease:
        lease.close()
        with pytest.raises(RuntimeError, match="requires an acquired lease"):
            lease.restore(
                target_device=torch.device("cuda:0"),
                restore_fn=lambda **_kwargs: {},
            )

    client = _Client(_response())
    with acquire_preload_lease(authority, runtime=_Runtime(client)) as lease:
        attached = lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        with pytest.raises(RuntimeError, match="requires an acquired lease"):
            lease.restore(
                target_device=torch.device("cuda:0"),
                restore_fn=lambda **_kwargs: {},
            )
        attached.transfer_to_runtime()
        with pytest.raises(RuntimeError, match="requires a restored attachment"):
            attached.transfer_to_runtime()


def test_acquire_preload_lease_rejects_mismatched_acquire_response():
    authority = _authority()
    response = _response()
    response.current_value.binding_value_id = "other-value"
    client = _Client(response)

    with (
        pytest.raises(RuntimeError, match="different binding value"),
        acquire_preload_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.released_tokens == [b"lease"]


def test_acquire_preload_lease_releases_mismatched_reservation_response():
    authority = _authority(reservation_bytes=4096)
    client = _Client(_response(reservation_bytes=8192))

    with (
        pytest.raises(RuntimeError, match="reservation byte mismatch"),
        acquire_preload_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.released_tokens == [b"lease"]


def test_external_preload_public_helpers_build_extra_from_prefetched_binding():
    member = _authority().member
    prefetched = _prefetched(member, reservation_bytes=8192)
    target = _target(member)

    extra = external_preload_extra_from_prefetched_binding(
        prefetched=prefetched,
        target=target,
        expected_member=member,
    )
    authority = parse_external_preload_authority(extra)

    assert external_preload_mode(extra) == "external"
    assert authority.member == member
    assert authority.reservation_bytes == 8192
    assert authority.expected.target_layout_hash == "target-layout-hash"
    assert authority.expected.tensor_schema_hash == "tensor-schema"
    assert authority.expected.serving_build_digest == "serving-build"
    assert authority.expected.resolved_spec_digest == "spec-digest"
    assert external_preload_trusted_reservation_bytes(extra) == 8192
    assert (
        external_preload_trusted_reservation_bytes(
            SimpleNamespace(model_loader_extra_config=extra)
        )
        == 8192
    )
    assert '"mode":"external"' in external_preload_extra_json(
        prefetched=prefetched,
        target=target,
        expected_member=member,
    )


def test_external_preload_extra_preserves_group_realization_acquire():
    member = _authority().member
    target = _target(member)
    prefetched = _prefetched(member).model_copy(
        update={
            "staged_value": True,
            "group_realization_acquire": GroupRealizationAcquireRef(
                transaction_id="txn-1",
                version_set_id="version-set-1",
                part_id="part-0",
                staging_token="token-1",
                wait_for_publish=True,
                wait_timeout_ms=1234,
            ),
        }
    )

    extra = external_preload_extra_from_prefetched_binding(
        prefetched=prefetched,
        target=target,
        expected_member=member,
    )
    authority = parse_external_preload_authority(extra)

    assert authority.group_realization_acquire is not None
    assert authority.group_realization_acquire.transaction_id == "txn-1"
    assert authority.group_realization_acquire.wait_for_publish is True


def test_external_preload_extra_rejects_unexpected_member():
    member = _authority().member
    unexpected = member.model_copy(update={"member_id": "other"})

    with pytest.raises(ValueError, match="does not match expected placement"):
        external_preload_extra_from_prefetched_binding(
            prefetched=_prefetched(member),
            target=_target(member),
            expected_member=unexpected,
        )


class _PromotionBinding:
    def __init__(self, statuses: list[SimpleNamespace]) -> None:
        self.statuses = list(statuses)
        self.start_calls: list[str] = []
        self.poll_calls: list[dict[str, str | None]] = []

    def start_promote_current_value(self, *, binding_value_id: str):
        self.start_calls.append(binding_value_id)
        return self.statuses.pop(0)

    def get_promotion_status(
        self,
        *,
        verification_job_id: str | None,
        binding_value_id: str,
    ):
        self.poll_calls.append(
            {
                "verification_job_id": verification_job_id,
                "binding_value_id": binding_value_id,
            }
        )
        return self.statuses.pop(0)


def test_promote_current_value_and_wait_returns_verified_result():
    binding = _PromotionBinding(
        [
            SimpleNamespace(state="running", verification_job_id="job-1"),
            SimpleNamespace(state="succeeded", verification_job_id="job-1"),
        ]
    )
    polled: list[str] = []

    result = promote_current_value_and_wait(
        binding=binding,
        current_value=SimpleNamespace(binding_value_id="value-1"),
        sleep_fn=lambda _seconds: None,
        on_poll=lambda status: polled.append(status.state),
    )

    assert result.verification_state == "verified"
    assert result.verification_job_id == "job-1"
    assert binding.start_calls == ["value-1"]
    assert binding.poll_calls == [
        {
            "verification_job_id": "job-1",
            "binding_value_id": "value-1",
        }
    ]
    assert polled == ["succeeded"]


def test_promote_current_value_and_wait_scopes_wrap_start_and_poll_calls():
    events: list[str] = []

    @contextmanager
    def _scope(name: str):
        payload = {"name": name}
        events.append(f"{name}:enter")
        try:
            yield payload
        finally:
            events.append(f"{name}:exit")

    class ScopedPromotionBinding(_PromotionBinding):
        def start_promote_current_value(self, *, binding_value_id: str):
            events.append("start:call")
            return super().start_promote_current_value(
                binding_value_id=binding_value_id
            )

        def get_promotion_status(
            self,
            *,
            verification_job_id: str | None,
            binding_value_id: str,
        ):
            events.append("poll:call")
            return super().get_promotion_status(
                verification_job_id=verification_job_id,
                binding_value_id=binding_value_id,
            )

    binding = ScopedPromotionBinding(
        [
            SimpleNamespace(state="running", verification_job_id="job-1"),
            SimpleNamespace(state="succeeded", verification_job_id="job-1"),
        ]
    )

    result = promote_current_value_and_wait(
        binding=binding,
        current_value=SimpleNamespace(binding_value_id="value-1"),
        sleep_fn=lambda _seconds: None,
        start_scope=lambda: _scope("start"),
        poll_scope=lambda: _scope("poll"),
        on_start=lambda status, profile: events.append(
            f"start:observe:{profile['name']}:{status.state}"
        ),
        on_poll=lambda status, profile: events.append(
            f"poll:observe:{profile['name']}:{status.state}"
        ),
    )

    assert result.verification_state == "verified"
    assert events == [
        "start:enter",
        "start:call",
        "start:observe:start:running",
        "start:exit",
        "poll:enter",
        "poll:call",
        "poll:observe:poll:succeeded",
        "poll:exit",
    ]


def test_promote_current_value_and_wait_raises_on_failure():
    binding = _PromotionBinding(
        [
            SimpleNamespace(
                state="failed",
                verification_job_id="job-1",
                failure_reason="bad checksum",
            )
        ]
    )

    with pytest.raises(RuntimeError, match="bad checksum"):
        promote_current_value_and_wait(
            binding=binding,
            current_value=SimpleNamespace(binding_value_id="value-1"),
            sleep_fn=lambda _seconds: None,
        )
