#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorcast.serving import (
    ExternalPreloadExpectedDigests,
    acquire_preload_lease,
)
from tensorcast.serving.preload import ParsedExternalPreloadAuthority
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    ServingBindingMemberRef,
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
    assert client.acquire_calls[0]["expected_tensor_schema_hash"] == \
        "schema-hash"
    assert client.released_tokens == [b"lease"]


def test_restore_failure_releases_acquired_lease():
    authority = _authority()
    client = _Client(_response())

    def fail_restore(**_kwargs):
        raise RuntimeError("restore failed")

    with pytest.raises(RuntimeError, match="restore failed"), \
            acquire_preload_lease(authority, runtime=_Runtime(client)) as lease:
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
            restore_fn=lambda **_kwargs: {
                "w": torch.empty((1, ), dtype=torch.float32)
            },
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
            restore_fn=lambda **_kwargs: {
                "w": torch.empty((1, ), dtype=torch.float32)
            },
        )
        runtime_handle = attached.transfer_to_runtime()
        attached.close()
        assert runtime_handle.binding_layout_id == "layout-1"
        runtime_handle.close()
        runtime_handle.close()

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
            restore_fn=lambda **_kwargs: {
                "w": torch.empty((1, ), dtype=torch.float32)
            },
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

    with pytest.raises(RuntimeError, match="different binding value"), \
            acquire_preload_lease(authority, runtime=_Runtime(client)):
        pass

    assert client.released_tokens == []
