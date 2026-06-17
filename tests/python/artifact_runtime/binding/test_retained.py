#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import tensorcast.retained_realization as retained_realization_module
import tensorcast.retained_realization_authority as retained_authority_module
from tensorcast.artifact_runtime.binding.retained import (
    acquire_retained_binding,
    acquire_retained_binding_lease,
    promote_current_value_and_wait,
    retained_binding_acquire_mode,
)
from tensorcast.retained_realization import (
    RetainedRealizationClaim,
    RetainedRealizationExpectedDigests,
    parse_retained_realization_authority,
    parse_retained_realization_claim,
    retained_realization_claim_extra_from_handoff,
    retained_realization_claim_extra_json_from_handoff,
    retained_realization_claim_mode,
    retained_realization_trusted_reservation_bytes,
)
from tensorcast.retained_realization_authority import (
    ParsedRetainedRealizationAuthority,
    RetainedRealizationAuthority,
)
from tensorcast.retained_realization_authority import (
    RetainedRealizationExpectedDigests as RetainedRealizationAuthorityExpectedDigests,
)
from tensorcast.types import (
    BindingReservationCapability,
    BindingValueRef,
    BindingValueVerificationState,
    GroupRealizationAcquireRef,
    PrefetchHandoff,
    PrefetchRetentionPolicy,
    RealizationTarget,
    RuntimeBindingMemberRef,
    RuntimeBindingResolvedLayout,
    RuntimeBindingSourceRef,
    RuntimeBindingSourceReuseDecision,
    RuntimeTopologyRef,
)


def _authority(
    *,
    reservation_bytes: int = 4096,
    member_index: int = 0,
    member_count: int = 1,
    expires_at_ms: int | None = None,
) -> ParsedRetainedRealizationAuthority:
    suffix = member_index + 1
    member = RuntimeBindingMemberRef(
        member_id=f"member-{member_index}",
        member_index=member_index,
        member_count=member_count,
        group_id="group-1",
    )
    binding_ref = BindingValueRef(
        binding_id=f"binding-{suffix}",
        binding_layout_id="layout-1",
        binding_value_id=f"value-{suffix}",
        seal_generation=1,
    )
    capability = BindingReservationCapability(
        capability_id=f"capability-{suffix}",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid=f"gpu-{member_index}",
        member=member,
        reservation_bytes=reservation_bytes,
        scope_digest="scope-1",
        expires_at_ms=expires_at_ms,
    )
    return ParsedRetainedRealizationAuthority(
        group_id="group-1",
        local_serving_ref=f"binding-local:binding-{suffix}:value-{suffix}",
        binding_value_ref=binding_ref,
        reservation_capability=capability,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid=f"gpu-{member_index}",
        member=member,
        reservation_bytes=reservation_bytes,
        expected=RetainedRealizationAuthorityExpectedDigests(
            target_layout_hash="layout-hash",
            tensor_schema_hash="schema-hash",
            runtime_build_digest="build-digest",
            resolved_spec_digest="spec-digest",
        ),
        readiness="runtime_local_ready",
        verification_state="local_only",
    )


def _authority_payload(
    authority: ParsedRetainedRealizationAuthority,
) -> dict[str, object]:
    return {
        "group_id": authority.group_id,
        "member_ref": authority.member.model_dump(mode="python"),
        "daemon_id": authority.daemon_id,
        "daemon_session_id": authority.daemon_session_id,
        "device_uuid": authority.device_uuid,
        "binding_value_ref": authority.binding_value_ref.model_dump(mode="python"),
        "reservation_capability": authority.reservation_capability.model_dump(
            mode="python"
        ),
        "group_realization_acquire": None
        if authority.group_realization_acquire is None
        else authority.group_realization_acquire.model_dump(mode="python"),
        "local_serving_ref": authority.local_serving_ref,
        "readiness": authority.readiness,
        "verification_state": authority.verification_state,
        "serving_artifact_id": authority.serving_artifact_id,
        "trusted_reservation_bytes": authority.reservation_bytes,
        "expected": authority.expected.model_dump(mode="python"),
    }


def _set_nested(
    payload: dict[str, object], path: tuple[str, ...], value: object
) -> None:
    current = payload
    for key in path[:-1]:
        current = cast(dict[str, object], current[key])
    current[path[-1]] = value


def test_serving_retained_binding_does_not_export_legacy_authority_aliases() -> None:
    import tensorcast.artifact_runtime.binding.retained as retained_binding_module

    assert not hasattr(
        retained_binding_module, "ParsedRetainedServingBindingAuthority"
    )
    assert not hasattr(retained_binding_module, "RetainedServingBindingAuthority")
    assert not hasattr(
        retained_binding_module, "RetainedServingBindingExpectedDigests"
    )


def test_retained_realization_authority_module_hides_serving_aliases() -> None:
    public_names = set(retained_authority_module.__all__)

    assert "ParsedRetainedServingBindingAuthority" not in public_names
    assert "RetainedServingBindingAuthority" not in public_names
    assert "RetainedServingBindingExpectedDigests" not in public_names
    assert not hasattr(
        retained_authority_module, "ParsedRetainedServingBindingAuthority"
    )
    assert not hasattr(retained_authority_module, "RetainedServingBindingAuthority")
    assert not hasattr(
        retained_authority_module, "RetainedServingBindingExpectedDigests"
    )


def test_retained_realization_module_hides_prefetched_compat_helpers() -> None:
    public_names = set(retained_realization_module.__all__)

    for removed_name in (
        "retained_realization_claim_extra_from_prefetched_binding",
        "retained_realization_claim_extra_json",
    ):
        assert removed_name not in public_names
        assert not hasattr(retained_realization_module, removed_name)


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


def _topology() -> RuntimeTopologyRef:
    return RuntimeTopologyRef(
        schema_topology_digest="topology-schema",
        admission_topology_digest="topology-admission",
    )


def _source() -> RuntimeBindingSourceRef:
    return RuntimeBindingSourceRef(
        source_kind="checkpoint_artifact",
        artifact_selection_digest="selection-digest",
        source_artifact_ref="mi2:checkpoint",
        source_schema_hash="source-schema",
    )


def _target(
    member: RuntimeBindingMemberRef,
    *,
    topology: RuntimeTopologyRef | None = None,
) -> RealizationTarget:
    resolved_topology = topology or _topology()
    source = _source()
    source_reuse = RuntimeBindingSourceReuseDecision(
        mode="checkpoint_to_runtime",
        representation_contract_hash="repr-contract",
    )
    resolved_layout = RuntimeBindingResolvedLayout(
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
    return RealizationTarget(
        runtime="vllm",
        device="cuda:0",
        device_uuid="GPU-0",
        source=source,
        topology=resolved_topology,
        member=member,
        model_config_digest="model-config",
        runtime_build_digest="serving-build",
        resolved_layout=resolved_layout,
    )


def _prefetched(
    member: RuntimeBindingMemberRef,
    *,
    reservation_bytes: int = 4096,
) -> PrefetchHandoff:
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
    return PrefetchHandoff(
        local_serving_ref="binding-local:binding-1:value-1",
        binding_value_ref=binding_ref,
        daemon_id="daemon-1",
        daemon_session_id="session-1",
        device_uuid="GPU-0",
        member=member,
        reservation_bytes=reservation_bytes,
        reservation_capability=capability,
        readiness="runtime_local_ready",
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

    def acquire_binding_value_by_local_ref(self, **kwargs):
        self.acquire_calls.append(kwargs)
        return self.response

    def release_placement_lease(self, *, lease_token: bytes, **_kwargs):
        self.released_tokens.append(bytes(lease_token))


class _Runtime:
    def __init__(self, client: _Client) -> None:
        self.client = client

    def ensure_client(self):
        return self.client


def test_acquire_retained_binding_lease_releases_unrestored_lease_on_context_exit():
    authority = _authority()
    client = _Client(_response())

    with acquire_retained_binding_lease(
        authority,
        runtime=_Runtime(client),
        caller_pid=123,
    ) as lease:
        assert lease.binding_value_ref == authority.binding_value_ref
        assert lease.member_ref == authority.member
        assert lease.reservation_bytes == 4096
        assert lease.release_contract.release_policy == (
            "close_runtime_attachment",
            "release_placement_lease",
        )
        assert lease.release_contract.released is False

    assert client.acquire_calls[0]["caller_pid"] == 123
    assert client.acquire_calls[0]["expected_tensor_schema_hash"] == "schema-hash"
    assert client.released_tokens == [b"lease"]
    assert lease.release_contract.released is True


def test_acquire_retained_binding_uses_authority():
    authority = _authority()
    client = _Client(_response())

    with acquire_retained_binding(
        authority=authority,
        runtime=_Runtime(client),
        caller_pid=456,
    ) as lease:
        assert lease.binding_value_ref == authority.binding_value_ref

    assert client.acquire_calls[0]["caller_pid"] == 456
    assert client.acquire_calls[0]["binding_value_ref"] == authority.binding_value_ref
    assert client.released_tokens == [b"lease"]


def test_acquire_retained_binding_rejects_expired_capability_before_daemon_call():
    authority = _authority(expires_at_ms=1)
    client = _Client(_response())

    with (
        pytest.raises(ValueError, match="reservation_capability has expired"),
        acquire_retained_binding_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.acquire_calls == []
    assert client.released_tokens == []


def test_retained_binding_debug_status_tracks_capability_ttl_and_lifecycle():
    authority = _authority(expires_at_ms=4_102_444_800_000)
    client = _Client(_response())

    with acquire_retained_binding_lease(
        authority, runtime=_Runtime(client)
    ) as lease:
        acquired_status = lease.debug_status()
        assert lease.status() == "acquired"
        assert acquired_status["state"] == "acquired"
        assert acquired_status["reservation_capability_id"] == "capability-1"
        assert acquired_status["reservation_expires_at_ms"] == 4_102_444_800_000
        assert acquired_status["readiness"] == "runtime_local_ready"
        assert acquired_status["verification_state"] == "local_only"
        assert acquired_status["lease_token_present"] is True
        assert acquired_status["release_policy"] == (
            "close_runtime_attachment",
            "release_placement_lease",
        )
        assert acquired_status["released"] is False

        attached = lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        assert lease.status() == "restored"
        assert attached.status() == "restored"
        assert attached.debug_status()["state"] == "restored"

        runtime_handle = attached.transfer_to_runtime()
        assert runtime_handle.status() == "runtime_owned"
        assert runtime_handle.debug_status()["state"] == "runtime_owned"
        runtime_handle.close()
        assert runtime_handle.status() == "closed"
        assert runtime_handle.debug_status()["released"] is True

    assert client.released_tokens == [b"lease"]


def test_retained_prefetch_retention_policy_round_trips_ttl_and_idle_retire():
    policy = PrefetchRetentionPolicy(
        expire_if_unacquired_after_ms=10_000,
        idle_ttl_after_last_release_ms=500,
        materialization_timeout_ms=30_000,
        allow_acquire_after_creator_exit=True,
    )

    round_tripped = PrefetchRetentionPolicy.from_proto(policy.to_proto())

    assert round_tripped.expire_if_unacquired_after_ms == 10_000
    assert round_tripped.idle_ttl_after_last_release_ms == 500
    assert round_tripped.materialization_timeout_ms == 30_000
    assert round_tripped.allow_acquire_after_creator_exit is True


def test_acquire_retained_binding_acquires_local_ready(monkeypatch):
    member = RuntimeBindingMemberRef(
        member_id="member-0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )
    client = _Client(_response())
    import tensorcast.api.store as store_api

    monkeypatch.setattr(
        store_api, "device_uuid_for", lambda device_index: f"gpu-{device_index}"
    )

    with acquire_retained_binding(
        local_serving_ref="binding-local:binding-1:value-1",
        target_device=torch.device("cuda:3"),
        expected_member=member,
        expected_tensor_schema_hash="schema-hash",
        expected_serving_build_digest="build-digest",
        runtime=_Runtime(client),
        caller_pid=789,
    ) as lease:
        assert lease.authority.readiness == "runtime_local_ready"

    assert (
        client.acquire_calls[0]["local_serving_ref"]
        == "binding-local:binding-1:value-1"
    )
    assert client.acquire_calls[0]["expected_device_uuid"] == "gpu-3"
    assert client.acquire_calls[0]["caller_pid"] == 789
    assert client.released_tokens == [b"lease"]


def test_restore_failure_releases_acquired_lease():
    authority = _authority()
    client = _Client(_response())

    def fail_restore(**_kwargs):
        raise RuntimeError("restore failed")

    with (
        pytest.raises(RuntimeError, match="restore failed"),
        acquire_retained_binding_lease(
            authority, runtime=_Runtime(client)
        ) as lease,
    ):
        lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=fail_restore,
        )

    assert client.released_tokens == [b"lease"]


def test_attached_close_releases_once_after_successful_restore():
    authority = _authority()
    client = _Client(_response())

    with acquire_retained_binding_lease(
        authority, runtime=_Runtime(client)
    ) as lease:
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

    with acquire_retained_binding_lease(
        authority, runtime=_Runtime(client)
    ) as lease:
        attached = lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        assert attached.authority is authority
        runtime_handle = attached.transfer_to_runtime()
        assert runtime_handle.authority is authority
        assert runtime_handle.release_contract is lease.release_contract
        attached.close()
        assert runtime_handle.binding_layout_id == "layout-1"
        runtime_handle.close()
        runtime_handle.close()

    assert client.released_tokens == [b"lease"]
    assert runtime_handle.release_contract.released is True


def test_restored_lease_releases_on_context_exit_when_not_transferred():
    authority = _authority()
    client = _Client(_response())

    with (
        pytest.raises(RuntimeError, match="attach failed"),
        acquire_retained_binding_lease(
            authority, runtime=_Runtime(client)
        ) as lease,
    ):
        lease.restore(
            target_device=torch.device("cuda:0"),
            restore_fn=lambda **_kwargs: {"w": torch.empty((1,), dtype=torch.float32)},
        )
        raise RuntimeError("attach failed")

    assert client.released_tokens == [b"lease"]


def test_retained_binding_lifecycle_rejects_invalid_transitions():
    authority = _authority()
    client = _Client(_response())

    with acquire_retained_binding_lease(
        authority, runtime=_Runtime(client)
    ) as lease:
        lease.close()
        with pytest.raises(RuntimeError, match="requires an acquired lease"):
            lease.restore(
                target_device=torch.device("cuda:0"),
                restore_fn=lambda **_kwargs: {},
            )

    client = _Client(_response())
    with acquire_retained_binding_lease(
        authority, runtime=_Runtime(client)
    ) as lease:
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


def test_acquire_retained_binding_lease_rejects_mismatched_acquire_response():
    authority = _authority()
    response = _response()
    response.current_value.binding_value_id = "other-value"
    client = _Client(response)

    with (
        pytest.raises(RuntimeError, match="different binding value"),
        acquire_retained_binding_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.released_tokens == [b"lease"]


def test_acquire_retained_binding_lease_releases_mismatched_reservation_response():
    authority = _authority(reservation_bytes=4096)
    client = _Client(_response(reservation_bytes=8192))

    with (
        pytest.raises(RuntimeError, match="reservation byte mismatch"),
        acquire_retained_binding_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.released_tokens == [b"lease"]


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (
            ("reservation_capability", "binding_value_ref", "binding_value_id"),
            "other-value",
            "binding_value_ref",
        ),
        (
            ("reservation_capability", "daemon_id"),
            "other-daemon",
            "daemon_id mismatch",
        ),
        (
            ("reservation_capability", "daemon_session_id"),
            "other-session",
            "daemon_session_id mismatch",
        ),
        (
            ("reservation_capability", "device_uuid"),
            "other-gpu",
            "device_uuid mismatch",
        ),
        (
            ("reservation_capability", "member", "member_id"),
            "other-member",
            "member mismatch",
        ),
        (
            ("reservation_capability", "reservation_bytes"),
            8192,
            "reservation_bytes",
        ),
        (("expected", "tensor_schema_hash"), "", "expected digest fields"),
    ],
)
def test_parse_retained_binding_authority_rejects_inconsistent_authority(
    path,
    value,
    match,
):
    payload = _authority_payload(_authority())
    _set_nested(payload, path, value)
    extra = {
        "retained_binding_acquire": {
            "mode": "external",
            "authority": payload,
        },
    }

    with pytest.raises(ValueError, match=match):
        parse_retained_realization_authority(extra)


def test_parse_retained_binding_authority_rejects_member_group_mismatch():
    payload = _authority_payload(_authority())
    _set_nested(payload, ("member_ref", "group_id"), "other-group")
    _set_nested(
        payload, ("reservation_capability", "member", "group_id"), "other-group"
    )
    extra = {
        "retained_binding_acquire": {
            "mode": "external",
            "authority": payload,
        },
    }

    with pytest.raises(ValueError, match="member_ref.group_id"):
        parse_retained_realization_authority(extra)


def test_parse_retained_binding_authority_requires_published_artifact_scope():
    authority = replace(
        _authority(),
        readiness="runtime_published_ready",
        serving_artifact_id=None,
    )
    extra = {
        "retained_binding_acquire": {
            "mode": "external",
            "authority": _authority_payload(authority),
        },
    }

    with pytest.raises(ValueError, match="serving_artifact_id"):
        parse_retained_realization_authority(extra)


def test_acquire_retained_binding_rejects_reserved_authority_before_daemon_call():
    authority = replace(_authority(), readiness="runtime_reserved")
    client = _Client(_response())

    with (
        pytest.raises(ValueError, match="runtime_reserved"),
        acquire_retained_binding_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.acquire_calls == []
    assert client.released_tokens == []


def test_acquire_retained_binding_requires_group_publish_wait_before_attach():
    authority = replace(
        _authority(),
        group_realization_acquire=GroupRealizationAcquireRef(
            transaction_id="txn-1",
            version_set_id="version-set-1",
            part_id="part-0",
            staging_token="token-1",
            wait_for_publish=False,
        ),
    )
    client = _Client(_response())

    with (
        pytest.raises(ValueError, match="wait for group publish"),
        acquire_retained_binding_lease(authority, runtime=_Runtime(client)),
    ):
        pass

    assert client.acquire_calls == []
    assert client.released_tokens == []


def test_acquire_retained_binding_passes_group_publish_wait_authority():
    group_acquire = GroupRealizationAcquireRef(
        transaction_id="txn-1",
        version_set_id="version-set-1",
        part_id="part-0",
        staging_token="token-1",
        wait_for_publish=True,
        wait_timeout_ms=1234,
    )
    authority = replace(_authority(), group_realization_acquire=group_acquire)
    client = _Client(_response())

    with acquire_retained_binding_lease(authority, runtime=_Runtime(client)):
        pass

    assert client.acquire_calls[0]["group_realization_acquire"] == group_acquire
    assert client.released_tokens == [b"lease"]


def test_retained_binding_public_helpers_build_extra_from_handoff():
    member = _authority().member
    prefetched = _prefetched(member, reservation_bytes=8192)
    target = _target(member)

    extra = retained_realization_claim_extra_from_handoff(
        handoff=prefetched,
        target=target,
        expected_member=member,
    )
    authority = parse_retained_realization_authority(extra)

    assert "retained_binding_acquire" in extra
    assert retained_binding_acquire_mode(extra) == "external"
    assert isinstance(extra["retained_binding_acquire"]["authority"], dict)
    assert (
        RetainedRealizationAuthority.model_validate(
            extra["retained_binding_acquire"]["authority"]
        ).trusted_reservation_bytes
        == 8192
    )
    assert authority.member == member
    assert authority.reservation_bytes == 8192
    assert authority.expected.target_layout_hash == "target-layout-hash"
    assert authority.expected.tensor_schema_hash == "tensor-schema"
    assert authority.expected.runtime_build_digest == "serving-build"
    assert authority.expected.resolved_spec_digest == "spec-digest"
    assert retained_realization_trusted_reservation_bytes(extra) == 8192
    assert (
        retained_realization_trusted_reservation_bytes(
            SimpleNamespace(model_loader_extra_config=extra)
        )
        == 8192
    )
    assert '"mode":"external"' in retained_realization_claim_extra_json_from_handoff(
        handoff=prefetched,
        target=target,
        expected_member=member,
    )
    assert '"retained_binding_acquire"' in retained_realization_claim_extra_json_from_handoff(
        handoff=prefetched,
        target=target,
        expected_member=member,
    )


def test_retained_realization_claim_helpers_use_primary_authority_contract():
    member = _authority().member
    handoff = _prefetched(member, reservation_bytes=8192)
    target = _target(member)

    extra = retained_realization_claim_extra_from_handoff(
        handoff=handoff,
        target=target,
        expected_member=member,
    )
    claim = parse_retained_realization_claim(extra, expected_member=member)
    authority = parse_retained_realization_authority(extra, expected_member=member)

    assert isinstance(claim, RetainedRealizationClaim)
    assert claim.authority == authority
    assert claim.as_authority() == authority
    assert parse_retained_realization_authority(extra, expected_member=member) == (
        authority
    )
    assert claim.group_id == authority.group_id
    assert claim.local_ref == authority.local_serving_ref
    assert claim.binding_value_ref == authority.binding_value_ref
    assert claim.reservation_capability == authority.reservation_capability
    assert claim.daemon_id == authority.daemon_id
    assert claim.daemon_session_id == authority.daemon_session_id
    assert claim.device_uuid == authority.device_uuid
    assert claim.member == member
    assert claim.reservation_bytes == 8192
    assert isinstance(claim.expected, RetainedRealizationExpectedDigests)
    assert claim.expected.tensor_schema_hash == "tensor-schema"
    assert claim.readiness == "runtime_local_ready"
    assert claim.verification_state == "local_only"
    assert claim.serving_artifact_id == authority.serving_artifact_id
    assert claim.group_realization_acquire == authority.group_realization_acquire
    assert retained_realization_claim_mode(extra) == "external"
    assert retained_realization_trusted_reservation_bytes(extra) == 8192
    assert (
        retained_realization_trusted_reservation_bytes(
            SimpleNamespace(model_loader_extra_config=extra)
        )
        == 8192
    )
    assert retained_realization_claim_extra_from_handoff(
        handoff=handoff,
        target=target,
        expected_member=member,
    ) == extra
    assert json.loads(
        retained_realization_claim_extra_json_from_handoff(
            handoff=handoff,
            target=target,
            expected_member=member,
        )
    ) == extra


def test_retained_realization_claim_trusted_bytes_fail_closed_on_mismatch():
    payload = _authority_payload(_authority())
    _set_nested(payload, ("reservation_capability", "reservation_bytes"), 8192)
    extra = {
        "retained_binding_acquire": {
            "mode": "external",
            "authority": payload,
        },
    }

    with pytest.raises(ValueError, match="reservation_bytes"):
        retained_realization_trusted_reservation_bytes(extra)


def test_retained_binding_authority_set_selects_expected_member():
    authority0 = _authority(
        reservation_bytes=4096,
        member_index=0,
        member_count=2,
    )
    authority1 = _authority(
        reservation_bytes=8192,
        member_index=1,
        member_count=2,
    )
    extra = {
        "retained_binding_acquire": {
            "mode": "external",
            "authorities": [
                _authority_payload(authority0),
                _authority_payload(authority1),
            ],
        },
    }

    selected = parse_retained_realization_authority(
        extra,
        expected_member=authority1.member,
    )

    assert selected.member == authority1.member
    assert selected.reservation_bytes == 8192
    assert (
        retained_realization_trusted_reservation_bytes(
            extra,
            expected_member=authority1.member,
        )
        == 8192
    )


def test_retained_binding_authority_set_requires_expected_member():
    authority0 = _authority(member_index=0, member_count=2)
    authority1 = _authority(member_index=1, member_count=2)
    extra = {
        "retained_binding_acquire": {
            "mode": "external",
            "authorities": [
                _authority_payload(authority0),
                _authority_payload(authority1),
            ],
        },
    }

    with pytest.raises(ValueError, match="expected serving member"):
        parse_retained_realization_authority(extra)


def test_retained_binding_extra_preserves_group_realization_acquire():
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

    extra = retained_realization_claim_extra_from_handoff(
        handoff=prefetched,
        target=target,
        expected_member=member,
    )
    authority = parse_retained_realization_authority(extra)

    assert authority.group_realization_acquire is not None
    assert authority.group_realization_acquire.transaction_id == "txn-1"
    assert authority.group_realization_acquire.wait_for_publish is True


def test_retained_binding_extra_rejects_unexpected_member():
    member = _authority().member
    unexpected = member.model_copy(update={"member_id": "other"})

    with pytest.raises(ValueError, match="does not match expected placement"):
        retained_realization_claim_extra_from_handoff(
            handoff=_prefetched(member),
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
