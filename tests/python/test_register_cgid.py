#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

import pytest
import torch

from tensorcast.api._config import PlanType, RegisterArtifactOptions
from tensorcast.api._register import _register_artifact_core
from tensorcast.api._errors import TensorCastError
from tensorcast.common.identity import ArtifactIdKind
from tensorcast.types import (
    ArtifactDescriptor,
    BeginRegisterArtifactResult,
    CoalescedHandshake,
    CommitResult,
)


class _FakeDaemonCtl:
    def __init__(self) -> None:
        self.client_ids: list[str | None] = []
        self.begin_total_sizes: list[int] = []

    def begin_register_artifact(
        self,
        *,
        device_id: int,
        total_size_bytes: int,
        ttl_ms: int | None = None,
        tensor_index_key: str | None = None,
        tensor_index_data: bytes | None = None,
        encoding: str = "json",
        schema_version: str = "v3",
        client_artifact_id: str | None = None,
        plan=None,
        policy=None,
        timeout_s: float = 30.0,
        view=None,
    ) -> BeginRegisterArtifactResult:
        self.client_ids.append(client_artifact_id)
        self.begin_total_sizes.append(int(total_size_bytes))
        return BeginRegisterArtifactResult(
            registration_id="reg-1",
            device_id=device_id,
            total_size=int(total_size_bytes),
            handshake=CoalescedHandshake(daemon_ipc_handle=b""),
        )

    def commit_registered_artifact(self, registration_id: str, *, timeout_s: float = 30.0) -> CommitResult:
        if not self.client_ids:
            raise AssertionError("begin_register_artifact was not invoked before commit")
        artifact_id = self.client_ids[-1] or "mi2:auto"
        total_size = self.begin_total_sizes[-1] if self.begin_total_sizes else 0
        descriptor = ArtifactDescriptor(
            artifact_id=artifact_id,
            index_multihash=None,
            data_multihash=None,
            schema_version=None,
            encoding=None,
            total_size=total_size,
            id_kind=ArtifactIdKind.CGID if artifact_id.startswith("cgid:") else ArtifactIdKind.MI2,
        )
        return CommitResult(descriptor=descriptor, existed=False)

    def abort_registered_artifact(self, registration_id: str, *, timeout_s: float = 15.0) -> bool:
        return True


def _patch_coalesced_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorcast.api import _register as register_mod

    def _fake_upload(self, *, artifact, ctx, layout, handle, handshake, cancel_event=None):
        return dict(artifact)

    monkeypatch.setattr(register_mod._CoalescedUploader, "upload", _fake_upload, raising=False)


def test_register_artifact_with_cgid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_coalesced_upload(monkeypatch)
    client = _FakeDaemonCtl()
    tensors = {"a": torch.zeros((4, 4), dtype=torch.float32)}
    options = RegisterArtifactOptions(plan=PlanType.VRAM_COALESCED, lease_in_place=False)

    result = _register_artifact_core(
        artifact=tensors,
        options=options,
        device_id=0,
        ttl_ms=None,
        client_artifact_id="cgid:kvdeadbeef",
        force_lease_in_place=False,
        prevalidate_disk=False,
        client=client,
        daemon_address="fake-daemon",
    )

    assert client.client_ids == ["cgid:kvdeadbeef"]
    assert result.descriptor.artifact_id == "cgid:kvdeadbeef"
    assert result.descriptor.id_kind is ArtifactIdKind.CGID
    assert result.descriptor.index_multihash is None
    assert result.descriptor.data_multihash is None


def test_register_artifact_rejects_non_cgid(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_coalesced_upload(monkeypatch)
    client = _FakeDaemonCtl()
    tensors = {"a": torch.zeros((2, 2), dtype=torch.float32)}
    options = RegisterArtifactOptions(plan=PlanType.VRAM_COALESCED, lease_in_place=False)

    with pytest.raises(TensorCastError) as exc_info:
        _register_artifact_core(
            artifact=tensors,
            options=options,
            device_id=0,
            ttl_ms=None,
            client_artifact_id="mi2:not-allowed",
            force_lease_in_place=False,
            prevalidate_disk=False,
            client=client,
            daemon_address="fake-daemon",
        )

    assert "cgid" in str(exc_info.value)
