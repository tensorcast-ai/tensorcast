#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from tensorcast.api._config import PlanType, RegisterArtifactOptions
from tensorcast.api._register import (
    _prepare_build,
    _stable_dram_feed_upload_workers,
    _StableDramUploader,
    make_plan_model,
)
from tensorcast.types import StableDramHandshake


class _FakeDaemonCtl:
    def __init__(self) -> None:
        self.calls: list[dict[str, int]] = []

    def feed_register_artifact_view_spans(
        self,
        registration_id: str,
        spans: Any,
        *,
        chunk_bytes: int | None = None,
        timeout_s: float | None = None,
    ) -> bool:
        del timeout_s
        resolved_chunk = int(chunk_bytes) if chunk_bytes is not None else -1
        for base_offset, data in spans:
            self.calls.append(
                {
                    "registration_id": registration_id,
                    "nbytes": int(memoryview(data).nbytes),
                    "chunk_bytes": resolved_chunk,
                    "base_offset": int(base_offset),
                }
            )
        return True


@dataclass
class _FakeRegisteredArtifact:
    registration_id: str
    client: Any


def test_make_plan_model_allows_stable_dram_without_gpu_staging() -> None:
    options = RegisterArtifactOptions(
        plan=PlanType.DRAM_STABLE,
        stage_on_gpu=False,
        release_gpu_on_commit=False,
    )
    plan = make_plan_model(options)
    assert plan.kind == "dram_stable"
    assert plan.stage_on_gpu is False
    assert plan.release_gpu_on_commit is False


def test_stable_dram_uploader_streams_cpu_chunks_when_no_staging_handle() -> None:
    artifact = {
        "a": torch.arange(8, dtype=torch.float32),
        "b": torch.arange(6, dtype=torch.float16).reshape(2, 3),
    }
    ctx, layout, _ = _prepare_build(artifact, device_id=0)
    # Source index offsets are not canonical upload offsets; force non-canonical
    # values to ensure uploader always uses layout offsets for stream base_offset.
    for name, (_, size_bytes) in tuple(ctx.tensor_source_index.items()):
        ctx.tensor_source_index[name] = (140_000_000_000_000 + int(size_bytes), size_bytes)

    offsets_for_device = layout.offsets.get(0, {})
    expected = sorted(
        (int(offsets_for_device[name]), int(ctx.tensor_source_index[name][1]))
        for name in artifact
    )

    fake_ctl = _FakeDaemonCtl()
    handle = _FakeRegisteredArtifact(registration_id="reg-stable-stream", client=fake_ctl)
    uploader = _StableDramUploader()
    handshake = StableDramHandshake(staging_cuda_ipc_handle=b"")

    result = uploader.upload(
        artifact=artifact,
        ctx=ctx,
        layout=layout,
        handle=handle,
        handshake=handshake,
        cancel_event=None,
    )

    assert result is artifact
    got = sorted((call["base_offset"], call["nbytes"]) for call in fake_ctl.calls)
    assert got == expected
    assert all(call["registration_id"] == "reg-stable-stream" for call in fake_ctl.calls)


def test_stable_dram_uploader_uses_1mb_chunk_with_concurrent_workers(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv("TENSORCAST_FEED_VIEW_UPLOAD_WORKERS", "8")
    monkeypatch.delenv("TENSORCAST_FEED_VIEW_CHUNK_BYTES", raising=False)

    artifact = {
        "a": torch.arange(8, dtype=torch.float32),
        "b": torch.arange(6, dtype=torch.float16).reshape(2, 3),
    }
    ctx, layout, _ = _prepare_build(artifact, device_id=0)

    fake_ctl = _FakeDaemonCtl()
    handle = _FakeRegisteredArtifact(registration_id="reg-stable-stream", client=fake_ctl)
    uploader = _StableDramUploader()
    handshake = StableDramHandshake(staging_cuda_ipc_handle=b"")

    result = uploader.upload(
        artifact=artifact,
        ctx=ctx,
        layout=layout,
        handle=handle,
        handshake=handshake,
        cancel_event=None,
    )

    assert result is artifact
    assert fake_ctl.calls
    assert all(call["chunk_bytes"] == 1 * 1024 * 1024 for call in fake_ctl.calls)


def test_stable_dram_upload_workers_auto_tune_without_env(monkeypatch: Any) -> None:
    monkeypatch.delenv("TENSORCAST_FEED_VIEW_UPLOAD_WORKERS", raising=False)
    assert _stable_dram_feed_upload_workers(total_bytes=1 * 1024**3, tensor_count=6) == 1
    assert _stable_dram_feed_upload_workers(total_bytes=8 * 1024**3, tensor_count=6) == 4
    assert _stable_dram_feed_upload_workers(total_bytes=40 * 1024**3, tensor_count=6) == 8
