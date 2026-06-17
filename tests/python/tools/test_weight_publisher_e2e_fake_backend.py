#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast.tools import weight_publisher_e2e as e2e


def test_materialization_device_auto_uses_fake_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_CUDA_BACKEND", "fake")
    assert e2e._materialization_device("auto") == "cuda:0"


def test_materialization_device_explicit_value_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_CUDA_BACKEND", "fake")
    assert e2e._materialization_device("cpu") == "cpu"


def test_check_options_enforces_local_only_under_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_CUDA_BACKEND", "fake")
    publisher = e2e.WeightUpdatePublisher.__new__(e2e.WeightUpdatePublisher)
    assert publisher._check_options() == e2e.GetArtifactOptions(source="local_only")


def test_pre_publish_trim_is_enforced_with_keep_last_minus_one(tmp_path: Path) -> None:
    publisher = e2e.WeightUpdatePublisher(
        model_name="m",
        key_template="k:{model_name}:v{weight_version}",
        keep_last=2,
        history_path=tmp_path / "history.json",
        run_root=tmp_path,
        check_poll_interval_s=0.1,
        check_timeout_s=1.0,
        strict_drop_check=False,
        payload_mode="tp_ranked",
        tp_world_size=8,
        tp_total_bytes=1024,
        publish_device="cpu",
    )
    assert publisher._publisher._effective_pre_publish_keep_last() == 1
