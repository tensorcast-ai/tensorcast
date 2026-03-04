#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest
import torch

from tensorcast.api.errors import ArtifactError
from tensorcast.api.store.registration import RegistrationPipeline


def test_require_put_tensors_rejects_cpu_without_cuda_or_fake_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TENSORCAST_CUDA_BACKEND", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ArtifactError, match="put requires CUDA to be available"):
        RegistrationPipeline._require_put_tensors({"w": torch.ones(1, dtype=torch.float32)})


def test_require_put_tensors_allows_cpu_when_fake_backend_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TENSORCAST_CUDA_BACKEND", "fake")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    RegistrationPipeline._require_put_tensors({"w": torch.ones(1, dtype=torch.float32)})
