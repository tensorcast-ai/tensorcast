#  Copyright (c) 2026, TensorCast Team.

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import torch
from torch import nn

from tensorcast.pytorch.trace_capture import TraceActivation, trace_model_load
from tensorcast.artifact_runtime.recipe.materialization import apply_copy_plan
from tensorcast.artifact_runtime.recipe.trace_ir import MultiRange, Range


@dataclass(frozen=True)
class _TensorMeta:
    dtype: torch.dtype
    shape: tuple[int, ...]


def test_trace_capture_handles_param_data_and_const_fill() -> None:

    class _Dummy(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((10,), device="meta"))

        def load_weights(self, weights) -> None:
            _, tensor = next(iter(weights))
            narrowed = tensor.narrow(0, 0, 5)
            self.w[:narrowed.shape[0]].data.copy_(narrowed)
            self.w[narrowed.shape[0]:].data.fill_(0)

    trace_plan = trace_model_load(
        _Dummy(),
        ["x"],
        {"x": _TensorMeta(dtype=torch.float16, shape=(10,))},
    )

    assert len(trace_plan.copy_plan) == 2
    assert trace_plan.expected_src_names == {"x"}
    assert trace_plan.expected_dst_names == {"w"}
    assert trace_plan.tensorcast_slices["x"] == Range(dim=0, start=0, end=5)

    serving = {"w": torch.full((10,), 123, dtype=torch.float16)}
    apply_copy_plan(trace_plan, {"x": torch.arange(5, dtype=torch.float16)}, serving)
    assert torch.equal(
        serving["w"],
        torch.tensor([0, 1, 2, 3, 4, 0, 0, 0, 0, 0], dtype=torch.float16),
    )


def test_trace_capture_supports_fill_from_ckpt_scalar() -> None:

    class _Dummy(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((4,), device="meta"))

        def load_weights(self, weights) -> None:
            _, tensor = next(iter(weights))
            self.w.data.fill_(tensor.item())

    trace_plan = trace_model_load(
        _Dummy(),
        ["x"],
        {"x": _TensorMeta(dtype=torch.float32, shape=())},
    )

    serving = {"w": torch.empty((4,), dtype=torch.float32)}
    apply_copy_plan(trace_plan, {"x": torch.tensor(3.5, dtype=torch.float32)}, serving)
    assert torch.equal(serving["w"], torch.full((4,), 3.5, dtype=torch.float32))


def test_trace_capture_keeps_multirange_for_real_multi_dim_slice() -> None:

    class _Dummy(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((8, 4), device="meta"))

        def load_weights(self, weights) -> None:
            _, tensor = next(iter(weights))
            src = tensor[3].narrow(0, 1, 2)
            dst = self.w[3].data.narrow(0, 1, 2)
            dst.copy_(src)

    trace_plan = trace_model_load(
        _Dummy(),
        ["x"],
        {"x": _TensorMeta(dtype=torch.float16, shape=(8, 4))},
    )

    entry = trace_plan.copy_plan[0]
    assert isinstance(entry.ckpt_range, MultiRange)
    assert "x" not in trace_plan.tensorcast_slices


def test_trace_capture_uses_framework_activation() -> None:
    active_var = contextvars.ContextVar("trace_active", default=False)

    class _Dummy(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.zeros((4,)))

        def load_weights(self, weights) -> None:

            def worker(weight_pair) -> None:
                _, tensor = weight_pair
                self.w.data.copy_(tensor)

            if active_var.get():
                for weight_pair in weights:
                    worker(weight_pair)
                return

            with ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(worker, weight_pair) for weight_pair in weights
                ]
                for future in futures:
                    future.result()

    trace_plan = trace_model_load(
        _Dummy(),
        ["x"],
        {"x": _TensorMeta(dtype=torch.float32, shape=(4,))},
        activation=TraceActivation(
            set_active=lambda: active_var.set(True),
            reset_active=active_var.reset,
        ),
    )

    assert len(trace_plan.copy_plan) == 1
    assert trace_plan.expected_src_names == {"x"}
    assert trace_plan.expected_dst_names == {"w"}
