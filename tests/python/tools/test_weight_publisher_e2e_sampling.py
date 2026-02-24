#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import pytest
import torch

from tensorcast.tools.weight_publisher_e2e import _build_sample_indices


def test_build_sample_indices_large_total_stays_in_bounds() -> None:
    total = 2_147_483_644
    points = 4096

    indices = _build_sample_indices(
        total=total,
        points=points,
        device=torch.device("cpu"),
    )

    assert tuple(indices.shape) == (points,)
    assert int(indices.min().item()) == 0
    assert int(indices.max().item()) == total - 1
    assert bool(torch.all(indices[1:] >= indices[:-1]).item())


def test_build_sample_indices_single_point() -> None:
    indices = _build_sample_indices(
        total=123,
        points=1,
        device=torch.device("cpu"),
    )
    assert indices.tolist() == [0]


@pytest.mark.parametrize(
    ("total", "points", "expected"),
    [
        (10, 2, [0, 9]),
        (10, 4, [0, 3, 6, 9]),
        (11, 4, [0, 3, 6, 10]),
    ],
)
def test_build_sample_indices_expected_integer_spacing(
    total: int,
    points: int,
    expected: list[int],
) -> None:
    indices = _build_sample_indices(
        total=total,
        points=points,
        device=torch.device("cpu"),
    )
    assert indices.tolist() == expected


@pytest.mark.parametrize(
    ("total", "points"),
    [
        (0, 1),
        (-1, 1),
        (1, 0),
        (1, -1),
    ],
)
def test_build_sample_indices_rejects_non_positive_inputs(total: int, points: int) -> None:
    with pytest.raises(ValueError):
        _build_sample_indices(
            total=total,
            points=points,
            device=torch.device("cpu"),
        )
