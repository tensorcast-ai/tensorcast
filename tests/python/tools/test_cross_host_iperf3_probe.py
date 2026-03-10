# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_probe_module() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "examples" / "cross_host" / "cross_host_iperf3_probe.py"
    spec = importlib.util.spec_from_file_location("cross_host_iperf3_probe", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load cross_host_iperf3_probe.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_iperf_gibps_prefers_sum_received() -> None:
    probe = _load_probe_module()
    payload = {
        "end": {
            "sum_received": {"bits_per_second": 8.0 * (1024**3) * 2.5},
            "sum_sent": {"bits_per_second": 8.0 * (1024**3) * 2.0},
        }
    }
    gibps = probe._extract_iperf_gibps(payload)
    assert abs(gibps - 2.5) < 1e-6


def test_compute_summary_uses_pair_floor_p50_for_single_link_ref() -> None:
    probe = _load_probe_module()
    pair1 = probe.PairProbeResult(
        getter_name="get1",
        getter_process="p1",
        getter_ip="100.1.1.1",
        seed_to_getter=probe.ProbeDirectionResult(
            direction="seed_to_getter",
            server_process="p1",
            client_process="seed",
            server_host="100.1.1.1",
            port=63900,
            gibps=2.2,
            error=None,
            client_bits_per_second=None,
            server_bits_per_second=None,
        ),
        getter_to_seed=probe.ProbeDirectionResult(
            direction="getter_to_seed",
            server_process="seed",
            client_process="p1",
            server_host="100.1.1.10",
            port=63901,
            gibps=2.0,
            error=None,
            client_bits_per_second=None,
            server_bits_per_second=None,
        ),
    )
    pair2 = probe.PairProbeResult(
        getter_name="get2",
        getter_process="p2",
        getter_ip="100.1.1.2",
        seed_to_getter=probe.ProbeDirectionResult(
            direction="seed_to_getter",
            server_process="p2",
            client_process="seed",
            server_host="100.1.1.2",
            port=63902,
            gibps=1.8,
            error=None,
            client_bits_per_second=None,
            server_bits_per_second=None,
        ),
        getter_to_seed=probe.ProbeDirectionResult(
            direction="getter_to_seed",
            server_process="seed",
            client_process="p2",
            server_host="100.1.1.10",
            port=63903,
            gibps=2.4,
            error=None,
            client_bits_per_second=None,
            server_bits_per_second=None,
        ),
    )

    summary = probe._compute_summary([pair1, pair2])
    # pair floors are [1.8, 2.0], and current percentile index policy chooses the
    # lower middle value for p50 in a 2-sample series.
    assert summary["single_link_ref_gibps"] == 1.8
    assert summary["bidirectional_floor_gibps"] == 1.8
