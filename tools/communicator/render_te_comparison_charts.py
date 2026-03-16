#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Render communicator vs Transfer Engine comparison charts as SVG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path("/data/workspace/tensorcast-280")
IMAGE_DIR = REPO_ROOT / "docs/benchmarks/image"
DATA_DIR = REPO_ROOT / "docs/benchmarks/data"

COMM_CASES = {
    "map8_id": Path("/data/tc/comm-map8-id-rerun-attempt1-043514/result.json"),
    "map8_within_swap": Path("/data/tc/comm-map8-within-swap-clean-attempt1-050919/result.json"),
    "map8_half_swap": Path("/data/tc/comm-map8-half-swap-attempt2-052059/result.json"),
    "map8_target_bad_local": Path("/data/tc/comm-map8-target-bad-local-attempt1-052344/result.json"),
    "single_big_read": Path("/data/tc/comm-tune-0360-0496-bigread-t1-qp4-latfix-attempt1-025158/result.json"),
}

MOONCAKE = {
    "full8_raw_gbps": 1568.75,
    "full8_te_best_gbps": 1482.64,
    "map8_identity_gbps": 35.22 * 8.0,
    "map8_within_swap_gbps": 41.36 * 8.0,
    "map8_half_swap_gbps": 32.18 * 8.0,
    "map8_target_bad_local_gbps": 34.88 * 8.0,
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def communicator_case_metrics(path: Path) -> dict[str, float]:
    payload = load_json(path)
    lanes = payload.get("lanes", [])
    if lanes:
        total_bytes = sum(int(lane["summary"]["bytes"]) for lane in lanes)
        max_lane_wall_us = max(float(lane["summary"]["wall_us"]) for lane in lanes)
        lane_count = max(1, len(lanes))
        data_plane_GBps = (float(total_bytes) / 1.0e9) / (max_lane_wall_us / 1.0e6)
        aggregate = payload["aggregate"]
        case_wall_GBps = float(aggregate.get("bw_GBps_case_wall", aggregate.get("bw_gbps_case_wall", 0.0)))
        return {
            "lane_count": float(lane_count),
            "total_bytes": float(total_bytes),
            "data_plane_GBps": data_plane_GBps,
            "data_plane_gbps": data_plane_GBps * 8.0,
            "case_wall_GBps": case_wall_GBps,
            "case_wall_gbps": case_wall_GBps * 8.0,
            "per_lane_GBps": data_plane_GBps / float(lane_count),
            "per_lane_gbps": (data_plane_GBps * 8.0) / float(lane_count),
        }
    summary = payload["parsed"]["summary"]
    data_plane_GBps = float(summary.get("bw_GBps", summary["bw_gbps"]))
    return {
        "data_plane_GBps": data_plane_GBps,
        "data_plane_gbps": data_plane_GBps * 8.0,
        "case_wall_GBps": data_plane_GBps,
        "case_wall_gbps": data_plane_GBps * 8.0,
        "per_lane_GBps": data_plane_GBps,
        "per_lane_gbps": data_plane_GBps * 8.0,
    }


def svg_text(x: float, y: float, text: str, size: int = 14, weight: str = "normal", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}">{text}</text>'
    )


def render_bar_chart(
    *,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    unit: str,
    output_path: Path,
    note: str | None = None,
) -> None:
    width = 980
    height = 620
    left = 90
    top = 110
    chart_w = 820
    chart_h = 380
    bottom = top + chart_h
    max_value = max(values) * 1.12 if values else 1.0
    tick_count = 5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#faf8f2"/>',
        svg_text(width / 2, 36, title, size=26, weight="bold", anchor="middle"),
        svg_text(width / 2, 62, subtitle, size=14, anchor="middle"),
    ]
    if note:
        parts.append(svg_text(width / 2, 84, note, size=12, anchor="middle"))

    for i in range(tick_count + 1):
        value = max_value * i / tick_count
        y = bottom - (chart_h * i / tick_count)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#d6d1c4" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 5, f"{value:.0f}", size=12, anchor="end"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left + chart_w}" y2="{bottom}" stroke="#222" stroke-width="2"/>')
    parts.append(svg_text(left - 56, top - 20, unit, size=13))

    bar_width = chart_w / max(1, len(labels)) * 0.58
    gap = chart_w / max(1, len(labels))
    for idx, (label, value, color) in enumerate(zip(labels, values, colors)):
        x = left + gap * idx + (gap - bar_width) / 2
        bar_h = 0 if max_value <= 0 else chart_h * value / max_value
        y = bottom - bar_h
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>')
        parts.append(svg_text(x + bar_width / 2, y - 8, f"{value:.2f}", size=12, weight="bold", anchor="middle"))
        for line_idx, segment in enumerate(label.split("\n")):
            parts.append(svg_text(x + bar_width / 2, bottom + 24 + line_idx * 16, segment, size=12, anchor="middle"))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_grouped_bar_chart(
    *,
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[tuple[str, list[float], str]],
    output_path: Path,
    unit: str,
    note: str | None = None,
) -> None:
    width = 1100
    height = 650
    left = 90
    top = 110
    chart_w = 900
    chart_h = 400
    bottom = top + chart_h
    max_value = max(max(values) for _, values, _ in series) * 1.12
    tick_count = 5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f5ef"/>',
        svg_text(width / 2, 36, title, size=26, weight="bold", anchor="middle"),
        svg_text(width / 2, 62, subtitle, size=14, anchor="middle"),
    ]
    if note:
        parts.append(svg_text(width / 2, 84, note, size=12, anchor="middle"))

    for i in range(tick_count + 1):
        value = max_value * i / tick_count
        y = bottom - (chart_h * i / tick_count)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#d6d1c4" stroke-width="1"/>'
        )
        parts.append(svg_text(left - 12, y + 5, f"{value:.2f}", size=12, anchor="end"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222" stroke-width="2"/>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{left + chart_w}" y2="{bottom}" stroke="#222" stroke-width="2"/>')
    parts.append(svg_text(left - 56, top - 20, unit, size=13))

    group_width = chart_w / max(1, len(categories))
    inner_gap = 18
    total_bar_width = group_width * 0.68
    single_bar_width = (total_bar_width - inner_gap * (len(series) - 1)) / len(series)

    for cat_idx, category in enumerate(categories):
        group_x = left + group_width * cat_idx + (group_width - total_bar_width) / 2
        for series_idx, (_, values, color) in enumerate(series):
            value = values[cat_idx]
            x = group_x + series_idx * (single_bar_width + inner_gap)
            bar_h = chart_h * value / max_value
            y = bottom - bar_h
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{single_bar_width:.1f}" height="{bar_h:.1f}" rx="4" fill="{color}"/>'
            )
            parts.append(svg_text(x + single_bar_width / 2, y - 8, f"{value:.2f}", size=11, anchor="middle"))
        for line_idx, segment in enumerate(category.split("\n")):
            parts.append(svg_text(left + group_width * cat_idx + group_width / 2, bottom + 24 + line_idx * 16, segment, size=12, anchor="middle"))

    legend_x = left + chart_w - 200
    legend_y = 92
    for idx, (name, _, color) in enumerate(series):
        y = legend_y + idx * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="16" height="16" fill="{color}" rx="2"/>')
        parts.append(svg_text(legend_x + 24, y, name, size=12))

    parts.append("</svg>")
    output_path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    comm_metrics = {name: communicator_case_metrics(path) for name, path in COMM_CASES.items()}

    comparison = {
        "mooncake": MOONCAKE,
        "communicator": comm_metrics,
    }
    (DATA_DIR / "20260316-communicator-vs-te-comparison.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    render_bar_chart(
        title="Full-8 Framework Throughput Reference",
        subtitle="Use as a framework-scale reference, not strict apples-to-apples",
        labels=["Raw RDMA\nMooncake", "Transfer Engine\nfull-8 best", "Communicator\nfull-8 best"],
        values=[
            MOONCAKE["full8_raw_gbps"],
            MOONCAKE["full8_te_best_gbps"],
            max(
                comm_metrics["map8_id"]["data_plane_gbps"],
                comm_metrics["map8_within_swap"]["data_plane_gbps"],
                comm_metrics["map8_half_swap"]["data_plane_gbps"],
            ),
        ],
        colors=["#204b57", "#7f5539", "#2a7f62"],
        unit="Gbps",
        note="Mooncake full-8 best is write; communicator full-8 best is read",
        output_path=IMAGE_DIR / "communicator_vs_te_full8_ceiling_20260316.svg",
    )

    render_grouped_bar_chart(
        title="Full-8 Mapping Sensitivity",
        subtitle="Normalized to identity = 1.0 on each framework",
        categories=["identity", "within-half\nswap", "cross-half\nswap", "target\nbad local"],
        series=[
            (
                "Mooncake TE",
                [
                    1.0,
                    MOONCAKE["map8_within_swap_gbps"] / MOONCAKE["map8_identity_gbps"],
                    MOONCAKE["map8_half_swap_gbps"] / MOONCAKE["map8_identity_gbps"],
                    MOONCAKE["map8_target_bad_local_gbps"] / MOONCAKE["map8_identity_gbps"],
                ],
                "#7f5539",
            ),
            (
                "Communicator",
                [
                    1.0,
                    comm_metrics["map8_within_swap"]["data_plane_gbps"] / comm_metrics["map8_id"]["data_plane_gbps"],
                    comm_metrics["map8_half_swap"]["data_plane_gbps"] / comm_metrics["map8_id"]["data_plane_gbps"],
                    comm_metrics["map8_target_bad_local"]["data_plane_gbps"] / comm_metrics["map8_id"]["data_plane_gbps"],
                ],
                "#2a7f62",
            ),
        ],
        unit="normalized",
        note="Communicator matches the local-affinity rule, but shows weaker remote permutation sensitivity",
        output_path=IMAGE_DIR / "communicator_vs_te_mapping_norm_20260316.svg",
    )

    render_bar_chart(
        title="Per-lane Reference",
        subtitle="Single-object and full-8 lane-level reference values",
        labels=[
            "TE full-8\nper-lane avg",
            "Comm full-8\nper-lane avg",
            "Comm single\nbig read",
            "Comm bad-local\nper-lane avg",
        ],
        values=[
            (MOONCAKE["full8_te_best_gbps"] / 8.0) / 8.0,
            max(
                comm_metrics["map8_id"]["per_lane_GBps"],
                comm_metrics["map8_within_swap"]["per_lane_GBps"],
                comm_metrics["map8_half_swap"]["per_lane_GBps"],
            ),
            comm_metrics["single_big_read"]["per_lane_GBps"],
            comm_metrics["map8_target_bad_local"]["per_lane_GBps"],
        ],
        colors=["#7f5539", "#2a7f62", "#204b57", "#a33f2f"],
        unit="GB/s",
        note="TE value is write-path full-8 average; communicator single-big-read is strict direct read_tensor",
        output_path=IMAGE_DIR / "communicator_vs_te_per_lane_20260316.svg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
