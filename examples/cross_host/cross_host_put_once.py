#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import contextlib
import json
import time

import torch

import tensorcast as tc
from tensorcast import startup

DTYPE_BY_NAME: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _sync_if_needed(device: str, enabled: bool) -> None:
    if not enabled:
        return
    if device.startswith("cuda:") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _bytes_to_gib_per_sec(num_bytes: int, sec: float) -> float:
    if num_bytes <= 0 or sec <= 0:
        return 0.0
    return float(num_bytes) / float(1024**3) / float(sec)


def _build_payload(
    *,
    total_bytes: int,
    tensor_count: int,
    dtype: torch.dtype,
    device: str,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int]:
    element_size = int(torch.empty((), dtype=dtype).element_size())
    total_elems = total_bytes // element_size
    if total_elems < tensor_count:
        raise ValueError(
            "Requested payload too small for tensor_count and dtype; "
            f"total_bytes={total_bytes}, tensor_count={tensor_count}, dtype={dtype}"
        )

    base_elems = total_elems // tensor_count
    extra = total_elems % tensor_count

    payload: dict[str, torch.Tensor] = {}
    generated_elems = 0
    floating = bool(torch.empty((), dtype=dtype).is_floating_point())
    for idx in range(tensor_count):
        elems = base_elems + (1 if idx < extra else 0)
        generated_elems += elems
        name = f"tensor_{idx:04d}"
        tensor = torch.empty((int(elems),), dtype=dtype, device=device)
        if floating:
            tensor.fill_(float((seed + idx) % 37) + 0.25)
            if elems > 0:
                tensor[0] = float(seed)
        else:
            tensor.fill_(int((seed + idx) % 97))
            if elems > 0:
                tensor[0] = int(seed)
        payload[name] = tensor

    effective_bytes = generated_elems * element_size
    return payload, int(effective_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-host helper: run one put from a local worker process."
    )
    parser.add_argument("--daemon", required=True, help="Put daemon address host:port")
    parser.add_argument("--key", required=True, help="Benchmark artifact key")
    parser.add_argument("--size-mib", type=int, default=2048)
    parser.add_argument("--tensor-count", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=tuple(DTYPE_BY_NAME.keys()),
        default="float16",
    )
    parser.add_argument("--put-device", default="cuda:0")
    parser.add_argument("--put-policy", default="pinned")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--sync-cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    dtype = DTYPE_BY_NAME[str(args.dtype)]
    requested_bytes = int(args.size_mib) * 1024 * 1024
    payload, effective_bytes = _build_payload(
        total_bytes=requested_bytes,
        tensor_count=int(args.tensor_count),
        dtype=dtype,
        device=str(args.put_device),
        seed=int(args.seed),
    )

    startup.init(mode="connect", address=str(args.daemon))
    try:
        _sync_if_needed(str(args.put_device), bool(args.sync_cuda))
        put_start = time.perf_counter()
        registered = tc.put(
            payload,
            key=str(args.key),
            policy=str(args.put_policy),
        )
        _sync_if_needed(str(args.put_device), bool(args.sync_cuda))
        put_end = time.perf_counter()
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()

    put_sec = float(put_end - put_start)
    output = {
        "key": str(args.key),
        "artifact_id": str(registered.artifact_id),
        "requested_bytes": int(requested_bytes),
        "effective_bytes": int(effective_bytes),
        "put_sec": put_sec,
        "put_gibps": _bytes_to_gib_per_sec(int(effective_bytes), put_sec),
        "put_policy": str(args.put_policy),
        "put_device": str(args.put_device),
    }
    print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
