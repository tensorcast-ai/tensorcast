#  Copyright (c) 2026, TensorCast Team.

import argparse
import json
from typing import Any

import tensorcast as tc
from tensorcast._c_ext import compute_view_index_bytes
from tensorcast.common.selection_identity import compute_logical_layout_hash

SUPPORTED_TP_VALUES = (1, 2, 4, 8)
DEFAULT_KEY = "model:qwen3-0.6b-base:v123"
DEFAULT_DEVICE = "cuda:0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=str, default=DEFAULT_KEY)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--tp",
        type=int,
        choices=SUPPORTED_TP_VALUES,
        default=1,
        help="Tensor parallel size. Supported values: 1/2/4/8.",
    )
    parser.add_argument(
        "--tp-rank",
        type=int,
        default=None,
        help="Tensor parallel rank index for view slicing. Omit to load all ranks.",
    )
    parser.add_argument(
        "--use-view",
        action="store_true",
        help="Force view path when tp=1.",
    )
    return parser.parse_args()


def pick_split_dim(shape: tuple[int, ...], tp: int) -> int | None:
    if len(shape) < 2:
        return None
    candidate_dims = [
        dim for dim, extent in enumerate(shape) if extent >= tp and extent % tp == 0
    ]
    if not candidate_dims:
        return None
    max_extent = max(shape[dim] for dim in candidate_dims)
    for dim in reversed(candidate_dims):
        if shape[dim] == max_extent:
            return dim
    return None


def build_tp_slices(
    handle: Any,
    *,
    tp: int,
    tp_rank: int,
) -> dict[str, list[tuple[int, slice]]]:
    if tp == 1:
        return {}

    desc = handle.describe()
    slices: dict[str, list[tuple[int, slice]]] = {}
    for name in handle.tensor_names:
        shape = tuple(int(dim) for dim in desc.tensor_metas[name].shape)
        split_dim = pick_split_dim(shape, tp)
        if split_dim is None:
            continue
        extent = shape[split_dim]
        shard_size = extent // tp
        shard_start = tp_rank * shard_size
        shard_stop = shard_start + shard_size
        slices[name] = [(split_dim, slice(shard_start, shard_stop, None))]
    return slices


def _normalized_view_ops(view_handle: Any) -> dict[str, list[dict[str, int | str]]]:
    view_spec = view_handle._view_spec
    if view_spec is None or view_spec.proto is None:
        return {}
    normalized: dict[str, list[dict[str, int | str]]] = {}
    for name, ops in view_spec.proto.tensors.items():
        op_list: list[dict[str, int | str]] = []
        for op in ops.ops:
            if op.HasField("narrow"):
                op_list.append(
                    {
                        "type": "narrow",
                        "dim": int(op.narrow.dim),
                        "start": int(op.narrow.start),
                        "length": int(op.narrow.length),
                    }
                )
            elif op.HasField("transpose"):
                op_list.append(
                    {
                        "type": "transpose",
                        "dim0": int(op.transpose.dim0),
                        "dim1": int(op.transpose.dim1),
                    }
                )
        if op_list:
            normalized[str(name)] = op_list
    return normalized


def build_selection_debug_info(
    *,
    handle: Any,
    view_handle: Any,
    tensor_slices: dict[str, list[tuple[int, slice]]],
    tp: int,
    tp_rank: int,
) -> dict[str, object]:
    view_metadata = view_handle._view_metadata
    view_index_hint = bytes(
        view_metadata.view_index_bytes if view_metadata is not None else b""
    )
    tensor_names = tuple(
        view_metadata.tensor_names if view_metadata is not None else handle.tensor_names
    )
    canonical_index_bytes = bytes(handle._canonical_index_bytes or b"")

    debug: dict[str, object] = {
        "tp": int(tp),
        "tp_rank": int(tp_rank),
        "total_tensors": int(len(handle.tensor_names)),
        "split_tensors": int(len(tensor_slices)),
        "split_tensor_sample": [
            {
                "name": name,
                "dim": int(spec[0][0]),
                "start": int(spec[0][1].start or 0),
                "stop": int(spec[0][1].stop or 0),
            }
            for name, spec in list(tensor_slices.items())[:5]
        ],
        "tensor_names_count": int(len(tensor_names)),
        "view_index_hint_len": int(len(view_index_hint)),
        "canonical_index_len": int(len(canonical_index_bytes)),
    }

    if view_index_hint:
        debug["view_index_hint_hash"] = compute_logical_layout_hash(
            index_bytes=view_index_hint,
            needs_view_index=True,
        ).hex()

    if not canonical_index_bytes:
        return debug

    normalized_ops = _normalized_view_ops(view_handle)
    if not normalized_ops and not tensor_names:
        return debug

    recomputed_payload = compute_view_index_bytes(
        canonical_index_bytes,
        normalized_ops,
        list(tensor_names) if tensor_names else None,
    )
    recomputed_view_index = bytes(recomputed_payload["view_index_bytes"])
    debug["recomputed_view_index_len"] = int(len(recomputed_view_index))
    debug["recomputed_view_index_hash"] = compute_logical_layout_hash(
        index_bytes=recomputed_view_index,
        needs_view_index=True,
    ).hex()
    if view_index_hint:
        debug["view_index_hint_matches_recomputed"] = bool(
            view_index_hint == recomputed_view_index
        )

    return debug


def main() -> None:
    args = parse_args()

    tc.init(mode="connect")
    handle = tc.artifact(key=args.key)
    total_tensors = len(handle.tensor_names)

    if args.tp == 1:
        if args.tp_rank not in {None, 0}:
            raise ValueError(f"tp-rank must be 0 when tp=1, got {args.tp_rank}")
        if not args.use_view:
            loaded = handle.tensor_dict(device=args.device)
            print(loaded)
            return

    if args.tp_rank is not None and (args.tp_rank < 0 or args.tp_rank >= args.tp):
        raise ValueError(f"tp-rank must be in [0, {args.tp - 1}], got {args.tp_rank}")

    if args.tp_rank is not None:
        tp_ranks = [args.tp_rank]
    elif args.tp == 1:
        tp_ranks = [0]
    else:
        tp_ranks = list(range(args.tp))

    loaded_by_rank: dict[int, Any] = {}
    for tp_rank in tp_ranks:
        tensor_slices = build_tp_slices(handle, tp=args.tp, tp_rank=tp_rank)
        print(
            f"using view slicing: tp={args.tp}, tp_rank={tp_rank}, "
            f"split_tensors={len(tensor_slices)}, total_tensors={total_tensors}"
        )
        view_handle = handle.view(slices=tensor_slices)
        try:
            loaded_by_rank[tp_rank] = view_handle.tensor_dict(device=args.device)
        except Exception:
            debug_info = build_selection_debug_info(
                handle=handle,
                view_handle=view_handle,
                tensor_slices=tensor_slices,
                tp=args.tp,
                tp_rank=tp_rank,
            )
            print("selection debug info:")
            print(json.dumps(debug_info, indent=2, sort_keys=True))
            raise

    if len(tp_ranks) == 1:
        print(loaded_by_rank[tp_ranks[0]])
    else:
        print(loaded_by_rank)


if __name__ == "__main__":
    main()
