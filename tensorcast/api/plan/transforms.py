#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class TransformSpec:
    name: str
    args: Mapping[str, str | int]
    layout_hash: str | None = None

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "args": {k: self.args[k] for k in sorted(self.args)},
            "layout_hash": self.layout_hash or "",
        }
        digest = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()
        return f"tc.transform.v1:{digest}"

    def to_proto(self):
        from tensorcast.proto.plan.v1 import plan_pb2

        proto = plan_pb2.TransformSpec(name=str(self.name))
        if self.layout_hash is not None:
            proto.layout_hash = str(self.layout_hash)
        for key in sorted(self.args):
            value = self.args[key]
            arg = proto.args.add()
            arg.key = str(key)
            if isinstance(value, int):
                arg.int_value = int(value)
            else:
                arg.string_value = str(value)
        return proto

    @staticmethod
    def from_proto(proto) -> "TransformSpec":
        args: dict[str, str | int] = {}
        for arg in proto.args:
            kind = arg.WhichOneof("value")
            if kind == "int_value":
                args[str(arg.key)] = int(arg.int_value)
            else:
                args[str(arg.key)] = str(arg.string_value)
        layout_hash = str(proto.layout_hash) if proto.layout_hash else None
        return TransformSpec(name=str(proto.name), args=args, layout_hash=layout_hash)


__all__ = [
    "TransformSpec",
]
