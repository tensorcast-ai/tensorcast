#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from tensorcast.types import PureTransformPublicationSpec


@dataclass(frozen=True, slots=True)
class TransformSpec:
    name: str
    args: Mapping[str, str | int]
    layout_hash: str | None = None
    publication_spec: PureTransformPublicationSpec | None = None

    def fingerprint(self) -> str:
        payload = {
            "name": self.name,
            "args": {k: self.args[k] for k in sorted(self.args)},
            "layout_hash": self.layout_hash or "",
            "publication_spec": (
                self.publication_spec.to_proto().SerializeToString().hex()
                if self.publication_spec is not None
                else ""
            ),
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
        if self.publication_spec is not None:
            proto.publication_spec.CopyFrom(self.publication_spec.to_proto())
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
        publication_spec = (
            PureTransformPublicationSpec.from_proto(proto.publication_spec)
            if proto.HasField("publication_spec")
            else None
        )
        return TransformSpec(
            name=str(proto.name),
            args=args,
            layout_hash=layout_hash,
            publication_spec=publication_spec,
        )


__all__ = [
    "TransformSpec",
]
