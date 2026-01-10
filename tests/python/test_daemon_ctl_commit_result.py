#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

from types import SimpleNamespace

from tensorcast.daemon_ctl import DaemonCtl
from tensorcast.proto.daemon.v2 import store_daemon_pb2


def test_commit_registered_artifact_maps_local_stable_tier(monkeypatch) -> None:
    ctl = DaemonCtl("127.0.0.1:9")  # address unused (unary call patched)
    ctl.stub = SimpleNamespace(CommitRegisteredArtifact=object())  # type: ignore[assignment]

    resp = store_daemon_pb2.CommitRegisteredArtifactResponse()
    resp.artifact_descriptor.artifact_id = "mi2:index:data"
    resp.artifact_descriptor.total_size = 8
    resp.artifact_descriptor.id_kind = 1
    resp.local_stable_tier.status = store_daemon_pb2.LOCAL_STABLE_TIER_STATUS_READY
    resp.local_stable_tier.message = "ok"

    def fake_unary_call(*_args, **_kwargs):
        return resp

    monkeypatch.setattr(ctl, "_unary_call", fake_unary_call)

    out = ctl.commit_registered_artifact("reg-1", timeout_s=0.1)
    assert out.local_stable_tier is not None
    assert out.local_stable_tier.status == "ready"
    assert out.local_stable_tier.message == "ok"

