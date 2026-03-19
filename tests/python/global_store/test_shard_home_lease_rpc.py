#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import grpc

from tensorcast.proto.global_store.v1 import global_store_pb2


def test_shard_home_lease_acquire_conflict_refresh(servicer, test_context) -> None:
    shard_id = 123
    ttl_ms = 5_000

    req_a = global_store_pb2.AcquireShardHomeLeaseRequest(
        shard_id=shard_id,
        holder_daemon_id="daemon-a",
        ttl_ms=ttl_ms,
    )
    resp_a1 = servicer.AcquireShardHomeLease(req_a, test_context)
    assert test_context.code is None
    assert resp_a1.acquired is True
    assert resp_a1.lease.shard_id == shard_id
    assert resp_a1.lease.holder_daemon_id == "daemon-a"
    assert resp_a1.lease.lease_generation == 1
    assert resp_a1.lease.lease_token

    req_b = global_store_pb2.AcquireShardHomeLeaseRequest(
        shard_id=shard_id,
        holder_daemon_id="daemon-b",
        ttl_ms=ttl_ms,
    )
    ctx_b = type(test_context)()
    resp_b = servicer.AcquireShardHomeLease(req_b, ctx_b)
    assert ctx_b.code is None
    assert resp_b.acquired is False
    assert resp_b.lease.shard_id == shard_id
    assert resp_b.lease.holder_daemon_id == "daemon-a"
    assert resp_b.lease.lease_generation == 1
    assert resp_b.lease.lease_token == ""

    # Same holder refresh should keep token + generation stable.
    ctx_a2 = type(test_context)()
    resp_a2 = servicer.AcquireShardHomeLease(req_a, ctx_a2)
    assert ctx_a2.code is None
    assert resp_a2.acquired is True
    assert resp_a2.lease.lease_generation == 1
    assert resp_a2.lease.lease_token == resp_a1.lease.lease_token


def test_shard_home_lease_keepalive_release_and_reacquire(servicer, test_context) -> None:
    shard_id = 999
    ttl_ms = 1_000

    req = global_store_pb2.AcquireShardHomeLeaseRequest(
        shard_id=shard_id,
        holder_daemon_id="daemon-a",
        ttl_ms=ttl_ms,
    )
    resp = servicer.AcquireShardHomeLease(req, test_context)
    assert resp.acquired is True
    token = resp.lease.lease_token
    assert token

    keepalive_req = global_store_pb2.KeepaliveShardHomeLeaseRequest(
        lease_token=token,
        ttl_ms=ttl_ms,
    )
    ctx_k = type(test_context)()
    keepalive_resp = servicer.KeepaliveShardHomeLease(keepalive_req, ctx_k)
    assert ctx_k.code is None
    assert keepalive_resp.lease.shard_id == shard_id
    assert keepalive_resp.lease.lease_generation == 1
    assert keepalive_resp.lease.lease_token == token

    # Invalid token should map to NOT_FOUND.
    bad_ctx = type(test_context)()
    bad_resp = servicer.KeepaliveShardHomeLease(
        global_store_pb2.KeepaliveShardHomeLeaseRequest(
            lease_token="missing",
            ttl_ms=ttl_ms,
        ),
        bad_ctx,
    )
    assert bad_ctx.code == grpc.StatusCode.NOT_FOUND
    assert bad_resp.lease.shard_id == 0

    rel_ctx = type(test_context)()
    rel_resp = servicer.ReleaseShardHomeLease(
        global_store_pb2.ReleaseShardHomeLeaseRequest(lease_token=token),
        rel_ctx,
    )
    assert rel_ctx.code is None
    assert rel_resp.released is True

    # After release, GetShardHomeLease should be NOT_FOUND (inactive).
    get_ctx = type(test_context)()
    get_resp = servicer.GetShardHomeLease(
        global_store_pb2.GetShardHomeLeaseRequest(shard_id=shard_id),
        get_ctx,
    )
    assert get_ctx.code == grpc.StatusCode.NOT_FOUND
    assert get_resp.lease.shard_id == 0

    # Reacquire bumps generation.
    reacq_ctx = type(test_context)()
    reacq_resp = servicer.AcquireShardHomeLease(req, reacq_ctx)
    assert reacq_ctx.code is None
    assert reacq_resp.acquired is True
    assert reacq_resp.lease.lease_generation == 2


def test_shard_home_lease_batch_get(servicer, test_context) -> None:
    ttl_ms = 5_000
    shards = [1, 2, 3]

    for shard_id in shards:
        servicer.AcquireShardHomeLease(
            global_store_pb2.AcquireShardHomeLeaseRequest(
                shard_id=shard_id,
                holder_daemon_id="daemon-a",
                ttl_ms=ttl_ms,
            ),
            type(test_context)(),
        )

    ctx = type(test_context)()
    resp = servicer.BatchGetShardHomeLeases(
        global_store_pb2.BatchGetShardHomeLeasesRequest(shard_ids=shards),
        ctx,
    )
    assert ctx.code is None
    # All should be active and returned.
    got = {l.shard_id: l for l in resp.leases}
    assert set(got.keys()) == set(shards)
    for shard_id in shards:
        lease = got[shard_id]
        assert lease.holder_daemon_id == "daemon-a"
        assert lease.lease_generation == 1
