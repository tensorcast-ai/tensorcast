#  Copyright (c) 2026, TensorCast Team.

from tensorcast.serving import RecipeBuildIdentity, RecipeBuildSession
from tensorcast.serving.recipe_build import (
    compute_recipe_cache_key,
    compute_trace_cache_key,
)


def _identity(**updates):
    payload = {
        "model_hash": "model-hash",
        "model_id": "model",
        "model_revision": None,
        "dtype": "torch.float16",
        "runtime_version": "runtime-v1",
        "framework_name": "vllm",
        "framework_version": "framework-v1",
        "adapter_version": "adapter-v1",
        "serving_abi_version": "abi-v1",
        "trace_cache_schema_version": 7,
        "tp_rank": 0,
        "tp_world_size": 1,
        "topology_ref": {"topology": "a"},
        "member_ref": {"member": "a"},
        "placement": {"tp_rank": 0},
    }
    payload.update(updates)
    return RecipeBuildIdentity(**payload)


def test_recipe_build_session_keys_track_framework_and_placement():
    identity = _identity()
    session = RecipeBuildSession(identity)

    trace_key = session.trace_cache_key(metadata_fingerprint="meta-a")
    recipe_key = session.recipe_cache_key(metadata_fingerprint="meta-a")

    assert trace_key == compute_trace_cache_key(
        identity, metadata_fingerprint="meta-a"
    )
    assert recipe_key == compute_recipe_cache_key(
        identity, metadata_fingerprint="meta-a"
    )
    assert recipe_key != RecipeBuildSession(
        _identity(adapter_version="adapter-v2")
    ).recipe_cache_key(metadata_fingerprint="meta-a")
    assert recipe_key != RecipeBuildSession(
        _identity(placement={"tp_rank": 1}, tp_rank=1)
    ).recipe_cache_key(metadata_fingerprint="meta-a")


def test_recipe_build_session_paths_include_cache_key_and_rank():
    session = RecipeBuildSession(_identity(tp_rank=3, tp_world_size=4))

    trace_path = session.trace_cache_path(
        metadata_fingerprint="meta-a",
        cache_dir="/tmp/tensorcast",
    )
    recipe_path = session.recipe_cache_path(
        metadata_fingerprint="meta-a",
        cache_dir="/tmp/tensorcast",
    )

    assert trace_path.startswith("/tmp/tensorcast/tensorcast_trace_")
    assert trace_path.endswith("_tp3.json")
    assert recipe_path.startswith("/tmp/tensorcast/tensorcast_recipe_")
    assert recipe_path.endswith("_tp3.json")
