#  Copyright (c) 2026, TensorCast Team.

from types import SimpleNamespace

import torch
from torch import nn

from tensorcast.serving import ServingPlacement
from tensorcast.serving.recipe_build import (
    RecipeBuildCacheConfig,
    RecipeBuildIdentity,
    RecipeBuildSession,
    compute_recipe_cache_key,
    compute_trace_cache_key,
)
from tensorcast.types import ServingBindingMemberRef, ServingTopologyRef


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
        "topology_ref": {
            "topology": "a"
        },
        "member_ref": {
            "member": "a"
        },
        "placement": {
            "tp_rank": 0
        },
    }
    payload.update(updates)
    return RecipeBuildIdentity(**payload)


def test_recipe_build_session_keys_track_framework_and_placement():
    identity = _identity()
    session = RecipeBuildSession(identity)

    trace_key = session.trace_cache_key(metadata_fingerprint="meta-a")
    recipe_key = session.recipe_cache_key(metadata_fingerprint="meta-a")

    assert trace_key == compute_trace_cache_key(identity,
                                                metadata_fingerprint="meta-a")
    assert recipe_key == compute_recipe_cache_key(
        identity, metadata_fingerprint="meta-a")
    assert recipe_key != RecipeBuildSession(
        _identity(adapter_version="adapter-v2")).recipe_cache_key(
            metadata_fingerprint="meta-a")
    assert recipe_key != RecipeBuildSession(
        _identity(placement={"tp_rank": 1},
                  tp_rank=1)).recipe_cache_key(metadata_fingerprint="meta-a")


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


def test_recipe_build_session_owns_cache_io(monkeypatch):
    session = RecipeBuildSession(_identity())
    calls = []

    import tensorcast.serving.builder.recipe_cache as recipe_cache
    import tensorcast.serving.builder.trace_cache as trace_cache

    monkeypatch.setattr(
        trace_cache, "load_trace_plan_cache", lambda path: calls.append(
            ("load_trace", path)) or "trace")
    monkeypatch.setattr(
        trace_cache, "write_trace_plan_cache",
        lambda path, trace: calls.append(("write_trace", path, trace)))
    monkeypatch.setattr(
        recipe_cache, "load_compiled_recipe_cache", lambda path: calls.append(
            ("load_recipe", path)) or "recipe")
    monkeypatch.setattr(
        recipe_cache, "write_compiled_recipe_cache",
        lambda path, recipe: calls.append(("write_recipe", path, recipe)))

    assert session.load_trace_plan_cache("/tmp/trace.json") == "trace"
    session.write_trace_plan_cache("/tmp/trace.json", "trace-plan")
    assert session.load_compiled_recipe_cache("/tmp/recipe.json") == "recipe"
    session.write_compiled_recipe_cache("/tmp/recipe.json", "compiled")
    trace_write = session.store_trace_plan_cache(
        cache_path="/tmp/trace-2.json",
        trace_plan="trace-plan-2",
    )
    recipe_write = session.store_compiled_recipe_cache(
        cache_path="/tmp/recipe-2.json",
        recipe="compiled-2",
    )

    assert calls == [
        ("load_trace", "/tmp/trace.json"),
        ("write_trace", "/tmp/trace.json", "trace-plan"),
        ("load_recipe", "/tmp/recipe.json"),
        ("write_recipe", "/tmp/recipe.json", "compiled"),
        ("write_trace", "/tmp/trace-2.json", "trace-plan-2"),
        ("write_recipe", "/tmp/recipe-2.json", "compiled-2"),
    ]
    assert trace_write.cache_path == "/tmp/trace-2.json"
    assert trace_write.cache_write_sec >= 0.0
    assert recipe_write.cache_path == "/tmp/recipe-2.json"
    assert recipe_write.cache_write_sec >= 0.0


def test_recipe_build_session_owns_cache_lookup_and_memory(monkeypatch):
    session = RecipeBuildSession(_identity())
    source_catalog = SimpleNamespace(metadata_fingerprint="meta-a")
    calls = []

    monkeypatch.setattr(
        RecipeBuildSession,
        "load_trace_plan_cache",
        staticmethod(lambda path: calls.append(
            ("load_trace", path)) or "trace-plan"),
    )
    trace_memory = {}
    trace_result = session.lookup_trace_plan_cache(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast", ),
        memory_cache=trace_memory,
    )

    assert trace_result.value == "trace-plan"
    assert trace_result.disk_hit
    assert trace_result.cache_path == session.trace_cache_write_paths(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast", ),
    )[0]
    assert trace_memory[trace_result.cache_key] == "trace-plan"
    assert calls == [("load_trace", trace_result.cache_path)]

    calls.clear()
    memory_result = session.lookup_trace_plan_cache(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast", ),
        memory_cache=trace_memory,
    )

    assert memory_result.value == "trace-plan"
    assert memory_result.memory_hit
    assert memory_result.cache_path is None
    assert calls == []

    recipe = SimpleNamespace(
        source_metadata_fingerprint="meta-a",
        logical_topology=None,
    )
    monkeypatch.setattr(
        RecipeBuildSession,
        "load_compiled_recipe_cache",
        staticmethod(lambda path: calls.append(
            ("load_recipe", path)) or recipe),
    )
    recipe_memory = {}
    recipe_result = session.lookup_compiled_recipe_cache(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast", ),
        memory_cache=recipe_memory,
    )

    assert recipe_result.value is recipe
    assert recipe_result.disk_hit
    assert recipe_memory[recipe_result.cache_key] is recipe
    assert calls == [("load_recipe", recipe_result.cache_path)]

    remembered_key = session.remember_compiled_recipe_cache(
        source_catalog=source_catalog,
        recipe=recipe,
        memory_cache=recipe_memory,
    )
    assert recipe_memory[remembered_key] is recipe


def test_recipe_build_session_owns_compile_identity_and_cached_rebind():
    session = RecipeBuildSession(_identity(tp_rank=2, tp_world_size=4))

    serving_facts = type(
        "Facts",
        (),
        {
            "framework_name": "vllm",
            "adapter_version": "adapter-v1",
            "serving_abi_version": "abi-v1",
            "framework_version": "framework-v1",
        },
    )()

    identity = session.compile_identity(serving_facts=serving_facts)

    assert identity.model_id == "model"
    assert identity.logical_topology.tensor_parallel_rank == 2
    assert identity.logical_topology.tensor_parallel_world_size == 4
    assert identity.extra == {"placement": {"tp_rank": 0}}

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Catalog:
        source_artifact_ref: str
        metadata_fingerprint: str

    # Use a dataclass recipe so rebind_cached_recipe_template can preserve type
    # through dataclasses.replace, matching the real CompiledServingRecipe.
    import tensorcast as tc
    from tensorcast.serving.builder.compiler import (
        CompiledServingRecipe,
        TensorcastSemanticValidationSpec,
        TensorcastServingFacts,
    )
    from tensorcast.serving.builder.trace_ir import TracePlan

    real_recipe = CompiledServingRecipe(
        compile_key="old",
        source_artifact_ref="old",
        source_metadata_fingerprint="old",
        serving_facts=TensorcastServingFacts(
            framework_name="vllm",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=tc.ServingSupportLevel.BUILDER_PUBLICATION_READY,
            runtime_only_tensor_names=(),
            process_after_load_class=tc.FinalizeClass.RUNTIME_ONLY,
            post_bind_finalize_class=tc.FinalizeClass.RUNTIME_ONLY,
            framework_version="framework-v1",
        ),
        trace_plan=TracePlan(
            copy_plan=(),
            expected_src_names=frozenset(),
            expected_dst_names=frozenset(),
            tensorcast_slices=(),
            src_hull=(),
        ),
        tensor_schema=(),
        source_hull=(),
        realization_plan=(),
        realization_fallback_plan=(),
        logical_topology=None,
        semantic_validation_spec=TensorcastSemanticValidationSpec.empty(),
    )

    rebound = session.rebind_cached_recipe_template(
        real_recipe,
        source_catalog=_Catalog(
            source_artifact_ref="mi2:test:source",
            metadata_fingerprint="meta-b",
        ),
    )

    assert rebound.compile_key != "old"
    assert rebound.source_artifact_ref == "mi2:test:source"
    assert rebound.source_metadata_fingerprint == "meta-b"
    assert rebound.logical_topology.tensor_parallel_rank == 2


def test_recipe_build_session_owns_recipe_metadata_collection():
    import tensorcast as tc
    from tensorcast.serving.builder.compiler import (
        TensorcastSemanticValidationSpec,
    )

    class _Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((2, ), dtype=torch.float32))
            self.runtime_only = nn.Parameter(
                torch.empty((1, ), dtype=torch.float32))

    class _Adapter:

        def framework_name(self):
            return "fakefw"

        def framework_version(self):
            return "fakefw-v1"

        def adapter_version(self):
            return "adapter-v1"

        def serving_abi_version(self, model_config):
            assert model_config == "model-config"
            return "abi-v1"

        def support_level(self, model, model_config):
            assert model_config == "model-config"
            return tc.ServingSupportLevel.BUILDER_PUBLICATION_READY

        def runtime_only_tensor_names(self, model):
            return ("runtime_only", )

        def process_after_load_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def post_bind_finalize_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def semantic_probes(self, model, model_config):
            return {"model": type(model).__name__, "config": model_config}

    session = RecipeBuildSession(_identity())
    model = _Model()
    adapter = _Adapter()

    facts = session.collect_serving_facts(model, "model-config", adapter)
    assert facts.framework_name == "fakefw"
    assert facts.runtime_only_tensor_names == ("runtime_only", )

    schema = session.collect_tensor_schema(
        model,
        runtime_only_tensor_names=facts.runtime_only_tensor_names,
        is_reserved_serving_tensor_name=lambda name: name.startswith(
            "__tensorcast_meta__."),
    )
    assert [entry.name for entry in schema] == ["w"]
    assert schema[0].dtype == "torch.float32"

    explicit = TensorcastSemanticValidationSpec(kind="explicit",
                                                payload={"ok": True})
    assert session.resolve_semantic_validation_spec(
        model,
        "model-config",
        adapter,
        explicit,
    ) is explicit
    resolved = session.resolve_semantic_validation_spec(
        model,
        "model-config",
        adapter,
        None,
    )
    assert resolved.kind == "framework_semantic_probes"
    assert resolved.payload == {"config": "model-config", "model": "_Model"}


def test_recipe_build_session_build_recipe_runs_core_orchestration():
    import tensorcast as tc
    from tensorcast.serving.builder.trace_ir import CopyPlanEntry, TracePlan
    from tensorcast.serving.source_catalog import (
        SourceCatalog,
        SourceTensorMeta,
        compute_source_metadata_fingerprint,
    )

    class _Model(nn.Module):

        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((4, ), device="meta"))
            self.runtime_only = nn.Parameter(torch.empty((1, ), device="meta"))

    cleanup_calls = []

    class _Adapter:

        def framework_name(self):
            return "fakefw"

        def framework_version(self):
            return "fakefw-v1"

        def adapter_version(self):
            return "adapter-v1"

        def serving_abi_version(self, model_config):
            assert model_config.model == "fake-model"
            return "abi-v1"

        def support_level(self, model, model_config):
            assert model_config.model == "fake-model"
            return tc.ServingSupportLevel.BUILDER_PUBLICATION_READY

        def runtime_only_tensor_names(self, model):
            return ("runtime_only", )

        def process_after_load_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def post_bind_finalize_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def semantic_probes(self, model, model_config):
            return None

        def cleanup_after_recipe_build(self,
                                       model,
                                       model_config,
                                       *,
                                       framework_config=None):
            cleanup_calls.append(
                (type(model).__name__, model_config.model, framework_config))

    meta_by_name = {
        "x":
        SourceTensorMeta(
            dtype=torch.float16,
            shape=(4, ),
            stride=(1, ),
            storage_offset=0,
        )
    }
    ordered_names = ("x", )
    source_catalog = SourceCatalog(
        ordered_names=ordered_names,
        meta_by_name=meta_by_name,
        selected_files=(),
        source_artifact_ref="mi2:test:source",
        canonical_index_hash="",
        metadata_fingerprint=compute_source_metadata_fingerprint(
            ordered_names=ordered_names,
            meta_by_name=meta_by_name,
        ),
        canonical_index_bytes=b"",
    )
    trace_plan = TracePlan(
        copy_plan=[
            CopyPlanEntry(op="copy",
                          ckpt_name="x",
                          ckpt_range=None,
                          dst_name="w",
                          dst_range=None)
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={},
    )
    events = []
    session = RecipeBuildSession(_identity(framework_name="fakefw"))
    result = session.build_recipe(
        model_config=SimpleNamespace(
            model="fake-model",
            revision=None,
            dtype=torch.float16,
            compute_hash=lambda: "fake-model-hash",
        ),
        framework_config="framework-config",
        source_catalog=source_catalog,
        framework_adapter=_Adapter(),
        build_meta_model=_Model,
        cache_config=RecipeBuildCacheConfig(
            cache_dirs=(),
            allow_cache=False,
            allow_recipe_cache=False,
            allow_trace=True,
            trace_tp_slices=True,
            trace_cache_schema_version=7,
        ),
        is_reserved_serving_tensor_name=lambda name: name.startswith(
            "__tensorcast_meta__."),
        trace_capture_fn=lambda *_args: trace_plan,
        profile_sink=events.append,
    )

    assert result.recipe.source_artifact_ref == "mi2:test:source"
    assert result.recipe.trace_plan is trace_plan
    assert [entry.name for entry in result.recipe.tensor_schema] == ["w"]
    assert result.diagnostics["compile_key"] == result.recipe.compile_key
    assert result.diagnostics["trace_build_sec"] >= 0.0
    assert cleanup_calls == [("_Model", "fake-model", "framework-config")]
    assert any(event["stage"] == "recipe.summary" for event in events)


def test_recipe_build_session_owns_cached_recipe_context_match():
    session = RecipeBuildSession(_identity())
    source_catalog = type(
        "Catalog",
        (),
        {
            "metadata_fingerprint": "meta-a",
        },
    )()
    recipe = type(
        "Recipe",
        (),
        {
            "source_metadata_fingerprint":
            "meta-a",
            "logical_topology":
            type(
                "Topology",
                (),
                {
                    "tensor_parallel_rank": 1,
                    "tensor_parallel_world_size": 2,
                },
            )(),
        },
    )()
    placement = type(
        "Placement",
        (),
        {
            "tp_rank": 1,
            "tp_world_size": 2,
        },
    )()

    assert session.cached_recipe_matches_context(
        recipe,
        source_catalog=source_catalog,
        placement=placement,
    )
    assert not session.cached_recipe_matches_context(
        recipe,
        source_catalog=type(
            "Catalog",
            (),
            {
                "metadata_fingerprint": "meta-b",
            },
        )(),
        placement=placement,
    )
    assert not session.cached_recipe_matches_context(
        recipe,
        source_catalog=source_catalog,
        placement=type(
            "Placement",
            (),
            {
                "tp_rank": 0,
                "tp_world_size": 2,
            },
        )(),
    )


def test_recipe_cache_match_accepts_serving_placement_identity_payload():
    session = RecipeBuildSession(_identity())
    source_catalog = SimpleNamespace(metadata_fingerprint="meta-a")
    recipe = SimpleNamespace(
        source_metadata_fingerprint="meta-a",
        logical_topology=SimpleNamespace(
            tensor_parallel_rank=1,
            tensor_parallel_world_size=4,
        ),
    )
    placement = ServingPlacement(
        topology=ServingTopologyRef(
            schema_topology_digest="topology-digest",
            logical_topology_ref="tensorcast://placement/topology",
        ),
        member=ServingBindingMemberRef(
            member_id="dp0:pp0:tp1",
            member_index=9,
            member_count=16,
            group_id="group-1",
        ),
        framework_payload={
            "family": "vllm_parallelism",
            "version": "v1",
        },
        identity_payload={
            "tensor_parallel_rank": 1,
            "tensor_parallel_size": 4,
        },
    )

    assert session.cached_recipe_matches_context(
        recipe,
        source_catalog=source_catalog,
        placement=placement,
    )
    assert not session.cached_recipe_matches_context(
        recipe,
        source_catalog=source_catalog,
        placement=ServingPlacement(
            topology=placement.topology,
            member=placement.member,
            framework_payload=placement.framework_payload,
            identity_payload={
                "tensor_parallel_rank": 2,
                "tensor_parallel_size": 4,
            },
        ),
    )


def test_recipe_build_session_owns_recipe_summary_fields():
    trace_plan = type(
        "TracePlan",
        (),
        {
            "copy_plan": (1, 2),
            "expected_src_names": {"a", "b"},
            "expected_dst_names": {"x"},
            "tensorcast_slices": {
                "x": object()
            },
        },
    )()
    recipe = type(
        "Recipe",
        (),
        {
            "trace_plan": trace_plan,
            "tensor_schema": ("w", "x"),
            "realization_plan_count": 3,
            "realization_plan": (),
            "realization_fallback_plan": (1, ),
        },
    )()

    assert RecipeBuildSession.trace_plan_summary_fields(trace_plan) == {
        "copy_plan_count": 2,
        "expected_src_count": 2,
        "expected_dst_count": 1,
        "tensorcast_slice_count": 1,
    }
    assert RecipeBuildSession.recipe_summary_fields(recipe) == {
        "tensor_schema_count": 2,
        "copy_plan_count": 2,
        "expected_src_count": 2,
        "expected_dst_count": 1,
        "tensorcast_slice_count": 1,
        "realization_plan_count": 3,
        "realization_fallback_count": 1,
    }
