#  Copyright (c) 2026, TensorCast Team.

from types import SimpleNamespace

import torch
from torch import nn

from tensorcast.artifact_runtime.dto import RuntimePlacement
from tensorcast.artifact_runtime.recipe.build import (
    COMPILED_RECIPE_MEMORY_CACHE,
    DEFAULT_RECIPE_BUILD_MEMORY_CACHE_ENTRIES,
    TRACE_PLAN_MEMORY_CACHE,
    RecipeBuildCacheConfig,
    RecipeBuildMemoryCache,
    RecipeBuildSession,
    RuntimeBindingPlan,
    compute_recipe_cache_key,
    compute_trace_cache_key,
)
from tensorcast.types import (
    RuntimeBindingMemberRef,
    RuntimeSupportLevel,
    RuntimeTopologyRef,
)


def _identity(**updates):
    topology = RuntimeTopologyRef(
        schema_topology_digest="topology-a",
        logical_topology_ref="tensorcast://topology/a",
    )
    member = RuntimeBindingMemberRef(
        member_id="dp0:pp0:tp0",
        member_index=0,
        member_count=1,
        group_id="group-1",
    )
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
        "topology_ref": topology,
        "member_ref": member,
        "placement": {"tp_rank": 0},
    }
    payload.update(updates)
    return RuntimeBindingPlan(**payload)


def test_recipe_build_session_keys_track_framework_and_placement():
    identity = _identity()
    session = RecipeBuildSession(identity)

    trace_key = session.trace_cache_key(metadata_fingerprint="meta-a")
    recipe_key = session.recipe_cache_key(metadata_fingerprint="meta-a")

    assert trace_key == compute_trace_cache_key(identity, metadata_fingerprint="meta-a")
    assert recipe_key == compute_recipe_cache_key(
        identity, metadata_fingerprint="meta-a"
    )
    assert recipe_key != RecipeBuildSession(
        _identity(adapter_version="adapter-v2")
    ).recipe_cache_key(metadata_fingerprint="meta-a")
    assert recipe_key != RecipeBuildSession(
        _identity(placement={"tp_rank": 1}, tp_rank=1)
    ).recipe_cache_key(metadata_fingerprint="meta-a")
    assert trace_key != RecipeBuildSession(
        _identity(extra={"variant": "a"})
    ).trace_cache_key(metadata_fingerprint="meta-a")
    assert recipe_key != RecipeBuildSession(
        _identity(extra={"variant": "a"})
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


def test_serving_binding_plan_absorbs_resolved_spec_cache_entry():
    identity = _identity()
    entry = SimpleNamespace(
        source_schema_hash="source-schema",
        source_reuse={"mode": "serving_transform"},
        model_config_digest="model-digest",
        load_config_digest="load-digest",
        serving_build_digest="build-digest",
        binding_layout_id="layout-id",
        target_layout_hash="layout-hash",
        tensor_schema_hash="tensor-schema",
        spec_digest="spec-digest",
        topology=identity.topology_ref,
        member=identity.member_ref,
    )

    plan = identity.with_resolved_spec_cache_entry(entry)

    assert plan.resolved_spec_cache_entry is entry
    assert plan.source_schema_hash == "source-schema"
    assert plan.source_reuse_decision == {"mode": "serving_transform"}
    assert plan.model_config_digest == "model-digest"
    assert plan.load_config_digest == "load-digest"
    assert plan.serving_build_digest == "build-digest"
    assert plan.binding_layout_id == "layout-id"
    assert plan.target_layout_hash == "layout-hash"
    assert plan.tensor_schema_hash == "tensor-schema"
    assert plan.resolved_spec_digest == "spec-digest"
    assert plan.topology_ref == identity.topology_ref
    assert plan.member_ref == identity.member_ref
    assert plan.base_payload()["resolved_spec_digest"] == "spec-digest"


def test_recipe_build_session_owns_cache_io(monkeypatch):
    session = RecipeBuildSession(_identity())
    calls = []

    import tensorcast.artifact_runtime.recipe.cache as recipe_cache
    import tensorcast.artifact_runtime.recipe.trace_cache as trace_cache

    monkeypatch.setattr(
        trace_cache,
        "load_trace_plan_cache",
        lambda path: calls.append(("load_trace", path)) or "trace",
    )
    monkeypatch.setattr(
        trace_cache,
        "write_trace_plan_cache",
        lambda path, trace: calls.append(("write_trace", path, trace)),
    )
    monkeypatch.setattr(
        recipe_cache,
        "load_compiled_recipe_cache",
        lambda path: calls.append(("load_recipe", path)) or "recipe",
    )
    monkeypatch.setattr(
        recipe_cache,
        "write_compiled_recipe_cache",
        lambda path, recipe: calls.append(("write_recipe", path, recipe)),
    )

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
        staticmethod(lambda path: calls.append(("load_trace", path)) or "trace-plan"),
    )
    trace_memory = {}
    trace_result = session.lookup_trace_plan_cache(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast",),
        memory_cache=trace_memory,
    )

    assert trace_result.value == "trace-plan"
    assert trace_result.disk_hit
    assert (
        trace_result.cache_path
        == session.trace_cache_write_paths(
            source_catalog=source_catalog,
            cache_dirs=("/tmp/tensorcast",),
        )[0]
    )
    assert trace_memory[trace_result.cache_key] == "trace-plan"
    assert calls == [("load_trace", trace_result.cache_path)]

    calls.clear()
    memory_result = session.lookup_trace_plan_cache(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast",),
        memory_cache=trace_memory,
    )

    assert memory_result.value == "trace-plan"
    assert memory_result.memory_hit
    assert memory_result.cache_path is None
    assert calls == []

    recipe = SimpleNamespace(
        source_metadata_fingerprint="meta-a",
        topology_ref=None,
        member_ref=None,
    )
    monkeypatch.setattr(
        RecipeBuildSession,
        "load_compiled_recipe_cache",
        staticmethod(lambda path: calls.append(("load_recipe", path)) or recipe),
    )
    recipe_memory = {}
    recipe_result = session.lookup_compiled_recipe_cache(
        source_catalog=source_catalog,
        cache_dirs=("/tmp/tensorcast",),
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


def test_recipe_build_memory_cache_is_bounded_lru():
    cache = RecipeBuildMemoryCache(max_entries=2)
    cache["a"] = 1
    cache["b"] = 2

    assert cache.get("a") == 1

    cache["c"] = 3

    assert len(cache) == 2
    assert "a" in cache
    assert "b" not in cache
    assert "c" in cache
    assert cache.max_entries == 2
    assert cache.size == 2


def test_default_recipe_build_memory_caches_are_bounded():
    assert isinstance(TRACE_PLAN_MEMORY_CACHE, RecipeBuildMemoryCache)
    assert isinstance(COMPILED_RECIPE_MEMORY_CACHE, RecipeBuildMemoryCache)
    assert TRACE_PLAN_MEMORY_CACHE.max_entries == (
        DEFAULT_RECIPE_BUILD_MEMORY_CACHE_ENTRIES
    )
    assert COMPILED_RECIPE_MEMORY_CACHE.max_entries == (
        DEFAULT_RECIPE_BUILD_MEMORY_CACHE_ENTRIES
    )


def test_recipe_build_session_owns_compile_identity_and_cached_rebind():
    session = RecipeBuildSession(_identity(tp_rank=2, tp_world_size=4))

    runtime_facts = type(
        "Facts",
        (),
        {
            "framework_name": "vllm",
            "adapter_version": "adapter-v1",
            "serving_abi_version": "abi-v1",
            "framework_version": "framework-v1",
        },
    )()

    identity = session.compile_identity(runtime_facts=runtime_facts)

    assert identity.model_id == "model"
    assert identity.tp_rank == 2
    assert identity.tp_world_size == 4
    assert identity.topology_ref == session.identity.topology_ref
    assert identity.member_ref == session.identity.member_ref
    assert identity.placement == {"tp_rank": 0}

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _Catalog:
        source_artifact_ref: str
        metadata_fingerprint: str
        canonical_index_hash: str

    # Use a dataclass recipe so rebind_cached_recipe_template can preserve type
    # through dataclasses.replace, matching the real CompiledRuntimeRecipe.
    import tensorcast as tc
    from tensorcast.artifact_runtime.recipe.compiler import (
        CompiledRuntimeRecipe,
        TensorcastRuntimeFacts,
        TensorcastSemanticValidationSpec,
        TensorSchemaEntry,
        realization_plan_digest,
        target_tensor_schema_hash,
    )
    from tensorcast.artifact_runtime.recipe.trace_ir import TracePlan

    real_recipe = CompiledRuntimeRecipe(
        compile_key="old",
        source_artifact_ref="old",
        source_metadata_fingerprint="old",
        runtime_facts=TensorcastRuntimeFacts(
            framework_name="vllm",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=RuntimeSupportLevel.BUILDER_PUBLICATION_READY,
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
        topology_ref=None,
        member_ref=None,
        semantic_validation_spec=TensorcastSemanticValidationSpec.empty(),
    )
    compile_payload = identity.compile_payload(
        source_artifact_ref="mi2:test:source",
        source_metadata_fingerprint="meta-b",
        runtime_facts=real_recipe.runtime_facts,
        tensor_schema=(
            TensorSchemaEntry(
                name="w",
                dtype="torch.float32",
                shape=(1,),
                stride=(1,),
            ),
        ),
        semantic_validation_spec=TensorcastSemanticValidationSpec.empty(),
    )

    assert compile_payload["source_artifact_ref"] == "mi2:test:source"
    assert compile_payload["metadata_fingerprint"] == "meta-b"
    assert compile_payload["topology_ref"] == session.identity.topology_ref.model_dump(
        mode="python"
    )
    assert compile_payload["member_ref"] == session.identity.member_ref.model_dump(
        mode="python"
    )
    assert compile_payload["tensor_schema"] == [
        {
            "name": "w",
            "dtype": "torch.float32",
            "shape": [1],
            "stride": [1],
        }
    ]

    rebound = session.rebind_cached_recipe_template(
        real_recipe,
        source_catalog=_Catalog(
            source_artifact_ref="mi2:test:source",
            metadata_fingerprint="meta-b",
            canonical_index_hash="source-schema-b",
        ),
    )

    assert rebound.compile_key != "old"
    assert rebound.source_artifact_ref == "mi2:test:source"
    assert rebound.source_metadata_fingerprint == "meta-b"
    assert rebound.topology_ref == session.identity.topology_ref
    assert rebound.member_ref == session.identity.member_ref
    assert rebound.binding_plan is not None
    assert rebound.binding_plan.source_artifact_ref == "mi2:test:source"
    assert rebound.binding_plan.source_metadata_fingerprint == "meta-b"
    assert rebound.binding_plan.source_schema_hash == "source-schema-b"
    assert rebound.binding_plan.tensor_schema_hash == target_tensor_schema_hash(
        rebound.tensor_schema
    )
    assert rebound.binding_plan.realization_plan_digest == realization_plan_digest(
        rebound.realization_plan_proto
    )
    assert rebound.binding_plan.runtime_facts is rebound.runtime_facts
    assert rebound.binding_plan.trace_plan is rebound.trace_plan
    assert rebound.binding_plan.tensor_schema == rebound.tensor_schema
    assert rebound.binding_plan.source_hull == rebound.source_hull
    assert rebound.binding_plan.realization_plan == rebound.realization_plan
    assert rebound.binding_plan.realization_fallback_plan == (
        rebound.realization_fallback_plan
    )
    assert rebound.binding_plan.realization_plan_proto == (
        rebound.realization_plan_proto
    )
    assert rebound.binding_plan.realization_plan_count == (
        rebound.realization_plan_count
    )
    assert rebound.binding_plan.semantic_validation_spec is (
        rebound.semantic_validation_spec
    )
    assert rebound.binding_plan.topology_ref == session.identity.topology_ref
    assert rebound.binding_plan.member_ref == session.identity.member_ref
    resolved_compile_payload = rebound.binding_plan.compile_payload(
        source_artifact_ref=rebound.source_artifact_ref,
        source_metadata_fingerprint=rebound.source_metadata_fingerprint,
        runtime_facts=rebound.runtime_facts,
        tensor_schema=rebound.tensor_schema,
        semantic_validation_spec=rebound.semantic_validation_spec,
    )
    identity_compile_payload = identity.compile_payload(
        source_artifact_ref=rebound.source_artifact_ref,
        source_metadata_fingerprint=rebound.source_metadata_fingerprint,
        runtime_facts=rebound.runtime_facts,
        tensor_schema=rebound.tensor_schema,
        semantic_validation_spec=rebound.semantic_validation_spec,
    )
    assert resolved_compile_payload["source_schema_hash"] == "source-schema-b"
    assert resolved_compile_payload["tensor_schema_hash"] == (
        rebound.binding_plan.tensor_schema_hash
    )
    assert resolved_compile_payload["realization_plan_digest"] == (
        rebound.binding_plan.realization_plan_digest
    )
    assert resolved_compile_payload != identity_compile_payload


def test_recipe_build_session_owns_recipe_metadata_collection():
    import tensorcast as tc
    from tensorcast.artifact_runtime.recipe.compiler import (
        TensorcastSemanticValidationSpec,
    )

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((2,), dtype=torch.float32))
            self.runtime_only = nn.Parameter(torch.empty((1,), dtype=torch.float32))

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
            return RuntimeSupportLevel.BUILDER_PUBLICATION_READY

        def runtime_only_tensor_names(self, model):
            return ("runtime_only",)

        def process_after_load_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def post_bind_finalize_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def semantic_probes(self, model, model_config):
            return {"model": type(model).__name__, "config": model_config}

    session = RecipeBuildSession(_identity())
    model = _Model()
    adapter = _Adapter()

    facts = session.collect_runtime_facts(model, "model-config", adapter)
    assert facts.framework_name == "fakefw"
    assert facts.runtime_only_tensor_names == ("runtime_only",)

    schema = session.collect_tensor_schema(
        model,
        runtime_only_tensor_names=facts.runtime_only_tensor_names,
        is_reserved_runtime_tensor_name=lambda name: name.startswith(
            "__tensorcast_meta__."
        ),
    )
    assert [entry.name for entry in schema] == ["w"]
    assert schema[0].dtype == "torch.float32"

    explicit = TensorcastSemanticValidationSpec(kind="explicit", payload={"ok": True})
    assert (
        session.resolve_semantic_validation_spec(
            model,
            "model-config",
            adapter,
            explicit,
        )
        is explicit
    )
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
    from tensorcast.artifact_runtime.recipe.trace_ir import CopyPlanEntry, TracePlan
    from tensorcast.artifact_runtime.source import (
        SourceCatalog,
        SourceTensorMeta,
        compute_source_metadata_fingerprint,
    )

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((4,), device="meta"))
            self.runtime_only = nn.Parameter(torch.empty((1,), device="meta"))

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
            return RuntimeSupportLevel.BUILDER_PUBLICATION_READY

        def runtime_only_tensor_names(self, model):
            return ("runtime_only",)

        def process_after_load_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def post_bind_finalize_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def semantic_probes(self, model, model_config):
            return None

        def cleanup_after_recipe_build(
            self, model, model_config, *, framework_config=None
        ):
            cleanup_calls.append(
                (type(model).__name__, model_config.model, framework_config)
            )

    meta_by_name = {
        "x": SourceTensorMeta(
            dtype=torch.float16,
            shape=(4,),
            stride=(1,),
            storage_offset=0,
        )
    }
    ordered_names = ("x",)
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
            CopyPlanEntry(
                op="copy", ckpt_name="x", ckpt_range=None, dst_name="w", dst_range=None
            )
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
        is_reserved_runtime_tensor_name=lambda name: name.startswith(
            "__tensorcast_meta__."
        ),
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


def test_recipe_build_session_debug_trace_dump_requires_explicit_flag(tmp_path):
    import tensorcast as tc
    from tensorcast.artifact_runtime.recipe.trace_ir import CopyPlanEntry, TracePlan
    from tensorcast.artifact_runtime.source import (
        SourceCatalog,
        SourceTensorMeta,
        compute_source_metadata_fingerprint,
    )

    class _Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.empty((4,), device="meta"))

    class _Adapter:
        def framework_name(self):
            return "fakefw"

        def framework_version(self):
            return "fakefw-v1"

        def adapter_version(self):
            return "adapter-v1"

        def serving_abi_version(self, model_config):
            return "abi-v1"

        def support_level(self, model, model_config):
            return RuntimeSupportLevel.BUILDER_PUBLICATION_READY

        def runtime_only_tensor_names(self, model):
            return ()

        def process_after_load_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def post_bind_finalize_class(self, model, model_config):
            return tc.FinalizeClass.RUNTIME_ONLY

        def semantic_probes(self, model, model_config):
            return None

    ordered_names = ("x",)
    meta_by_name = {
        "x": SourceTensorMeta(
            dtype=torch.float16,
            shape=(4,),
            stride=(1,),
            storage_offset=0,
        )
    }
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
            CopyPlanEntry(
                op="copy", ckpt_name="x", ckpt_range=None, dst_name="w", dst_range=None
            )
        ],
        expected_src_names={"x"},
        expected_dst_names={"w"},
        tensorcast_slices={},
        src_hull={},
    )

    def _build(*, debug_dump_trace: bool):
        session = RecipeBuildSession(_identity(framework_name="fakefw"))
        return session.build_recipe(
            model_config=SimpleNamespace(
                model="fake-model",
                revision=None,
                dtype=torch.float16,
                compute_hash=lambda: "fake-model-hash",
            ),
            framework_config=None,
            source_catalog=source_catalog,
            framework_adapter=_Adapter(),
            build_meta_model=_Model,
            cache_config=RecipeBuildCacheConfig(
                cache_dirs=(),
                allow_cache=False,
                allow_recipe_cache=False,
                allow_trace=True,
                debug_output_dir=tmp_path,
                debug_dump_trace=debug_dump_trace,
            ),
            is_reserved_runtime_tensor_name=lambda _name: False,
            trace_capture_fn=lambda *_args, **_kwargs: trace_plan,
        )

    _build(debug_dump_trace=False)
    assert not list(tmp_path.glob("tensorcast_trace_plan_*.json"))

    _build(debug_dump_trace=True)
    assert list(tmp_path.glob("tensorcast_trace_plan_*.json"))


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
            "source_metadata_fingerprint": "meta-a",
            "topology_ref": None,
            "member_ref": RuntimeBindingMemberRef(
                member_id="member-1",
                member_index=1,
                member_count=2,
            ),
        },
    )()
    placement = type(
        "Placement",
        (),
        {
            "member": RuntimeBindingMemberRef(
                member_id="member-1",
                member_index=1,
                member_count=2,
            ),
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
                "member": RuntimeBindingMemberRef(
                    member_id="member-0",
                    member_index=0,
                    member_count=2,
                ),
            },
        )(),
    )


def test_recipe_cache_match_uses_serving_member_identity():
    session = RecipeBuildSession(_identity())
    source_catalog = SimpleNamespace(metadata_fingerprint="meta-a")
    recipe = SimpleNamespace(
        source_metadata_fingerprint="meta-a",
        topology_ref=None,
        member_ref=RuntimeBindingMemberRef(
            member_id="dp0:pp0:tp1",
            member_index=9,
            member_count=16,
            group_id="group-1",
        ),
    )
    placement = RuntimePlacement(
        topology=RuntimeTopologyRef(
            schema_topology_digest="topology-digest",
            logical_topology_ref="tensorcast://placement/topology",
        ),
        member=RuntimeBindingMemberRef(
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
        placement=RuntimePlacement(
            topology=placement.topology,
            member=RuntimeBindingMemberRef(
                member_id="dp0:pp0:tp2",
                member_index=10,
                member_count=16,
                group_id="group-1",
            ),
            framework_payload=placement.framework_payload,
            identity_payload={
                "tensor_parallel_rank": 1,
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
            "tensorcast_slices": {"x": object()},
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
            "realization_fallback_plan": (1,),
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
