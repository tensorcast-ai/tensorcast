#  Copyright (c) 2026, TensorCast Team.

from types import SimpleNamespace

import torch

from tensorcast.serving import (
    CompiledServingRecipe,
    RecipeBuildIdentity,
    RecipeBuildSession,
    TensorcastSemanticValidationSpec,
    TensorcastServingFacts,
    TensorSchemaEntry,
    TracePlan,
    bind_serving_artifact,
    prepare_local_ready_serving,
    swap_serving_artifact,
)
from tensorcast.types import FinalizeClass, ServingSupportLevel


class _FakeArtifactView:
    def __init__(self, parent, names=None):
        self.parent = parent
        self.names = tuple(names or ())

    def bind(self, **kwargs):
        return {"names": self.names, "kwargs": kwargs}


class _FakeArtifact:
    def subset(self, names):
        return _FakeArtifactView(self, names)


class _FakeBinding:
    def __init__(self):
        self.realized = None
        self.closed = False

    def realize_from(self, source_view, *, realization_plan, options):
        self.realized = (source_view, realization_plan, options)
        return "epoch-1"

    def close(self):
        self.closed = True


class _FakeSource:
    def subset(self, names):
        return ("subset", tuple(names))


def _recipe():
    return CompiledServingRecipe(
        compile_key="compile",
        source_artifact_ref="mi2:source",
        source_metadata_fingerprint="meta",
        serving_facts=TensorcastServingFacts(
            framework_name="fakefw",
            framework_version="fakefw-v1",
            adapter_version="adapter-v1",
            serving_abi_version="abi-v1",
            support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
            runtime_only_tensor_names=(),
            process_after_load_class=FinalizeClass.RUNTIME_ONLY,
            post_bind_finalize_class=FinalizeClass.RUNTIME_ONLY,
        ),
        trace_plan=TracePlan(
            copy_plan=[],
            expected_src_names={"w"},
            expected_dst_names={"w"},
            tensorcast_slices={},
            src_hull={},
        ),
        tensor_schema=(
            TensorSchemaEntry(
                name="w",
                dtype="torch.float16",
                shape=(1,),
                stride=(1,),
            ),
        ),
        source_hull=(),
        realization_plan=(),
        realization_fallback_plan=(),
        logical_topology=None,
        semantic_validation_spec=TensorcastSemanticValidationSpec.empty(),
        realization_plan_proto=b"",
        realization_plan_count=0,
    )


def test_fake_second_framework_reuses_core_direct_and_local_ready_facades():
    identity = RecipeBuildIdentity(
        model_hash="hash",
        model_id="fake-model",
        model_revision=None,
        dtype="torch.float16",
        runtime_version="fake-runtime-v1",
        framework_name="fakefw",
        framework_version="fakefw-v1",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        trace_cache_schema_version=1,
        tp_rank=0,
        tp_world_size=1,
    )
    session = RecipeBuildSession(identity)
    assert session.recipe_cache_key(metadata_fingerprint="meta")

    resolved = SimpleNamespace(artifact=_FakeArtifact())
    bound = bind_serving_artifact(
        resolved_artifact=resolved,
        tensor_names=("w",),
        device="cuda:0",
        serving_runtime_policy=None,
        options=None,
    )
    assert bound["names"] == ("w",)

    swapped = []
    swap_serving_artifact(
        binding=SimpleNamespace(
            swap=lambda artifact, **kwargs: swapped.append((artifact, kwargs))
        ),
        resolved_artifact=resolved,
        serving_runtime_policy=None,
        options=None,
    )
    assert swapped and swapped[0][0] is resolved.artifact

    fake_binding = _FakeBinding()
    realized = prepare_local_ready_serving(
        recipe=_recipe(),
        source_subject=_FakeSource(),
        target_device=torch.device("cuda:0"),
        manifest_tensor_name="__tensorcast_meta__.manifest",
        manifest_bytes=None,
        options=None,
        binding_factory=lambda *args, **kwargs: fake_binding,
    )
    assert realized.binding is fake_binding
    assert realized.update_epoch == "epoch-1"
    assert fake_binding.realized is not None
