#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import concurrent.futures
import weakref

import pytest
import torch

import tensorcast.api.store as store_api
from tensorcast.api._config import PlanType
from tensorcast.api.operation import OperationStatus
from tensorcast.api.plan import PlanResult, PlanStepRef, PlanStepResult
from tensorcast.api.store import (
    AssemblyCloseoutContract,
    AssemblyRequirementSetRef,
    PublishedModelVersion,
    RegisteredServingPublication,
    RepresentationPublishContract,
    RepresentationPublishSpec,
    RuntimeArtifactBuildIntent,
    RuntimeArtifactManifest,
    Store,
    build_binding_finalize_admission_facts,
    build_representation_publish_requirements,
    build_serving_manifest_ref,
    prepare_pure_transform_serving_registration,
)
from tensorcast.api.store.artifact import Artifact
from tensorcast.api.store.common import (
    canonical_index_from_bytes,
    canonical_index_to_bytes,
)
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.types import ReplicaInfo
from tensorcast.proto.common.v1 import common_pb2
from tensorcast.proto.daemon.v2 import store_daemon_pb2
from tensorcast.proto.operation.v1 import operation_pb2
from tensorcast.types import (
    ArtifactDescriptor as PublishedArtifactDescriptor,
)
from tensorcast.types import (
    AssemblyAttemptRef,
    BindingValueRef,
    BuilderMode,
    ServingPublicationSubject,
    ServingSupportLevel,
)


class FakeAttemptClient:
    def __init__(self) -> None:
        self.start_calls: list[dict[str, object]] = []
        self.seal_calls: list[dict[str, object]] = []
        self.wait_calls: list[dict[str, object]] = []
        self.layout_calls: list[str] = []
        self.ensure_canonical_layout_calls: list[dict[str, object]] = []
        self.ensure_canonical_layout_result = "layout-provisioned"
        self.layout_ids_by_artifact: dict[str, tuple[str, ...]] = {
            "mi2:test:source": ("layout-source",),
            "mi2:test:serving": (),
        }
        self.wait_payload = store_daemon_pb2.SealAssemblyResult(
            artifact=common_pb2.ArtifactDescriptor(
                artifact_id="mi2:test:artifact",
                index_multihash="bafkindex",
                data_multihash="bafkdata",
                schema_version="v3",
                encoding="json",
                total_size=128,
            ),
            source_version_key="models/demo/source/v1",
        )

    def start_assembly_attempt(self, **kwargs: object) -> AssemblyAttemptRef:
        self.start_calls.append(dict(kwargs))
        operation_ref = operation_pb2.OperationRef(
            operation_id="bafkattemptop",
            kind="assembly_attempt",
            target_artifact_id="cgid:assembly-workspace-1",
            authority_scope_kind="assembly_attempt",
            authority_scope_id="cgid:assembly-attempt-1",
            attachment_kind="assembly_attempt",
            recovery_class="cluster_durable",
            fencing_digest="bafkattemptintent",
        )
        return AssemblyAttemptRef(
            attempt_id="cgid:assembly-attempt-1",
            workspace_assembly_id="cgid:assembly-workspace-1",
            layout_id=str(kwargs["layout_id"]),
            attempt_intent_digest="bafkattemptintent",
            coordinator_generation=1,
            coordinator_operation=operation_ref,
        )

    def seal_assembly_attempt(
        self,
        *,
        attempt_id: str,
        timeout_s: float = 10.0,
    ) -> store_daemon_pb2.SealAssemblyAttemptResponse:
        self.seal_calls.append(
            {
                "attempt_id": attempt_id,
                "timeout_s": timeout_s,
            }
        )
        return store_daemon_pb2.SealAssemblyAttemptResponse(
            operation=operation_pb2.OperationRef(
                operation_id="bafkattemptop",
                kind="assembly_attempt",
                target_artifact_id="cgid:assembly-workspace-9",
                authority_scope_kind="assembly_attempt",
                authority_scope_id=attempt_id,
                attachment_kind="assembly_attempt",
                recovery_class="cluster_durable",
                fencing_digest="bafk-intent-9",
            )
        )

    def list_artifact_layouts(
        self,
        artifact_id: str,
        *,
        timeout_s: float = 10.0,
    ) -> list[str]:
        del timeout_s
        self.layout_calls.append(artifact_id)
        return list(self.layout_ids_by_artifact.get(artifact_id, ()))

    def ensure_canonical_layout(self, **kwargs: object) -> str:
        self.ensure_canonical_layout_calls.append(dict(kwargs))
        return self.ensure_canonical_layout_result

    def get_operation(
        self,
        operation_id: str,
        *,
        operation_ref: operation_pb2.OperationRef | None = None,
        timeout_s: float = 10.0,
    ) -> operation_pb2.GetOperationResponse:
        del operation_id
        del operation_ref
        del timeout_s
        response = operation_pb2.GetOperationResponse()
        response.status.state = operation_pb2.OPERATION_STATE_PENDING
        return response

    def wait_operation(
        self,
        operation_id: str,
        *,
        operation_ref: operation_pb2.OperationRef | None = None,
        timeout_ms: int,
        timeout_s: float,
    ) -> operation_pb2.GetOperationResponse:
        self.wait_calls.append(
            {
                "operation_id": operation_id,
                "operation_ref": operation_ref,
                "timeout_ms": timeout_ms,
                "timeout_s": timeout_s,
            }
        )
        response = operation_pb2.GetOperationResponse()
        response.status.state = operation_pb2.OPERATION_STATE_SUCCESS
        response.status.result.Pack(self.wait_payload)
        return response


class FakeRuntime:
    def __init__(self, client: FakeAttemptClient) -> None:
        self._client = client
        self.daemon_endpoint = "fake://daemon"
        self.closed = False
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def ensure_client(self) -> FakeAttemptClient:
        return self._client

    def track_future(self, future: concurrent.futures.Future[object]) -> None:
        return None


class _StoreStub:
    closed = False
    _runtime = None


def _canonical_index_bytes() -> bytes:
    return b'{"w":[0,4,[1],[1],"torch.float32",0]}'


def _serving_build_intent() -> RuntimeArtifactBuildIntent:
    return RuntimeArtifactBuildIntent(
        representation_contract_hash="bafkrepresentation",
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        build_pipeline_version="pipeline-v1",
        source_artifact_ref="mi2:test:source",
    )


def _binding_finalize_build_intent() -> RuntimeArtifactBuildIntent:
    return RuntimeArtifactBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-vbinding",
        serving_abi_version="abi-vbinding",
        build_pipeline_version="pipeline-vbinding",
        source_artifact_ref="mi2:test:source",
    )


def _representation_publish_bundle() -> RepresentationPublishSpec:
    manifest = RuntimeArtifactManifest(
        framework_name="torch",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=1,
        source_artifact_ref="mi2:test:source",
        builder_mode=BuilderMode.PURE_TRANSFORM,
        build_pipeline_version="pipeline-v1",
    )
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            serving_artifact_id="mi2:test:serving",
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash=manifest.representation_contract_hash,
        serving_build_digest=manifest.serving_build_digest,
    )
    closeout_contract = AssemblyCloseoutContract(
        kind="representation_publish",
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        serving_manifest_ref=contract.serving_manifest_ref,
        representation_publish_contract=contract,
    )
    return RepresentationPublishSpec(
        serving_artifact_id=contract.serving_artifact_id,
        serving_manifest_ref=contract.serving_manifest_ref,
        serving_manifest=manifest,
        serving_manifest_bytes=manifest.to_bytes(),
        representation_publish_contract=contract,
        closeout_contract=closeout_contract,
    )


def _plan_publication_result(
    bundle: RepresentationPublishSpec,
) -> tuple[PlanResult, PlanStepRef[RepresentationPublishSpec]]:
    step_ref = PlanStepRef("s-pure")
    result = PlanResult(
        ok=True,
        request_id="req-plan-publish",
        steps={
            step_ref.step_id: PlanStepResult(
                step_id=step_ref.step_id,
                target_id="inst-a",
                action="transform_register",
                status=OperationStatus(
                    state="success",
                    message="transform_register completed",
                    as_of_ms=1,
                ),
                artifact_result=bundle,
            )
        },
    )
    return result, step_ref


def test_register_pure_transform_publication_registers_manifest_bearing_artifact() -> (
    None
):
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    tensors = {"w": torch.ones((1,), dtype=torch.float32)}
    build_intent = _serving_build_intent()
    prepared = prepare_pure_transform_serving_registration(
        build_intent=build_intent,
        source_artifact=None,
        tensors=tensors,
    )
    register_calls: list[dict[str, object]] = []

    def _put(
        registered_tensors: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: object | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        register_calls.append(
            {
                "artifact_id": artifact_id,
                "key": key,
                "policy": policy,
                "device": device,
                "tensor_names": tuple(sorted(registered_tensors.keys())),
            }
        )
        return RegisteredArtifact(
            artifact_id="mi2:test:serving",
            replica=ReplicaInfo(
                replica_id="mi2:test:serving",
                replica_type="COALESCED_VRAM",
                device=torch.device("cuda", 0),
                plan=PlanType.VRAM_COALESCED,
                size_bytes=int(
                    sum(
                        int(tensor.numel() * tensor.element_size())
                        for tensor in tensors.values()
                    )
                ),
            ),
            canonical_index=prepared.canonical_index,
            lease=None,
        )

    store.put = _put  # type: ignore[method-assign]

    result = store.register_pure_transform_publication(
        tensors,
        build_intent=build_intent,
        contract_family="canonical_full",
        key="models/demo/serving",
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
    )

    assert register_calls == [
        {
            "artifact_id": None,
            "key": "models/demo/serving",
            "policy": None,
            "device": None,
            "tensor_names": ("__tensorcast_meta__.manifest_json", "w"),
        }
    ]
    assert result.registered_artifact.artifact_id == "mi2:test:serving"
    assert (
        result.prepared_registration.serving_manifest_ref
        == build_serving_manifest_ref()
    )
    assert result.publication.contract_family == "canonical_full"
    assert (
        result.publication.serving_manifest
        == result.prepared_registration.serving_manifest
    )
    assert result.publication.closeout_contract.kind == "representation_publish"


def test_complete_pure_transform_publication_runs_register_and_closeout() -> None:
    client = FakeAttemptClient()
    client.wait_payload = store_daemon_pb2.SealAssemblyResult(
        artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:source",
            index_multihash="bafksourceindex",
            data_multihash="bafksourcedata",
            schema_version="v3",
            encoding="json",
            total_size=128,
        ),
        serving_artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:serving",
            index_multihash="bafkservingindex",
            data_multihash="bafkservingdata",
            schema_version="v3",
            encoding="json",
            total_size=256,
        ),
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref(),
    )
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    tensors = {"w": torch.ones((1,), dtype=torch.float32)}
    build_intent = _serving_build_intent()
    prepared = prepare_pure_transform_serving_registration(
        build_intent=build_intent,
        source_artifact=None,
        tensors=tensors,
    )

    def _put(
        registered_tensors: dict[str, torch.Tensor],
        *,
        artifact_id: str | None = None,
        key: str | None = None,
        policy: object | None = None,
        device: int | torch.device | None = None,
    ) -> RegisteredArtifact:
        del registered_tensors
        del artifact_id
        del key
        del policy
        del device
        return RegisteredArtifact(
            artifact_id="mi2:test:serving",
            replica=ReplicaInfo(
                replica_id="mi2:test:serving",
                replica_type="COALESCED_VRAM",
                device=torch.device("cuda", 0),
                plan=PlanType.VRAM_COALESCED,
                size_bytes=4,
            ),
            canonical_index=prepared.canonical_index,
            lease=None,
        )

    store.put = _put  # type: ignore[method-assign]

    result = store.complete_pure_transform_publication(
        tensors,
        build_intent=build_intent,
        contract_family="canonical_full",
        key="models/demo/serving",
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        timeout_s=5.0,
    )

    assert result.serving_artifact_id == "mi2:test:serving"
    assert client.layout_calls == ["mi2:test:source"]
    requirements = client.start_calls[0]["requirements"]
    assert isinstance(requirements, AssemblyRequirementSetRef)
    assert requirements == AssemblyRequirementSetRef.canonical_full()
    assert len(client.seal_calls) == 1
    assert len(client.wait_calls) == 1


def test_complete_pure_transform_publication_canonical_full_routes_source_artifact_contribution() -> (
    None
):
    store_ref = _StoreStub()
    source_artifact = Artifact(
        store_ref=weakref.ref(store_ref),
        artifact_id="mi2:test:source",
        canonical_index_bytes=_canonical_index_bytes(),
        canonical_index=canonical_index_from_bytes(_canonical_index_bytes()),
    )
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    captured: dict[str, object] = {}
    prepared = prepare_pure_transform_serving_registration(
        build_intent=_serving_build_intent(),
        source_artifact=source_artifact,
        tensors={"w": torch.ones((1,), dtype=torch.float32)},
    )

    def _register(
        tensors: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> RegisteredServingPublication:
        del tensors
        del kwargs
        bundle = _representation_publish_bundle().model_copy(
            update={"contract_family": "canonical_full"}
        )
        return RegisteredServingPublication(
            registered_artifact=RegisteredArtifact(
                artifact_id="mi2:test:serving",
                replica=ReplicaInfo(
                    replica_id="mi2:test:serving",
                    replica_type="COALESCED_VRAM",
                    device=torch.device("cuda", 0),
                    plan=PlanType.VRAM_COALESCED,
                    size_bytes=4,
                ),
                canonical_index=prepared.canonical_index,
                lease=None,
            ),
            prepared_registration=prepared,
            publication=bundle,
        )

    def _start_repo_owned(**kwargs: object) -> AssemblyAttemptRef:
        captured["start_kwargs"] = kwargs
        return client.start_assembly_attempt(
            layout_id=str(kwargs["layout_id"]),
            requirements=AssemblyRequirementSetRef.canonical_full(),
        )

    def _contribute_source(**kwargs: object) -> tuple[object, ...]:
        captured["source_kwargs"] = kwargs
        return ()

    def _seal_attempt(
        attempt: AssemblyAttemptRef,
        *,
        ctx: object | None = None,
    ) -> object:
        del ctx
        captured["seal_attempt"] = attempt
        return object()

    def _wait_attempt(
        attempt: object,
        *,
        timeout_s: float | None = None,
        ctx: object | None = None,
    ) -> PublishedModelVersion:
        del attempt
        del timeout_s
        del ctx
        return PublishedModelVersion(
            assembly_id="cgid:assembly-1",
            source_artifact_id="mi2:test:source",
            source_descriptor=PublishedArtifactDescriptor(
                artifact_id="mi2:test:source",
                total_size=4,
            ),
            serving_artifact_id="mi2:test:serving",
            serving_manifest_ref=build_serving_manifest_ref(),
        )

    store.register_pure_transform_publication = _register  # type: ignore[method-assign]
    store.start_repo_owned_representation_publish_attempt = _start_repo_owned  # type: ignore[method-assign]
    store._contribute_source_current_values_to_attempt_and_keep_bindings = (
        _contribute_source  # type: ignore[method-assign]
    )
    store.seal_assembly_attempt = _seal_attempt  # type: ignore[method-assign]
    store.wait_assembly_attempt = _wait_attempt  # type: ignore[method-assign]

    result = store.complete_pure_transform_publication(
        {"w": torch.ones((1,), dtype=torch.float32)},
        build_intent=_serving_build_intent(),
        source_artifact=source_artifact,
        contract_family="canonical_full",
        source_contribution_device="cuda:0",
        source_contribution_artifacts=(source_artifact,),
        layout_id="layout-source",
    )

    assert result.serving_artifact_id == "mi2:test:serving"
    assert captured["start_kwargs"]["source_artifact"] is source_artifact
    assert captured["source_kwargs"]["source_artifacts"] == (source_artifact,)
    assert captured["source_kwargs"]["device"] == "cuda:0"


def test_complete_pure_transform_publication_routes_structural_view_contributions() -> (
    None
):
    store_ref = _StoreStub()
    canonical_bytes = b'{"w":[0,16,[4],[1],"torch.float32",0]}'
    source_artifact = Artifact(
        store_ref=weakref.ref(store_ref),
        artifact_id="mi2:test:source",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    )
    source_view_a = source_artifact.view(slices={"w": [slice(0, 2)]})
    source_view_b = source_artifact.view(slices={"w": [slice(2, 4)]})
    view_id_a = source_view_a._ensure_view_metadata_cache(require_view_id=True).view_id
    view_id_b = source_view_b._ensure_view_metadata_cache(require_view_id=True).view_id

    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    captured: dict[str, object] = {}

    def _register_publication(
        tensors: dict[str, torch.Tensor],
        **kwargs: object,
    ) -> RegisteredServingPublication:
        del tensors
        del kwargs
        bundle = _representation_publish_bundle().model_copy(
            update={"contract_family": "pp"}
        )
        return RegisteredServingPublication(
            registered_artifact=RegisteredArtifact(
                artifact_id="mi2:test:serving",
                replica=ReplicaInfo(
                    replica_id="mi2:test:serving",
                    replica_type="COALESCED_VRAM",
                    device=torch.device("cuda", 0),
                    plan=PlanType.VRAM_COALESCED,
                    size_bytes=4,
                ),
                canonical_index=canonical_index_from_bytes(_canonical_index_bytes()),
                lease=None,
            ),
            prepared_registration=prepare_pure_transform_serving_registration(
                build_intent=_serving_build_intent(),
                source_artifact=source_artifact,
                tensors={"w": torch.ones((1,), dtype=torch.float32)},
            ),
            publication=bundle,
        )

    def _start_repo_owned(**kwargs: object) -> AssemblyAttemptRef:
        captured["start_kwargs"] = kwargs
        return client.start_assembly_attempt(
            layout_id=str(kwargs["layout_id"]),
            requirements=kwargs["structural_view_ids"],
        )

    def _contribute(**kwargs: object) -> tuple[object, ...]:
        captured["contribute_kwargs"] = kwargs
        return ()

    def _seal_attempt(
        attempt: AssemblyAttemptRef,
        *,
        ctx: object | None = None,
    ) -> object:
        del ctx
        captured["seal_attempt"] = attempt
        return object()

    def _wait_attempt(
        attempt: object,
        *,
        timeout_s: float | None = None,
        ctx: object | None = None,
    ) -> PublishedModelVersion:
        del attempt
        del timeout_s
        del ctx
        return PublishedModelVersion(
            assembly_id="cgid:assembly-1",
            source_artifact_id="mi2:test:source",
            source_descriptor=PublishedArtifactDescriptor(
                artifact_id="mi2:test:source",
                total_size=4,
            ),
            serving_artifact_id="mi2:test:serving",
            serving_manifest_ref=build_serving_manifest_ref(),
        )

    store.register_pure_transform_publication = _register_publication  # type: ignore[method-assign]
    store.start_repo_owned_representation_publish_attempt = _start_repo_owned  # type: ignore[method-assign]
    store._contribute_source_artifacts_to_attempt = _contribute  # type: ignore[method-assign]
    store.seal_assembly_attempt = _seal_attempt  # type: ignore[method-assign]
    store.wait_assembly_attempt = _wait_attempt  # type: ignore[method-assign]

    result = store.complete_pure_transform_publication(
        {"w": torch.ones((1,), dtype=torch.float32)},
        build_intent=_serving_build_intent(),
        source_artifact=source_artifact,
        contract_family="pp",
        source_contribution_device="cuda:0",
        source_contribution_artifacts=(source_view_a, source_view_b),
        layout_id="layout-structural",
    )

    assert result.serving_artifact_id == "mi2:test:serving"
    assert captured["start_kwargs"]["structural_view_ids"] == (view_id_a, view_id_b)
    contribution_artifacts = captured["contribute_kwargs"]["source_artifacts"]
    assert contribution_artifacts == (source_view_a, source_view_b)


def test_start_assembly_attempt_returns_attempt_ref() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.pp_from_structural_views(
        ["view-a", "view-b"]
    )

    attempt = store.start_assembly_attempt(
        layout_id="layout-1",
        requirements=requirements,
    )

    assert attempt.attempt_id == "cgid:assembly-attempt-1"
    assert attempt.workspace_assembly_id == "cgid:assembly-workspace-1"
    assert attempt.layout_id == "layout-1"
    assert attempt.attempt_intent_digest == "bafkattemptintent"
    assert attempt.coordinator_operation.operation_id == "bafkattemptop"
    assert attempt.coordinator_operation.kind == "assembly_attempt"
    assert client.start_calls == [
        {
            "layout_id": "layout-1",
            "requirements": requirements,
        }
    ]


def test_start_assembly_attempt_requires_explicit_requirements() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)

    with pytest.raises(ValueError) as exc_info:
        store.start_assembly_attempt(layout_id="layout-1")

    assert "requirements are required" in str(exc_info.value)


def test_start_representation_publish_attempt_uses_bundle_closeout() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.canonical_full()
    bundle = _representation_publish_bundle()

    attempt = store.start_representation_publish_attempt(
        layout_id="layout-publish",
        requirements=requirements,
        publication=bundle,
    )

    assert attempt.layout_id == "layout-publish"
    assert len(client.start_calls) == 1
    assert client.start_calls[0]["layout_id"] == "layout-publish"
    assert client.start_calls[0]["requirements"] == requirements
    assert client.start_calls[0]["closeout_contract"] == bundle.closeout_contract


def test_start_representation_publish_attempt_infers_unique_layout_from_source_artifact() -> (
    None
):
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.canonical_full()
    bundle = _representation_publish_bundle()

    attempt = store.start_representation_publish_attempt(
        requirements=requirements,
        publication=bundle,
    )

    assert attempt.layout_id == "layout-source"
    assert client.layout_calls == ["mi2:test:source"]
    assert client.start_calls[0]["layout_id"] == "layout-source"


def test_start_representation_publish_attempt_rejects_ambiguous_inferred_layout() -> (
    None
):
    client = FakeAttemptClient()
    client.layout_ids_by_artifact["mi2:test:source"] = ("layout-a", "layout-b")
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.canonical_full()
    bundle = _representation_publish_bundle()

    with pytest.raises(Exception) as exc_info:
        store.start_representation_publish_attempt(
            requirements=requirements,
            publication=bundle,
        )

    assert "ambiguous" in str(exc_info.value)


def test_start_representation_publish_attempt_provisions_canonical_source_layout() -> (
    None
):
    client = FakeAttemptClient()
    client.layout_ids_by_artifact["mi2:test:source"] = ()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.canonical_full()
    bundle = _representation_publish_bundle().model_copy(
        update={"contract_family": "canonical_full"}
    )

    attempt = store.start_representation_publish_attempt(
        requirements=requirements,
        publication=bundle,
    )

    assert client.ensure_canonical_layout_calls == [
        {
            "artifact_id": "mi2:test:source",
            "index_multihash": "test",
            "attach_to_artifact": True,
        }
    ]
    assert attempt.layout_id == "layout-provisioned"
    assert client.start_calls[0]["layout_id"] == "layout-provisioned"


def test_start_representation_publish_attempt_provisions_binding_subject_layout_from_canonical_index() -> (
    None
):
    client = FakeAttemptClient()
    client.ensure_canonical_layout_result = "layout-binding-native"
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.canonical_full()
    canonical_index = canonical_index_from_bytes(
        b'{"serving.weight":[0,4,[1],[1],"torch.float32",0]}'
    )
    manifest = RuntimeArtifactManifest(
        framework_name="torch",
        adapter_version="adapter-vbinding",
        serving_abi_version="abi-vbinding",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=1,
        source_artifact_ref="mi2:test:source",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        build_pipeline_version="pipeline-vbinding",
    )
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            binding_value_ref=BindingValueRef(
                binding_id="binding-1",
                binding_layout_id="layout-1",
                binding_value_id="value-1",
                seal_generation=2,
            )
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash=manifest.representation_contract_hash,
        serving_build_digest=manifest.serving_build_digest,
    )
    publication = RepresentationPublishSpec(
        serving_artifact_id=None,
        serving_manifest_ref=contract.serving_manifest_ref,
        serving_manifest=manifest,
        serving_manifest_bytes=manifest.to_bytes(),
        canonical_index=canonical_index,
        representation_publish_contract=contract,
        closeout_contract=AssemblyCloseoutContract(
            kind="representation_publish",
            serving_manifest_ref=contract.serving_manifest_ref,
            representation_publish_contract=contract,
        ),
        admission_facts=build_binding_finalize_admission_facts(
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
            same_binding_fast_path_validated=True,
        ),
        contract_family="canonical_full",
    )
    attempt = store.start_representation_publish_attempt(
        requirements=requirements,
        publication=publication,
        layout_artifact_id="mi2:test:source",
    )

    assert len(client.ensure_canonical_layout_calls) == 1
    ensure_call = client.ensure_canonical_layout_calls[0]
    assert ensure_call["canonical_index_data"] == canonical_index_to_bytes(
        canonical_index
    )
    assert ensure_call["attach_to_artifact"] is False
    assert client.layout_calls == []
    assert attempt.layout_id == "layout-binding-native"
    assert client.start_calls[0]["layout_id"] == "layout-binding-native"


def test_complete_representation_publish_attempt_runs_start_seal_wait() -> None:
    client = FakeAttemptClient()
    client.wait_payload = store_daemon_pb2.SealAssemblyResult(
        artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:source",
            index_multihash="bafksourceindex",
            data_multihash="bafksourcedata",
            schema_version="v3",
            encoding="json",
            total_size=128,
        ),
        serving_artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:serving",
            index_multihash="bafkservingindex",
            data_multihash="bafkservingdata",
            schema_version="v3",
            encoding="json",
            total_size=256,
        ),
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref(),
    )
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    requirements = AssemblyRequirementSetRef.canonical_full()
    bundle = _representation_publish_bundle()

    result = store.complete_representation_publish_attempt(
        requirements=requirements,
        publication=bundle,
        timeout_s=5.0,
    )

    assert result.source_artifact_id == "mi2:test:source"
    assert result.serving_artifact_id == "mi2:test:serving"
    assert result.representation_contract_hash == "bafkrepresentation"
    assert result.serving_build_digest == "bafkbuilddigest"
    assert client.layout_calls == ["mi2:test:source"]
    assert len(client.start_calls) == 1
    assert len(client.seal_calls) == 1
    assert len(client.wait_calls) == 1


def test_start_canonical_representation_publish_attempt_uses_canonical_full() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    bundle = _representation_publish_bundle()

    attempt = store.start_canonical_representation_publish_attempt(
        publication=bundle,
    )

    assert attempt.layout_id == "layout-source"
    assert len(client.start_calls) == 1
    requirements = client.start_calls[0]["requirements"]
    assert isinstance(requirements, AssemblyRequirementSetRef)
    assert requirements == AssemblyRequirementSetRef.canonical_full()


def test_complete_repo_owned_representation_publish_attempt_routes_by_bundle_family() -> (
    None
):
    client = FakeAttemptClient()
    client.wait_payload = store_daemon_pb2.SealAssemblyResult(
        artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:source",
            index_multihash="bafksourceindex",
            data_multihash="bafksourcedata",
            schema_version="v3",
            encoding="json",
            total_size=128,
        ),
        serving_artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:serving",
            index_multihash="bafkservingindex",
            data_multihash="bafkservingdata",
            schema_version="v3",
            encoding="json",
            total_size=256,
        ),
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref(),
    )
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    bundle = _representation_publish_bundle().model_copy(
        update={"contract_family": "canonical_full"}
    )

    result = store.complete_repo_owned_representation_publish_attempt(
        publication=bundle,
        timeout_s=5.0,
    )

    assert result.serving_artifact_id == "mi2:test:serving"
    requirements = client.start_calls[0]["requirements"]
    assert isinstance(requirements, AssemblyRequirementSetRef)
    assert requirements == AssemblyRequirementSetRef.canonical_full()


def test_complete_plan_repo_owned_representation_publish_attempt_routes_plan_bundle() -> (
    None
):
    client = FakeAttemptClient()
    client.wait_payload = store_daemon_pb2.SealAssemblyResult(
        artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:source",
            index_multihash="bafksourceindex",
            data_multihash="bafksourcedata",
            schema_version="v3",
            encoding="json",
            total_size=128,
        ),
        serving_artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:serving",
            index_multihash="bafkservingindex",
            data_multihash="bafkservingdata",
            schema_version="v3",
            encoding="json",
            total_size=256,
        ),
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref(),
    )
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    plan_result, publication_step = _plan_publication_result(
        _representation_publish_bundle().model_copy(
            update={"contract_family": "canonical_full"}
        )
    )

    result = store.complete_plan_repo_owned_representation_publish_attempt(
        plan_result=plan_result,
        publication_step=publication_step,
        timeout_s=5.0,
    )

    assert result.serving_artifact_id == "mi2:test:serving"
    requirements = client.start_calls[0]["requirements"]
    assert isinstance(requirements, AssemblyRequirementSetRef)
    assert requirements == AssemblyRequirementSetRef.canonical_full()


def test_build_representation_publish_requirements_derives_structural_view_id() -> None:
    store = _StoreStub()
    canonical_bytes = _canonical_index_bytes()
    source = Artifact(
        store_ref=weakref.ref(store),
        artifact_id="mi2:test:source",
        canonical_index_bytes=canonical_bytes,
        canonical_index=canonical_index_from_bytes(canonical_bytes),
    ).subset(["w"])

    requirements = build_representation_publish_requirements(
        contract_family="pp",
        source_artifact=source,
    )

    view_metadata = source._ensure_view_metadata_cache(require_view_id=True)
    assert view_metadata is not None
    assert requirements == AssemblyRequirementSetRef.pp_from_structural_views(
        [str(view_metadata.view_id)]
    )


def test_start_structural_representation_publish_attempt_uses_repo_owned_requirements() -> (
    None
):
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    bundle = _representation_publish_bundle()
    source = Artifact(
        store_ref=weakref.ref(_StoreStub()),
        artifact_id="mi2:test:source",
        canonical_index_bytes=_canonical_index_bytes(),
        canonical_index=canonical_index_from_bytes(_canonical_index_bytes()),
    ).subset(["w"])

    attempt = store.start_structural_representation_publish_attempt(
        contract_family="pp",
        publication=bundle,
        source_artifact=source,
    )

    assert attempt.layout_id == "layout-source"
    requirements = client.start_calls[0]["requirements"]
    assert isinstance(requirements, AssemblyRequirementSetRef)
    view_metadata = source._ensure_view_metadata_cache(require_view_id=True)
    assert view_metadata is not None
    assert requirements == AssemblyRequirementSetRef.pp_from_structural_views(
        [str(view_metadata.view_id)]
    )


def test_complete_canonical_representation_publish_attempt_runs_canonical_full() -> (
    None
):
    client = FakeAttemptClient()
    client.wait_payload = store_daemon_pb2.SealAssemblyResult(
        artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:source",
            index_multihash="bafksourceindex",
            data_multihash="bafksourcedata",
            schema_version="v3",
            encoding="json",
            total_size=128,
        ),
        serving_artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:serving",
            index_multihash="bafkservingindex",
            data_multihash="bafkservingdata",
            schema_version="v3",
            encoding="json",
            total_size=256,
        ),
        source_version_key="models/demo/source/v1",
        serving_version_key="models/demo/serving/v1",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref(),
    )
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    bundle = _representation_publish_bundle()

    result = store.complete_canonical_representation_publish_attempt(
        publication=bundle,
        timeout_s=5.0,
    )

    assert result.serving_artifact_id == "mi2:test:serving"
    requirements = client.start_calls[0]["requirements"]
    assert isinstance(requirements, AssemblyRequirementSetRef)
    assert requirements == AssemblyRequirementSetRef.canonical_full()


def test_seal_assembly_attempt_decodes_source_lineage() -> None:
    client = FakeAttemptClient()
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    operation_ref = operation_pb2.OperationRef(
        operation_id="bafkattemptop",
        kind="assembly_attempt",
        target_artifact_id="cgid:assembly-workspace-9",
        authority_scope_kind="assembly_attempt",
        authority_scope_id="cgid:assembly-attempt-9",
        attachment_kind="assembly_attempt",
        recovery_class="cluster_durable",
        fencing_digest="bafk-intent-9",
    )
    attempt = AssemblyAttemptRef(
        attempt_id="cgid:assembly-attempt-9",
        workspace_assembly_id="cgid:assembly-workspace-9",
        layout_id="layout-9",
        attempt_intent_digest="bafk-intent-9",
        coordinator_generation=1,
        coordinator_operation=operation_ref,
    )

    operation = store.seal_assembly_attempt(attempt)
    result = operation.wait(timeout_s=5.0)

    assert result.assembly_id == "cgid:assembly-workspace-9"
    assert result.source_artifact_id == "mi2:test:artifact"
    assert result.source_descriptor.artifact_id == "mi2:test:artifact"
    assert result.source_version_key == "models/demo/source/v1"
    assert result.serving_artifact_id is None
    assert result.representation_contract_hash is None
    assert client.seal_calls == [
        {"attempt_id": "cgid:assembly-attempt-9", "timeout_s": 10.0}
    ]
    assert len(client.wait_calls) == 1
    wait_call = client.wait_calls[0]
    assert wait_call["operation_id"] == "bafkattemptop"
    assert wait_call["operation_ref"] == operation_ref
    assert 4900 <= int(wait_call["timeout_ms"]) <= 5000
    assert 9.0 <= float(wait_call["timeout_s"]) <= 10.0


def test_requirement_family_builders_encode_distinct_contracts() -> None:
    pp = AssemblyRequirementSetRef.pp_from_structural_views(
        ["view-b", "view-a", "view-a"]
    )
    ep = AssemblyRequirementSetRef.ep_from_structural_views(["view-b", "view-a"])
    canonical = AssemblyRequirementSetRef.canonical_full()

    assert tuple(req.slot_id for req in pp.inline_requirements) == (
        "view-a",
        "view-b",
    )
    assert tuple(req.coverage_contract for req in pp.inline_requirements) == (
        "pp_structural_view",
        "pp_structural_view",
    )

    assert tuple(req.slot_id for req in ep.inline_requirements) == (
        "view-a",
        "view-b",
    )
    assert tuple(req.coverage_contract for req in ep.inline_requirements) == (
        "ep_structural_view",
        "ep_structural_view",
    )

    assert canonical.requirement_count == 1
    assert canonical.inline_requirements[0].slot_id == "__canonical_full__"
    assert canonical.inline_requirements[0].coverage_contract == "canonical_full"


def test_seal_assembly_attempt_decodes_representation_publish_lineage() -> None:
    client = FakeAttemptClient()
    client.wait_payload = store_daemon_pb2.SealAssemblyResult(
        artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:source",
            index_multihash="bafksourceindex",
            data_multihash="bafksourcedata",
            schema_version="v3",
            encoding="json",
            total_size=128,
        ),
        serving_artifact=common_pb2.ArtifactDescriptor(
            artifact_id="mi2:test:serving",
            index_multihash="bafkservingindex",
            data_multihash="bafkservingdata",
            schema_version="v3",
            encoding="json",
            total_size=256,
        ),
        source_version_key="models/demo/source/v2",
        serving_version_key="models/demo/serving/v2",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref(),
    )
    runtime = FakeRuntime(client)
    store = Store("fake://daemon", runtime=runtime)
    operation_ref = operation_pb2.OperationRef(
        operation_id="bafkattemptop",
        kind="assembly_attempt",
        target_artifact_id="cgid:assembly-workspace-9",
        authority_scope_kind="assembly_attempt",
        authority_scope_id="cgid:assembly-attempt-9",
        attachment_kind="assembly_attempt",
        recovery_class="cluster_durable",
        fencing_digest="bafk-intent-9",
    )
    attempt = AssemblyAttemptRef(
        attempt_id="cgid:assembly-attempt-9",
        workspace_assembly_id="cgid:assembly-workspace-9",
        layout_id="layout-9",
        attempt_intent_digest="bafk-intent-9",
        coordinator_generation=1,
        coordinator_operation=operation_ref,
    )

    result = store.seal_assembly_attempt(attempt).wait(timeout_s=5.0)

    assert result.source_artifact_id == "mi2:test:source"
    assert result.source_version_key == "models/demo/source/v2"
    assert result.serving_artifact_id == "mi2:test:serving"
    assert result.serving_descriptor is not None
    assert result.serving_descriptor.artifact_id == "mi2:test:serving"
    assert result.serving_version_key == "models/demo/serving/v2"
    assert result.representation_contract_hash == "bafkrepresentation"
    assert result.serving_build_digest == "bafkbuilddigest"
    assert result.serving_manifest_ref == build_serving_manifest_ref()


def test_representation_publish_closeout_contract_requires_typed_child() -> None:
    with pytest.raises(ValueError) as exc_info:
        AssemblyCloseoutContract(kind="representation_publish")

    assert "representation_publish_contract" in str(exc_info.value)


def test_tensor_entry_publication_helpers_use_canonical_publication_names() -> None:
    assert hasattr(Store, "complete_pure_transform_publication")
    assert hasattr(Store, "register_pure_transform_publication")
    assert hasattr(store_api, "complete_pure_transform_publication")
    assert hasattr(store_api, "register_pure_transform_publication")
    assert not hasattr(Store, "complete_binding_finalize_publication")
    assert not hasattr(Store, "register_binding_finalize_publication")
    assert not hasattr(store_api, "complete_binding_finalize_publication")
    assert not hasattr(store_api, "register_binding_finalize_publication")
    assert not hasattr(Store, "complete_pure_transform_publication_bridge")
    assert not hasattr(Store, "complete_binding_finalize_publication_bridge")
    assert not hasattr(Store, "register_pure_transform_publication_bridge")
    assert not hasattr(Store, "register_binding_finalize_publication_bridge")
    assert not hasattr(store_api, "complete_pure_transform_publication_bridge")
    assert not hasattr(store_api, "complete_binding_finalize_publication_bridge")
    assert not hasattr(store_api, "register_pure_transform_publication_bridge")
    assert not hasattr(store_api, "register_binding_finalize_publication_bridge")


def test_representation_publish_contract_from_proto_requires_subject() -> None:
    proto = store_daemon_pb2.RepresentationPublishContract(
        serving_artifact_id="mi2:test:serving",
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
    )

    with pytest.raises(
        ValueError,
        match="RepresentationPublishContract requires a serving publication subject",
    ):
        RepresentationPublishContract.from_proto(proto)


def test_representation_publish_closeout_contract_accepts_matching_typed_child() -> (
    None
):
    contract = AssemblyCloseoutContract(
        kind="representation_publish",
        source_version_key="models/demo/source/v3",
        serving_version_key="models/demo/serving/v3",
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_publish_contract=RepresentationPublishContract(
            subject=ServingPublicationSubject(
                serving_artifact_id="mi2:test:serving",
            ),
            serving_manifest_ref=build_serving_manifest_ref(),
            representation_contract_hash="bafkrepresentation",
            serving_build_digest="bafkbuilddigest",
        ),
    )

    proto = contract.to_proto()

    assert (
        proto.representation_publish_contract.subject.serving_artifact_id
        == "mi2:test:serving"
    )
    assert (
        proto.representation_publish_contract.serving_manifest_ref
        == build_serving_manifest_ref()
    )
    assert (
        proto.representation_publish_contract.representation_contract_hash
        == "bafkrepresentation"
    )
    assert (
        proto.representation_publish_contract.serving_build_digest == "bafkbuilddigest"
    )


def test_representation_publish_closeout_contract_rejects_outer_serving_artifact_id() -> (
    None
):
    with pytest.raises(
        ValueError,
        match="representation_publish closeout contracts must not set serving_artifact_id",
    ):
        AssemblyCloseoutContract(
            kind="representation_publish",
            serving_artifact_id="mi2:test:serving",
            serving_manifest_ref=build_serving_manifest_ref(),
            representation_publish_contract=RepresentationPublishContract(
                subject=ServingPublicationSubject(
                    serving_artifact_id="mi2:test:serving",
                ),
                serving_manifest_ref=build_serving_manifest_ref(),
                representation_contract_hash="bafkrepresentation",
                serving_build_digest="bafkbuilddigest",
            ),
        )


def test_representation_publish_closeout_contract_accepts_binding_subject_child() -> (
    None
):
    contract = AssemblyCloseoutContract(
        kind="representation_publish",
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_publish_contract=RepresentationPublishContract(
            subject=ServingPublicationSubject(
                binding_value_ref=BindingValueRef(
                    binding_id="binding-1",
                    binding_layout_id="layout-1",
                    binding_value_id="value-1",
                    seal_generation=2,
                )
            ),
            serving_manifest_ref=build_serving_manifest_ref(),
            representation_contract_hash="bafkrepresentation",
            serving_build_digest="bafkbuilddigest",
        ),
    )

    proto = contract.to_proto()

    assert not proto.serving_artifact_id
    assert (
        proto.representation_publish_contract.subject.binding_value.binding_id
        == "binding-1"
    )


def test_representation_publish_contract_from_proto_preserves_binding_value_subject() -> (
    None
):
    proto = store_daemon_pb2.RepresentationPublishContract(
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
    )
    proto.subject.binding_value.binding_id = "binding-1"
    proto.subject.binding_value.binding_layout_id = "layout-1"
    proto.subject.binding_value.binding_value_id = "value-1"
    proto.subject.binding_value.seal_generation = 2

    contract = RepresentationPublishContract.from_proto(proto)

    assert contract.subject.serving_artifact_id is None
    assert contract.subject.binding_value_ref is not None
    assert contract.subject.binding_value_ref.binding_id == "binding-1"
    assert contract.subject.binding_value_ref.binding_layout_id == "layout-1"
    assert contract.subject.binding_value_ref.binding_value_id == "value-1"
    assert contract.subject.binding_value_ref.seal_generation == 2
