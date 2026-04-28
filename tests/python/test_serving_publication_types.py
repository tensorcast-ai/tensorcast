#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import inspect

import pytest
import torch

import tensorcast as tc
from tensorcast.api._config import PlanType
from tensorcast.api.plan.transforms import TransformSpec
from tensorcast.api.store import (
    build_binding_finalize_admission_facts,
    build_binding_finalize_publication_bundle,
    build_pure_transform_publication_bundle_from_registered_artifact,
    build_pure_transform_publication_spec,
    build_pure_transform_serving_args,
    build_pure_transform_transform_spec,
    build_serving_publication_bundle_from_registered_artifact,
    compute_pure_transform_representation_contract_hash,
    compute_serving_tensor_schema_hash,
    count_canonical_serving_tensors,
    prepare_binding_finalize_serving_registration,
    prepare_pure_transform_serving_registration,
    prepare_serving_registration,
)
from tensorcast.api.store.handles import RegisteredArtifact
from tensorcast.api.store.types import (
    ArtifactError,
    CanonicalIndex,
    CanonicalIndexEntry,
    ReplicaInfo,
)
from tensorcast.types import (
    SERVING_BUILD_DIGEST_VERSION,
    SERVING_MANIFEST_TENSOR_NAME,
    AssemblyCloseoutContract,
    BindingValueRef,
    BuilderMode,
    FinalizeClass,
    PublishedModelVersion,
    RepresentationPublishContract,
    RepresentationPublishSpec,
    ServingAdmissionFacts,
    ServingArtifactManifest,
    ServingBuildIntent,
    ServingPublicationSubject,
    ServingRuntimePolicy,
    ServingSupportLevel,
    build_serving_manifest_ref,
    coerce_serving_runtime_policy,
    parse_serving_manifest_ref,
)
from tensorcast.types import ArtifactDescriptor as PublishedArtifactDescriptor


def _canonical_index(
    *entries: CanonicalIndexEntry,
    total_size_bytes: int | None = None,
) -> CanonicalIndex:
    if total_size_bytes is None:
        total_size_bytes = sum(int(entry.size_bytes) for entry in entries)
    return CanonicalIndex(
        entries=entries,
        total_size_bytes=int(total_size_bytes),
        avbs_hash="bafkavbs",
    )


def test_serving_build_digest_ignores_source_and_semantic_hash_inputs() -> None:
    intent_a = ServingBuildIntent(
        representation_contract_hash="bafksemantic-a",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        build_pipeline_version="pipeline-v1",
        source_artifact_ref="mi2:source-a",
    )
    intent_b = ServingBuildIntent(
        representation_contract_hash="bafksemantic-b",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v1",
        serving_abi_version="abi-v1",
        build_pipeline_version="pipeline-v1",
        source_artifact_ref="mi2:source-b",
    )

    assert (
        intent_a.compute_serving_build_digest()
        == intent_b.compute_serving_build_digest()
    )


def test_tensorcast_top_level_exports_cover_vllm_serving_contract() -> None:
    assert tc.prepare_serving_registration is prepare_serving_registration
    assert (
        tc.prepare_binding_finalize_serving_registration
        is prepare_binding_finalize_serving_registration
    )
    assert (
        tc.build_serving_publication_bundle_from_registered_artifact
        is build_serving_publication_bundle_from_registered_artifact
    )
    assert tc.PublishedModelVersion is PublishedModelVersion
    assert tc.RepresentationPublishContract is RepresentationPublishContract
    assert tc.ServingAdmissionFacts is ServingAdmissionFacts
    assert tc.ServingArtifactManifest is ServingArtifactManifest
    assert tc.SERVING_BUILD_DIGEST_VERSION == SERVING_BUILD_DIGEST_VERSION
    assert tc.ServingRuntimePolicy is ServingRuntimePolicy


def test_serving_artifact_manifest_round_trips_via_json_payload() -> None:
    intent = ServingBuildIntent(
        representation_contract_hash="bafksemantic",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v2",
        serving_abi_version="abi-v2",
        build_pipeline_version="pipeline-v2",
        source_artifact_ref="mi2:source",
    )
    manifest = ServingArtifactManifest.from_build_intent(
        intent=intent,
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=17,
    )

    restored = ServingArtifactManifest.from_bytes(manifest.to_bytes())

    assert restored == manifest
    assert restored.serving_manifest_ref == build_serving_manifest_ref()
    assert restored.serving_build_digest == intent.compute_serving_build_digest()
    assert restored.serving_build_digest_version == SERVING_BUILD_DIGEST_VERSION


def test_representation_publish_contract_matches_serving_manifest() -> None:
    intent = ServingBuildIntent(
        representation_contract_hash="bafksemantic",
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v3",
        serving_abi_version="abi-v3",
        build_pipeline_version="pipeline-v3",
    )
    manifest = ServingArtifactManifest.from_build_intent(
        intent=intent,
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=9,
    )
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            serving_artifact_id="mi2:test:serving",
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash=manifest.representation_contract_hash,
        serving_build_digest=manifest.serving_build_digest,
    )

    contract.validate_against_manifest(manifest)
    assert (
        parse_serving_manifest_ref(contract.serving_manifest_ref)
        == "__tensorcast_meta__.manifest_json"
    )
    runtime_policy = contract.to_runtime_policy()
    assert runtime_policy.require_manifest is True
    assert runtime_policy.serving_manifest_ref == build_serving_manifest_ref()
    assert (
        runtime_policy.expected_representation_contract_hash
        == manifest.representation_contract_hash
    )


def test_serving_admission_facts_require_fast_path_validation() -> None:
    with pytest.raises(ValueError, match="same_binding_fast_path_validated=True"):
        ServingAdmissionFacts(
            finalize_class=FinalizeClass.REPRESENTATION_CHANGING,
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
            same_binding_fast_path_validated=False,
        )


def test_representation_publish_contract_accepts_binding_value_subject() -> None:
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            binding_value_ref=BindingValueRef(
                binding_id="binding-1",
                binding_layout_id="layout-1",
                binding_value_id="value-1",
                seal_generation=7,
            )
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
    )

    assert contract.serving_artifact_id is None
    assert contract.binding_value_ref is not None
    restored = RepresentationPublishContract.from_publication_proto(
        contract.to_publication_proto()
    )
    assert restored.binding_value_ref is not None
    assert restored.binding_value_ref.binding_id == "binding-1"


def test_binding_subject_contract_rejects_runtime_policy_until_promoted() -> None:
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            binding_value_ref=BindingValueRef(
                binding_id="binding-2",
                binding_layout_id="layout-2",
                binding_value_id="value-2",
                seal_generation=3,
            )
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
    )

    with pytest.raises(ValueError, match="binding publication subjects"):
        contract.to_runtime_policy()


def test_build_binding_finalize_admission_facts_requires_same_binding_proof() -> None:
    facts = build_binding_finalize_admission_facts(
        support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
        same_binding_fast_path_validated=True,
    )

    assert facts.finalize_class == FinalizeClass.REPRESENTATION_CHANGING
    assert facts.same_binding_fast_path_validated is True
    assert facts.support_level == ServingSupportLevel.BUILDER_PUBLICATION_READY


def test_build_binding_finalize_publication_bundle_has_no_artifact_subject_parameter() -> (
    None
):
    assert (
        "serving_artifact"
        not in inspect.signature(build_binding_finalize_publication_bundle).parameters
    )

    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        framework_name="torch",
        adapter_version="adapter-mounted-source",
        serving_abi_version="abi-mounted-source",
        build_pipeline_version="pipeline-mounted-source",
        source_artifact_ref="mi2:test:source",
        builder_mode=BuilderMode.BINDING_FINALIZE,
    )

    with pytest.raises(ArtifactError, match="binding_value_ref"):
        build_binding_finalize_publication_bundle(
            build_intent=intent,
            canonical_index=canonical_index,
            admission_facts=build_binding_finalize_admission_facts(
                support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
                same_binding_fast_path_validated=True,
            ),
        )


def test_build_binding_finalize_publication_bundle_accepts_binding_value_subject() -> (
    None
):
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-mounted-source",
        serving_abi_version="abi-mounted-source",
        build_pipeline_version="pipeline-mounted-source",
        source_artifact_ref="msa1:test-session~policy~safetensors~deadbeef",
    )

    bundle = build_binding_finalize_publication_bundle(
        build_intent=intent,
        source_artifact="msa1:test-session~policy~safetensors~deadbeef",
        publication_subject=BindingValueRef(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
        ),
        canonical_index=canonical_index,
        admission_facts=build_binding_finalize_admission_facts(
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
            same_binding_fast_path_validated=True,
        ),
    )

    assert bundle.admission_facts is not None
    assert bundle.admission_facts.same_binding_fast_path_validated is True
    assert bundle.representation_publish_contract.binding_value_ref is not None


def test_compute_serving_tensor_schema_hash_excludes_reserved_manifest_tensor() -> None:
    canonical_without_manifest = CanonicalIndex(
        entries=(
            CanonicalIndexEntry(
                name="weights",
                dtype=torch.float16,
                shape=(8,),
                stride=(1,),
                storage_offset=0,
                segment_offset=0,
                size_bytes=16,
            ),
        ),
        total_size_bytes=16,
        avbs_hash="bafkavbs",
    )
    canonical_with_manifest = CanonicalIndex(
        entries=canonical_without_manifest.entries
        + (
            CanonicalIndexEntry(
                name="__tensorcast_meta__.manifest_json",
                dtype=torch.uint8,
                shape=(32,),
                stride=(1,),
                storage_offset=0,
                segment_offset=16,
                size_bytes=32,
            ),
        ),
        total_size_bytes=48,
        avbs_hash="bafkavbs",
    )

    assert compute_serving_tensor_schema_hash(
        canonical_with_manifest
    ) == compute_serving_tensor_schema_hash(canonical_without_manifest)
    assert count_canonical_serving_tensors(canonical_with_manifest) == 1


def test_compute_pure_transform_representation_contract_hash_accepts_tensor_mapping() -> (
    None
):
    source_canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
    )
    serving_tensors = {
        "weights": torch.empty((8,), dtype=torch.float16),
        SERVING_MANIFEST_TENSOR_NAME: torch.tensor(
            list(b'{"schema_version":1}'),
            dtype=torch.uint8,
        ),
    }

    hash_from_tensors = compute_pure_transform_representation_contract_hash(
        source_artifact=source_canonical_index,
        serving_artifact=serving_tensors,
    )
    hash_from_index = compute_pure_transform_representation_contract_hash(
        source_artifact=source_canonical_index,
        serving_artifact=_canonical_index(
            CanonicalIndexEntry(
                name="weights",
                dtype=torch.float16,
                shape=(8,),
                stride=(1,),
                storage_offset=0,
                segment_offset=0,
                size_bytes=16,
            ),
        ),
    )

    assert hash_from_tensors == hash_from_index


def test_build_pure_transform_publication_bundle_from_registered_artifact() -> None:
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    registered_artifact = RegisteredArtifact(
        artifact_id="mi2:test:serving",
        replica=ReplicaInfo(
            replica_id="mi2:test:serving",
            replica_type="COALESCED_VRAM",
            device=torch.device("cuda", 0),
            plan=PlanType.VRAM_COALESCED,
            size_bytes=80,
        ),
        canonical_index=canonical_index,
        lease=None,
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkrepresentation",
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v4",
        serving_abi_version="abi-v4",
        build_pipeline_version="pipeline-v4",
        source_artifact_ref="mi2:test:source",
    )

    bundle = build_pure_transform_publication_bundle_from_registered_artifact(
        build_intent=intent,
        serving_artifact=registered_artifact,
        source_version_key="models/demo/source/v4",
        serving_version_key="models/demo/serving/v4",
    )

    assert isinstance(bundle, RepresentationPublishSpec)
    assert bundle.serving_artifact_id == "mi2:test:serving"
    assert bundle.manifest_tensor_name == "__tensorcast_meta__.manifest_json"
    assert bundle.serving_manifest_ref == build_serving_manifest_ref()
    assert bundle.serving_manifest.canonical_tensor_count == 1
    assert (
        bundle.serving_manifest.tensor_schema_hash
        == compute_serving_tensor_schema_hash(canonical_index)
    )
    assert (
        bundle.representation_publish_contract.serving_artifact_id == "mi2:test:serving"
    )
    assert (
        bundle.closeout_contract.representation_publish_contract
        == bundle.representation_publish_contract
    )
    assert bundle.closeout_contract.kind == "representation_publish"
    assert (
        ServingArtifactManifest.from_bytes(bundle.serving_manifest_bytes)
        == bundle.serving_manifest
    )
    assert (
        bundle.representation_publish_contract.serving_build_digest_version
        == SERVING_BUILD_DIGEST_VERSION
    )


def test_compute_pure_transform_representation_contract_hash_normalizes_logical_topology() -> (
    None
):
    source_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="bias",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=16,
        ),
    )
    serving_index = _canonical_index(
        CanonicalIndexEntry(
            name="bias",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
    )

    hash_a = compute_pure_transform_representation_contract_hash(
        source_artifact=source_index,
        serving_artifact=serving_index,
        logical_topology_json='{"family":"tp","version":"v1","dimensions":[{"name":"tp","size":2},{"name":"pp","size":1}]}',
    )
    hash_b = compute_pure_transform_representation_contract_hash(
        source_artifact=source_index,
        serving_artifact=serving_index,
        logical_topology_json='{"dimensions":[{"name":"pp","size":1},{"name":"tp","size":2}],"version":"v1","family":"tp"}',
    )
    hash_c = compute_pure_transform_representation_contract_hash(
        source_artifact=source_index,
        serving_artifact=serving_index,
        logical_topology_json='{"family":"tp","version":"v1","dimensions":[{"name":"tp","size":4},{"name":"pp","size":1}]}',
    )

    assert hash_a == hash_b
    assert hash_a != hash_c


def test_build_pure_transform_publication_bundle_auto_derives_representation_hash() -> (
    None
):
    source_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
    )
    serving_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    registered_artifact = RegisteredArtifact(
        artifact_id="mi2:test:auto-serving",
        replica=ReplicaInfo(
            replica_id="mi2:test:auto-serving",
            replica_type="COALESCED_VRAM",
            device=torch.device("cuda", 0),
            plan=PlanType.VRAM_COALESCED,
            size_bytes=80,
        ),
        canonical_index=serving_index,
        lease=None,
    )
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v4-auto",
        serving_abi_version="abi-v4-auto",
        build_pipeline_version="pipeline-v4-auto",
        source_artifact_ref="mi2:test:auto-source",
    )

    bundle = build_pure_transform_publication_bundle_from_registered_artifact(
        build_intent=intent,
        source_artifact=source_index,
        contract_family="pp",
        serving_artifact=registered_artifact,
        source_version_key="models/demo/source/auto",
        serving_version_key="models/demo/serving/auto",
        logical_topology_json='{"family":"tp","version":"v1","dimensions":[{"name":"tp","size":2}]}',
    )

    expected_hash = compute_pure_transform_representation_contract_hash(
        source_artifact=source_index,
        serving_artifact=registered_artifact,
        logical_topology_json='{"family":"tp","version":"v1","dimensions":[{"name":"tp","size":2}]}',
    )
    assert bundle.representation_publish_contract.representation_contract_hash == (
        expected_hash
    )
    assert bundle.contract_family == "pp"
    assert bundle.serving_manifest.representation_contract_hash == expected_hash


def test_prepare_pure_transform_serving_registration_embeds_manifest_tensor() -> None:
    source_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
    )
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v4-prep",
        serving_abi_version="abi-v4-prep",
        build_pipeline_version="pipeline-v4-prep",
        source_artifact_ref="mi2:test:prep-source",
    )

    prepared = prepare_pure_transform_serving_registration(
        build_intent=intent,
        source_artifact=source_index,
        tensors={"weights": torch.zeros(8, dtype=torch.float16)},
    )

    assert prepared.manifest_tensor_name == "__tensorcast_meta__.manifest_json"
    assert "__tensorcast_meta__.manifest_json" in prepared.tensors
    assert len(prepared.serving_manifest_bytes) % 8 == 0
    assert prepared.canonical_index.total_size_bytes == sum(
        int(entry.size_bytes) for entry in prepared.canonical_index.entries
    )
    assert (
        ServingArtifactManifest.from_bytes(prepared.serving_manifest_bytes)
        == prepared.serving_manifest
    )
    assert (
        ServingArtifactManifest.from_bytes(
            bytes(prepared.tensors["__tensorcast_meta__.manifest_json"].tolist())
        )
        == prepared.serving_manifest
    )
    assert prepared.representation_contract_hash == (
        compute_pure_transform_representation_contract_hash(
            source_artifact=source_index,
            serving_artifact=prepared.canonical_index,
        )
    )


def test_prepare_serving_registration_supports_binding_finalize() -> None:
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v4-binding",
        serving_abi_version="abi-v4-binding",
        build_pipeline_version="pipeline-v4-binding",
        source_artifact_ref="mi2:test:binding-source",
    )

    prepared = prepare_serving_registration(
        build_intent=intent,
        tensors={"weights": torch.ones(8, dtype=torch.float16)},
        representation_contract_hash="bafkbindingrepr",
    )

    assert prepared.manifest_tensor_name == "__tensorcast_meta__.manifest_json"
    assert "__tensorcast_meta__.manifest_json" in prepared.tensors
    assert prepared.serving_manifest.builder_mode == BuilderMode.BINDING_FINALIZE
    assert prepared.serving_manifest.framework_name == "torch"
    assert prepared.representation_contract_hash == "bafkbindingrepr"


def test_prepare_binding_finalize_serving_registration_requires_binding_finalize() -> (
    None
):
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v4-wrong",
        serving_abi_version="abi-v4-wrong",
        build_pipeline_version="pipeline-v4-wrong",
        source_artifact_ref="mi2:test:binding-source",
    )

    with pytest.raises(Exception, match="builder_mode=BINDING_FINALIZE"):
        prepare_binding_finalize_serving_registration(
            build_intent=intent,
            tensors={"weights": torch.ones(8, dtype=torch.float16)},
            representation_contract_hash="bafkbindingrepr",
        )


def test_prepare_binding_finalize_serving_registration_supports_binding_finalize() -> (
    None
):
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v4-binding-helper",
        serving_abi_version="abi-v4-binding-helper",
        build_pipeline_version="pipeline-v4-binding-helper",
        source_artifact_ref="mi2:test:binding-source",
    )

    prepared = prepare_binding_finalize_serving_registration(
        build_intent=intent,
        tensors={"weights": torch.ones(8, dtype=torch.float16)},
        representation_contract_hash="bafkbindingrepr",
    )

    assert prepared.serving_manifest.builder_mode == BuilderMode.BINDING_FINALIZE


def test_prepare_serving_registration_keeps_manifest_on_tensor_device() -> None:
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v4-device",
        serving_abi_version="abi-v4-device",
        build_pipeline_version="pipeline-v4-device",
        source_artifact_ref="mi2:test:binding-source",
    )

    prepared = prepare_serving_registration(
        build_intent=intent,
        tensors={"weights": torch.ones(8, dtype=torch.float16, device=device)},
        representation_contract_hash="bafkbindingrepr",
    )

    assert prepared.tensors["weights"].device == device
    assert prepared.tensors["__tensorcast_meta__.manifest_json"].device == device


def test_build_serving_publication_bundle_from_registered_artifact_rejects_binding_finalize() -> (
    None
):
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    registered_artifact = RegisteredArtifact(
        artifact_id="mi2:test:serving-binding-finalize",
        replica=ReplicaInfo(
            replica_id="mi2:test:serving-binding-finalize",
            replica_type="COALESCED_VRAM",
            device=torch.device("cuda", 0),
            plan=PlanType.VRAM_COALESCED,
            size_bytes=80,
        ),
        canonical_index=canonical_index,
        lease=None,
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v4-bundle",
        serving_abi_version="abi-v4-bundle",
        build_pipeline_version="pipeline-v4-bundle",
        source_artifact_ref="mi2:test:binding-source",
    )

    with pytest.raises(ValueError, match="binding_value_ref subject"):
        build_serving_publication_bundle_from_registered_artifact(
            build_intent=intent,
            serving_artifact=registered_artifact,
            source_version_key="models/demo/source/v4",
            serving_version_key="models/demo/serving/v4",
        )


def test_build_binding_finalize_publication_bundle_uses_admission_facts() -> None:
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v4-bundle-helper",
        serving_abi_version="abi-v4-bundle-helper",
        build_pipeline_version="pipeline-v4-bundle-helper",
        source_artifact_ref="mi2:test:binding-source",
    )
    admission_facts = build_binding_finalize_admission_facts(
        support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        topology_admission_digest="bafktopology",
        same_binding_fast_path_validated=True,
    )

    bundle = build_binding_finalize_publication_bundle(
        build_intent=intent,
        publication_subject=BindingValueRef(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
        ),
        canonical_index=canonical_index,
        source_version_key="models/demo/source/v4",
        serving_version_key="models/demo/serving/v4",
        admission_facts=admission_facts,
    )

    assert bundle.serving_manifest.builder_mode == BuilderMode.BINDING_FINALIZE
    assert bundle.admission_facts == admission_facts


def test_build_binding_finalize_publication_bundle_rejects_serving_key_without_runtime_ready() -> (
    None
):
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-v4-bundle-helper",
        serving_abi_version="abi-v4-bundle-helper",
        build_pipeline_version="pipeline-v4-bundle-helper",
        source_artifact_ref="mi2:test:binding-source",
    )

    with pytest.raises(ValueError, match="serving_version_key activation"):
        build_binding_finalize_publication_bundle(
            build_intent=intent,
            publication_subject=BindingValueRef(
                binding_id="binding-1",
                binding_layout_id="layout-1",
                binding_value_id="value-1",
                seal_generation=1,
            ),
            canonical_index=canonical_index,
            serving_version_key="models/demo/serving/v4",
            admission_facts=build_binding_finalize_admission_facts(
                support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
                same_binding_fast_path_validated=True,
            ),
        )


def test_build_pure_transform_serving_args_encodes_repo_owned_keys() -> None:
    intent = ServingBuildIntent(
        representation_contract_hash="bafkrepresentation",
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v5",
        serving_abi_version="abi-v5",
        build_pipeline_version="pipeline-v5",
        source_artifact_ref="mi2:test:source",
    )

    args = build_pure_transform_serving_args(
        build_intent=intent,
        contract_family="canonical_full",
        source_version_key="models/demo/source/v5",
        serving_version_key="models/demo/serving/v5",
        extra_args={"quant": 4},
    )

    assert args["tc_serving_enable"] == 1
    assert args["tc_serving_representation_contract_hash"] == "bafkrepresentation"
    assert args["tc_serving_framework_name"] == "torch"
    assert args["tc_serving_adapter_version"] == "adapter-v5"
    assert args["tc_serving_abi_version"] == "abi-v5"
    assert args["tc_serving_build_pipeline_version"] == "pipeline-v5"
    assert args["tc_serving_contract_family"] == "canonical_full"
    assert args["tc_serving_source_version_key"] == "models/demo/source/v5"
    assert args["tc_serving_serving_version_key"] == "models/demo/serving/v5"
    assert args["quant"] == 4


def test_build_pure_transform_serving_args_omits_unresolved_representation_hash() -> (
    None
):
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v5-auto",
        serving_abi_version="abi-v5-auto",
        build_pipeline_version="pipeline-v5-auto",
    )

    args = build_pure_transform_serving_args(
        build_intent=intent,
        serving_version_key="models/demo/serving/v5-auto",
    )

    assert args["tc_serving_enable"] == 1
    assert "tc_serving_representation_contract_hash" not in args
    assert args["tc_serving_serving_version_key"] == "models/demo/serving/v5-auto"


def test_published_model_version_builds_serving_runtime_policy() -> None:
    version = PublishedModelVersion(
        assembly_id="cgid:test-assembly",
        source_artifact_id="mi2:test:source",
        source_descriptor=PublishedArtifactDescriptor(
            artifact_id="mi2:test:source",
            total_size=16,
        ),
        serving_artifact_id="mi2:test:serving",
        serving_descriptor=PublishedArtifactDescriptor(
            artifact_id="mi2:test:serving",
            total_size=32,
        ),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref("__serving_manifest__.json"),
    )

    policy = version.require_serving_runtime_policy()

    assert isinstance(policy, ServingRuntimePolicy)
    assert policy.require_manifest is True
    assert policy.serving_manifest_ref == "tensor:__serving_manifest__.json"
    assert policy.expected_representation_contract_hash == "bafkrepresentation"
    assert policy.expected_serving_build_digest == "bafkbuilddigest"


def test_coerce_serving_runtime_policy_accepts_manifest_lineage_models() -> None:
    manifest = ServingArtifactManifest(
        framework_name="torch",
        adapter_version="adapter-v6",
        serving_abi_version="abi-v6",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=1,
        serving_manifest_ref=build_serving_manifest_ref("__alt_manifest__.json"),
        builder_mode=BuilderMode.PURE_TRANSFORM,
        build_pipeline_version="pipeline-v6",
    )

    policy = coerce_serving_runtime_policy(manifest)

    assert isinstance(policy, ServingRuntimePolicy)
    assert policy.serving_manifest_ref == "tensor:__alt_manifest__.json"
    assert policy.expected_representation_contract_hash == "bafkrepresentation"
    assert policy.expected_serving_build_digest == "bafkbuilddigest"


def test_coerce_serving_runtime_policy_accepts_contract_and_version() -> None:
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            serving_artifact_id="mi2:test:serving",
        ),
        serving_manifest_ref=build_serving_manifest_ref("__alt_manifest__.json"),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
    )
    version = PublishedModelVersion(
        assembly_id="cgid:test-assembly",
        source_artifact_id="mi2:test:source",
        source_descriptor=PublishedArtifactDescriptor(
            artifact_id="mi2:test:source",
            total_size=16,
        ),
        serving_artifact_id="mi2:test:serving",
        serving_descriptor=PublishedArtifactDescriptor(
            artifact_id="mi2:test:serving",
            total_size=32,
        ),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_manifest_ref=build_serving_manifest_ref("__alt_manifest__.json"),
    )

    contract_policy = coerce_serving_runtime_policy(contract)
    version_policy = coerce_serving_runtime_policy(version)

    assert contract_policy == ServingRuntimePolicy(
        require_manifest=True,
        serving_manifest_ref="tensor:__alt_manifest__.json",
        expected_representation_contract_hash="bafkrepresentation",
        expected_serving_build_digest="bafkbuilddigest",
    )
    assert version_policy == contract_policy


def test_coerce_serving_runtime_policy_accepts_runtime_ready_representation_publish_spec() -> (
    None
):
    manifest = ServingArtifactManifest(
        framework_name="torch",
        adapter_version="adapter-v6-runtime",
        serving_abi_version="abi-v6-runtime",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=1,
        builder_mode=BuilderMode.PURE_TRANSFORM,
        build_pipeline_version="pipeline-v6-runtime",
    )
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            serving_artifact_id="mi2:test:serving",
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        serving_build_digest_version=SERVING_BUILD_DIGEST_VERSION,
    )
    spec = RepresentationPublishSpec(
        serving_artifact_id="mi2:test:serving",
        serving_manifest_ref=build_serving_manifest_ref(),
        serving_manifest=manifest,
        serving_manifest_bytes=manifest.to_bytes(),
        representation_publish_contract=contract,
        closeout_contract=AssemblyCloseoutContract(
            kind="representation_publish",
            representation_publish_contract=contract,
        ),
        admission_facts=ServingAdmissionFacts(
            finalize_class=FinalizeClass.RUNTIME_ONLY,
            support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        ),
    )

    policy = coerce_serving_runtime_policy(spec)

    assert policy == ServingRuntimePolicy(
        require_manifest=True,
        serving_manifest_ref=build_serving_manifest_ref(),
        expected_representation_contract_hash="bafkrepresentation",
        expected_serving_build_digest="bafkbuilddigest",
    )


def test_coerce_serving_runtime_policy_rejects_builder_only_representation_publish_spec() -> (
    None
):
    manifest = ServingArtifactManifest(
        framework_name="torch",
        adapter_version="adapter-v6-runtime-blocked",
        serving_abi_version="abi-v6-runtime-blocked",
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
        tensor_schema_hash="bafktensorschema",
        canonical_tensor_count=1,
        builder_mode=BuilderMode.PURE_TRANSFORM,
        build_pipeline_version="pipeline-v6-runtime-blocked",
    )
    contract = RepresentationPublishContract(
        subject=ServingPublicationSubject(
            serving_artifact_id="mi2:test:serving",
        ),
        serving_manifest_ref=build_serving_manifest_ref(),
        representation_contract_hash="bafkrepresentation",
        serving_build_digest="bafkbuilddigest",
    )
    spec = RepresentationPublishSpec(
        serving_artifact_id="mi2:test:serving",
        serving_manifest_ref=build_serving_manifest_ref(),
        serving_manifest=manifest,
        serving_manifest_bytes=manifest.to_bytes(),
        representation_publish_contract=contract,
        closeout_contract=AssemblyCloseoutContract(
            kind="representation_publish",
            representation_publish_contract=contract,
        ),
        admission_facts=ServingAdmissionFacts(
            finalize_class=FinalizeClass.RUNTIME_ONLY,
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
        ),
    )

    with pytest.raises(ValueError, match="RUNTIME_BIND_SWAP_READY"):
        coerce_serving_runtime_policy(spec)


def test_build_pure_transform_transform_spec_wraps_transform_args() -> None:
    intent = ServingBuildIntent(
        representation_contract_hash="bafkrepresentation",
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v6",
        serving_abi_version="abi-v6",
        build_pipeline_version="pipeline-v6",
    )

    spec = build_pure_transform_transform_spec(
        transform_name="identity.v1",
        build_intent=intent,
        contract_family="pp",
        transform_args={"quant": 8},
        serving_version_key="models/demo/serving/v6",
        layout_hash="layout-v6",
    )

    assert isinstance(spec, TransformSpec)
    assert spec.name == "identity.v1"
    assert spec.layout_hash == "layout-v6"
    assert spec.args["quant"] == 8
    assert spec.publication_spec is not None
    assert spec.publication_spec.contract_family == "pp"
    assert spec.publication_spec.serving_version_key == "models/demo/serving/v6"


def test_build_pure_transform_transform_spec_can_omit_representation_hash() -> None:
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v6-auto",
        serving_abi_version="abi-v6-auto",
        build_pipeline_version="pipeline-v6-auto",
    )

    spec = build_pure_transform_transform_spec(
        transform_name="identity.v1",
        build_intent=intent,
    )

    assert spec.publication_spec is not None
    assert spec.publication_spec.build_intent.representation_contract_hash is None


def test_build_pure_transform_publication_spec_wraps_typed_inputs() -> None:
    admission_facts = ServingAdmissionFacts(
        finalize_class=FinalizeClass.RUNTIME_ONLY,
        support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        topology_admission_digest="bafktopology",
    )
    intent = ServingBuildIntent(
        builder_mode=BuilderMode.PURE_TRANSFORM,
        framework_name="torch",
        adapter_version="adapter-v7",
        serving_abi_version="abi-v7",
        build_pipeline_version="pipeline-v7",
    )

    publication_spec = build_pure_transform_publication_spec(
        build_intent=intent,
        contract_family="canonical_full",
        source_version_key="models/demo/source/v7",
        serving_version_key="models/demo/serving/v7",
        serving_manifest_ref=build_serving_manifest_ref("__alt_manifest__.json"),
        structural_view_ids=("view-a",),
        admission_facts=admission_facts,
    )

    assert publication_spec.build_intent.framework_name == "torch"
    assert publication_spec.contract_family == "canonical_full"
    assert publication_spec.source_version_key == "models/demo/source/v7"
    assert publication_spec.serving_version_key == "models/demo/serving/v7"
    assert publication_spec.serving_manifest_ref == "tensor:__alt_manifest__.json"
    assert publication_spec.structural_view_ids == ("view-a",)
    assert publication_spec.admission_facts == admission_facts


def test_representation_publish_spec_round_trips_admission_facts_and_digest_version() -> (
    None
):
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    registered_artifact = RegisteredArtifact(
        artifact_id="mi2:test:spec-roundtrip",
        replica=ReplicaInfo(
            replica_id="mi2:test:spec-roundtrip",
            replica_type="COALESCED_VRAM",
            device=torch.device("cuda", 0),
            plan=PlanType.VRAM_COALESCED,
            size_bytes=80,
        ),
        canonical_index=canonical_index,
        lease=None,
    )
    admission_facts = ServingAdmissionFacts(
        finalize_class=FinalizeClass.RUNTIME_ONLY,
        support_level=ServingSupportLevel.RUNTIME_BIND_SWAP_READY,
        topology_admission_digest="bafktopology",
    )
    bundle = build_pure_transform_publication_bundle_from_registered_artifact(
        build_intent=ServingBuildIntent(
            representation_contract_hash="bafkrepresentation",
            builder_mode=BuilderMode.PURE_TRANSFORM,
            framework_name="torch",
            adapter_version="adapter-v8",
            serving_abi_version="abi-v8",
            build_pipeline_version="pipeline-v8",
            source_artifact_ref="mi2:test:source",
        ),
        serving_artifact=registered_artifact,
        serving_version_key="models/demo/serving/v8",
        admission_facts=admission_facts,
    )

    restored = RepresentationPublishSpec.from_proto(bundle.to_proto())

    assert restored.admission_facts == admission_facts
    assert (
        restored.representation_publish_contract.serving_build_digest_version
        == SERVING_BUILD_DIGEST_VERSION
    )


def test_topology_admission_digest_does_not_change_representation_or_build_identity() -> (
    None
):
    canonical_index = _canonical_index(
        CanonicalIndexEntry(
            name="weights",
            dtype=torch.float16,
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            segment_offset=0,
            size_bytes=16,
        ),
        CanonicalIndexEntry(
            name="__tensorcast_meta__.manifest_json",
            dtype=torch.uint8,
            shape=(64,),
            stride=(1,),
            storage_offset=0,
            segment_offset=16,
            size_bytes=64,
        ),
        total_size_bytes=80,
    )
    intent = ServingBuildIntent(
        representation_contract_hash="bafkbindingrepr",
        builder_mode=BuilderMode.BINDING_FINALIZE,
        framework_name="torch",
        adapter_version="adapter-vtopology",
        serving_abi_version="abi-vtopology",
        build_pipeline_version="pipeline-vtopology",
    )

    binding_value = BindingValueRef(
        binding_id="binding-topology",
        binding_layout_id="layout-topology",
        binding_value_id="value-topology",
        seal_generation=1,
    )
    bundle_a = build_binding_finalize_publication_bundle(
        build_intent=intent,
        publication_subject=binding_value,
        canonical_index=canonical_index,
        admission_facts=build_binding_finalize_admission_facts(
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
            topology_admission_digest="bafktopology-a",
            same_binding_fast_path_validated=True,
        ),
    )
    bundle_b = build_binding_finalize_publication_bundle(
        build_intent=intent,
        publication_subject=binding_value,
        canonical_index=canonical_index,
        admission_facts=build_binding_finalize_admission_facts(
            support_level=ServingSupportLevel.BUILDER_PUBLICATION_READY,
            topology_admission_digest="bafktopology-b",
            same_binding_fast_path_validated=True,
        ),
    )

    assert (
        bundle_a.representation_publish_contract.representation_contract_hash
        == bundle_b.representation_publish_contract.representation_contract_hash
    )
    assert (
        bundle_a.representation_publish_contract.serving_build_digest
        == bundle_b.representation_publish_contract.serving_build_digest
    )
    assert bundle_a.admission_facts != bundle_b.admission_facts
