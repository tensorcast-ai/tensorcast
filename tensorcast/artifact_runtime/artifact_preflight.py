#  Copyright (c) 2026, TensorCast Team.
"""Runtime artifact resolution and manifest preflight helpers.

The lifecycle integration layer should orchestrate loads and swaps; this module
owns the artifact-centered checks that decide whether a resolved runtime
artifact is admissible for the current framework placement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import tensorcast as tc
import tensorcast.artifact_runtime.contract as tc_contract
from tensorcast.artifact_runtime.artifact.resolver import (
    ResolvedRuntimeArtifact,
    RuntimeArtifactResolver,
)
from tensorcast.artifact_runtime.dto import RuntimePlacement
from tensorcast.artifact_runtime.errors import (
    ArtifactRuntimeIntegrationError,
    ManifestMismatchError,
)


@dataclass(frozen=True)
class RuntimeArtifactPreflight:
    resolved_artifact: ResolvedRuntimeArtifact
    runtime_artifact_policy: Any | None


def artifact_locator_kind(artifact_locator: object) -> str:
    if isinstance(artifact_locator, Mapping):
        return str(artifact_locator.get("kind") or "")
    return str(getattr(artifact_locator, "kind", "") or "")


def runtime_policy(policy: Any | None) -> Any | None:
    to_runtime_policy = getattr(policy, "to_runtime_policy", None)
    if callable(to_runtime_policy):
        return to_runtime_policy()
    return policy


def runtime_policy_with_placement(
    policy: Any | None,
    placement: RuntimePlacement | None,
) -> Any | None:
    digest = _optional_text(
        getattr(getattr(placement, "topology", None), "schema_topology_digest", None)
    )
    if digest is None:
        return policy
    if policy is None:
        return tc.RuntimeArtifactPolicy(
            require_manifest=True,
            expected_topology_admission_digest=digest,
        )
    model_copy = getattr(policy, "model_copy", None)
    if callable(model_copy):
        return model_copy(
            update={
                "require_manifest": True,
                "expected_topology_admission_digest": digest,
            }
        )
    return policy


def runtime_policy_from_manifest(
    policy: Any | None,
    resolved: Any,
    placement: RuntimePlacement | None = None,
) -> Any | None:
    if policy is not None:
        return runtime_policy_with_placement(policy, placement)
    manifest = getattr(resolved, "manifest", None)
    to_runtime_policy = getattr(manifest, "to_runtime_policy", None)
    if callable(to_runtime_policy):
        return runtime_policy_with_placement(to_runtime_policy(), placement)
    return runtime_policy_with_placement(None, placement)


def validate_resolved_artifact_placement(
    resolved_artifact: Any,
    *,
    placement: RuntimePlacement | None,
) -> None:
    manifest = getattr(resolved_artifact, "manifest", None)
    if manifest is None:
        return
    manifest_topology_digest = _optional_text(
        getattr(manifest, "topology_admission_digest", None)
    )
    placement_topology_digest = _optional_text(
        getattr(getattr(placement, "topology", None), "schema_topology_digest", None)
    )
    if manifest_topology_digest is not None:
        if placement_topology_digest is None:
            raise ManifestMismatchError(
                "TensorCast runtime artifact topology admission digest "
                "requires current framework placement"
            )
        if manifest_topology_digest != placement_topology_digest:
            raise ManifestMismatchError(
                "TensorCast runtime artifact topology admission digest mismatch: "
                f"manifest={manifest_topology_digest}, "
                f"current={placement_topology_digest}"
            )

    manifest_logical_topology = _optional_text(
        getattr(manifest, "logical_topology_json", None)
    )
    if manifest_logical_topology is None:
        return
    if placement is None:
        raise ManifestMismatchError(
            "TensorCast runtime artifact logical topology requires current "
            "framework placement"
        )
    try:
        current_logical_topology = tc_contract.logical_topology_json(
            placement.topology,
            framework_payload=dict(getattr(placement, "framework_payload", {})),
        )
    except Exception as exc:
        raise ManifestMismatchError(
            "TensorCast runtime artifact logical topology could not be "
            "computed from current framework placement"
        ) from exc
    if _json_object_payload(
        manifest_logical_topology, field_name="logical_topology_json"
    ) != _json_object_payload(
        current_logical_topology, field_name="current logical topology"
    ):
        raise ManifestMismatchError(
            "TensorCast runtime artifact logical topology mismatch"
        )


def resolve_runtime_artifact(
    artifact_ref: str,
    *,
    resolver: RuntimeArtifactResolver | None = None,
    manifest_tensor_name: str | None = None,
    schema_version: int | None = None,
    expected_tensor_schema_hash: str | None = None,
    runtime_artifact_policy: Any | None = None,
) -> ResolvedRuntimeArtifact:
    """Resolve a runtime artifact and optionally cross-check runtime schema."""

    resolved_resolver = resolver or RuntimeArtifactResolver(
        manifest_tensor_name=manifest_tensor_name or tc.SERVING_MANIFEST_TENSOR_NAME,
        schema_version=(
            schema_version
            if schema_version is not None
            else int(tc.RuntimeArtifactManifest.model_fields["schema_version"].default)
        ),
    )
    resolved = resolved_resolver.resolve(str(artifact_ref))
    if expected_tensor_schema_hash is not None:
        resolved_resolver.cross_check(
            resolved,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            runtime_artifact_policy=runtime_artifact_policy,
        )
    return resolved


def read_runtime_artifact_manifest(
    artifact: Any,
    *,
    artifact_ref: str,
    resolver: RuntimeArtifactResolver,
) -> ResolvedRuntimeArtifact:
    """Read a runtime manifest from an already opened artifact handle."""

    return resolver.read_manifest(artifact, artifact_ref=str(artifact_ref))


def cross_check_runtime_artifact(
    resolved_artifact: ResolvedRuntimeArtifact,
    *,
    resolver: RuntimeArtifactResolver,
    expected_tensor_schema_hash: str,
    runtime_artifact_policy: Any | None = None,
) -> ResolvedRuntimeArtifact:
    """Validate manifest, descriptor schema, and runtime policy agreement."""

    return resolver.cross_check(
        resolved_artifact,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        runtime_artifact_policy=runtime_artifact_policy,
    )


def resolve_artifact_input(
    *,
    resolver: RuntimeArtifactResolver | None,
    resolved_artifact: ResolvedRuntimeArtifact | None,
    artifact_ref: str | None,
    artifact_locator: Any | None,
    expected_tensor_schema_hash: str | None,
    runtime_artifact_policy: Any | None,
    placement: RuntimePlacement | None = None,
) -> ResolvedRuntimeArtifact:
    if resolved_artifact is not None:
        if artifact_ref is not None and str(resolved_artifact.artifact_ref) != str(
            artifact_ref
        ):
            raise ManifestMismatchError(
                "TensorCast resolved runtime artifact ref mismatch: "
                f"resolved={resolved_artifact.artifact_ref}, "
                f"requested={artifact_ref}"
            )
        validate_resolved_artifact_placement(
            resolved_artifact,
            placement=placement,
        )
        if resolver is not None and expected_tensor_schema_hash:
            return cross_check_runtime_artifact(
                resolved_artifact,
                resolver=resolver,
                expected_tensor_schema_hash=expected_tensor_schema_hash,
                runtime_artifact_policy=runtime_artifact_policy,
            )
        return resolved_artifact

    resolved_ref = artifact_ref
    if resolved_ref is None and artifact_locator is not None:
        resolve_artifact_ref = getattr(artifact_locator, "resolve_artifact_ref", None)
        if callable(resolve_artifact_ref):
            if artifact_locator_kind(artifact_locator) == "ranked_version_key":
                resolved_ref = resolve_artifact_ref(placement=placement)
            else:
                resolved_ref = resolve_artifact_ref()
        else:
            resolved_ref = str(artifact_locator)
    if not resolved_ref:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration request requires resolved_artifact, "
            "artifact_ref, or artifact_locator"
        )

    resolved = resolve_runtime_artifact(
        str(resolved_ref),
        resolver=resolver,
        expected_tensor_schema_hash=expected_tensor_schema_hash,
        runtime_artifact_policy=runtime_artifact_policy,
    )
    validate_resolved_artifact_placement(
        resolved,
        placement=placement,
    )
    return resolved


def preflight_runtime_artifact(
    *,
    resolver: RuntimeArtifactResolver | None,
    resolved_artifact: ResolvedRuntimeArtifact | None,
    artifact_ref: str | None,
    artifact_locator: Any | None,
    expected_tensor_schema_hash: str | None,
    policy: Any | None,
    placement: RuntimePlacement | None = None,
) -> RuntimeArtifactPreflight:
    base_policy = runtime_policy(policy)
    resolved = resolve_artifact_input(
        resolver=resolver,
        resolved_artifact=resolved_artifact,
        artifact_ref=artifact_ref,
        artifact_locator=artifact_locator,
        expected_tensor_schema_hash=None,
        runtime_artifact_policy=None,
        placement=placement,
    )
    runtime_artifact_policy = runtime_policy_from_manifest(
        base_policy,
        resolved,
        placement=placement,
    )
    if expected_tensor_schema_hash is not None:
        resolved = resolve_artifact_input(
            resolver=resolver,
            resolved_artifact=resolved,
            artifact_ref=artifact_ref,
            artifact_locator=artifact_locator,
            expected_tensor_schema_hash=expected_tensor_schema_hash,
            runtime_artifact_policy=runtime_artifact_policy,
            placement=placement,
        )
    return RuntimeArtifactPreflight(
        resolved_artifact=resolved,
        runtime_artifact_policy=runtime_artifact_policy,
    )


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def _json_object_payload(value: Any, *, field_name: str) -> Any:
    try:
        payload = json.loads(str(value))
    except Exception as exc:
        raise ManifestMismatchError(
            f"TensorCast runtime artifact {field_name} is invalid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ManifestMismatchError(
            f"TensorCast runtime artifact {field_name} must be a JSON object"
        )
    return payload


__all__ = [
    "RuntimeArtifactPreflight",
    "artifact_locator_kind",
    "cross_check_runtime_artifact",
    "preflight_runtime_artifact",
    "read_runtime_artifact_manifest",
    "resolve_artifact_input",
    "resolve_runtime_artifact",
    "runtime_policy",
    "runtime_policy_from_manifest",
    "runtime_policy_with_placement",
    "validate_resolved_artifact_placement",
]
