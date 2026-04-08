#  Copyright (c) 2026, TensorCast Team.

"""Compatibility wrapper for the legacy KV-named artifact adapter surface."""

from __future__ import annotations

from tensorcast.engine_adapter.artifact_api import (
    MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA,
    MANIFEST_ARTIFACT_SET_BRIDGE_VERSION,
    PUBLISH_MANIFEST_SCHEMA,
    BatchOutcome,
    BatchResult,
    EngineArtifactAdapter,
    EngineOwnedManifest,
    HydrateResult,
    ManifestArtifactSetBridge,
    ManifestResult,
    OpenByteArtifact,
    PublishManifest,
    PublishResult,
    PutIfAbsentInvariant,
    SealedByteArtifact,
    compute_key_set_digest_hex,
    open_byte_artifact,
    seal_byte_artifact,
)

__all__ = [
    "BatchOutcome",
    "BatchResult",
    "EngineOwnedManifest",
    "EngineArtifactAdapter",
    "HydrateResult",
    "MANIFEST_ARTIFACT_SET_BRIDGE_SCHEMA",
    "MANIFEST_ARTIFACT_SET_BRIDGE_VERSION",
    "ManifestArtifactSetBridge",
    "ManifestResult",
    "OpenByteArtifact",
    "PUBLISH_MANIFEST_SCHEMA",
    "PublishManifest",
    "PublishResult",
    "PutIfAbsentInvariant",
    "SealedByteArtifact",
    "compute_key_set_digest_hex",
    "open_byte_artifact",
    "seal_byte_artifact",
]
