#  Copyright (c) 2026, TensorCast Team.

"""Structured artifact runtime errors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class TensorCastRuntimeError(RuntimeError):
    """Base class for machine-readable artifact runtime failures."""

    code = "tensorcast_runtime_error"
    operation = "artifact_runtime"
    retryable = False
    worker_suspect = False

    def __init__(
        self,
        message: str = "",
        *,
        operation: str | None = None,
        retryable: bool | None = None,
        worker_suspect: bool | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.operation = operation or self.operation
        self.retryable = self.retryable if retryable is None else retryable
        self.worker_suspect = (
            self.worker_suspect if worker_suspect is None else worker_suspect
        )
        self.details = dict(details or {})


class ArtifactRuntimeIntegrationError(TensorCastRuntimeError):
    """Base class for structured runtime integration failures."""


class ArtifactRuntimeNotImplementedError(ArtifactRuntimeIntegrationError):
    """Raised when a deep core-owned lifecycle method is not implemented yet."""

    code = "not_implemented"
    operation = "artifact_runtime"


class ConfigConflictError(ArtifactRuntimeIntegrationError):
    """Runtime config requests mutually exclusive lifecycle execution modes."""

    code = "config_conflict"
    operation = "config_planning"


class CapabilityMissingError(ArtifactRuntimeIntegrationError):
    """Required host capability is absent for a requested lifecycle path."""

    code = "capability_missing"
    operation = "capability_validation"


def capability_missing(
    message: str,
    *,
    level: str,
    capability: str,
    operation: str,
    required_methods: Sequence[str] = (),
    next_action: str,
) -> CapabilityMissingError:
    return CapabilityMissingError(
        message,
        operation=operation,
        details={
            "level": level,
            "capability": capability,
            "operation": operation,
            "required_methods": tuple(required_methods),
            "next_action": next_action,
        },
    )


class AdmissionRejectedError(ArtifactRuntimeIntegrationError):
    """Core admission rejected a runtime lifecycle request."""

    code = "admission_rejected"
    operation = "admission"


class PlacementAdmissionError(ArtifactRuntimeIntegrationError):
    """Placement identity or semantic placement proof is invalid."""

    code = "placement_admission"
    operation = "placement_admission"


class ArtifactLocatorResolutionError(ArtifactRuntimeIntegrationError):
    """Durable runtime artifact locator could not resolve to an artifact."""

    code = "artifact_locator_resolution"
    operation = "artifact_locator_resolution"


class ManifestMismatchError(ArtifactRuntimeIntegrationError):
    """Runtime artifact manifest content does not match requested facts."""

    code = "manifest_mismatch"
    operation = "manifest_validation"


class PolicyMismatchError(ArtifactRuntimeIntegrationError):
    """Runtime artifact policy does not match the artifact manifest."""

    code = "policy_mismatch"
    operation = "policy_validation"


class AuthorityValidationError(ArtifactRuntimeIntegrationError):
    """Retained realization authority failed validation."""

    code = "authority_validation"
    operation = "retained_acquire"


class SchemaMismatchError(ArtifactRuntimeIntegrationError):
    """Runtime tensor schema does not match the artifact schema."""

    code = "schema_mismatch"
    operation = "schema_validation"
    worker_suspect = True


class AttachFinalizeError(ArtifactRuntimeIntegrationError):
    """Framework attach, process-after-load, or finalize failed."""

    code = "attach_finalize"
    operation = "attach_finalize"
    worker_suspect = True


class RestoreBindingError(ArtifactRuntimeIntegrationError):
    """Retained binding restore failed before runtime ownership transfer."""

    code = "restore_binding"
    operation = "retained_acquire"


class OwnershipTransferError(ArtifactRuntimeIntegrationError):
    """Binding ownership transfer to runtime state failed."""

    code = "ownership_transfer"
    operation = "ownership_transfer"
    worker_suspect = True


class RuntimeSwapError(ArtifactRuntimeIntegrationError):
    """Runtime binding swap failed after execution started."""

    code = "runtime_swap"
    operation = "reload"
    worker_suspect = True


class SourceSubjectError(ArtifactRuntimeIntegrationError):
    """Source selector resolution or broadcast payload handling failed."""

    code = "source_subject"
    operation = "source_provider"


class SourceProviderError(ArtifactRuntimeIntegrationError):
    """Source provider, catalog, or cache policy failed."""

    code = "source_provider"
    operation = "source_provider"


class PublicationRequiredError(ArtifactRuntimeIntegrationError):
    """A local-ready identity was used where durable publication is required."""

    code = "publication_required"
    operation = "artifact_locator_validation"


class ReplicaPublicationError(ArtifactRuntimeIntegrationError):
    """Runtime-owned ephemeral replica publication failed."""

    code = "replica_publication"
    operation = "replica_publication"
    worker_suspect = True

    def __init__(
        self,
        message: str = "",
        *,
        attachment: object | None = None,
        operation: str | None = None,
        retryable: bool | None = None,
        worker_suspect: bool | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            operation=operation,
            retryable=retryable,
            worker_suspect=worker_suspect,
            details=details,
        )
        self.attachment = attachment


__all__ = [
    "AdmissionRejectedError",
    "ArtifactLocatorResolutionError",
    "AttachFinalizeError",
    "AuthorityValidationError",
    "CapabilityMissingError",
    "ConfigConflictError",
    "ManifestMismatchError",
    "OwnershipTransferError",
    "PlacementAdmissionError",
    "PolicyMismatchError",
    "PublicationRequiredError",
    "ReplicaPublicationError",
    "RestoreBindingError",
    "RuntimeSwapError",
    "SchemaMismatchError",
    "ArtifactRuntimeIntegrationError",
    "ArtifactRuntimeNotImplementedError",
    "SourceProviderError",
    "SourceSubjectError",
    "TensorCastRuntimeError",
    "capability_missing",
]
