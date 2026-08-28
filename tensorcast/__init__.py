#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import contextlib
import importlib
import importlib.abc

# -----------------------------------------------------------------------------
# Torch ABI fail-fast guard.
#
# tensorcast ships a C++ extension (`tensorcast._C`) and a `tensorcast_daemon`
# binary that link against a specific PyTorch C++ ABI. Mixing them with a
# different torch minor version usually produces undefined-symbol errors at
# dlopen time. We refuse to import early with a clearer message so users land
# on a fix before they hit confusing low-level failures.
#
# The build-time torch version is baked into `tensorcast/_version.py` by
# `setup.py:gen_version_file()` so each wheel/install carries its own ABI
# anchor. From-source developers automatically get the right anchor for their
# torch — no source edits needed when bumping torch versions.
#
# Set `TENSORCAST_SKIP_TORCH_ABI_CHECK=1` to bypass at your own risk.
# -----------------------------------------------------------------------------
import os as _os
import sys
import threading
import warnings as _warnings
from types import ModuleType
from typing import TYPE_CHECKING, Any, Callable

import torch as _torch


def _torch_major_minor(version: str) -> str:
    return ".".join(version.split("+", 1)[0].split(".")[:2])


if _os.environ.get("TENSORCAST_SKIP_TORCH_ABI_CHECK") != "1":
    try:
        from tensorcast._version import __torch_version__ as _build_torch
    except ImportError:
        _warnings.warn(
            "tensorcast/_version.py not found; skipping torch ABI check. "
            "Run `setup.py build_ext` (or `pip install`) to populate it.",
            stacklevel=2,
        )
    else:
        _build_mm = _torch_major_minor(_build_torch)
        _runtime_mm = _torch_major_minor(_torch.__version__)
        if _build_mm != _runtime_mm:
            raise ImportError(
                f"tensorcast was built against torch {_build_torch} "
                f"(major.minor={_build_mm}) but found torch {_torch.__version__} "
                f"(major.minor={_runtime_mm}). The native extension and daemon "
                "link against a specific torch C++ ABI; mixing produces "
                "undefined-symbol errors at dlopen time. "
                f"Fix: install torch {_build_mm}.x to match this wheel, or rebuild "
                "tensorcast from source against your torch version. To override "
                "at your own risk: TENSORCAST_SKIP_TORCH_ABI_CHECK=1."
            )

# -------------------------------------------------------------------------
# CUDA runtime version guard
#
# tensorcast is built against a specific CUDA toolkit (via PyTorch). The
# torch wheel carries its own CUDA runtime, but the driver must be new
# enough to support that CUDA version. We warn early so users know they
# need a driver update or a different torch index.
#
# Set TENSORCAST_SKIP_TORCH_ABI_CHECK=1 to bypass at your own risk.
# -------------------------------------------------------------------------
if _os.environ.get("TENSORCAST_SKIP_TORCH_ABI_CHECK") != "1":
    try:
        from tensorcast._version import __cuda_version__ as _build_cuda
    except ImportError:
        pass
    else:
        _runtime_cuda = _torch.version.cuda
        if _runtime_cuda is None:
            _cuda_tag = f"cu{_build_cuda.replace('.', '')}"
            raise ImportError(
                "tensorcast was built with CUDA support (CUDA "
                f"{_build_cuda}), but the installed torch does not report a CUDA "
                "version. Ensure you installed torch from a CUDA-enabled index, e.g. "
                "pip install torch==2.11.0 --index-url "
                f"https://download.pytorch.org/whl/{_cuda_tag}"
            )
        # Compare major.minor (ignore patch level differences)
        _build_cuda_mm = ".".join(_build_cuda.split(".")[:2])
        _runtime_cuda_mm = ".".join(_runtime_cuda.split(".")[:2])
        if _build_cuda_mm != _runtime_cuda_mm:
            _cuda_tag = f"cu{_build_cuda.replace('.', '')}"
            raise ImportError(
                f"tensorcast was built against CUDA {_build_cuda}, but the "
                f"installed torch uses CUDA {_runtime_cuda}. "
                "The NVIDIA driver may be too old, or torch was installed from the "
                "wrong index. Fix: reinstall torch from the correct CUDA index, e.g. "
                f"pip install torch==2.11.0 --index-url "
                f"https://download.pytorch.org/whl/{_cuda_tag} . "
                "To override at your own risk: TENSORCAST_SKIP_TORCH_ABI_CHECK=1."
            )

del _torch, _os, _warnings, _torch_major_minor

# -----------------------------------------------------------------------------
# Early patch for a PyTorch bug that can raise the following exception when
# importing `torch.overrides` multiple times within the same interpreter:
#
#   RuntimeError: function '_has_torch_function' already has a docstring
#
# The root cause is a duplicate call to `torch._C._add_docstr` which (by design)
# throws if a docstring already exists.  We defensively wrap the original
# implementation so that it silently ignores this specific error.  To avoid
# eagerly importing PyTorch (and its dependency chain) we install the patch via
# a meta path hook that runs immediately before ``tensorcast._C`` is imported.
# -----------------------------------------------------------------------------


class _TensorCastCExtensionBootstrap(importlib.abc.MetaPathFinder):
    """Installs prerequisites immediately before ``tensorcast._C`` loads."""

    _lock = threading.Lock()
    _prepared = False

    @classmethod
    def _ensure_prepared(cls) -> None:
        import torch  # pylint: disable=import-outside-toplevel

        from tensorcast._cuda_loader import ensure_nvrtc_builtins_preloaded

        ensure_nvrtc_builtins_preloaded()

        # Patch torch._C._add_docstr to tolerate duplicate docstrings (PyTorch bug).
        _c_mod: ModuleType = torch._C
        _orig_add_docstr: Callable[[Any, str], Any] = _c_mod._add_docstr

        try:
            already_patched = _orig_add_docstr._tensorcast_patched  # pyright: ignore[reportFunctionMemberAccess]
        except AttributeError:
            already_patched = False

        if already_patched:
            return

        def _safe_add_docstr(obj: Any, doc: str) -> Any:
            try:
                return _orig_add_docstr(obj, doc)
            except RuntimeError as exc:  # noqa: BLE001
                if "already has a docstring" in str(exc):
                    return obj
                raise

        _safe_add_docstr._tensorcast_patched = True  # pyright: ignore[reportFunctionMemberAccess]
        _c_mod._add_docstr = _safe_add_docstr

    def find_spec(self, fullname: str, path: Any, target: Any = None) -> Any:  # noqa: D401, ANN401
        if fullname != "tensorcast._C":
            return None

        finder_cls = type(self)

        if finder_cls._prepared:
            return None

        with finder_cls._lock:
            if not finder_cls._prepared:
                finder_cls._ensure_prepared()
                finder_cls._prepared = True
                with contextlib.suppress(ValueError):
                    sys.meta_path.remove(self)

        return None


def _install_c_extension_bootstrap() -> None:
    for finder in sys.meta_path:
        if isinstance(finder, _TensorCastCExtensionBootstrap):
            return

    sys.meta_path.insert(0, _TensorCastCExtensionBootstrap())


_install_c_extension_bootstrap()

# -----------------------------------------------------------------------------
# Build configuration helpers
# -----------------------------------------------------------------------------
# Validation runs on first C-extension load via tensorcast._c_ext to avoid
# importing the heavy extension at package import time.
from tensorcast._build_config import (  # noqa: E402, F401
    BuildConfigMismatchError,
    validate_cuda_backend_consistency,
)
from tensorcast._version import __version__  # noqa: E402, F401

# -----------------------------------------------------------------------------
# Lazy public API re-exports
# -----------------------------------------------------------------------------
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    "Artifact": ("tensorcast.api", "Artifact"),
    "ArtifactDescriptor": ("tensorcast.api", "ArtifactDescriptor"),
    "ArtifactError": ("tensorcast.api", "ArtifactError"),
    "ArtifactFuture": ("tensorcast.api", "ArtifactFuture"),
    "ArtifactRealizationHandle": ("tensorcast.api", "ArtifactRealizationHandle"),
    "ArtifactRealizationReport": ("tensorcast.api", "ArtifactRealizationReport"),
    "ArtifactRealizationSpec": ("tensorcast.api", "ArtifactRealizationSpec"),
    "AssemblyAttemptRef": ("tensorcast.api", "AssemblyAttemptRef"),
    "BindingReservationCapability": (
        "tensorcast.api",
        "BindingReservationCapability",
    ),
    "BindingValueRef": ("tensorcast.api", "BindingValueRef"),
    "GroupRealizationAcquireRef": ("tensorcast.api", "GroupRealizationAcquireRef"),
    "BindingRealizationEntry": ("tensorcast.api", "BindingRealizationEntry"),
    "BindingRealizationPlan": ("tensorcast.api", "BindingRealizationPlan"),
    "BuilderMode": ("tensorcast.api", "BuilderMode"),
    "BlobRef": ("tensorcast.api", "BlobRef"),
    "CanonicalIndex": ("tensorcast.api", "CanonicalIndex"),
    "CanonicalIndexEntry": ("tensorcast.api", "CanonicalIndexEntry"),
    "ExecutionTopologyContext": ("tensorcast.api", "ExecutionTopologyContext"),
    "FinalizeClass": ("tensorcast.api", "FinalizeClass"),
    "GetArtifactOptions": ("tensorcast.api", "GetArtifactOptions"),
    "PlanType": ("tensorcast.api", "PlanType"),
    "RegisterArtifactOptions": ("tensorcast.api", "RegisterArtifactOptions"),
    "RegisteredArtifact": ("tensorcast.api", "RegisteredArtifact"),
    "PrefetchRetentionPolicy": ("tensorcast.api", "PrefetchRetentionPolicy"),
    "PrefetchHandoff": ("tensorcast.api", "PrefetchHandoff"),
    "PrefetchHandoffMemberFailure": (
        "tensorcast.api",
        "PrefetchHandoffMemberFailure",
    ),
    "PrefetchHandoffSet": ("tensorcast.api", "PrefetchHandoffSet"),
    "RegisteredLease": ("tensorcast.api", "RegisteredLease"),
    "RegistrationResult": ("tensorcast.api", "RegistrationResult"),
    "RuntimeBindingMemberRef": ("tensorcast.api", "RuntimeBindingMemberRef"),
    "RuntimeBindingReadiness": ("tensorcast.api", "RuntimeBindingReadiness"),
    "RuntimeBindingResolvedLayout": (
        "tensorcast.api",
        "RuntimeBindingResolvedLayout",
    ),
    "RuntimeRealizationSpecCacheEntry": (
        "tensorcast.api",
        "RuntimeRealizationSpecCacheEntry",
    ),
    "RuntimeBindingSourceKind": ("tensorcast.api", "RuntimeBindingSourceKind"),
    "RuntimeBindingSourceMemberRef": (
        "tensorcast.api",
        "RuntimeBindingSourceMemberRef",
    ),
    "RuntimeBindingSourceRef": ("tensorcast.api", "RuntimeBindingSourceRef"),
    "RuntimeBindingSourceReuseDecision": (
        "tensorcast.api",
        "RuntimeBindingSourceReuseDecision",
    ),
    "RuntimeBindingSourceReuseMode": (
        "tensorcast.api",
        "RuntimeBindingSourceReuseMode",
    ),
    "RuntimeTopologyRef": ("tensorcast.api", "RuntimeTopologyRef"),
    "RealizationTarget": ("tensorcast.api", "RealizationTarget"),
    "RealizationTargetSet": ("tensorcast.api", "RealizationTargetSet"),
    "SealAssemblyResult": ("tensorcast.api", "SealAssemblyResult"),
    "SERVING_MANIFEST_TENSOR_NAME": ("tensorcast.api", "SERVING_MANIFEST_TENSOR_NAME"),
    "ViewRegistrationKind": ("tensorcast.api", "ViewRegistrationKind"),
    "Store": ("tensorcast.api", "Store"),
    "StoreOptions": ("tensorcast.api", "StoreOptions"),
    "build_indices_from_safetensors": (
        "tensorcast.api",
        "build_indices_from_safetensors",
    ),
    "calculate_tensor_device_offsets": (
        "tensorcast.api",
        "calculate_tensor_device_offsets",
    ),
    "CallContext": ("tensorcast.api", "CallContext"),
    "CollectiveLoadGroup": ("tensorcast.api", "CollectiveLoadGroup"),
    "GroupRealization": ("tensorcast.api", "GroupRealization"),
    "GroupVersionSetRef": ("tensorcast.api", "GroupVersionSetRef"),
    "GovernanceContext": ("tensorcast.api", "GovernanceContext"),
    "DirectorySnapshot": ("tensorcast.api", "DirectorySnapshot"),
    "Operation": ("tensorcast.api", "Operation"),
    "OperationError": ("tensorcast.api", "OperationError"),
    "OperationRefMetadata": ("tensorcast.api", "OperationRefMetadata"),
    "OperationStatus": ("tensorcast.api", "OperationStatus"),
    "OperationTimeoutError": ("tensorcast.api", "OperationTimeoutError"),
    "Plan": ("tensorcast.api", "Plan"),
    "PlanFailedError": ("tensorcast.api", "PlanFailedError"),
    "PlanResult": ("tensorcast.api", "PlanResult"),
    "PlanStepRef": ("tensorcast.api", "PlanStepRef"),
    "PlanStepResult": ("tensorcast.api", "PlanStepResult"),
    "PartialSealResult": ("tensorcast.api", "PartialSealResult"),
    "PublicDiskSourceHandle": ("tensorcast.api", "PublicDiskSourceHandle"),
    "PublishedModelVersion": ("tensorcast.api", "PublishedModelVersion"),
    "ExecutionDiagnostics": ("tensorcast.api", "ExecutionDiagnostics"),
    "BindingUpdateEpoch": ("tensorcast.api", "BindingUpdateEpoch"),
    "BindingValueVerificationState": (
        "tensorcast.api",
        "BindingValueVerificationState",
    ),
    "BindingPromotionStatusState": (
        "tensorcast.api",
        "BindingPromotionStatusState",
    ),
    "HashBackend": ("tensorcast.api", "HashBackend"),
    "HashLocation": ("tensorcast.api", "HashLocation"),
    "IdentityMintStrategy": ("tensorcast.api", "IdentityMintStrategy"),
    "RepresentationPublishContract": (
        "tensorcast.api",
        "RepresentationPublishContract",
    ),
    "RepresentationPublishSpec": (
        "tensorcast.api",
        "RepresentationPublishSpec",
    ),
    "RuntimeArtifactBuildIntent": ("tensorcast.api", "RuntimeArtifactBuildIntent"),
    "RuntimeArtifactManifest": ("tensorcast.api", "RuntimeArtifactManifest"),
    "RuntimeArtifactPolicy": ("tensorcast.api", "RuntimeArtifactPolicy"),
    "RuntimeArtifactPolicyInput": ("tensorcast.api", "RuntimeArtifactPolicyInput"),
    "SourceBoundCapability": ("tensorcast.api", "SourceBoundCapability"),
    "coerce_runtime_artifact_policy": (
        "tensorcast.api",
        "coerce_runtime_artifact_policy",
    ),
    "Instance": ("tensorcast.api", "Instance"),
    "InstanceExecutionRoute": ("tensorcast.api", "InstanceExecutionRoute"),
    "Worker": ("tensorcast.api", "Worker"),
    "TargetSpec": ("tensorcast.api", "TargetSpec"),
    "TransformSpec": ("tensorcast.api", "TransformSpec"),
    "Runtime": ("tensorcast.api", "Runtime"),
    "RuntimeAttachment": (
        "tensorcast.artifact_runtime.attachment",
        "RuntimeAttachment",
    ),
    "RuntimeBindingState": (
        "tensorcast.artifact_runtime.attachment",
        "RuntimeBindingState",
    ),
    "RuntimeAdmissionDecision": (
        "tensorcast.artifact_runtime.host",
        "RuntimeAdmissionDecision",
    ),
    "RuntimeAdmissionPolicy": (
        "tensorcast.artifact_runtime.host",
        "RuntimeAdmissionPolicy",
    ),
    "RuntimeAdmissionRequest": (
        "tensorcast.artifact_runtime.host",
        "RuntimeAdmissionRequest",
    ),
    "RuntimeHostCapabilities": (
        "tensorcast.artifact_runtime.host",
        "RuntimeHostCapabilities",
    ),
    "RuntimePlacement": ("tensorcast.artifact_runtime.host", "RuntimePlacement"),
    "RuntimeProfile": ("tensorcast.artifact_runtime.host", "RuntimeProfile"),
    "RuntimeTensorView": ("tensorcast.artifact_runtime.host", "RuntimeTensorView"),
    "ArtifactLocator": ("tensorcast.artifact_runtime.locator", "ArtifactLocator"),
    "RuntimeArtifactLocator": (
        "tensorcast.artifact_runtime.config",
        "RuntimeArtifactLocator",
    ),
    "RuntimeStartPlanError": (
        "tensorcast.artifact_runtime.config",
        "RuntimeStartPlanError",
    ),
    "RuntimePolicy": ("tensorcast.artifact_runtime.policy", "RuntimePolicy"),
    "RuntimeRealizationReport": (
        "tensorcast.artifact_runtime.diagnostics",
        "RuntimeRealizationReport",
    ),
    "TensorCastRuntimeConfig": (
        "tensorcast.artifact_runtime.config",
        "TensorCastRuntimeConfig",
    ),
    "plan_runtime_start": ("tensorcast.artifact_runtime.config", "plan_runtime_start"),
    "RuntimeRequestContext": (
        "tensorcast.artifact_runtime.intent",
        "RuntimeRequestContext",
    ),
    "ModelAttributeNames": ("tensorcast.artifact_runtime.state", "ModelAttributeNames"),
    "ModelAttributeRuntimeState": (
        "tensorcast.artifact_runtime.state",
        "ModelAttributeRuntimeState",
    ),
    "OneShotRuntimeHook": ("tensorcast.artifact_runtime.state", "OneShotRuntimeHook"),
    "BindingValueRefProjection": (
        "tensorcast.artifact_runtime.view",
        "BindingValueRefProjection",
    ),
    "RuntimeEndpointProjection": (
        "tensorcast.artifact_runtime.view",
        "RuntimeEndpointProjection",
    ),
    "RuntimeWorkerView": ("tensorcast.artifact_runtime.view", "RuntimeWorkerView"),
    "SourceSelectionProjection": (
        "tensorcast.artifact_runtime.view",
        "SourceSelectionProjection",
    ),
    "WeightVersionProjection": (
        "tensorcast.artifact_runtime.view",
        "WeightVersionProjection",
    ),
    "aggregate_runtime_view_outputs": (
        "tensorcast.artifact_runtime.view",
        "aggregate_runtime_view_outputs",
    ),
    "RuntimeReplicaPublicationSettings": (
        "tensorcast.artifact_runtime.publication.actions",
        "RuntimeReplicaPublicationSettings",
    ),
    "RetainedRealizationClaim": (
        "tensorcast.retained_realization",
        "RetainedRealizationClaim",
    ),
    "RetainedRealizationExpectedDigests": (
        "tensorcast.retained_realization",
        "RetainedRealizationExpectedDigests",
    ),
    "project_runtime_replica_publication_state": (
        "tensorcast.artifact_runtime.publication.actions",
        "project_runtime_replica_publication_state",
    ),
    "publish_runtime_replica": (
        "tensorcast.artifact_runtime.publication.actions",
        "publish_runtime_replica",
    ),
    "reload_runtime_attachment": (
        "tensorcast.artifact_runtime.reload",
        "reload_runtime_attachment",
    ),
    "merge_runtime_reload_extra_config": (
        "tensorcast.artifact_runtime.reload",
        "merge_runtime_reload_extra_config",
    ),
    "normalize_runtime_reload_request_payload": (
        "tensorcast.artifact_runtime.reload",
        "normalize_runtime_reload_request_payload",
    ),
    "retire_runtime_replica": (
        "tensorcast.artifact_runtime.publication.actions",
        "retire_runtime_replica",
    ),
    "runtime_replica_publication_settings": (
        "tensorcast.artifact_runtime.publication.actions",
        "runtime_replica_publication_settings",
    ),
    "SignalSnapshot": ("tensorcast.api", "SignalSnapshot"),
    "TensorCastDirectory": ("tensorcast.api", "TensorCastDirectory"),
    "TensorCastSignals": ("tensorcast.api", "TensorCastSignals"),
    "WorkerRoute": ("tensorcast.api", "WorkerRoute"),
    "WorkerStatus": ("tensorcast.api", "WorkerStatus"),
    "connect": ("tensorcast.api", "connect"),
    "context": ("tensorcast.api", "context"),
    "plan": ("tensorcast.api", "plan"),
    "RetentionHandle": ("tensorcast.retention", "RetentionHandle"),
    "acquire_retention_handle": ("tensorcast.retention", "acquire_retention_handle"),
    "parse_retained_realization_claim": (
        "tensorcast.retained_realization",
        "parse_retained_realization_claim",
    ),
    "renew_retention_handle": ("tensorcast.retention", "renew_retention_handle"),
    "retained_realization_claim_extra_from_handoff": (
        "tensorcast.retained_realization",
        "retained_realization_claim_extra_from_handoff",
    ),
    "retained_realization_claim_extra_json_from_handoff": (
        "tensorcast.retained_realization",
        "retained_realization_claim_extra_json_from_handoff",
    ),
    "retained_realization_claim_mode": (
        "tensorcast.retained_realization",
        "retained_realization_claim_mode",
    ),
    "retained_realization_trusted_reservation_bytes": (
        "tensorcast.retained_realization",
        "retained_realization_trusted_reservation_bytes",
    ),
    "release_retention_handle": ("tensorcast.retention", "release_retention_handle"),
    "artifact": ("tensorcast.api.store", "artifact"),
    "artifact_async": ("tensorcast.api.store", "artifact_async"),
    "binding_realization_plan_to_proto": (
        "tensorcast.api.store",
        "binding_realization_plan_to_proto",
    ),
    "deregister_artifact": ("tensorcast.api.store", "deregister_artifact"),
    "from_disk": ("tensorcast.api.store", "from_disk"),
    "from_resolved_public_disk_source": (
        "tensorcast.api.store",
        "from_resolved_public_disk_source",
    ),
    "import_from_disk": ("tensorcast.api.store", "import_from_disk"),
    "promote_mounted_source": ("tensorcast.api.store", "promote_mounted_source"),
    "realize_into_binding": ("tensorcast.api.store", "realize_into_binding"),
    "put": ("tensorcast.api.store", "put"),
    "normalize_binding_realization_plan": (
        "tensorcast.api.store",
        "normalize_binding_realization_plan",
    ),
    "put_async": ("tensorcast.api.store", "put_async"),
    "persist_artifact": ("tensorcast.api.store", "persist_artifact"),
    "persistence_operation": ("tensorcast.api.store", "persistence_operation"),
    "query_persistence_status": ("tensorcast.api.store", "query_persistence_status"),
    "register": ("tensorcast.api.store", "register"),
    "register_async": ("tensorcast.api.store", "register_async"),
    "register_view": ("tensorcast.api.store", "register_view"),
    "register_piece": ("tensorcast.api.store", "register_piece"),
    "register_pure_transform_publication": (
        "tensorcast.api.store",
        "register_pure_transform_publication",
    ),
    "register_vram_region": ("tensorcast.api.store", "register_vram_region"),
    "resolve_public_disk_source": (
        "tensorcast.api.store",
        "resolve_public_disk_source",
    ),
    "list_artifact_layouts": ("tensorcast.api.store", "list_artifact_layouts"),
    "seal_assembly": ("tensorcast.api.store", "seal_assembly"),
    "start_canonical_representation_publish_attempt": (
        "tensorcast.api.store",
        "start_canonical_representation_publish_attempt",
    ),
    "start_assembly_attempt": ("tensorcast.api.store", "start_assembly_attempt"),
    "start_representation_publish_attempt": (
        "tensorcast.api.store",
        "start_representation_publish_attempt",
    ),
    "start_plan_repo_owned_representation_publish_attempt": (
        "tensorcast.api.store",
        "start_plan_repo_owned_representation_publish_attempt",
    ),
    "start_repo_owned_representation_publish_attempt": (
        "tensorcast.api.store",
        "start_repo_owned_representation_publish_attempt",
    ),
    "start_structural_representation_publish_attempt": (
        "tensorcast.api.store",
        "start_structural_representation_publish_attempt",
    ),
    "store": ("tensorcast.api.store", "store"),
    "unregister_vram_region": ("tensorcast.api.store", "unregister_vram_region"),
    "wait_assembly_attempt": ("tensorcast.api.store", "wait_assembly_attempt"),
    "build_pure_transform_publication_bundle": (
        "tensorcast.api.store",
        "build_pure_transform_publication_bundle",
    ),
    "build_binding_finalize_admission_facts": (
        "tensorcast.api.store",
        "build_binding_finalize_admission_facts",
    ),
    "build_binding_finalize_publication_bundle": (
        "tensorcast.api.store",
        "build_binding_finalize_publication_bundle",
    ),
    "build_pure_transform_publication_bundle_from_registered_artifact": (
        "tensorcast.api.store",
        "build_pure_transform_publication_bundle_from_registered_artifact",
    ),
    "build_pure_transform_publication_spec": (
        "tensorcast.api.store",
        "build_pure_transform_publication_spec",
    ),
    "build_representation_publish_requirements": (
        "tensorcast.api.store",
        "build_representation_publish_requirements",
    ),
    "build_pure_transform_transform_spec": (
        "tensorcast.api.store",
        "build_pure_transform_transform_spec",
    ),
    "complete_canonical_representation_publish_attempt": (
        "tensorcast.api.store",
        "complete_canonical_representation_publish_attempt",
    ),
    "complete_binding_finalize_publication_from_binding": (
        "tensorcast.api.store",
        "complete_binding_finalize_publication_from_binding",
    ),
    "complete_pure_transform_publication": (
        "tensorcast.api.store",
        "complete_pure_transform_publication",
    ),
    "complete_pure_transform_publication_from_binding": (
        "tensorcast.api.store",
        "complete_pure_transform_publication_from_binding",
    ),
    "complete_representation_publish_attempt": (
        "tensorcast.api.store",
        "complete_representation_publish_attempt",
    ),
    "complete_plan_repo_owned_representation_publish_attempt": (
        "tensorcast.api.store",
        "complete_plan_repo_owned_representation_publish_attempt",
    ),
    "complete_repo_owned_representation_publish_attempt": (
        "tensorcast.api.store",
        "complete_repo_owned_representation_publish_attempt",
    ),
    "complete_structural_representation_publish_attempt": (
        "tensorcast.api.store",
        "complete_structural_representation_publish_attempt",
    ),
    "compute_pure_transform_representation_contract_hash": (
        "tensorcast.api.store",
        "compute_pure_transform_representation_contract_hash",
    ),
    "init": ("tensorcast.startup", "init"),
    "PortConfig": ("tensorcast.startup", "PortConfig"),
    "is_initialized": ("tensorcast.startup", "is_initialized"),
    "shutdown": ("tensorcast.startup", "shutdown"),
}

context: Any
plan: Any
runtime: Any
RuntimeAdmissionDecision: Any
RuntimeAdmissionPolicy: Any
RuntimeAdmissionRequest: Any
RuntimePlacement: Any
RuntimeProfile: Any
RuntimeTensorView: Any
coerce_runtime_artifact_policy: Any


def __getattr__(name: str) -> Any:
    if name == "runtime":
        module = importlib.import_module("tensorcast.runtime")
        globals()[name] = module
        return module
    if name not in _LAZY_ATTRS:
        raise AttributeError(name)
    module_name, attr_name = _LAZY_ATTRS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_ATTRS).union({"runtime"}))


if TYPE_CHECKING:
    from tensorcast.api import (  # noqa: F401
        Artifact,
        ArtifactDescriptor,
        ArtifactError,
        ArtifactFuture,
        ArtifactRealizationHandle,
        ArtifactRealizationReport,
        ArtifactRealizationSpec,
        BindingRealizationEntry,
        BindingRealizationPlan,
        BindingReservationCapability,
        BindingUpdateEpoch,
        BindingValueRef,
        BlobRef,
        CallContext,
        CanonicalIndex,
        CanonicalIndexEntry,
        CollectiveLoadGroup,
        DirectorySnapshot,
        ExecutionDiagnostics,
        ExecutionTopologyContext,
        GetArtifactOptions,
        GovernanceContext,
        GroupRealization,
        GroupRealizationAcquireRef,
        GroupVersionSetRef,
        HashBackend,
        HashLocation,
        IdentityMintStrategy,
        Instance,
        InstanceExecutionRoute,
        Operation,
        OperationError,
        OperationRefMetadata,
        OperationStatus,
        OperationTimeoutError,
        Plan,
        PlanFailedError,
        PlanResult,
        PlanStepRef,
        PlanStepResult,
        PlanType,
        PrefetchHandoff,
        PrefetchHandoffMemberFailure,
        PrefetchHandoffSet,
        PrefetchRetentionPolicy,
        PublicDiskSourceHandle,
        RealizationTarget,
        RealizationTargetSet,
        RegisterArtifactOptions,
        RegisteredArtifact,
        RegisteredLease,
        RegistrationResult,
        RetentionHandle,
        Runtime,
        RuntimeArtifactBuildIntent,
        RuntimeArtifactManifest,
        RuntimeArtifactPolicy,
        RuntimeArtifactPolicyInput,
        RuntimeBindingMemberRef,
        RuntimeBindingReadiness,
        RuntimeBindingResolvedLayout,
        RuntimeBindingSourceKind,
        RuntimeBindingSourceMemberRef,
        RuntimeBindingSourceRef,
        RuntimeBindingSourceReuseDecision,
        RuntimeBindingSourceReuseMode,
        RuntimeRealizationSpecCacheEntry,
        RuntimeTopologyRef,
        SignalSnapshot,
        SourceBoundCapability,
        Store,
        StoreOptions,
        TargetSpec,
        TensorCastDirectory,
        TensorCastSignals,
        TransformSpec,
        Worker,
        WorkerRoute,
        WorkerStatus,
        acquire_retention_handle,
        build_indices_from_safetensors,
        calculate_tensor_device_offsets,
        connect,
        release_retention_handle,
        renew_retention_handle,
    )
    from tensorcast.api.store import (  # type: ignore[no-redef]  # noqa: F401
        artifact,
        artifact_async,
        binding_realization_plan_to_proto,
        complete_pure_transform_publication,
        deregister_artifact,
        from_disk,
        import_from_disk,
        normalize_binding_realization_plan,
        persist_artifact,
        persistence_operation,
        promote_mounted_source,
        put,
        put_async,
        query_persistence_status,
        realize_into_binding,
        register,
        register_async,
        register_pure_transform_publication,
        register_view,
        register_vram_region,
        resolve_public_disk_source,
        store,
        unregister_vram_region,
    )
    from tensorcast.artifact_runtime.attachment import (  # noqa: F401
        RuntimeAttachment,
        RuntimeBindingState,
    )
    from tensorcast.artifact_runtime.config import (  # noqa: F401
        RuntimeArtifactLocator,
        RuntimeStartPlanError,
        TensorCastRuntimeConfig,
        plan_runtime_start,
    )
    from tensorcast.artifact_runtime.diagnostics import (
        RuntimeRealizationReport,  # noqa: F401
    )
    from tensorcast.artifact_runtime.host import RuntimeHostCapabilities  # noqa: F401
    from tensorcast.artifact_runtime.intent import RuntimeRequestContext  # noqa: F401
    from tensorcast.artifact_runtime.locator import ArtifactLocator  # noqa: F401
    from tensorcast.artifact_runtime.policy import RuntimePolicy  # noqa: F401
    from tensorcast.artifact_runtime.publication.actions import (  # noqa: F401
        RuntimeReplicaPublicationSettings,
        project_runtime_replica_publication_state,
        publish_runtime_replica,
        retire_runtime_replica,
        runtime_replica_publication_settings,
    )
    from tensorcast.artifact_runtime.reload import (  # noqa: F401
        merge_runtime_reload_extra_config,
        normalize_runtime_reload_request_payload,
        reload_runtime_attachment,
    )
    from tensorcast.artifact_runtime.state import (  # noqa: F401
        ModelAttributeNames,
        ModelAttributeRuntimeState,
        OneShotRuntimeHook,
    )
    from tensorcast.artifact_runtime.view import (  # noqa: F401
        BindingValueRefProjection,
        RuntimeEndpointProjection,
        RuntimeWorkerView,
        SourceSelectionProjection,
        WeightVersionProjection,
        aggregate_runtime_view_outputs,
    )
    from tensorcast.retained_realization import (  # noqa: F401
        RetainedRealizationClaim,
        RetainedRealizationExpectedDigests,
        parse_retained_realization_claim,
        retained_realization_claim_extra_from_handoff,
        retained_realization_claim_extra_json_from_handoff,
        retained_realization_claim_mode,
        retained_realization_trusted_reservation_bytes,
    )
    from tensorcast.startup import (  # noqa: F401
        PortConfig,
        init,
        is_initialized,
        shutdown,
    )


__all__ = [
    "__version__",
    "runtime",
    "init",
    "is_initialized",
    "shutdown",
    "Store",
    "StoreOptions",
    "RegisteredArtifact",
    "ArtifactError",
    "ArtifactFuture",
    "BindingReservationCapability",
    "BindingValueRef",
    "GroupRealizationAcquireRef",
    "BindingRealizationEntry",
    "BindingRealizationPlan",
    "BlobRef",
    "CanonicalIndex",
    "CanonicalIndexEntry",
    "ExecutionTopologyContext",
    "RegisteredLease",
    "RegistrationResult",
    "PlanType",
    "RegisterArtifactOptions",
    "GetArtifactOptions",
    "PrefetchRetentionPolicy",
    "PrefetchHandoff",
    "PrefetchHandoffMemberFailure",
    "PrefetchHandoffSet",
    "calculate_tensor_device_offsets",
    "build_indices_from_safetensors",
    "binding_realization_plan_to_proto",
    "CallContext",
    "CollectiveLoadGroup",
    "GroupRealization",
    "GroupVersionSetRef",
    "RealizationTarget",
    "RealizationTargetSet",
    "ExecutionDiagnostics",
    "BindingUpdateEpoch",
    "HashBackend",
    "Operation",
    "OperationError",
    "OperationRefMetadata",
    "OperationStatus",
    "OperationTimeoutError",
    "HashLocation",
    "IdentityMintStrategy",
    "Plan",
    "PlanFailedError",
    "PlanResult",
    "PlanStepRef",
    "PlanStepResult",
    "Worker",
    "Instance",
    "TargetSpec",
    "TransformSpec",
    "context",
    "normalize_binding_realization_plan",
    "plan",
    "from_disk",
    "from_resolved_public_disk_source",
    "import_from_disk",
    "promote_mounted_source",
    "realize_into_binding",
    "resolve_public_disk_source",
    "Artifact",
    "ArtifactDescriptor",
    "ArtifactRealizationHandle",
    "ArtifactRealizationReport",
    "ArtifactRealizationSpec",
    "PublicDiskSourceHandle",
    "SourceBoundCapability",
    "RuntimeBindingMemberRef",
    "RuntimeBindingReadiness",
    "RuntimeBindingResolvedLayout",
    "RuntimeRealizationSpecCacheEntry",
    "RuntimeBindingSourceKind",
    "RuntimeBindingSourceMemberRef",
    "RuntimeBindingSourceRef",
    "RuntimeBindingSourceReuseDecision",
    "RuntimeBindingSourceReuseMode",
    "RuntimeTopologyRef",
    "store",
    "register",
    "register_async",
    "register_view",
    "persist_artifact",
    "persistence_operation",
    "put",
    "put_async",
    "query_persistence_status",
    "artifact",
    "artifact_async",
    "register_vram_region",
    "unregister_vram_region",
    "deregister_artifact",
    "BuildConfigMismatchError",
    "RuntimeAttachment",
    "RuntimeAdmissionDecision",
    "RuntimeAdmissionPolicy",
    "RuntimeAdmissionRequest",
    "RuntimeArtifactBuildIntent",
    "RuntimeArtifactManifest",
    "RuntimeArtifactPolicy",
    "RuntimeArtifactPolicyInput",
    "RuntimeBindingState",
    "RuntimeHostCapabilities",
    "RuntimePlacement",
    "RuntimeProfile",
    "RuntimeRequestContext",
    "ArtifactLocator",
    "RuntimeArtifactLocator",
    "RuntimePolicy",
    "RuntimeRealizationReport",
    "RuntimeStartPlanError",
    "RuntimeTensorView",
    "TensorCastRuntimeConfig",
    "ModelAttributeNames",
    "ModelAttributeRuntimeState",
    "OneShotRuntimeHook",
    "BindingValueRefProjection",
    "RuntimeEndpointProjection",
    "RuntimeWorkerView",
    "SourceSelectionProjection",
    "WeightVersionProjection",
    "aggregate_runtime_view_outputs",
    "RuntimeReplicaPublicationSettings",
    "coerce_runtime_artifact_policy",
    "RetainedRealizationClaim",
    "RetainedRealizationExpectedDigests",
    "parse_retained_realization_claim",
    "project_runtime_replica_publication_state",
    "publish_runtime_replica",
    "reload_runtime_attachment",
    "merge_runtime_reload_extra_config",
    "normalize_runtime_reload_request_payload",
    "plan_runtime_start",
    "retained_realization_claim_extra_from_handoff",
    "retained_realization_claim_extra_json_from_handoff",
    "retained_realization_claim_mode",
    "retained_realization_trusted_reservation_bytes",
    "retire_runtime_replica",
    "runtime_replica_publication_settings",
]
