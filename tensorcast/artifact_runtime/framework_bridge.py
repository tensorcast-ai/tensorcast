#  Copyright (c) 2026, TensorCast Team.
"""Framework host facade used by artifact-runtime lifecycle orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast

import torch

import tensorcast.artifact_runtime.readiness as tc_readiness
from tensorcast.artifact_runtime.errors import (
    ArtifactRuntimeIntegrationError,
    SchemaMismatchError,
)
from tensorcast.artifact_runtime.errors import (
    capability_missing as _capability_missing,
)
from tensorcast.artifact_runtime.host import (
    FrameworkHost,
    FrameworkIdentity,
    IntegrationHost,
    TensorSurfaceHost,
)
from tensorcast.artifact_runtime.recipe.trace_ir import TracePlan
from tensorcast.types import FinalizeClass, RuntimeSupportLevel


def framework_host(host: IntegrationHost | None) -> FrameworkHost:
    if host is not None:
        return host.framework
    raise _capability_missing(
        "ArtifactRuntimeIntegration requires IntegrationHost.framework",
        level="level1-runtime",
        capability="framework",
        operation="framework_host",
        required_methods=(
            "identity",
            "build_meta_model",
            "assert_model_ready_for_runtime_binding",
        ),
        next_action=(
            "Construct ArtifactRuntimeSession with IntegrationHost.framework."
        ),
    )


def framework_identity(
    host: IntegrationHost | None,
    model_config: Any | None,
) -> FrameworkIdentity:
    return framework_host(host).identity(model_config)


def build_meta_model(
    host: IntegrationHost | None,
    framework_config: Any | None,
    model_config: Any | None,
) -> object:
    return framework_host(host).build_meta_model(
        framework_config,
        model_config,
    )


def tensor_surface(host: IntegrationHost | None) -> TensorSurfaceHost:
    if host is not None:
        if host.tensor_surface is None:
            raise _capability_missing(
                "IntegrationHost requires TensorSurfaceHost for runtime "
                "tensor operations",
                level="level1-runtime",
                capability="tensor_surface",
                operation="runtime_tensor_surface",
                required_methods=(
                    "attach_bound_tensors",
                    "collect_runtime_tensors",
                    "compute_runtime_tensor_schema_hash",
                ),
                next_action=(
                    "Pass IntegrationHost(tensor_surface=...) or use "
                    "TorchTensorHost for PyTorch module carriers."
                ),
            )
        return host.tensor_surface
    raise _capability_missing(
        "ArtifactRuntimeIntegration requires IntegrationHost.tensor_surface",
        level="level1-runtime",
        capability="tensor_surface",
        operation="runtime_tensor_surface",
        required_methods=(
            "attach_bound_tensors",
            "collect_runtime_tensors",
            "compute_runtime_tensor_schema_hash",
        ),
        next_action=(
            "Construct ArtifactRuntimeSession with IntegrationHost.tensor_surface."
        ),
    )


def require_target_device(target_device: Any | None) -> torch.device:
    if target_device is None:
        raise ArtifactRuntimeIntegrationError(
            "ArtifactRuntimeIntegration request requires target_device"
        )
    return torch.device(target_device)


def prepare_model_construction(
    host: IntegrationHost | None,
    framework_config: Any | None,
    model_config: Any | None,
) -> None:
    prepare = getattr(framework_host(host), "prepare_model_construction", None)
    if callable(prepare):
        prepare(framework_config, model_config)


def assert_model_ready_for_runtime_binding(
    host: IntegrationHost | None,
    model: Any,
    *,
    context: str,
) -> None:
    check = getattr(
        framework_host(host), "assert_model_ready_for_runtime_binding", None
    )
    if callable(check):
        check(model, context=context)


def align_runtime_tensor_names(
    host: IntegrationHost | None,
    model: Any,
    expected_names: Sequence[str],
) -> int:
    return int(
        tensor_surface(host).align_runtime_tensor_names(model, expected_names) or 0
    )


def collect_runtime_binding_tensors(
    host: IntegrationHost | None,
    model: Any,
    *,
    remove_duplicate: bool,
) -> Mapping[str, Any]:
    return tensor_surface(host).collect_runtime_tensors(
        model,
        remove_duplicate=remove_duplicate,
    )


def compute_runtime_tensor_schema_hash(
    host: IntegrationHost | None,
    tensors: Mapping[str, Any],
    *,
    remove_duplicate: bool,
) -> str:
    return tensor_surface(host).compute_runtime_tensor_schema_hash(
        tensors,
        remove_duplicate=remove_duplicate,
    )


def runtime_only_tensor_names(
    host: IntegrationHost | None,
    model: object,
) -> tuple[str, ...]:
    return tensor_surface(host).runtime_only_tensor_names(model)


def support_level(
    host: IntegrationHost | None,
    model: object,
    model_config: object,
) -> RuntimeSupportLevel:
    support = getattr(framework_host(host), "support_level", None)
    if callable(support):
        return tc_readiness.coerce_runtime_support_level(
            support(model, model_config),
            default=RuntimeSupportLevel.BLOCKED,
        )
    return RuntimeSupportLevel.BLOCKED


def process_after_load_class(
    host: IntegrationHost | None,
    model: object,
    model_config: object,
) -> FinalizeClass:
    framework = framework_host(host)
    process_after_load = getattr(framework, "process_after_load_class", None)
    if callable(process_after_load):
        return tc_readiness.coerce_finalize_class(
            process_after_load(model, model_config),
            default=FinalizeClass.UNKNOWN_BLOCKED,
        )
    finalize_policy = getattr(framework, "finalize_policy", None)
    if callable(finalize_policy):
        finalize_policy(model, model_config)
        return FinalizeClass.RUNTIME_ONLY
    return FinalizeClass.UNKNOWN_BLOCKED


def post_bind_finalize_class(
    host: IntegrationHost | None,
    model: object,
    model_config: object,
) -> FinalizeClass:
    framework = framework_host(host)
    post_bind_finalize = getattr(framework, "post_bind_finalize_class", None)
    if callable(post_bind_finalize):
        return tc_readiness.coerce_finalize_class(
            post_bind_finalize(model, model_config),
            default=FinalizeClass.RUNTIME_ONLY,
        )
    finalize_policy = getattr(framework, "finalize_policy", None)
    if callable(finalize_policy):
        finalize_policy(model, model_config)
        return FinalizeClass.RUNTIME_ONLY
    return FinalizeClass.RUNTIME_ONLY


def trace_model_load(
    host: IntegrationHost | None,
    model: object,
    ordered_names: Sequence[str],
    meta_by_name: Mapping[str, object],
    *,
    debug_dump_trace: bool = False,
) -> TracePlan:
    trace = getattr(framework_host(host), "trace_model_load", None)
    if not callable(trace):
        raise _capability_missing(
            "ArtifactRuntimeIntegration host requires RecipeTraceHost."
            "trace_model_load on recipe cache miss",
            level="level2-local-bootstrap",
            capability="recipe_trace",
            operation="local_bootstrap.trace_model_load",
            required_methods=("trace_model_load",),
            next_action=(
                "Implement RecipeTraceHost.trace_model_load or precompute "
                "a recipe through the admin/offline builder path."
            ),
        )
    return cast(
        TracePlan,
        trace(
            model,
            ordered_names,
            meta_by_name,
            debug_dump_trace=debug_dump_trace,
        ),
    )


def cleanup_after_recipe_build(
    host: IntegrationHost | None,
    model: object,
    model_config: object,
    *,
    framework_config: object | None = None,
) -> None:
    cleanup = getattr(framework_host(host), "cleanup_after_recipe_build", None)
    if callable(cleanup):
        cleanup(
            model,
            model_config,
            framework_config=framework_config,
        )


def semantic_probes(
    host: IntegrationHost | None,
    model: object,
    model_config: object,
) -> object:
    semantic_probe_fn = getattr(framework_host(host), "semantic_probes", None)
    if callable(semantic_probe_fn):
        return semantic_probe_fn(model, model_config)
    return None


def native_load_weights(
    host: IntegrationHost | None,
    model: object,
    weights: object,
) -> None:
    native_load = getattr(framework_host(host), "native_load_weights", None)
    if not callable(native_load):
        raise _capability_missing(
            "ArtifactRuntimeIntegration host requires NativeLoadHost for native "
            "checkpoint/source loading",
            level="level2-local-bootstrap",
            capability="native_load",
            operation="local_bootstrap.native_load_weights",
            required_methods=("native_load_weights",),
            next_action=(
                "Implement NativeLoadHost.native_load_weights for source "
                "bootstrap cache misses."
            ),
        )
    native_load(model, weights)


def recipe_framework_adapter(
    *,
    identity: FrameworkIdentity,
    support_level_fn: Callable[[object, object], RuntimeSupportLevel],
    runtime_only_tensor_names_fn: Callable[[object], tuple[str, ...]],
    process_after_load_class_fn: Callable[[object, object], FinalizeClass],
    post_bind_finalize_class_fn: Callable[[object, object], FinalizeClass],
    trace_model_load_fn: Callable[..., TracePlan],
    cleanup_after_recipe_build_fn: Callable[..., None],
    semantic_probes_fn: Callable[[object, object], object],
    native_load_weights_fn: Callable[[object, object], None],
) -> Any:
    return SimpleNamespace(
        framework_name=lambda: str(identity.framework_name),
        framework_version=lambda: str(identity.framework_version),
        adapter_version=lambda: str(identity.adapter_version),
        serving_abi_version=lambda _model_config=None: str(
            identity.serving_abi_version
        ),
        support_level=support_level_fn,
        runtime_only_tensor_names=runtime_only_tensor_names_fn,
        process_after_load_class=process_after_load_class_fn,
        post_bind_finalize_class=post_bind_finalize_class_fn,
        trace_model_load=trace_model_load_fn,
        cleanup_after_recipe_build=cleanup_after_recipe_build_fn,
        semantic_probes=semantic_probes_fn,
        native_load_weights=native_load_weights_fn,
    )


def assert_tensor_names_match_expected(
    tensors: Mapping[str, Any],
    expected_names: Sequence[str],
) -> None:
    expected = {str(name) for name in expected_names}
    if not expected:
        return
    actual = {str(name) for name in tensors}
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if not missing and not unexpected:
        return
    raise SchemaMismatchError(
        "TensorCast runtime tensor set does not match runtime artifact: "
        f"missing_count={len(missing)}, unexpected_count={len(unexpected)}"
    )


__all__ = [
    "align_runtime_tensor_names",
    "assert_model_ready_for_runtime_binding",
    "assert_tensor_names_match_expected",
    "build_meta_model",
    "cleanup_after_recipe_build",
    "collect_runtime_binding_tensors",
    "compute_runtime_tensor_schema_hash",
    "framework_host",
    "framework_identity",
    "native_load_weights",
    "post_bind_finalize_class",
    "prepare_model_construction",
    "process_after_load_class",
    "recipe_framework_adapter",
    "require_target_device",
    "runtime_only_tensor_names",
    "semantic_probes",
    "support_level",
    "tensor_surface",
    "trace_model_load",
]
