#  Copyright (c) 2026, TensorCast Team.

"""PyTorch primitives shared by TensorCast framework integrations."""

from tensorcast.pytorch.module_binding import (
    AttachResult,
    TensorInvariant,
    allocate_unbound_module_tensors,
    attach_tensors_to_module,
    collect_module_tensors,
)
from tensorcast.pytorch.trace_capture import (
    ScalarProxy,
    TraceActivation,
    TraceMode,
    build_tensorcast_slices,
    meta_weights_iterator,
    trace_model_load,
)

__all__ = [
    "AttachResult",
    "ScalarProxy",
    "TensorInvariant",
    "TraceActivation",
    "TraceMode",
    "allocate_unbound_module_tensors",
    "attach_tensors_to_module",
    "build_tensorcast_slices",
    "collect_module_tensors",
    "meta_weights_iterator",
    "trace_model_load",
]
