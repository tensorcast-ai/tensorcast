#  Copyright (c) 2025, StepCast Team.

from __future__ import annotations

# -----------------------------------------------------------------------------
# Early patch for a PyTorch bug that can raise the following exception when
# importing `torch.overrides` multiple times within the same interpreter:
#
#   RuntimeError: function '_has_torch_function' already has a docstring
#
# The root cause is a duplicate call to `torch._C._add_docstr` which (by design)
# throws if a docstring already exists.  We defensively wrap the original
# implementation so that it silently ignores this specific error.  The patch is
# applied *before* any downstream code imports `torch.overrides`.
# -----------------------------------------------------------------------------
from types import ModuleType
from typing import Any, Callable

import torch  # pylint: disable=wrong-import-position

# Try to access torch._C module and its _add_docstr function
_c_mod: ModuleType = torch._C
_orig_add_docstr: Callable[[Any, str], Any] = _c_mod._add_docstr

# Check if we've already patched this function
# We need to use try/except here because we can't know if our custom attribute exists
try:
    already_patched = _orig_add_docstr._scstore_patched  # pyright: ignore[reportFunctionMemberAccess]
except AttributeError:
    already_patched = False

# Ensure we only patch once even if this module is re-imported.
if not already_patched:

    def _safe_add_docstr(obj: Any, doc: str) -> Any:
        try:
            return _orig_add_docstr(obj, doc)
        except RuntimeError as exc:  # noqa: BLE001
            # Ignore the specific error triggered by duplicate docstrings.
            if "already has a docstring" in str(exc):
                return obj
            raise

    _safe_add_docstr._scstore_patched = True  # pyright: ignore[reportFunctionMemberAccess]
    _c_mod._add_docstr = _safe_add_docstr

# -----------------------------------------------------------------------------
# Public package interface remains unchanged below.
# -----------------------------------------------------------------------------

import scstore._store_engine as _store_engine  # noqa: E402
from scstore._version import __version__  # noqa: E402

# Import functions from config module
from scstore.config import (  # noqa: E402
    init,
    is_initialized,
)

__all__ = [
    "__version__",
    "init",
    "is_initialized",
    "_store_engine",
]
