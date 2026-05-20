#  Copyright (c) 2026, TensorCast Team.
"""Compatibility import for TensorCast serving lifecycle implementation.

Framework integrations should prefer ``tensorcast.serving.runtime`` and
``tensorcast.serving.hosts``.  The implementation lives in the private
``tensorcast.serving._runtime_impl`` package so this public module remains a
stable compatibility import without becoming the recommended integration API.
"""

from __future__ import annotations

import sys

from tensorcast.serving._runtime_impl import lifecycle as _lifecycle

sys.modules[__name__] = _lifecycle
