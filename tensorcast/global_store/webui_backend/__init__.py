#  Copyright (c) 2025, TensorCast Team.

"""Global Store Web UI module."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorcast.global_store.webui_backend.app import WebUIApp

__all__ = ["WebUIApp"]
