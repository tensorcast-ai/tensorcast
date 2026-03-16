#  Copyright (c) 2025-2026, TensorCast Team.

"""Shard-home lease record used for byte artifact routing."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ShardHomeLease:
    """Represents ownership of a shard_id by a daemon under a fencing generation."""

    shard_id: int = 0
    holder_daemon_id: str = ""
    lease_token: str = ""
    lease_generation: int = 0
    expires_at: datetime | None = None
