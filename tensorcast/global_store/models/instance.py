#  Copyright (c) 2026, TensorCast Team.

"""Engine instance registry model."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping


@dataclass
class Instance:
    """Represents an engine process instance registered with the Global Store."""

    instance_id: str = ""
    daemon_id: str = ""
    worker_id: str | None = None
    engine: str = ""
    signals_endpoint: str | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    registered_at: datetime | None = None
    last_heartbeat: datetime | None = None
    inactive_at: datetime | None = None
