#  Copyright (c) 2025, StepCast Team.

"""Transport domain artifact."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4


@dataclass
class Transport:
    """Represents an active artifact transport between nodes."""

    # Identity
    transport_id: UUID = field(default_factory=uuid4)
    replica_id: UUID = field(default_factory=uuid4)
    artifact_id: str = ""  # content-addressed ID
    disk_path: str | None = None

    # Source information
    source_node_id: str = ""
    source_address: str = ""
    source_port: int = 0

    # Timestamps
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Status
    status: str = "in_progress"  # in_progress, completed, failed

    @property
    def age_seconds(self) -> float:
        """Get the age of this transport in seconds."""
        if not self.created_at:
            return 0.0

        # Ensure we compare *aware* vs *aware* or *naive* vs *naive* datetimes
        now = (
            datetime.now(tz=self.created_at.tzinfo)
            if self.created_at.tzinfo is not None
            else datetime.now()
        )
        return (now - self.created_at).total_seconds()

    @property
    def duration_seconds(self) -> Optional[float]:
        """Get the duration of this transport in seconds if completed."""
        if not self.created_at or not self.completed_at:
            return None
        return (self.completed_at - self.created_at).total_seconds()
