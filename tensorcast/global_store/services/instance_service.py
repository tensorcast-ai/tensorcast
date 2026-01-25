#  Copyright (c) 2026, TensorCast Team.

"""Service for instance registry operations."""

import time

from tensorcast.global_store.config import get_config
from tensorcast.global_store.exceptions import ValidationError
from tensorcast.global_store.models import Instance
from tensorcast.global_store.repositories import InstanceRepository, WorkerRepository
from tensorcast.logger import init_logger

logger = init_logger(__name__)


class InstanceService:
    def __init__(
        self,
        instance_repository: InstanceRepository,
        worker_repository: WorkerRepository,
    ) -> None:
        self.instance_repository = instance_repository
        self.worker_repository = worker_repository
        self.config = get_config()

    def register_instance(self, instance: Instance) -> Instance:
        if not instance.instance_id:
            raise ValidationError("instance_id is required")
        if not instance.daemon_id:
            raise ValidationError("daemon_id is required")
        if not instance.engine:
            raise ValidationError("engine is required")

        if not instance.worker_id:
            worker = self.worker_repository.find_by_daemon_id(instance.daemon_id)
            if worker:
                instance.worker_id = worker.worker_id

        existing = self.instance_repository.find_by_id(
            instance.instance_id, include_inactive=True
        )
        if existing:
            existing.daemon_id = instance.daemon_id
            existing.worker_id = instance.worker_id
            existing.engine = instance.engine
            existing.signals_endpoint = instance.signals_endpoint
            existing.labels = instance.labels
            return self.instance_repository.update(existing)

        return self.instance_repository.create(instance)

    def heartbeat(self, instance_id: str, worker_id: str | None = None) -> bool:
        if not instance_id:
            raise ValidationError("instance_id is required")
        resolved_worker_id = worker_id
        if resolved_worker_id is None:
            inst = self.instance_repository.find_by_id(
                instance_id, include_inactive=True
            )
            if inst is not None and inst.daemon_id:
                worker = self.worker_repository.find_by_daemon_id(inst.daemon_id)
                if worker:
                    resolved_worker_id = worker.worker_id
        return self.instance_repository.heartbeat(
            instance_id, worker_id=resolved_worker_id
        )

    def unregister_instance(self, instance_id: str) -> bool:
        if not instance_id:
            raise ValidationError("instance_id is required")
        return self.instance_repository.mark_inactive(instance_id)

    def list_active_instances(
        self, *, include_unavailable: bool = False
    ) -> list[Instance]:
        return self.instance_repository.list_active(
            include_unavailable=include_unavailable
        )

    def cleanup_inactive_instances(self) -> int:
        timeout_sec = self.config.heartbeat_timeout_ms / 1000.0
        cutoff = max(0.0, time.time() - timeout_sec)
        return self.instance_repository.cleanup_stale_instances(cutoff)


__all__ = ["InstanceService"]
