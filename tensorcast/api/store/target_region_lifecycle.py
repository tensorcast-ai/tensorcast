#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from tensorcast.api.store.realization_kernel import (
    RealizationReleaseContract,
    RealizationResourceEnvelope,
    envelope_for_target_region_registration,
    release_contract_for,
)
from tensorcast.api.store.region_utils import collect_storage_bases
from tensorcast.api.store.types import ArtifactError

logger = logging.getLogger(__name__)


class TargetRegionRegistrar(Protocol):
    def __call__(
        self,
        *,
        device_id: int,
        base_ptr: int,
        size_bytes: int,
        ttl_ms: int,
    ) -> object: ...


class TargetRegionUnregister(Protocol):
    def __call__(self, region_id: str, *, force: bool | None = None) -> bool: ...


class TargetRegionStore(Protocol):
    def register_vram_region(
        self,
        *,
        device_id: int,
        base_ptr: int,
        size_bytes: int,
        ttl_ms: int,
    ) -> object: ...

    def unregister_vram_region(
        self,
        region_id: str,
        *,
        force: bool | None = None,
    ) -> bool: ...


def _is_active_reference_cleanup_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return "region has active references" in message or "active reference" in message


def unregister_target_region_ids_best_effort(
    *,
    unregister_region: TargetRegionUnregister,
    region_ids: Sequence[str],
    context: str,
) -> None:
    for region_id in region_ids:
        try:
            released = unregister_region(region_id)
            if not released:
                logger.warning(
                    "%s: unregister_vram_region returned False (region_id=%s)",
                    context,
                    region_id,
                )
            continue
        except Exception as exc:  # noqa: BLE001
            if not _is_active_reference_cleanup_error(exc):
                logger.warning(
                    "%s: unregister_vram_region failed (region_id=%s): %s",
                    context,
                    region_id,
                    exc,
                )
                continue
            try:
                forced = unregister_region(region_id, force=True)
            except Exception as force_exc:  # noqa: BLE001
                logger.warning(
                    "%s: active-reference cleanup failed for region_id=%s "
                    "(normal=%s, force=%s)",
                    context,
                    region_id,
                    exc,
                    force_exc,
                )
                continue
            if not forced:
                logger.warning(
                    "%s: region cleanup deferred due to active references "
                    "(region_id=%s)",
                    context,
                    region_id,
                )


@dataclass(slots=True)
class TargetRegionRegistration:
    unregister_region: TargetRegionUnregister
    region_ids: tuple[str, ...]
    envelope: RealizationResourceEnvelope
    release_contract: RealizationReleaseContract | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def release_policy(self) -> tuple[str, ...]:
        return self.envelope.release_policy

    def release(self, *, context: str) -> None:
        if self._released:
            return
        contract = release_contract_for(
            self.envelope,
            lambda: unregister_target_region_ids_best_effort(
                unregister_region=self.unregister_region,
                region_ids=self.region_ids,
                context=context,
            ),
        )
        contract.release()
        self.release_contract = contract
        self._released = contract.released


def target_region_registration_error(
    *,
    exc: Exception,
    operation_name: str,
    requested_regions: int,
    registered_regions: int,
) -> ArtifactError:
    detail = str(exc).strip() or exc.__class__.__name__
    lowered = detail.lower()
    if "capacity reached" in lowered or isinstance(exc, MemoryError):
        return ArtifactError(
            f"{operation_name} region registration exhausted daemon registry capacity: "
            f"requested_regions={requested_regions}, "
            f"registered_before_failure={registered_regions}, "
            f"cause={detail}. Increase the daemon max_vram_regions limit "
            f"for large {operation_name} workloads.",
            status_code="RESOURCE_EXHAUSTED",
            retryable=False,
        )
    return ArtifactError(
        f"{operation_name} failed to register target CUDA regions: "
        f"requested_regions={requested_regions}, "
        f"registered_before_failure={registered_regions}, "
        f"cause={detail}. {operation_name} requires user-owned CUDA memory.",
        status_code="FAILED_PRECONDITION",
        retryable=False,
    )


def register_target_regions_for_realization(
    *,
    register_region: TargetRegionRegistrar,
    unregister_region: TargetRegionUnregister,
    target_tensors: Mapping[str, torch.Tensor],
    device_id: int,
    ttl_ms: int,
    context: str,
    operation_name: str,
) -> TargetRegionRegistration:
    bases = collect_storage_bases(target_tensors)
    region_ids: list[str] = []
    envelope = envelope_for_target_region_registration(target_tensors)
    try:
        for base_ptr, nbytes in sorted(bases.items()):
            handle = register_region(
                device_id=device_id,
                base_ptr=base_ptr,
                size_bytes=nbytes,
                ttl_ms=int(ttl_ms),
            )
            region_ids.append(str(handle.region_id))
    except Exception as exc:  # noqa: BLE001
        registration = TargetRegionRegistration(
            unregister_region=unregister_region,
            region_ids=tuple(region_ids),
            envelope=envelope,
        )
        registration.release(context=context)
        raise target_region_registration_error(
            exc=exc,
            operation_name=operation_name,
            requested_regions=len(bases),
            registered_regions=len(region_ids),
        ) from exc
    return TargetRegionRegistration(
        unregister_region=unregister_region,
        region_ids=tuple(region_ids),
        envelope=envelope,
    )


def register_store_target_regions_for_realization(
    *,
    store: TargetRegionStore,
    target_tensors: Mapping[str, torch.Tensor],
    device_id: int,
    ttl_ms: int,
    context: str,
    operation_name: str = "bind_into",
) -> TargetRegionRegistration:
    return register_target_regions_for_realization(
        register_region=store.register_vram_region,
        unregister_region=store.unregister_vram_region,
        target_tensors=target_tensors,
        device_id=device_id,
        ttl_ms=ttl_ms,
        context=context,
        operation_name=operation_name,
    )


def release_target_region_ids_for_realization(
    *,
    unregister_region: TargetRegionUnregister,
    region_ids: Sequence[str],
    context: str,
) -> TargetRegionRegistration:
    registration = TargetRegionRegistration(
        unregister_region=unregister_region,
        region_ids=tuple(dict.fromkeys(str(region_id) for region_id in region_ids)),
        envelope=envelope_for_target_region_registration(total_bytes=0),
    )
    registration.release(context=context)
    return registration


__all__ = [
    "TargetRegionRegistration",
    "release_target_region_ids_for_realization",
    "register_store_target_regions_for_realization",
    "register_target_regions_for_realization",
    "target_region_registration_error",
    "unregister_target_region_ids_best_effort",
]
