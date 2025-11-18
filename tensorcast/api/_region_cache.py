#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Dict, List, Optional


@dataclass(frozen=True)
class RegionRecord:
    region_id: str
    device_id: int
    base_ptr: int
    size_bytes: int
    ttl_ms: int


_LOCK: RLock = RLock()
_REGIONS_BY_DEVICE: Dict[int, List[RegionRecord]] = {}


def register_region(
    *,
    region_id: str,
    device_id: int,
    base_ptr: int,
    size_bytes: int,
    ttl_ms: int,
) -> None:
    """Register or update a VRAM region in the client-side cache.

    This cache enables the SDK to detect when individual storages fall within a
    pre-registered region so it can avoid exporting redundant CUDA IPC handles.
    """
    rec = RegionRecord(
        region_id=str(region_id),
        device_id=int(device_id),
        base_ptr=int(base_ptr),
        size_bytes=int(size_bytes),
        ttl_ms=int(ttl_ms),
    )
    with _LOCK:
        lst = _REGIONS_BY_DEVICE.setdefault(int(device_id), [])
        for i, existing in enumerate(lst):
            if existing.region_id == rec.region_id:
                lst[i] = rec
                break
        else:
            lst.append(rec)


def unregister_region(region_id: str) -> None:
    """Remove a region from the client-side cache by id."""
    with _LOCK:
        for dev, lst in list(_REGIONS_BY_DEVICE.items()):
            new_list = [r for r in lst if r.region_id != region_id]
            if new_list:
                _REGIONS_BY_DEVICE[dev] = new_list
            else:
                _REGIONS_BY_DEVICE.pop(dev, None)


def get_regions_for_device(device_id: int) -> list[RegionRecord]:
    with _LOCK:
        lst = list(_REGIONS_BY_DEVICE.get(int(device_id), ()))
    return lst


def find_region_for(
    device_id: int, base_ptr: int, length_bytes: int
) -> Optional[RegionRecord]:
    """Find a region that fully covers [base_ptr, base_ptr + length_bytes).

    Returns None if no matching region exists.
    """
    start = int(base_ptr)
    end = start + int(length_bytes)
    with _LOCK:
        for rec in _REGIONS_BY_DEVICE.get(int(device_id), ()):
            r_start = rec.base_ptr
            r_end = rec.base_ptr + rec.size_bytes
            if start >= r_start and end <= r_end:
                return rec
    return None
