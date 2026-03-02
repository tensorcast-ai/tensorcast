#!/usr/bin/env python
# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import contextlib
import json
import time

from tensorcast import startup
from tensorcast.daemon_ctl import DaemonCtl


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-host helper: fetch one daemon detailed status snapshot."
    )
    parser.add_argument("--daemon", required=True, help="Daemon address host:port")
    args = parser.parse_args()

    startup.init(mode="connect", address=str(args.daemon))
    try:
        client = DaemonCtl(server_address=str(args.daemon))
        response = client.get_detailed_status()
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()

    gpu_replica_count = sum(
        int(len(device.loaded_replicas)) for device in response.gpu_devices
    )
    payload = {
        "daemon": str(args.daemon),
        "timestamp_epoch": float(time.time()),
        "is_healthy": bool(response.is_healthy),
        "is_shutting_down": bool(response.is_shutting_down),
        "total_replicas_loaded": int(response.total_replicas_loaded),
        "total_artifact_size_bytes": int(response.total_artifact_size_bytes),
        "cpu_replica_count": int(len(response.cpu_replicas)),
        "gpu_replica_count": int(gpu_replica_count),
        "gpu_device_count": int(len(response.gpu_devices)),
        "comm_total_transfers": int(response.communication_info.total_transfers),
        "comm_total_bytes_transferred": int(
            response.communication_info.total_bytes_transferred
        ),
        "comm_total_transfer_errors": int(
            response.communication_info.total_transfer_errors
        ),
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
