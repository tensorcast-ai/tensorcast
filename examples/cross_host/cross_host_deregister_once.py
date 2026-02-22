#!/usr/bin/env python
#  Copyright (c) 2026, TensorCast Team.

# Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import argparse
import contextlib
import json

import tensorcast as tc
from tensorcast import startup


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-host helper: best-effort deregister one artifact."
    )
    parser.add_argument("--daemon", required=True, help="Daemon address host:port")
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--device-id", type=int, default=None)
    args = parser.parse_args()

    removed = False
    drained = False
    attempts = 0
    message = None
    error = None
    startup.init(mode="connect", address=str(args.daemon))
    try:
        try:
            attempts = 1
            outcome = tc.deregister_artifact(
                str(args.artifact_id),
                wait=True,
                drain_timeout_s=30.0,
                device_id=int(args.device_id) if args.device_id is not None else None,
            )
            removed = bool(outcome.removed)
            drained = bool(outcome.drained)
            message = outcome.message
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
    finally:
        with contextlib.suppress(Exception):
            startup.shutdown()

    print(
        json.dumps(
            {
                "daemon": str(args.daemon),
                "artifact_id": str(args.artifact_id),
                "removed": bool(removed),
                "drained": bool(drained),
                "message": message,
                "attempts": int(attempts),
                "error": error,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
