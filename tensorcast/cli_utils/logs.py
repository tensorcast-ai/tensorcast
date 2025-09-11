#  Copyright (c) 2025, TensorCast Team.

"""Log viewing helpers for daemon sessions."""

from __future__ import annotations

import os
import time

import click

from .errors import ServiceError
from .paths import get_current_session_id, session_paths


def logs_tail(
    *, session_id: str | None = None, stderr: bool = False, follow: bool = False
) -> None:
    sid = session_id or get_current_session_id()
    if not sid:
        raise ServiceError("No session id provided and no current session found")
    inst = session_paths(sid)
    log_file = inst.logs / ("daemon.err" if stderr else "daemon.out")
    if not log_file.exists():
        click.echo(f"No log file found for daemon session: {log_file}")
        return
    click.echo(f"==> {log_file} <==")
    if not follow:
        try:
            with open(log_file, "rb") as f:
                data = f.read()
                lines = data.splitlines()[-200:]
                for ln in lines:
                    click.echo(ln.decode("utf-8", errors="replace"))
        except Exception as e:  # noqa: BLE001
            raise ServiceError(f"Failed to read logs: {e}") from e
        return
    try:
        with open(log_file, "rb") as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if line:
                    click.echo(line.decode("utf-8", errors="replace"), nl=False)
                else:
                    time.sleep(0.2)
    except KeyboardInterrupt:
        return
    except Exception as e:  # noqa: BLE001
        raise ServiceError(f"Failed to tail logs: {e}") from e
