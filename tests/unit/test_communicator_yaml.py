#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path

from tensorcast.communicator import config_io as pycfg  # noqa: E402


def test_python_helper_from_yaml(tmp_path: Path) -> None:
    yaml = tmp_path / "comm.yaml"
    yaml.write_text(
        """
enable_rdma: false
stager:
  stage_chunk_mb_cpu: 12
transport:
  tcp_conn_count: 3
        """
    )

    cfg = pycfg.from_yaml(yaml)
    # Provided fields
    assert cfg.enable_rdma is False
    assert cfg.stager.stage_chunk_mb_cpu == 12
    assert cfg.transport.tcp_conn_count == 3
    # Defaults
    assert cfg.stager.stage_cpu_for_rdma is True
    assert cfg.stager.stage_chunk_mb_gpu == 16
    assert cfg.transport.connect_timeout_sec == 10
