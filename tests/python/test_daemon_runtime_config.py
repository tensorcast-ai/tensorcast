#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from pathlib import Path

import pytest

from tensorcast.daemon_runtime_config import load_daemon_config


def test_load_daemon_config_defaults_public_disk_source_from_storage_path(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "daemon.yaml"
    cfg_path.write_text(
        """
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp/models"
  num_threads: 2
pinned_memory:
  allocation_timeout: 30s
  classes: []
""",
        encoding="utf-8",
    )

    cfg = load_daemon_config(cfg_path)

    assert len(cfg.public_disk_source.trusted_root_policies) == 1
    policy = cfg.public_disk_source.trusted_root_policies[0]
    assert policy.root_path == "/tmp/models"
    assert policy.policy_id.startswith("trusted_storage_root_")
    assert (
        policy.descriptor_reuse_mode
        == cfg.PUBLIC_DISK_SOURCE_DESCRIPTOR_REUSE_MODE_TRUSTED_HINT_ONLY
    )
    assert (
        policy.validation_mode
        == cfg.PUBLIC_DISK_SOURCE_VALIDATION_MODE_VALIDATE_BEFORE_READ
    )
    assert policy.lightweight_attestation_enabled is True
    assert (
        cfg.public_disk_source.unmatched_path_mode
        == cfg.PublicDiskSource.UNMATCHED_PATH_MODE_REJECT
    )


def test_load_daemon_config_rejects_overlapping_public_disk_source_roots(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "daemon.yaml"
    cfg_path.write_text(
        """
server:
  listen: {host: "127.0.0.1", port: 50052}
  p2p_listen: {host: "127.0.0.1", port: 65090}
  storage_path: "/tmp/models"
  num_threads: 2
public_disk_source:
  trusted_root_policies:
    - policy_id: root-a
      root_path: /tmp/models
      validation_mode: PUBLIC_DISK_SOURCE_VALIDATION_MODE_VALIDATE_BEFORE_READ
      lightweight_attestation_enabled: true
    - policy_id: root-b
      root_path: /tmp/models/subdir
      validation_mode: PUBLIC_DISK_SOURCE_VALIDATION_MODE_VALIDATE_BEFORE_READ
      lightweight_attestation_enabled: true
pinned_memory:
  allocation_timeout: 30s
  classes: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not overlap"):
        load_daemon_config(cfg_path)
