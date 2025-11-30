#  Copyright (c) 2025, TensorCast Team.

from __future__ import annotations

from pathlib import Path


def test_no_load_dict_from_disk_imports() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    package_root = repo_root / "tensorcast"
    offenders: list[Path] = []
    for path in package_root.rglob("*.py"):
        if path.name == "_io_disk.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "load_dict_from_disk" in text:
            offenders.append(path)
    assert not offenders, f"Production modules must not import load_dict_from_disk: {offenders}"
