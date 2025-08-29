#!/usr/bin/env python3
#  Copyright (c) 2025, TensorCast Team.

"""Torch version management utilities and CLI.

This module merges the functionality that previously lived in:
  • tools/update_torch_version.py
  • tools/torch_version_utils.py
  • tools/manage_torch_version.py

It exposes a single public surface so other parts of the codebase only need
`from torch_version_manager import ...`.

Key public helpers:
  • validate_torch_versions(raise_on_error: bool = True) -> (bool, dict)
  • update_torch_version(content: str, version: str) -> str
  • cache_uv_lock(torch_version: str, cuda_version: str | None = None)
  • restore_uv_lock(torch_version: str, cuda_version: str | None = None) -> bool

Running this file as a script provides the same CLI that was previously
available via ``python tools/manage_torch_version.py``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import toml

__all__ = [
    "validate_torch_versions",
    "update_torch_version",
    "cache_uv_lock",
    "restore_uv_lock",
]

###############################################################################
#  Low-level helpers (moved from torch_version_utils)                         #
###############################################################################

def _get_torch_version_from_python() -> str | None:
    """Return the torch version installed in the current (or .venv) env."""
    try:
        import torch  # local import to avoid heavy dependency at import time

        return torch.__version__.split("+")[0]
    except ImportError:
        # Try inside project venv (helps when running via system python)
        try:
            result = subprocess.run(
                [".venv/bin/python", "-c", "import torch, sys; sys.stdout.write(torch.__version__.split('+')[0])"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None


def _get_pyproject_torch_versions() -> Dict[str, str]:
    """Collect every torch version pin inside ``pyproject.toml``."""
    versions: Dict[str, str] = {}
    try:
        with open("pyproject.toml", "r", encoding="utf-8") as fp:
            data = toml.load(fp)

        # build-system.requires
        for req in data.get("build-system", {}).get("requires", []):
            if req.startswith("torch=="):
                versions["build-system"] = req.split("==")[1]

        # project.dependencies
        for dep in data.get("project", {}).get("dependencies", []):
            if dep.startswith("torch=="):
                versions["dependencies"] = dep.split("==")[1]

        # project.optional-dependencies.*
        for group, deps in data.get("project", {}).get("optional-dependencies", {}).items():
            for dep in deps:
                if dep.startswith("torch=="):
                    versions[f"optional-dependencies.{group}"] = dep.split("==")[1]
    except FileNotFoundError:
        print("Warning: pyproject.toml not found – cannot inspect torch versions", file=sys.stderr)
    except Exception as exc:  # noqa: BLE001
        print(f"Error reading pyproject.toml: {exc}", file=sys.stderr)
    return versions


###############################################################################
#  Public helpers                                                             #
###############################################################################

def validate_torch_versions(raise_on_error: bool = True) -> Tuple[bool, Dict[str, str]]:
    """Ensure every torch pin across the repo (env + pyproject) matches.

    Returns ``(is_consistent, version_map)``.
    """
    versions: Dict[str, str] = {}

    python_version = _get_torch_version_from_python()
    if python_version:
        versions[".venv"] = python_version

    versions.update(_get_pyproject_torch_versions())

    unique_versions = set(versions.values())
    is_consistent = len(unique_versions) <= 1

    if not is_consistent:
        msg_lines = ["Torch version mismatch detected:"] + [f"  {k}: {v}" for k, v in sorted(versions.items())]
        msg = "\n".join(msg_lines)
        if raise_on_error:
            raise ValueError(msg)
        else:
            print(msg, file=sys.stderr)
    return is_consistent, versions


# ---- uv.lock caching -------------------------------------------------------

def _get_uv_lock_filename(torch_version: str, cuda_version: Optional[str] = None) -> str:
    """Generate a unique filename for a given torch/cuda combo."""
    if cuda_version and not cuda_version.startswith("cu"):
        cuda_version = f"cu{cuda_version.replace('.', '')}"
    cuda_str = f"-{cuda_version}" if cuda_version else "-default"
    return f"uv.lock.torch{torch_version.replace('.', '')}{cuda_str}"


def cache_uv_lock(torch_version: str, cuda_version: Optional[str] = None) -> None:
    """Copy ``uv.lock`` into *tools/uv-lock-cache* for later reuse."""
    lock_path = Path("uv.lock")
    if not lock_path.exists():
        print("No uv.lock found to cache", file=sys.stderr)
        return

    cache_dir = Path("tools/uv-lock-cache")
    cache_dir.mkdir(exist_ok=True)
    dst = cache_dir / _get_uv_lock_filename(torch_version, cuda_version)
    import shutil

    shutil.copy2(lock_path, dst)
    print(f"Cached uv.lock to {dst}")


def restore_uv_lock(torch_version: str, cuda_version: Optional[str] = None) -> bool:
    """Restore a cached lock file.

    Returns True when a cache hit occurred.
    """
    cache_dir = Path("tools/uv-lock-cache")
    src = cache_dir / _get_uv_lock_filename(torch_version, cuda_version)
    if src.exists():
        import shutil

        shutil.copy2(src, "uv.lock")
        print(f"Restored uv.lock from {src}")
        return True
    return False


# ---- pyproject.toml manipulation ------------------------------------------

def update_torch_version(content: str, version: str) -> str:
    """Replace every ``torch==x.y.z`` occurrence in *content* with *version*."""
    pattern = r'(\"torch==)[0-9.]+(\")'
    replacement = rf"\g<1>{version}\g<2>"
    updated = re.sub(pattern, replacement, content)

    original_count = len(re.findall(pattern, content))
    updated_count = len(re.findall(rf'\"torch=={re.escape(version)}\"', updated))
    print(f"Updated {original_count} torch version references to {version} (now {updated_count} occurrences)")
    return updated


###############################################################################
#  CLI (keeps previous manage_torch_version interface)                        #
###############################################################################

def _show_status() -> None:
    print("Current Torch Version Status")
    print("=" * 50)

    is_consistent, versions = validate_torch_versions(raise_on_error=False)
    for source, version in sorted(versions.items()):
        status = "✓" if is_consistent else "✗"
        print(f"{status} {source:30} {version}")

    print("=" * 50)
    if is_consistent:
        print("✓ All versions are consistent")
    else:
        print("✗ Version mismatch detected!")
        print("\nTo fix, run one of:")
        venv_version = versions.get(".venv") or versions.get("Python import")
        if venv_version:
            print(f"  1. Update pyproject.toml to match .venv:\n     python tools/torch_version_manager.py update-pyproject {venv_version}")
        print("  2. Update .venv to match pyproject.toml:\n     python tools/torch_version_manager.py sync-venv")


def _update_pyproject(version: str) -> None:
    print(f"Updating pyproject.toml to torch {version}…")
    with open("pyproject.toml", "r", encoding="utf-8") as fp:
        content = fp.read()
    updated = update_torch_version(content, version)
    with open("pyproject.toml", "w", encoding="utf-8") as fp:
        fp.write(updated)
    print("✓ Updated pyproject.toml")


def _sync_venv() -> None:
    print("Syncing .venv with pyproject.toml…")
    try:
        subprocess.run(["uv", "sync"], check=True)
        print("✓ Environment synced successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to sync environment", file=sys.stderr)
        sys.exit(1)


def _manage_cache(action: str, torch_version: Optional[str] = None, cuda_version: Optional[str] = None) -> None:
    if action == "save":
        if not torch_version:
            print("Error: torch_version is required for save action", file=sys.stderr)
            return
        cache_uv_lock(torch_version, cuda_version)
    elif action == "restore":
        if not torch_version:
            print("Error: torch_version is required for restore action", file=sys.stderr)
            return
        if restore_uv_lock(torch_version, cuda_version):
            print(f"✓ Restored uv.lock for torch {torch_version} {cuda_version or 'CPU'}")
        else:
            print(f"✗ No cached uv.lock found for torch {torch_version} {cuda_version or 'CPU'}")
    elif action == "list":
        cache_dir = Path("tools/uv-lock-cache")
        if cache_dir.exists():
            files = sorted(cache_dir.glob("uv.lock.*"))
            if files:
                print("Cached uv.lock files:")
                for f in files:
                    print(f"  {f.name}")
            else:
                print("No cached uv.lock files found")
        else:
            print("Cache directory does not exist")


###############################################################################
#  Entry-point                                                                #
###############################################################################

def main(argv: list[str] | None = None) -> None:  # noqa: D401
    """CLI wrapper – replicates `tools/manage_torch_version.py`."""
    parser = argparse.ArgumentParser(
        description="Manage torch versions across the project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show current status
  %(prog)s status

  # Update pyproject.toml to torch 2.6.0
  %(prog)s update-pyproject 2.6.0

  # Sync .venv with pyproject.toml
  %(prog)s sync-venv

  # Cache current uv.lock
  %(prog)s cache save 2.6.0 --cuda cu118

  # Restore cached uv.lock
  %(prog)s cache restore 2.6.0 --cuda cu118

  # List cached lock files
  %(prog)s cache list
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # status (default)
    subparsers.add_parser("status", help="Show current torch version status")

    # update-pyproject
    upd = subparsers.add_parser("update-pyproject", help="Update torch version in pyproject.toml")
    upd.add_argument("version", help="Torch version to set (e.g., 2.6.0)")

    # sync-venv
    subparsers.add_parser("sync-venv", help="Sync .venv with pyproject.toml using uv")

    # cache
    cache_parser = subparsers.add_parser("cache", help="Manage uv.lock cache")
    cache_sub = cache_parser.add_subparsers(dest="cache_action", help="Cache action")

    save = cache_sub.add_parser("save", help="Save current uv.lock to cache")
    save.add_argument("torch_version", help="Torch version")
    save.add_argument("--cuda", help="CUDA version (e.g., cu118)")

    restore = cache_sub.add_parser("restore", help="Restore uv.lock from cache")
    restore.add_argument("torch_version", help="Torch version")
    restore.add_argument("--cuda", help="CUDA version (e.g., cu118)")

    cache_sub.add_parser("list", help="List cached lock files")

    args = parser.parse_args(argv)

    if args.command in {"status", None}:
        _show_status()
    elif args.command == "update-pyproject":
        _update_pyproject(args.version)
    elif args.command == "sync-venv":
        _sync_venv()
    elif args.command == "cache":
        if args.cache_action == "save":
            _manage_cache("save", args.torch_version, args.cuda)
        elif args.cache_action == "restore":
            _manage_cache("restore", args.torch_version, args.cuda)
        elif args.cache_action == "list":
            _manage_cache("list")
        else:
            cache_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()