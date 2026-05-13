#!/usr/bin/env python3
#  Copyright (c) 2025-2026, TensorCast Team.

"""Post-process a freshly built tensorcast wheel for distribution.

Stages:
  1. `wheel unpack` the input wheel.
  2. patchelf --set-rpath on the daemon binary and `tensorcast` native `.so`s so
     they resolve LibTorch / NVIDIA libs from the user's site-packages at
     runtime via `$ORIGIN`-relative paths.
  3. `strip --strip-unneeded` the daemon binary to drop debug bloat.
  4. `os.chmod(0o755)` the daemon binary so its executable bit survives the
     repack (wheel RECORD does not preserve unix mode bits otherwise).
  5. `wheel pack` the directory back into a wheel.
  6. (Stage B / docker / IN_DOCKER=1) Run `auditwheel repair --plat
     manylinux_2_28_x86_64` excluding torch/CUDA libs to produce a PyPI-eligible
     `manylinux_2_28_x86_64` wheel.

Stage A (default, host build) skips auditwheel and leaves the wheel with its
original `linux_x86_64` platform tag — good for local install / e2e
verification, not for PyPI upload.

Usage:
  python tools/wheel_post_process.py dist/tensorcast-0.1.0-cp310-cp310-linux_x86_64.whl
  IN_DOCKER=1 python tools/wheel_post_process.py dist/...whl   # also run auditwheel
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# RPATH layout: tensorcast/_C*.so sits in `tensorcast/`, tensorcast/lib/*.so
# also in `tensorcast/`, tensorcast/bin/tensorcast_daemon sits one level
# deeper, so binary RPATH needs one extra `..`.
#
# Inside a venv's site-packages, the relevant siblings are:
#   site-packages/tensorcast/...                <- our package
#   site-packages/torch/lib/libtorch_cpu.so     <- pip install torch
#   site-packages/nvidia/cuda_runtime/lib/...   <- pip install nvidia-cuda-runtime-cu12
#
# From `site-packages/tensorcast/_C.so`:
#   $ORIGIN/..              -> site-packages/
#   $ORIGIN/../torch/lib    -> torch libs
# From `site-packages/tensorcast/bin/tensorcast_daemon`:
#   $ORIGIN/..              -> site-packages/tensorcast/
#   $ORIGIN/../..           -> site-packages/
#   $ORIGIN/../../torch/lib -> torch libs

_RPATH_PARTS_FROM_PACKAGE_ROOT = [
    "torch/lib",
    "nvidia/cuda_runtime/lib",
    "nvidia/cudnn/lib",
    "nvidia/cublas/lib",
    "nvidia/cuda_cupti/lib",
    "nvidia/cuda_nvrtc/lib",
    "nvidia/cufft/lib",
    "nvidia/curand/lib",
    "nvidia/cusolver/lib",
    "nvidia/cusparse/lib",
    "nvidia/cusparselt/lib",
    "nvidia/nccl/lib",
    "nvidia/nvjitlink/lib",
    "nvidia/nvtx/lib",
]


def _rpath(levels_up: int) -> str:
    up = "/".join([".."] * levels_up)
    head = f"$ORIGIN/{up}" if levels_up > 0 else "$ORIGIN"
    parts = [head]
    parts.extend(f"$ORIGIN/{up}/{p}" if levels_up > 0 else f"$ORIGIN/{p}"
                 for p in _RPATH_PARTS_FROM_PACKAGE_ROOT)
    parts.append(f"$ORIGIN/{up}/tensorcast/lib" if levels_up > 0 else "$ORIGIN/tensorcast/lib")
    return ":".join(parts)


# `tensorcast/_C*.so` and `tensorcast/lib/*.so` are one level inside the package
# root (site-packages/tensorcast/). To reach site-packages we go up by 1.
RPATH_PACKAGE_ROOT = _rpath(levels_up=1)
# `tensorcast/bin/tensorcast_daemon` is two levels inside site-packages.
RPATH_BIN = _rpath(levels_up=2)


# Libraries we exclude from auditwheel — they belong to the user's torch /
# nvidia / driver install, not ours.
AUDITWHEEL_EXCLUDES = [
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_python.so",
    "libtorch_global_deps.so",
    "libc10.so",
    "libc10_cuda.so",
    "libcudart.so.12",
    "libcudnn.so.9",
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libnccl.so.2",
    "libcupti.so.12",
    "libcuda.so.1",
    "libcufile.so.0",
    "libcurand.so.10",
    "libcusparse.so.12",
    "libcusparseLt.so.0",
    "libcusolver.so.11",
    "libcufft.so.11",
    "libnvjitlink.so.12",
    "libnvrtc.so.12",
    "libnvToolsExt.so.1",
    "libnvidia-ml.so.1",
]


class ToolMissingError(RuntimeError):
    """Raised when a required external tool is not on PATH."""


def _require(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise ToolMissingError(
            f"required tool `{name}` is not on PATH. "
            "The tensorcast release toolchain (wheel / auditwheel / patchelf / twine) "
            "lives in the `release` dependency group. Run `uv sync --group release` "
            "(or use `tools/release.sh post-process`, which does this for you)."
        )
    return path


def _run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _patchelf_set_rpath(elf: Path, rpath: str) -> None:
    patchelf = _require("patchelf")
    _run([patchelf, "--force-rpath", "--set-rpath", rpath, str(elf)])


def _strip_unneeded(elf: Path) -> None:
    strip = _require("strip")
    _run([strip, "--strip-unneeded", str(elf)])


def _chmod_executable(path: Path) -> None:
    print(f"+ chmod 0755 {path}")
    os.chmod(path, 0o755)


def _wheel_unpack(wheel: Path, out_dir: Path) -> Path:
    """Unpack `wheel` into `out_dir/<name>-<version>/` and return that directory."""
    wheel_cli = _require("wheel")
    _run([wheel_cli, "unpack", "--dest", str(out_dir), str(wheel)])
    children = [p for p in out_dir.iterdir() if p.is_dir()]
    if len(children) != 1:
        raise RuntimeError(
            f"expected exactly one unpacked dir under {out_dir}, got {children}"
        )
    return children[0]


def _wheel_pack(unpacked_dir: Path, dest_dir: Path) -> Path:
    """Pack `unpacked_dir` back into a wheel in `dest_dir` and return its path.

    `wheel pack` rebuilds the same filename as the input, so a simple
    before/after set diff misses the produced wheel when it overwrites an
    existing entry. We instead track mtimes: any wheel whose mtime advances
    past the moment we started the pack is the one wheel pack just wrote.
    """
    wheel_cli = _require("wheel")
    mark = time.time()
    _run([wheel_cli, "pack", "--dest-dir", str(dest_dir), str(unpacked_dir)])
    produced = [
        whl for whl in dest_dir.glob("*.whl")
        if whl.stat().st_mtime >= mark - 1.0
    ]
    if len(produced) != 1:
        raise RuntimeError(
            f"wheel pack produced unexpected output set: {produced}"
        )
    return produced[0]


def _auditwheel_repair(wheel: Path, dest_dir: Path, plat: str) -> Path:
    auditwheel = _require("auditwheel")
    cmd = [auditwheel, "repair", "--plat", plat, "-w", str(dest_dir)]
    for libname in AUDITWHEEL_EXCLUDES:
        cmd.extend(["--exclude", libname])
    cmd.append(str(wheel))
    before = set(dest_dir.glob("*.whl"))
    _run(cmd)
    after = set(dest_dir.glob("*.whl"))
    produced = after - before
    if len(produced) != 1:
        raise RuntimeError(
            f"auditwheel produced unexpected output set: {produced}"
        )
    return produced.pop()


def _patch_directory(unpacked: Path) -> None:
    pkg_root = unpacked / "tensorcast"
    daemon = pkg_root / "bin" / "tensorcast_daemon"
    if daemon.exists():
        _patchelf_set_rpath(daemon, RPATH_BIN)
        _strip_unneeded(daemon)
        _chmod_executable(daemon)
    else:
        print(f"Warning: daemon binary not found at {daemon}; skipping")

    # `tensorcast/_C*.so` and other top-level package .so files
    for so in pkg_root.glob("*.so"):
        _patchelf_set_rpath(so, RPATH_PACKAGE_ROOT)

    lib_dir = pkg_root / "lib"
    if lib_dir.exists():
        for so in lib_dir.glob("*.so*"):
            if so.is_symlink():
                continue
            _patchelf_set_rpath(so, RPATH_PACKAGE_ROOT)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Wheel produced by setup.py build")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Destination for the post-processed wheel (default: same dir as input)",
    )
    parser.add_argument(
        "--auditwheel",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Whether to run auditwheel repair. `auto` enables it when IN_DOCKER=1.",
    )
    parser.add_argument(
        "--plat",
        default="manylinux_2_28_x86_64",
        help="Target manylinux platform tag (only used when running auditwheel)",
    )
    args = parser.parse_args(argv)

    wheel: Path = args.wheel.resolve()
    if not wheel.is_file():
        print(f"error: wheel not found: {wheel}", file=sys.stderr)
        return 1

    dest_dir: Path = (args.dest_dir or wheel.parent).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    workdir = wheel.parent / "_post_process_work"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir()

    try:
        unpacked = _wheel_unpack(wheel, workdir)
        _patch_directory(unpacked)
        repacked = _wheel_pack(unpacked, dest_dir)
        print(f"\nRepacked wheel: {repacked}")

        run_auditwheel = args.auditwheel == "yes" or (
            args.auditwheel == "auto" and os.environ.get("IN_DOCKER") == "1"
        )
        if run_auditwheel:
            final = _auditwheel_repair(repacked, dest_dir, args.plat)
            print(f"\nFinal manylinux wheel: {final}")
        else:
            print(
                "\nSkipping auditwheel (Stage A / host build). "
                "Run inside the manylinux_2_28 docker with IN_DOCKER=1 to produce a "
                "PyPI-eligible wheel."
            )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
