#!/usr/bin/env python3
#  Copyright (c) 2025-2026, TensorCast Team.

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path
import re
import sys


PACKAGE_MAP: dict[str, tuple[str, ...]] = {
    "cudnn":      ("nvidia-cudnn-cu13", "nvidia-cudnn-cu12"),
    "nccl":       ("nvidia-nccl-cu13", "nvidia-nccl-cu12"),
    "cusparselt": ("nvidia-cusparselt-cu13", "nvidia-cusparselt-cu12"),
    # CUDA 13 dropped the -cu12 suffix on the core libs; cu12 lockfile still uses it.
    "cupti":      ("nvidia-cuda-cupti", "nvidia-cuda-cupti-cu12"),
    "cublas":     ("nvidia-cublas", "nvidia-cublas-cu12"),
    "cufft":      ("nvidia-cufft", "nvidia-cufft-cu12"),
    "cudart":     ("nvidia-cuda-runtime", "nvidia-cuda-runtime-cu12"),
    "curand":     ("nvidia-curand", "nvidia-curand-cu12"),
    "cusparse":   ("nvidia-cusparse", "nvidia-cusparse-cu12"),
    "cusolver":   ("nvidia-cusolver", "nvidia-cusolver-cu12"),
    "nvjitlink":  ("nvidia-nvjitlink", "nvidia-nvjitlink-cu12"),
    "nvrtc":      ("nvidia-cuda-nvrtc", "nvidia-cuda-nvrtc-cu12"),
    "nvtx":       ("nvidia-nvtx", "nvidia-nvtx-cu12"),
    "libtorch":   ("torch",),
    "triton":     ("triton",),
}

PYPI_PREFIX = "https://files.pythonhosted.org/packages/"

WHEEL_LINE_RE = re.compile(r'url = "([^"]+)", hash = "sha256[:=]([0-9a-fA-F]+)"')
NAME_RE = re.compile(r'name = "([^"]+)"')
INTEGRITY_RE = re.compile(r'integrity = "([^"]+)"')


@dataclass(frozen=True)
class Wheel:
    url: str
    sha256_hex: str


@dataclass(frozen=True)
class ArchiveUpdate:
    integrity: str
    urls: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update MODULE.bazel http_archive urls and integrity from uv.lock.",
    )
    parser.add_argument("--lockfile", default="uv.lock", help="Path to uv.lock")
    parser.add_argument("--module", default="MODULE.bazel", help="Path to MODULE.bazel")
    parser.add_argument("--python-tag", default="cp310", help="Python tag to select")
    parser.add_argument("--arch", default="x86_64", help="Architecture tag to select")
    parser.add_argument(
        "--verify-module",
        action="store_true",
        help="Verify integrity conversion against current MODULE.bazel before writing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned updates without writing MODULE.bazel.",
    )
    return parser.parse_args()


def parse_uv_lock(lock_path: Path, wanted: set[str]) -> dict[str, list[Wheel]]:
    wheels_by_package = {name: [] for name in wanted}
    current_package: str | None = None
    with lock_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("[[package]]"):
                current_package = None
                continue
            if line.startswith("name = "):
                match = NAME_RE.search(line)
                current_package = match.group(1) if match else None
                continue
            if current_package in wanted and ".whl" in line:
                match = WHEEL_LINE_RE.search(line)
                if match:
                    wheels_by_package[current_package].append(
                        Wheel(url=match.group(1), sha256_hex=match.group(2).lower()),
                    )
    return wheels_by_package


def select_wheel(wheels: list[Wheel], python_tag: str, arch: str) -> Wheel:
    if not wheels:
        raise ValueError("No wheels available to select from.")
    if len(wheels) == 1:
        return wheels[0]

    def filename(wheel: Wheel) -> str:
        return wheel.url.rsplit("/", 1)[-1]

    def is_pure_py3(wheel: Wheel) -> bool:
        # NVIDIA wheels (and similar) ship as `*-py3-none-<platform>.whl` without a cpython tag.
        # Treat any such wheel as compatible with the requested cpython tag.
        return "-py3-none-" in filename(wheel)

    def matches(wheel: Wheel, want_python: bool, want_arch: bool, want_linux: bool) -> bool:
        name = filename(wheel)
        if want_python and f"-{python_tag}-" not in name and not is_pure_py3(wheel):
            return False
        if want_arch and arch not in name:
            return False
        if want_linux and "linux" not in name:
            return False
        return True

    for want_python, want_arch, want_linux in (
        (True, True, True),
        (True, True, False),
        (True, False, False),
    ):
        candidates = [wheel for wheel in wheels if matches(wheel, want_python, want_arch, want_linux)]
        if candidates:
            return candidates[0]

    raise ValueError(f"Unable to select wheel for python_tag={python_tag}, arch={arch}")


def sri_from_sha256_hex(sha256_hex: str) -> str:
    digest = bytes.fromhex(sha256_hex)
    encoded = base64.b64encode(digest).decode("ascii")
    round_trip = base64.b64decode(encoded).hex()
    if round_trip != sha256_hex.lower():
        raise ValueError("sha256 hex/base64 round-trip validation failed")
    return f"sha256-{encoded}"


def build_urls(primary_url: str) -> list[str]:
    return [primary_url]


def parse_http_archive_blocks(lines: list[str]) -> dict[str, tuple[int, int]]:
    blocks: dict[str, tuple[int, int]] = {}
    index = 0
    while index < len(lines):
        if lines[index].lstrip().startswith("http_archive("):
            start = index
            depth = lines[index].count("(") - lines[index].count(")")
            index += 1
            while index < len(lines) and depth > 0:
                depth += lines[index].count("(") - lines[index].count(")")
                index += 1
            end = index - 1
            block_lines = lines[start : end + 1]
            name = extract_name(block_lines)
            if name:
                blocks[name] = (start, end)
            continue
        index += 1
    return blocks


def extract_name(block_lines: list[str]) -> str | None:
    for line in block_lines:
        stripped = line.lstrip()
        if stripped.startswith("name = "):
            match = NAME_RE.search(line)
            if match:
                return match.group(1)
    return None


def extract_integrity(block_lines: list[str]) -> str | None:
    for line in block_lines:
        if line.lstrip().startswith("integrity = "):
            match = INTEGRITY_RE.search(line)
            if match:
                return match.group(1)
    return None


def replace_integrity(block_lines: list[str], integrity: str) -> list[str]:
    for index, line in enumerate(block_lines):
        if line.lstrip().startswith("integrity = "):
            indent = line[: len(line) - len(line.lstrip())]
            block_lines[index] = f'{indent}integrity = "{integrity}",\n'
            return block_lines
    raise ValueError("integrity line not found in http_archive block")


def replace_urls(block_lines: list[str], urls: list[str]) -> list[str]:
    start = None
    end = None
    for index, line in enumerate(block_lines):
        if line.lstrip().startswith("urls = ["):
            start = index
            for scan in range(index + 1, len(block_lines)):
                if block_lines[scan].lstrip().startswith("]"):
                    end = scan
                    break
            break
    if start is None or end is None:
        raise ValueError("urls list not found in http_archive block")

    indent = block_lines[start][: len(block_lines[start]) - len(block_lines[start].lstrip())]
    item_indent = f"{indent}    "
    url_lines = [f'{item_indent}"{url}",\n' for url in urls]
    return block_lines[: start + 1] + url_lines + block_lines[end:]


def build_updates(
    wheels_by_package: dict[str, list[Wheel]],
    python_tag: str,
    arch: str,
) -> dict[str, ArchiveUpdate]:
    updates: dict[str, ArchiveUpdate] = {}
    for module_name, candidates in PACKAGE_MAP.items():
        wheels: list[Wheel] = []
        for package_name in candidates:
            wheels = wheels_by_package.get(package_name, [])
            if wheels:
                break
        if not wheels:
            raise ValueError(
                f"None of {candidates} have wheel entries in uv.lock for module: {module_name}"
            )
        wheel = select_wheel(wheels, python_tag, arch)
        integrity = sri_from_sha256_hex(wheel.sha256_hex)
        urls = build_urls(wheel.url)
        updates[module_name] = ArchiveUpdate(integrity=integrity, urls=urls)
    return updates


def apply_updates(
    module_path: Path,
    updates: dict[str, ArchiveUpdate],
    verify_module: bool,
    dry_run: bool,
) -> None:
    original_text = module_path.read_text(encoding="utf-8")
    lines = original_text.splitlines(keepends=True)
    blocks = parse_http_archive_blocks(lines)

    missing = [name for name in updates if name not in blocks]
    if missing:
        raise ValueError(f"Missing http_archive entries in MODULE.bazel: {', '.join(missing)}")

    mismatches: list[str] = []
    for name, update in updates.items():
        start, end = blocks[name]
        block_lines = lines[start : end + 1]
        if verify_module:
            existing = extract_integrity(block_lines)
            if existing != update.integrity:
                mismatches.append(
                    f"{name}: module={existing or 'missing'} lock={update.integrity}",
                )

    if mismatches:
        mismatch_text = "\n".join(mismatches)
        raise ValueError(f"Integrity verification failed:\n{mismatch_text}")

    for name, (start, end) in sorted(blocks.items(), key=lambda item: item[1][0], reverse=True):
        update = updates.get(name)
        if not update:
            continue
        block_lines = lines[start : end + 1]
        block_lines = replace_integrity(block_lines, update.integrity)
        block_lines = replace_urls(block_lines, update.urls)
        lines[start : end + 1] = block_lines

    updated_text = "".join(lines)
    if updated_text == original_text:
        print("MODULE.bazel already up to date.")
        return

    for name, update in updates.items():
        print(f"Updated {name}: integrity={update.integrity} urls={len(update.urls)}")

    if dry_run:
        print("Dry run enabled; MODULE.bazel not written.")
        return

    module_path.write_text(updated_text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    lock_path = Path(args.lockfile)
    module_path = Path(args.module)
    wheels_by_package = parse_uv_lock(
        lock_path,
        {pkg for candidates in PACKAGE_MAP.values() for pkg in candidates},
    )
    updates = build_updates(wheels_by_package, args.python_tag, args.arch)
    apply_updates(module_path, updates, args.verify_module, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
