#!/usr/bin/env python3
#  Copyright (c) 2025, TensorCast Team.

"""
Simple CLI to fetch direct download URLs from PyPI for a given package and version.

Examples:
  - List all files for torch 2.6.0:
      python tools/pypi_fetch.py torch 2.6.0

  - Filter by substrings (e.g., cp310 and manylinux1_x86_64):
      python tools/pypi_fetch.py torch 2.6.0 --match cp310 --match manylinux1_x86_64

Returns raw URLs by default; use --json for structured output.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Dict, Any

import urllib.request
import urllib.error


PYPI_JSON_URL = "https://pypi.org/pypi/{package}/{version}/json"


def fetch_pypi_release_files(package: str, version: str) -> List[Dict[str, Any]]:
    url = PYPI_JSON_URL.format(package=package, version=version)
    try:
        with urllib.request.urlopen(url) as resp:
            if resp.status != 200:
                raise RuntimeError(f"HTTP {resp.status} for {url}")
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise SystemExit(f"Failed to fetch PyPI metadata: HTTP {e.code} for {url}")
    except urllib.error.URLError as e:
        raise SystemExit(f"Failed to fetch PyPI metadata: {e.reason}")

    files = data.get("releases", {}).get(version, [])
    if not files:
        # Some packages use yanked or different structures; also try urls list
        files = data.get("urls", [])
    return files


def filter_files(files: List[Dict[str, Any]], match_substrings: List[str]) -> List[Dict[str, Any]]:
    if not match_substrings:
        return files
    filtered: List[Dict[str, Any]] = []
    for f in files:
        filename = f.get("filename", "")
        if all(substr in filename for substr in match_substrings):
            filtered.append(f)
    return filtered


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Fetch direct PyPI file URLs for a package version")
    parser.add_argument("package", help="Package name, e.g., torch")
    parser.add_argument("version", help="Version, e.g., 2.6.0")
    parser.add_argument("--match", action="append", default=[], help="Substring to match in filename (can repeat)")
    parser.add_argument("--json", dest="as_json", action="store_true", help="Print JSON objects instead of plain URLs")
    parser.add_argument("--first", action="store_true", help="Print only the first match")

    args = parser.parse_args(argv)

    files = fetch_pypi_release_files(args.package, args.version)
    if not files:
        print("No files found for given package/version", file=sys.stderr)
        return 2

    files = filter_files(files, args.match)
    if not files:
        print("No files matched given substrings", file=sys.stderr)
        return 3

    if args.first:
        files = files[:1]

    if args.as_json:
        # Output a compact JSON list of objects with key fields
        minimal = [
            {
                "filename": f.get("filename"),
                "url": f.get("url"),
                "digests": f.get("digests", {}),
                "yanked": f.get("yanked", False),
                "size": f.get("size"),
                "python_version": f.get("python_version"),
                "packagetype": f.get("packagetype"),
            }
            for f in files
        ]
        print(json.dumps(minimal, ensure_ascii=False))
    else:
        for f in files:
            print(f.get("url"))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


