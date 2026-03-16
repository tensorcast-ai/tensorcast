#!/usr/bin/env python3
#  Copyright (c) 2026, TensorCast Team.

"""Execute a command with explicit CPU affinity and NUMA memory policy."""

from __future__ import annotations

import argparse
import ctypes
import os
from pathlib import Path
import subprocess
import sys


MPOL_BIND = 2
SYS_SET_MEMPOLICY = 238


def parse_cpu_list(text: str) -> set[int]:
    cpus: set[int] = set()
    for item in text.strip().split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            cpus.update(range(start, end + 1))
            continue
        cpus.add(int(item))
    return cpus


def load_node_cpus(numa_node: int) -> set[int]:
    cpu_list_path = Path(f"/sys/devices/system/node/node{numa_node}/cpulist")
    if cpu_list_path.exists():
        return parse_cpu_list(cpu_list_path.read_text(encoding="utf-8"))

    proc = subprocess.run(
        ["bash", "-lc", "lscpu --parse=cpu,node | egrep -v '^(#|$)'"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(f"missing cpulist for NUMA node {numa_node}: {cpu_list_path}")
    cpus: set[int] = set()
    for line in proc.stdout.splitlines():
        cpu_text, node_text = [part.strip() for part in line.split(",", 1)]
        if not node_text:
            continue
        if int(node_text) == numa_node:
            cpus.add(int(cpu_text))
    if cpus:
        return cpus

    proc = subprocess.run(
        ["bash", "-lc", "lscpu"],
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise FileNotFoundError(f"could not resolve CPUs for NUMA node {numa_node}")
    prefix = f"NUMA node{numa_node} CPU(s):"
    for line in proc.stdout.splitlines():
        if not line.startswith(prefix):
            continue
        return parse_cpu_list(line.split(":", 1)[1].strip())
    return cpus


def set_memory_policy(numa_node: int) -> None:
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.syscall.argtypes = [
        ctypes.c_long,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_ulong,
    ]
    libc.syscall.restype = ctypes.c_long
    nodemask = (ctypes.c_ulong * 1)()
    nodemask[0] = 1 << numa_node
    result = libc.syscall(
        ctypes.c_long(SYS_SET_MEMPOLICY),
        ctypes.c_int(MPOL_BIND),
        ctypes.cast(nodemask, ctypes.POINTER(ctypes.c_ulong)),
        ctypes.c_ulong(ctypes.sizeof(ctypes.c_ulong) * 8),
    )
    if result != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--numa-node", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("missing command", file=sys.stderr)
        return 2
    if args.numa_node < 0:
        print(f"invalid NUMA node: {args.numa_node}", file=sys.stderr)
        return 2

    cpus = load_node_cpus(args.numa_node)
    if not cpus:
        print(f"NUMA node {args.numa_node} has no CPUs", file=sys.stderr)
        return 2

    os.sched_setaffinity(0, cpus)
    set_memory_policy(args.numa_node)
    os.execvp(command[0], command)
    return 0


if __name__ == "__main__":
    sys.exit(main())
