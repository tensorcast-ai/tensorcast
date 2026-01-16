#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import ctypes
import ctypes.util
import gc
import os
import socket
import struct
import tempfile
import threading
from pathlib import Path

import pytest

from tensorcast._c_ext import restore_tensors_from_cpu_fd_with_lease

pytestmark = pytest.mark.requires_cuda_or_fake


def _recv_exact(sock: socket.socket, nbytes: int) -> bytes:
    out = bytearray()
    while len(out) < nbytes:
        chunk = sock.recv(nbytes - len(out))
        if not chunk:
            break
        out.extend(chunk)
    return bytes(out)


def _memfd_create(name: str) -> int:
    flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
    if hasattr(os, "memfd_create"):
        return os.memfd_create(name, flags)

    libc_name = ctypes.util.find_library("c")
    if libc_name is None:
        raise OSError("libc not found for memfd_create fallback")
    libc = ctypes.CDLL(libc_name, use_errno=True)
    if hasattr(libc, "memfd_create"):
        libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        libc.memfd_create.restype = ctypes.c_int
        fd = libc.memfd_create(name.encode(), flags)
    else:
        sys_nr = getattr(os, "SYS_memfd_create", None)
        if sys_nr is None:
            raise OSError("SYS_memfd_create unavailable for memfd_create fallback")
        libc.syscall.argtypes = [ctypes.c_long, ctypes.c_char_p, ctypes.c_uint]
        libc.syscall.restype = ctypes.c_long
        fd = libc.syscall(sys_nr, name.encode(), flags)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, "memfd_create failed")
    return int(fd)


def test_cpu_memfd_lease_release_is_raii() -> None:
    token = b"lease_token_test"
    sock_dir = Path(tempfile.mkdtemp(prefix="tc_local_handle_"))
    socket_path = str(sock_dir / "local_handle.sock")

    received: dict[str, object] = {"token": None}
    ready = threading.Event()

    def _serve_once() -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            srv.bind(socket_path)
            srv.listen(1)
            ready.set()
            conn, _addr = srv.accept()
            with conn:
                header = _recv_exact(conn, 1 + 4)
                if len(header) != 5:
                    return
                opcode = int(header[0])
                (tok_len,) = struct.unpack("=I", header[1:])
                tok = _recv_exact(conn, tok_len)
                received["token"] = (opcode, tok)
                try:
                    conn.sendall(bytes([0]))
                except BrokenPipeError:
                    return
        finally:
            srv.close()

    thread = threading.Thread(target=_serve_once, daemon=True)
    thread.start()
    assert ready.wait(timeout=3.0)

    size_bytes = 4096
    fd = _memfd_create("tc_cpu_memfd_lease_raii")
    try:
        os.ftruncate(fd, size_bytes)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, b"abcdefghijklmnopqrstuvwxyz" * (size_bytes // 26))
        os.lseek(fd, 0, os.SEEK_SET)

        meta_state_dict = {"t": ([size_bytes], [1], "torch.uint8", 0)}
        tensor_offsets = {"t": 0}
        tensors = restore_tensors_from_cpu_fd_with_lease(
            meta_state_dict,
            fd=fd,
            size_bytes=size_bytes,
            offset_bytes=0,
            tensor_device_offsets=tensor_offsets,
            lease_token=token,
            local_handle_socket_path=socket_path,
        )
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

    t = tensors["t"]
    assert bytes(t[:26].tolist()) == b"abcdefghijklmnopqrstuvwxyz"

    del t
    del tensors
    gc.collect()

    thread.join(timeout=3.0)
    assert received["token"] == (2, token)
