#  Copyright (c) 2025-2026, TensorCast Team.

from tensorcast.global_store import launcher


def test_auto_advertise_tracks_explicit_loopback_listen_host(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "_resolve_default_advertise_host",
        lambda: "198.51.100.156",
    )

    host, port, source = launcher._resolve_advertise_address(
        listen_host="127.0.0.1",
        bound_port=50051,
        explicit_host=None,
        explicit_port=None,
    )

    assert host == "127.0.0.1"
    assert port == 50051
    assert source == "listen_loopback"


def test_auto_advertise_uses_routable_default_for_unspecified_listen_host(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        launcher,
        "_resolve_default_advertise_host",
        lambda: "198.51.100.156",
    )

    host, port, source = launcher._resolve_advertise_address(
        listen_host="0.0.0.0",
        bound_port=50051,
        explicit_host=None,
        explicit_port=None,
    )

    assert host == "198.51.100.156"
    assert port == 50051
    assert source == "default"
