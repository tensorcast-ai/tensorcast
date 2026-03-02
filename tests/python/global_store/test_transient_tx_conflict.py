# Copyright (c) 2026, TensorCast Team.

from tensorcast.global_store.repositories.base import is_transient_tx_conflict


def test_transient_conflict_detects_insert_or_ignore_race() -> None:
    error = RuntimeError(
        "Transport missing after insert-or-ignore "
        "request_id=transport:canonical:1772136363092736077:7"
    )
    assert is_transient_tx_conflict(error)


def test_transient_conflict_rejects_unrelated_errors() -> None:
    error = RuntimeError("validation failed: request_id already used")
    assert not is_transient_tx_conflict(error)
