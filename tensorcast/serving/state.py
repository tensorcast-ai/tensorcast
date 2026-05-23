#  Copyright (c) 2026, TensorCast Team.

"""Reusable model-attribute state storage for serving runtime integrations."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tensorcast.serving.runtime_attachment import (
        RuntimeAttachment,
        RuntimeBindingState,
        RuntimeBindingView,
    )


def _text_name_set(names: Any) -> set[str]:
    if names is None:
        return set()
    if isinstance(names, str):
        return {names}
    if not isinstance(names, Iterable):
        return {str(names)}
    return {str(name) for name in names}


class OneShotRuntimeHook:
    """Thread-safe hook wrapper used by after-ready framework callbacks."""

    def __init__(self, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._lock = threading.Lock()
        self._started = False

    def __call__(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
        self._callback()


@dataclass(frozen=True)
class ModelAttributeNames:
    runtime_attachment: str
    representation_contract_hash: str
    tensor_schema_hash: str
    runtime_binding_failure: str
    runtime_binding_failure_generation: str
    after_ready_hooks: str
    runtime_binding_exclude_names: str
    lock: str

    @classmethod
    def from_prefix(cls, prefix: str) -> "ModelAttributeNames":
        normalized = str(prefix).strip().rstrip("_")
        if not normalized:
            raise ValueError("ModelAttributeNames prefix must be non-empty")
        return cls(
            runtime_attachment=f"{normalized}_runtime_attachment",
            representation_contract_hash=(f"{normalized}_representation_contract_hash"),
            tensor_schema_hash=f"{normalized}_tensor_schema_hash",
            runtime_binding_failure=f"{normalized}_runtime_binding_failure",
            runtime_binding_failure_generation=(
                f"{normalized}_runtime_binding_failure_generation"
            ),
            after_ready_hooks=f"{normalized}_after_ready_hooks",
            runtime_binding_exclude_names=(
                f"{normalized}_runtime_binding_exclude_names"
            ),
            lock=f"{normalized}_runtime_state_lock",
        )


def attachment_generation_key(
    attachment: RuntimeAttachment | None,
) -> tuple[object, ...] | None:
    if attachment is None:
        return None
    view = getattr(getattr(attachment, "state", None), "runtime_view", None)
    if view is None:
        view = getattr(attachment, "view", None)
    binding_value_ref = getattr(view, "binding_value_ref", None)
    if binding_value_ref is not None:
        return (
            getattr(binding_value_ref, "binding_id", None),
            getattr(binding_value_ref, "binding_layout_id", None),
            getattr(binding_value_ref, "binding_value_id", None),
            getattr(binding_value_ref, "seal_generation", None),
        )
    return (
        getattr(view, "serving_artifact_ref", None),
        getattr(view, "local_serving_ref", None),
        getattr(view, "representation_contract_hash", None),
        getattr(view, "tensor_schema_hash", None),
        id(attachment),
    )


def _runtime_view(
    state: RuntimeBindingState,
) -> RuntimeBindingView:
    view = state.runtime_view
    if view is None:
        raise RuntimeError(
            "TensorCast core RuntimeBindingState is missing runtime_view"
        )
    return view


class ModelAttributeRuntimeState:
    """Owns framework-local state stored on a runtime model object."""

    def __init__(self, names: ModelAttributeNames | str) -> None:
        self.names = (
            names
            if isinstance(names, ModelAttributeNames)
            else ModelAttributeNames.from_prefix(names)
        )

    def _lock(self, model: Any) -> threading.RLock:
        lock = getattr(model, self.names.lock, None)
        if lock is None:
            lock = threading.RLock()
            setattr(model, self.names.lock, lock)
        return lock

    def get_runtime_binding(self, model: Any) -> RuntimeBindingState | None:
        attachment = self.get_runtime_attachment(model)
        if attachment is not None:
            return attachment.state
        return None

    def get_runtime_attachment(self, model: Any) -> RuntimeAttachment | None:
        attachment = getattr(model, self.names.runtime_attachment, None)
        if attachment is None:
            return None
        if not hasattr(attachment, "state") or not hasattr(attachment, "view"):
            raise RuntimeError("Invalid TensorCast runtime attachment")
        return attachment

    def attach_runtime_attachment(
        self,
        model: Any,
        attachment: RuntimeAttachment,
        *,
        clear_failure: bool = True,
    ) -> None:
        with self._lock(model):
            self._set_runtime_attachment_unlocked(model, attachment)
            if clear_failure:
                self.clear_failure(model, attachment=attachment)

    def compare_and_attach_runtime_attachment(
        self,
        model: Any,
        *,
        expected_attachment: RuntimeAttachment,
        replacement_attachment: RuntimeAttachment,
        clear_failure_for_generation: bool = False,
    ) -> bool:
        with self._lock(model):
            current = self.get_runtime_attachment(model)
            if current is not expected_attachment:
                return False
            self._set_runtime_attachment_unlocked(model, replacement_attachment)
            if clear_failure_for_generation:
                self.clear_failure(model, attachment=expected_attachment)
            return True

    def _set_runtime_attachment_unlocked(
        self,
        model: Any,
        attachment: RuntimeAttachment,
    ) -> None:
        view = _runtime_view(attachment.state)
        setattr(model, self.names.runtime_attachment, attachment)
        setattr(
            model,
            self.names.representation_contract_hash,
            view.representation_contract_hash,
        )
        setattr(model, self.names.tensor_schema_hash, view.tensor_schema_hash)

    def mark_failure(
        self,
        model: Any,
        exc: BaseException,
        *,
        attachment: RuntimeAttachment | None = None,
    ) -> None:
        with self._lock(model):
            setattr(model, self.names.runtime_binding_failure, str(exc))
            setattr(
                model,
                self.names.runtime_binding_failure_generation,
                attachment_generation_key(attachment),
            )

    def clear_failure(
        self,
        model: Any,
        *,
        attachment: RuntimeAttachment | None = None,
    ) -> None:
        with self._lock(model):
            if attachment is not None:
                failure_generation = getattr(
                    model,
                    self.names.runtime_binding_failure_generation,
                    None,
                )
                if (
                    failure_generation is not None
                    and failure_generation != attachment_generation_key(attachment)
                ):
                    return
            setattr(model, self.names.runtime_binding_failure, None)
            setattr(model, self.names.runtime_binding_failure_generation, None)

    def add_after_ready_hook(
        self,
        model: Any,
        callback: Callable[[], None],
    ) -> OneShotRuntimeHook:
        hook = OneShotRuntimeHook(callback)
        with self._lock(model):
            existing = tuple(getattr(model, self.names.after_ready_hooks, ()) or ())
            setattr(model, self.names.after_ready_hooks, existing + (hook,))
        return hook

    def set_runtime_binding_exclude_names(self, model: Any, names: Any) -> None:
        merged = _text_name_set(names)
        existing = getattr(model, self.names.runtime_binding_exclude_names, None)
        if callable(existing):
            merged.update(_text_name_set(existing()))
        elif existing is not None:
            merged.update(_text_name_set(existing))
        setattr(model, self.names.runtime_binding_exclude_names, tuple(sorted(merged)))


@dataclass(frozen=True)
class RuntimeAttachmentRecord:
    """Snapshot of key-value runtime attachment state."""

    attachment: RuntimeAttachment | None = None
    failure: str | None = None
    failure_generation: tuple[object, ...] | None = None


class RuntimeAttachmentStore:
    """Thread-safe key-value attachment store for non model-object frameworks."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[object, RuntimeAttachmentRecord] = {}

    def get_record(self, key: object) -> RuntimeAttachmentRecord:
        with self._lock:
            return self._records.get(key, RuntimeAttachmentRecord())

    def get_runtime_attachment(self, key: object) -> RuntimeAttachment | None:
        return self.get_record(key).attachment

    def attach_runtime_attachment(
        self,
        key: object,
        attachment: RuntimeAttachment,
        *,
        clear_failure: bool = True,
    ) -> None:
        with self._lock:
            record = self._records.get(key, RuntimeAttachmentRecord())
            failure = record.failure
            failure_generation = record.failure_generation
            if clear_failure:
                failure = None
                failure_generation = None
            self._records[key] = RuntimeAttachmentRecord(
                attachment=attachment,
                failure=failure,
                failure_generation=failure_generation,
            )

    def compare_and_attach_runtime_attachment(
        self,
        key: object,
        *,
        expected_attachment: RuntimeAttachment,
        replacement_attachment: RuntimeAttachment,
        clear_failure_for_generation: bool = False,
    ) -> bool:
        with self._lock:
            record = self._records.get(key, RuntimeAttachmentRecord())
            if record.attachment is not expected_attachment:
                return False
            failure = record.failure
            failure_generation = record.failure_generation
            if clear_failure_for_generation and failure_generation == (
                attachment_generation_key(expected_attachment)
            ):
                failure = None
                failure_generation = None
            self._records[key] = RuntimeAttachmentRecord(
                attachment=replacement_attachment,
                failure=failure,
                failure_generation=failure_generation,
            )
            return True

    def mark_failure(
        self,
        key: object,
        exc: BaseException,
        *,
        attachment: RuntimeAttachment | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(key, RuntimeAttachmentRecord())
            self._records[key] = RuntimeAttachmentRecord(
                attachment=record.attachment,
                failure=str(exc),
                failure_generation=attachment_generation_key(attachment),
            )

    def clear_failure(
        self,
        key: object,
        *,
        attachment: RuntimeAttachment | None = None,
    ) -> None:
        with self._lock:
            record = self._records.get(key, RuntimeAttachmentRecord())
            if (
                attachment is not None
                and record.failure_generation is not None
                and record.failure_generation != attachment_generation_key(attachment)
            ):
                return
            self._records[key] = RuntimeAttachmentRecord(
                attachment=record.attachment,
                failure=None,
                failure_generation=None,
            )

    def get_failure(self, key: object) -> str | None:
        return self.get_record(key).failure

    def discard(self, key: object) -> None:
        with self._lock:
            self._records.pop(key, None)


__all__ = [
    "ModelAttributeNames",
    "ModelAttributeRuntimeState",
    "OneShotRuntimeHook",
    "RuntimeAttachmentRecord",
    "RuntimeAttachmentStore",
    "attachment_generation_key",
]
