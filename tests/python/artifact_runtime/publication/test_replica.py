#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

import tensorcast.artifact_runtime.lifecycle as integration_mod
from tensorcast.artifact_runtime.attachment import (
    RuntimeAttachment,
    RuntimeBindingState,
    RuntimeBindingView,
)
from tensorcast.artifact_runtime.errors import ReplicaPublicationError
from tensorcast.artifact_runtime.intent import RuntimeRequestContext
from tensorcast.artifact_runtime.locator import ArtifactLocator
from tensorcast.artifact_runtime.policy import RuntimePolicy
from tensorcast.artifact_runtime.publication.actions import (
    project_runtime_replica_publication_state,
    publish_runtime_replica,
    retire_runtime_replica,
    runtime_replica_publication_settings,
)
from tensorcast.artifact_runtime.reload import reload_runtime_attachment
from tensorcast.artifact_runtime.view import (
    PublishedReplicaProjection,
    RuntimeWorkerView,
)


def _profile_records(tmp_path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for path in tmp_path.glob("tensorcast_pid*.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


class _Operation:
    def __init__(self, result: object) -> None:
        self.operation_id = "publish-op-1"
        self.result = result
        self.timeout_s = None

    def wait(self, *, timeout_s: float | None = None) -> object:
        self.timeout_s = timeout_s
        return self.result


class _PublicationBinding:
    artifact_id = "mi2:test:serving"
    binding_layout_id = "layout-1"
    byte_space = SimpleNamespace(kind="cuda", id="0")
    device_uuid = "GPU-0"

    def __init__(self, *, seal_generation: int = 1) -> None:
        self.current_value = SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=seal_generation,
            serving_artifact_id="mi2:test:serving",
        )
        self.published_lease_id: str | None = None
        self.published_replica_id: str | None = None
        self.retired_with: float | None = None
        self.publish_calls = 0
        self.close_calls = 0

    def publish_replica_operation(self) -> _Operation:
        self.publish_calls += 1
        self.published_lease_id = "lease-1"
        self.published_replica_id = "replica-1"
        return _Operation(self.current_value)

    def retire(self, *, drain_timeout_s: float | None = None) -> None:
        self.retired_with = drain_timeout_s
        self.published_lease_id = None
        self.published_replica_id = None

    def close(self) -> None:
        self.close_calls += 1


class _MissingPublicationCapabilityBinding:
    artifact_id = "mi2:test:serving"
    binding_layout_id = "layout-1"

    def __init__(self) -> None:
        self.current_value = SimpleNamespace(
            binding_id="binding-1",
            binding_layout_id="layout-1",
            binding_value_id="value-1",
            seal_generation=1,
            serving_artifact_id="mi2:test:serving",
        )


class _FailingPublicationBinding(_PublicationBinding):
    def publish_replica_operation(self) -> _Operation:
        self.publish_calls += 1
        raise RuntimeError("publish failed")


def _settings(
    config: dict[str, object] | None = None,
):
    return runtime_replica_publication_settings(config)


def _publish(
    attachment: RuntimeAttachment,
    config: dict[str, object] | None = None,
    *,
    profile_sink: object | None = None,
) -> RuntimeAttachment:
    settings = _settings(config or {"replica_publication": {"mode": "required"}})
    return publish_runtime_replica(
        current_attachment=attachment,
        policy=settings.policy,
        ensure_runtime_initialized=settings.ensure_runtime_initialized,
        profile_sink=cast(Any, profile_sink),
    )


def _retire(
    attachment: RuntimeAttachment,
    config: dict[str, object] | None = None,
    *,
    reason: str = "retire",
    drain_timeout_s: float | None = None,
    profile_sink: object | None = None,
) -> RuntimeAttachment:
    settings = _settings(config or {"replica_publication": {"mode": "required"}})
    return retire_runtime_replica(
        current_attachment=attachment,
        reason=reason,
        drain_timeout_s=drain_timeout_s,
        default_drain_timeout_s=settings.drain_timeout_s,
        ensure_runtime_initialized=settings.ensure_runtime_initialized,
        profile_sink=cast(Any, profile_sink),
    )


def _reload(attachment: RuntimeAttachment) -> RuntimeAttachment:
    return reload_runtime_attachment(
        current_attachment=attachment,
        artifact_locator=ArtifactLocator.artifact_ref("mi2:next"),
        policy=RuntimePolicy(),
        runtime_host=cast(Any, object()),
        runtime_context=RuntimeRequestContext(),
        ensure_runtime_initialized=lambda: pytest.fail(
            "active publication rejection must precede runtime init"
        ),
    )


def _attachment(
    binding: object | None = None,
    *,
    serving_artifact_ref: str | None = "mi2:test:serving",
    seal_generation: int = 1,
) -> RuntimeAttachment:
    binding_value_ref = SimpleNamespace(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id="value-1",
        seal_generation=seal_generation,
    )
    runtime_view = RuntimeBindingView(
        serving_artifact_ref=serving_artifact_ref,
        representation_contract_hash="repr-hash",
        tensor_schema_hash="schema-hash",
        binding_value_ref=binding_value_ref,
        readiness="runtime_ready",
    )
    state = RuntimeBindingState(
        binding=binding,
        artifact_ref=serving_artifact_ref,
        runtime_view=runtime_view,
    )
    return RuntimeAttachment(
        model=object(),
        state=state,
        view=RuntimeWorkerView.from_runtime_view(runtime_view),
    )


def _with_published_replica(
    attachment: RuntimeAttachment,
    *,
    state: str = "published",
    lease_id: str | None = "lease-1",
    replica_id: str | None = "replica-1",
) -> RuntimeAttachment:
    weight_version = attachment.view.endpoint.weight_version
    projection = PublishedReplicaProjection(
        state=state,
        operation_id="publish-op-1",
        lease_id=lease_id,
        replica_id=replica_id,
        artifact_ref=weight_version.serving_artifact_ref,
        binding_layout_id=weight_version.binding_layout_id,
        binding_value_ref=weight_version.binding_value_ref,
    )
    endpoint = replace(
        attachment.view.endpoint,
        weight_version=replace(
            weight_version,
            published_replica=projection,
        ),
    )
    return replace(attachment, view=replace(attachment.view, endpoint=endpoint))


def test_runtime_config_parses_replica_publication_policy() -> None:
    settings = _settings(
        {
            "replica_publication": {
                "mode": "REQUIRED",
                "trigger": "after_vllm_ready",
                "timeout_s": 5,
                "drain_timeout_s": 7,
            },
        }
    )
    policy = settings.policy

    assert policy.mode == "required"
    assert policy.trigger == "after_vllm_ready"
    assert policy.timeout_s == 5
    assert policy.drain_timeout_s == 7


@pytest.mark.parametrize(
    "payload",
    [
        {
            "ttl_ms": 1000,
        },
        {
            "async_publish": False,
        },
        {
            "mode": "always",
        },
    ],
)
def test_runtime_config_rejects_invalid_replica_publication_policy(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        _settings({"replica_publication": payload})


def test_publish_current_replica_rejects_local_ready_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    attachment = _attachment(
        _PublicationBinding(),
        serving_artifact_ref=None,
    )

    with pytest.raises(ReplicaPublicationError, match="artifact-backed"):
        _publish(
            attachment,
            {
                "replica_publication": {
                    "mode": "required",
                    "drain_timeout_s": 3,
                },
            },
        )


def test_publish_current_replica_rejects_missing_publication_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    with pytest.raises(ReplicaPublicationError, match="publish_replica"):
        _publish(
            _attachment(_MissingPublicationCapabilityBinding()),
            {
                "replica_publication": {
                    "mode": "required",
                    "drain_timeout_s": 3,
                },
            },
        )


def test_publish_current_replica_rejects_artifact_scope_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.artifact_id = "mi2:other:serving"

    with pytest.raises(ReplicaPublicationError, match="does not match"):
        _publish(_attachment(binding), {"replica_publication": {"mode": "required"}})


def test_publish_current_replica_returns_published_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("TENSORCAST_PROFILE_DIR", str(tmp_path))
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    attachment = _attachment(binding)

    published = _publish(attachment, {"replica_publication": {"mode": "required"}})

    projection = published.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "published"
    assert projection.operation_id == "publish-op-1"
    assert projection.replica_id == "replica-1"
    assert projection.lease_id == "lease-1"
    assert projection.artifact_ref == "mi2:test:serving"
    assert projection.binding_value_ref is not None
    assert projection.binding_value_ref.seal_generation == 1
    report = cast(
        dict[str, Any],
        published.view.diagnostics["artifact_publication_report"],
    )
    assert report["target_kind"] == "publication"
    assert report["artifact_id"] == "mi2:test:serving"
    assert report["operation_backend"] == "runtime_publication"
    assert report["target_layout_digest"] == "layout-1"
    assert report["envelope"]["projection_kind"] == "published_replica"
    assert report["envelope"]["release_policy"] == (
        "retire_published_replica",
        "release_publication_lease",
    )
    assert report["publication"]["state"] == "published"
    assert report["publication"]["replica_id"] == "replica-1"
    assert report["publication"]["lease_id"] == "lease-1"
    assert report["publishability"]["publishable"] is True
    assert report["publishability"]["publish_requested"] is True
    assert report["publishability"]["published"] is True
    assert attachment.view.endpoint.weight_version.published_replica is None
    assert published.state.release_contract is not None
    assert published.state.release_contract.release_policy == (
        "retire_published_replica",
        "release_publication_lease",
    )
    assert published.state.publication_handle is not None
    assert published.state.publication_handle.release_contract is (
        published.state.release_contract
    )
    assert published.state.publication_handle.report.target_kind == "publication"
    assert published.state.publication_handle.report.publication is not None
    assert published.state.publication_handle.report.publication.state == "published"
    assert published.state.publication_handle.binding() is binding
    events = [
        record
        for record in _profile_records(tmp_path)
        if record.get("stage") == "artifact.realize"
    ]
    assert len(events) == 1
    assert events[0]["target_kind"] == "publication"
    assert events[0]["operation_backend"] == "runtime_publication"
    assert events[0]["publishable"] is True
    assert events[0]["published"] is True
    assert events[0]["envelope_projection_kind"] == "published_replica"
    published.state.close()
    published.state.close()
    assert binding.retired_with is None
    assert binding.close_calls == 1
    assert published.state.release_contract.released is True


def test_project_current_replica_publication_state_returns_typed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    attachment = _attachment(binding)

    publishing = project_runtime_replica_publication_state(
        current_attachment=attachment,
        state="publishing",
        reason="after_vllm_ready",
    )
    published = _publish(publishing, {"replica_publication": {"mode": "required"}})

    pending = publishing.view.endpoint.weight_version.published_replica
    assert pending is not None
    assert pending.state == "publishing"
    assert pending.reason == "after_vllm_ready"
    assert pending.artifact_ref == "mi2:test:serving"
    assert pending.binding_value_ref is not None
    assert pending.binding_value_ref.seal_generation == 1
    pending_report = cast(
        dict[str, Any],
        publishing.view.diagnostics["artifact_publication_report"],
    )
    assert pending_report["publication"]["state"] == "publishing"
    assert pending_report["publication"]["reason"] == "after_vllm_ready"
    assert publishing.state.publication_handle is not None
    assert publishing.state.publication_handle.report.target_kind == "publication"
    assert publishing.state.publication_handle.report.publication is not None
    assert publishing.state.publication_handle.report.publication.state == "publishing"
    assert publishing.state.publication_handle.release_contract is (
        publishing.state.release_contract
    )
    projection = published.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "published"


def test_publish_current_replica_error_carries_failed_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    attachment = _attachment(_FailingPublicationBinding())

    with pytest.raises(ReplicaPublicationError) as raised:
        _publish(attachment, {"replica_publication": {"mode": "required"}})

    failed = raised.value.attachment
    assert isinstance(failed, RuntimeAttachment)
    projection = failed.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "failed"
    assert projection.reason == "publish_error"


def test_publish_and_retire_emit_profile_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    events: list[dict[str, Any]] = []
    binding = _PublicationBinding()
    config = {
        "replica_publication": {
            "mode": "required",
        }
    }

    published = _publish(_attachment(binding), config, profile_sink=events.append)
    retired = _retire(
        published,
        config,
        reason="shutdown",
        profile_sink=events.append,
    )

    assert retired.view.endpoint.weight_version.published_replica is not None
    assert [event["event"] for event in events] == [
        "runtime_publication.publish.done",
        "runtime_publication.retire.done",
    ]
    assert events[0]["published_replica_state"] == "published"
    assert events[0]["serving_artifact_ref"] == "mi2:test:serving"
    assert events[0]["duration_s"] >= 0
    assert events[1]["published_replica_state"] == "retired"
    assert events[1]["reason"] == "shutdown"


def test_publish_current_replica_is_idempotent_for_matching_active_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    attachment = _attachment(binding)

    published = _publish(attachment, {"replica_publication": {"mode": "required"}})
    replayed = _publish(published, {"replica_publication": {"mode": "required"}})

    assert replayed is published
    assert binding.publish_calls == 1


def test_publish_current_replica_rejects_mismatched_active_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    published = _publish(
        _attachment(binding),
        {"replica_publication": {"mode": "required"}},
    )
    binding.published_lease_id = "lease-2"

    with pytest.raises(ReplicaPublicationError, match="does not match"):
        _publish(published, {"replica_publication": {"mode": "required"}})


def test_publish_current_replica_rejects_stale_publish_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding(seal_generation=2)
    attachment = _attachment(binding)

    with pytest.raises(ReplicaPublicationError, match="stale") as raised:
        _publish(attachment, {"replica_publication": {"mode": "required"}})

    stale = raised.value.attachment
    assert isinstance(stale, RuntimeAttachment)
    projection = stale.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "stale"
    assert projection.reason == "stale_publish_result"
    assert binding.published_lease_id is None
    assert binding.published_replica_id is None


def test_publish_current_replica_rejects_binding_value_scope_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.current_value = SimpleNamespace(
        binding_id="binding-1",
        binding_layout_id="layout-2",
        binding_value_id="value-1",
        seal_generation=1,
        serving_artifact_id="mi2:test:serving",
    )

    with pytest.raises(ReplicaPublicationError, match="stale"):
        _publish(_attachment(binding), {"replica_publication": {"mode": "required"}})


def test_retire_current_replica_is_idempotent_for_unpublished_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    attachment = _attachment(_PublicationBinding())

    assert (
        _retire(attachment, {"replica_publication": {"mode": "required"}}) is attachment
    )


def test_runtime_binding_state_close_retires_binding_only_publication() -> None:
    binding = _PublicationBinding()
    binding.published_lease_id = "lease-orphan"
    binding.published_replica_id = "replica-orphan"
    state = RuntimeBindingState(
        binding=binding,
        artifact_ref="mi2:test:serving",
    )

    state.close()

    assert binding.retired_with is None
    assert binding.published_lease_id is None
    assert binding.published_replica_id is None


@pytest.mark.parametrize(
    ("lease_id", "replica_id"),
    [
        ("lease-orphan", "replica-orphan"),
        (None, "replica-orphan"),
    ],
)
def test_retire_current_replica_handles_binding_only_publication(
    monkeypatch: pytest.MonkeyPatch,
    lease_id: str | None,
    replica_id: str | None,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.published_lease_id = lease_id
    binding.published_replica_id = replica_id
    config = {
        "replica_publication": {
            "mode": "required",
            "drain_timeout_s": 3,
        },
    }

    retired = _retire(
        _attachment(binding),
        config,
        reason="reload",
    )

    projection = retired.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "retired"
    assert projection.reason == "reload"
    assert projection.lease_id == lease_id
    assert projection.replica_id == replica_id
    assert binding.retired_with == 3


def test_retire_current_replica_refreshes_stale_terminal_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.published_lease_id = "lease-orphan"
    binding.published_replica_id = "replica-orphan"
    config = {
        "replica_publication": {
            "mode": "required",
            "drain_timeout_s": 3,
        },
    }
    attachment = _with_published_replica(
        _attachment(binding),
        state="retired",
        lease_id=None,
        replica_id=None,
    )

    retired = _retire(
        attachment,
        config,
        reason="reload",
    )

    projection = retired.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "retired"
    assert projection.reason == "reload"
    assert projection.lease_id == "lease-orphan"
    assert projection.replica_id == "replica-orphan"
    assert binding.retired_with == 3


def test_retire_current_replica_terminalizes_publishing_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.published_lease_id = "lease-1"
    binding.published_replica_id = "replica-1"
    config = {
        "replica_publication": {
            "mode": "required",
            "drain_timeout_s": 3,
        },
    }
    publishing = _with_published_replica(
        _attachment(binding),
        state="publishing",
    )

    retired = _retire(
        publishing,
        config,
        reason="reload",
    )

    projection = retired.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "retired"
    assert projection.reason == "reload"
    assert binding.retired_with == 3


def test_retire_current_replica_updates_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    config = {
        "replica_publication": {
            "mode": "required",
            "drain_timeout_s": 3,
        },
    }
    published = _publish(_attachment(binding), config)

    retired = _retire(
        published,
        config,
        reason="reload",
    )

    projection = retired.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "retired"
    assert projection.reason == "reload"
    assert binding.retired_with == 3


def test_reload_rejects_active_published_replica(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    published = _publish(
        _attachment(_PublicationBinding()),
        {"replica_publication": {"mode": "required"}},
    )

    with pytest.raises(ReplicaPublicationError, match="retiring"):
        _reload(published)


@pytest.mark.parametrize("projection_state", [None, "retired"])
def test_reload_rejects_binding_lease_without_active_projection(
    monkeypatch: pytest.MonkeyPatch,
    projection_state: str | None,
) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime_config.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.published_lease_id = "lease-orphan"
    binding.published_replica_id = "replica-orphan"
    attachment = _attachment(binding)
    if projection_state is not None:
        attachment = _with_published_replica(
            attachment,
            state=projection_state,
            lease_id="lease-orphan",
            replica_id="replica-orphan",
        )

    with pytest.raises(ReplicaPublicationError, match="retire"):
        _reload(attachment)
