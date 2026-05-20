#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import tensorcast.serving.integration as integration_mod
from tensorcast.serving.integration import (
    IntegrationHost,
    PublishedReplicaProjection,
    ReplicaPublicationError,
    RequestContext,
    RuntimeAttachment,
    RuntimeBindingState,
    RuntimeBindingView,
    RuntimeWorkerView,
    ServingArtifactSelector,
    ServingConfig,
    ServingRuntimeSession,
)


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

    def publish_replica_operation(self) -> _Operation:
        self.publish_calls += 1
        self.published_lease_id = "lease-1"
        self.published_replica_id = "replica-1"
        return _Operation(self.current_value)

    def retire(self, *, drain_timeout_s: float | None = None) -> None:
        self.retired_with = drain_timeout_s
        self.published_lease_id = None
        self.published_replica_id = None


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


def _session(
    config: dict[str, object] | None = None,
    *,
    profile_sink: object | None = None,
) -> ServingRuntimeSession:
    return ServingRuntimeSession.from_config(
        ServingConfig.from_mapping(config),
        host=IntegrationHost(framework=object(), placement=object()),
        profile_sink=profile_sink,
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
        readiness="serving",
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
    return replace(attachment, view=replace(attachment.view,
                                            endpoint=endpoint))


def test_serving_config_parses_replica_publication_policy() -> None:
    config = ServingConfig.from_mapping({
        "replica_publication": {
            "mode": "REQUIRED",
            "trigger": "after_vllm_ready",
            "timeout_s": 5,
            "drain_timeout_s": 7,
        },
    })

    assert config.replica_publication.mode == "required"
    assert config.replica_publication.trigger == "after_vllm_ready"
    assert config.replica_publication.timeout_s == 5
    assert config.replica_publication.drain_timeout_s == 7


@pytest.mark.parametrize("payload", [{
    "ttl_ms": 1000,
}, {
    "async_publish": False,
}, {
    "mode": "always",
}])
def test_serving_config_rejects_invalid_replica_publication_policy(
        payload: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ServingConfig.from_mapping({"replica_publication": payload})


def test_publish_current_replica_rejects_local_ready_attachment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    session = _session({"replica_publication": {"mode": "required"}})
    attachment = _attachment(
        _PublicationBinding(),
        serving_artifact_ref=None,
    )

    with pytest.raises(ReplicaPublicationError, match="artifact-backed"):
        session.publish_current_replica(current_attachment=attachment)


def test_publish_current_replica_rejects_missing_publication_capability(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    session = _session({"replica_publication": {"mode": "required"}})

    with pytest.raises(ReplicaPublicationError, match="publish_replica"):
        session.publish_current_replica(current_attachment=_attachment(
            _MissingPublicationCapabilityBinding()))


def test_publish_current_replica_rejects_artifact_scope_mismatch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.artifact_id = "mi2:other:serving"
    session = _session({"replica_publication": {"mode": "required"}})

    with pytest.raises(ReplicaPublicationError, match="does not match"):
        session.publish_current_replica(current_attachment=_attachment(binding))


def test_publish_current_replica_returns_published_projection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    attachment = _attachment(binding)
    session = _session({"replica_publication": {"mode": "required"}})

    published = session.publish_current_replica(current_attachment=attachment)

    projection = published.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "published"
    assert projection.operation_id == "publish-op-1"
    assert projection.replica_id == "replica-1"
    assert projection.lease_id == "lease-1"
    assert projection.artifact_ref == "mi2:test:serving"
    assert projection.binding_value_ref is not None
    assert projection.binding_value_ref.seal_generation == 1
    assert attachment.view.endpoint.weight_version.published_replica is None


def test_publish_and_retire_emit_profile_metrics(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    events: list[dict[str, object]] = []
    binding = _PublicationBinding()
    session = _session(
        {"replica_publication": {
            "mode": "required",
        }},
        profile_sink=events.append,
    )

    published = session.publish_current_replica(
        current_attachment=_attachment(binding))
    retired = session.retire_current_replica(
        current_attachment=published,
        reason="shutdown",
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
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    attachment = _attachment(binding)
    session = _session({"replica_publication": {"mode": "required"}})

    published = session.publish_current_replica(current_attachment=attachment)
    replayed = session.publish_current_replica(current_attachment=published)

    assert replayed is published
    assert binding.publish_calls == 1


def test_publish_current_replica_rejects_mismatched_active_projection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    session = _session({"replica_publication": {"mode": "required"}})
    published = session.publish_current_replica(
        current_attachment=_attachment(binding))
    binding.published_lease_id = "lease-2"

    with pytest.raises(ReplicaPublicationError, match="does not match"):
        session.publish_current_replica(current_attachment=published)


def test_publish_current_replica_rejects_stale_publish_result(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    session = _session({"replica_publication": {"mode": "required"}})
    attachment = _attachment(_PublicationBinding(seal_generation=2))

    with pytest.raises(ReplicaPublicationError, match="stale"):
        session.publish_current_replica(current_attachment=attachment)


def test_publish_current_replica_rejects_binding_value_scope_mismatch(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
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
    session = _session({"replica_publication": {"mode": "required"}})

    with pytest.raises(ReplicaPublicationError, match="stale"):
        session.publish_current_replica(current_attachment=_attachment(binding))


def test_retire_current_replica_is_idempotent_for_unpublished_attachment(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    session = _session({"replica_publication": {"mode": "required"}})
    attachment = _attachment(_PublicationBinding())

    assert session.retire_current_replica(
        current_attachment=attachment) is attachment


def test_retire_current_replica_terminalizes_publishing_projection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    binding.published_lease_id = "lease-1"
    binding.published_replica_id = "replica-1"
    session = _session({
        "replica_publication": {
            "mode": "required",
            "drain_timeout_s": 3,
        },
    })
    publishing = _with_published_replica(
        _attachment(binding),
        state="publishing",
    )

    retired = session.retire_current_replica(
        current_attachment=publishing,
        reason="reload",
    )

    projection = retired.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "retired"
    assert projection.reason == "reload"
    assert binding.retired_with == 3


def test_retire_current_replica_updates_projection(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    binding = _PublicationBinding()
    session = _session({
        "replica_publication": {
            "mode": "required",
            "drain_timeout_s": 3,
        },
    })
    published = session.publish_current_replica(
        current_attachment=_attachment(binding))

    retired = session.retire_current_replica(
        current_attachment=published,
        reason="reload",
    )

    projection = retired.view.endpoint.weight_version.published_replica
    assert projection is not None
    assert projection.state == "retired"
    assert projection.reason == "reload"
    assert binding.retired_with == 3


def test_reload_rejects_active_published_replica(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        integration_mod.tc_runtime.RuntimeSettings,
        "ensure_initialized",
        lambda self: None,
    )
    session = _session({"replica_publication": {"mode": "required"}})
    published = session.publish_current_replica(
        current_attachment=_attachment(_PublicationBinding()))

    with pytest.raises(ReplicaPublicationError, match="retiring"):
        session.reload(
            current_attachment=published,
            selector=ServingArtifactSelector.artifact_ref("mi2:next"),
            policy=None,
            context=RequestContext(),
        )
