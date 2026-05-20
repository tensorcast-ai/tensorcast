#  Copyright (c) 2026, TensorCast Team.

from types import SimpleNamespace

from tensorcast.serving.integration import (
    RuntimeAttachment,
    RuntimeBindingState,
    RuntimeBindingView,
    RuntimeWorkerView,
)
from tensorcast.serving.state import ModelAttributeRuntimeState


def _attachment(value_id: str) -> RuntimeAttachment:
    binding_value_ref = SimpleNamespace(
        binding_id="binding-1",
        binding_layout_id="layout-1",
        binding_value_id=value_id,
        seal_generation=1,
    )
    view = RuntimeBindingView(
        serving_artifact_ref=f"mi2:test:{value_id}",
        representation_contract_hash=f"repr-{value_id}",
        tensor_schema_hash=f"schema-{value_id}",
        binding_value_ref=binding_value_ref,
        readiness="serving",
    )
    return RuntimeAttachment(
        model=object(),
        state=RuntimeBindingState(runtime_view=view),
        view=RuntimeWorkerView.from_runtime_view(view),
    )


def test_model_attribute_runtime_state_compare_and_attach_fences_generation():
    model = SimpleNamespace()
    state = ModelAttributeRuntimeState("_test_tensorcast")
    first = _attachment("value-1")
    stale_replacement = _attachment("value-stale")
    current = _attachment("value-2")

    state.attach_runtime_attachment(model, first)
    state.attach_runtime_attachment(model, current)

    assert state.compare_and_attach_runtime_attachment(
        model,
        expected_attachment=first,
        replacement_attachment=stale_replacement,
    ) is False
    assert state.get_runtime_attachment(model) is current


def test_model_attribute_runtime_state_failure_clear_is_generation_scoped():
    model = SimpleNamespace()
    state = ModelAttributeRuntimeState("_test_tensorcast")
    first = _attachment("value-1")
    current = _attachment("value-2")

    state.mark_failure(model, RuntimeError("old failure"), attachment=first)
    state.clear_failure(model, attachment=current)

    assert model._test_tensorcast_runtime_binding_failure == "old failure"

    state.clear_failure(model, attachment=first)

    assert model._test_tensorcast_runtime_binding_failure is None
