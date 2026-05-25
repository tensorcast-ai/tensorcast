#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

_EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "runtime_reference_framework"
    / "reference_framework.py"
)


def _load_example_module():
    spec = importlib.util.spec_from_file_location(
        "runtime_reference_framework",
        _EXAMPLE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reference_framework_uses_public_artifact_runtime_surfaces_only():
    module = ast.parse(_EXAMPLE.read_text(encoding="utf-8"), filename=str(_EXAMPLE))
    imported: set[str] = set()
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)

    forbidden = {
        name
        for name in imported
        if name == "vllm"
        or name.startswith("vllm.")
        or name == "tensorcast.serving"
        or name.startswith("tensorcast.serving.")
        or name.startswith("tensorcast.artifact_runtime.recipe.builder")
        or name.startswith("tensorcast.artifact_runtime.admin")
        or name.startswith("tensorcast.artifact_runtime.lifecycle")
    }
    assert forbidden == set()


def test_reference_framework_runs_level1_conformance():
    module = _load_example_module()

    result = module.run_level1_conformance()

    assert result.level == "level1-artifact-runtime"
    assert result.checks["direct_start"]
    assert result.checks["reload"]
    assert result.checks["describe"]
