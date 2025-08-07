# AGENTS.md

This document provides guidance to **AI coding agents** (e.g., GPT‑4/5 in Cursor) when working with the **StepCast Store** repository.

---

## Purpose
StepCast Store is a production-grade, high-performance distributed model storage system. Automated code generation must meet the same quality bar as human contributions. This guide distills the in-repo engineering rules (see `.cursor/rules/`) into a concise reference for AI agents.

---

## Agent Workflow (Checklist)

### Principles
- Be explicit about the goal and the required end-state. No assumptions.
- Keep the setup concise. Organize context into labeled sections. Use delimiters/XML tags to separate parts clearly.
- Specify the output contract precisely (structure, format, fields, constraints).
- Avoid chain-of-thought requests (don’t ask it to “think step by step” or “explain reasoning”) unless the user explicitly needs a short rationale section.
- Prefer deterministic formats (XML, tables, bullet lists) when useful.
- If critical info is missing, include a single clarifying question section before the end-state.

### Understand the task
- **Clarify**: problem, constraints, acceptance criteria.
- **Locate code**: prefer semantic search; fall back to regex for exact symbols.
- **Skim interfaces first**: read public APIs before implementations.

### Plan before coding
- **Write a brief plan/todo**: break work into atomic, testable edits.
- **Validate design**: use the complexity checklist from `.cursor/rules/common.mdc`.

### Execute with tooling
- **Parallelize lookups**: batch searches/reads to minimize latency.
- **Apply edits incrementally**: use minimal diffs; avoid dumping entire files.
- **Keep interfaces stable** where required; update docs when behavior changes.

### Validate
- **Python**: run lints, types, tests.
  ```bash
  uv run ruff check .
  uv run mypy ./scstore
  uv run pytest
  ```
- **C++**: build and run tests.
  ```bash
  bazel build //...
  bazel test //tests/cpp:all
  ```
- **Protobufs**: regenerate after editing `proto/`.
  ```bash
  bash tools/build_proto_python.sh
  ```

### Finalize
- **Fix linter/type errors** (≤ 3 attempts).
- **Update or add tests** for any bug fix or feature.
- **Produce a succinct summary** of changes and impact.

---

## Repository Conventions
(derived from `.cursor/rules/`)

- **Languages**: C++20, Python 3.10+
- **Build systems**: Bazel (C++), setuptools + uv (Python)
- **Style**: clang-tidy & ruff; `snake_case` for functions/vars, `PascalCase` for classes/types
- **Error handling**: early returns; C++ uses `absl::Status`, Python uses exceptions
- **Optionals**: avoid unless essential—design for presence
- **Comment-first development**: write interface comments before code
- **Layering**: higher layers depend only on lower ones
- **Documentation**: update docs alongside code; follow Docusaurus sidebar rules

---

## Prompt Patterns That Work Well
```text
# Add a unit test for scstore/global_store/registry.py::load_model()
<insert current function snippet here>
<describe test scenario and expected assertions>
```

```text
# Refactor: replace deprecated torch.load with torch.serialization.load
Constraints: keep public API stable, update docs accordingly, all tests must pass.
```

```text
# Bugfix: race in C++ cache eviction
Context: <repro / failing test output>
Goal: ensure thread-safe eviction; add Catch2 test covering concurrent access.
Acceptance: bazel test //tests/cpp:all passes; no data races under TSAN.
```

---

## Common Pitfalls to Avoid
- **Silencing linter errors** instead of fixing root cause
- **Forgetting to regenerate protobufs** after changing `proto/`
- **Creating cyclic dependencies** across architectural layers
- **Ignoring concurrency/thread-safety** in C++ shared resources
- **Changing behavior without tests** capturing the new/expected behavior
- **Editing generated files** instead of sources

---

## Quick Links
- **Architecture details**: `.cursor/rules/architecture.mdc`
- **Build guide**: `.cursor/rules/build.mdc`
- **Coding standards**: `.cursor/rules/common.mdc`, `cpp.mdc`, `python.mdc`
- **Technical design template**: `.cursor/rules/technical_design.mdc`

---

## Project Structure & Module Organization
- `scstore/`: Python package and CLI (`scstore-cli`), daemon utilities, `global_store/`, and C++ extension shims in `csrc/`
- `core/`: C++20 core library (Bazel target `//core:libscstore.so`)
- `tests/python/`, `tests/cpp/`: Python and C++ tests
- `proto/`: Protobuf definitions; Python stubs under `scstore/proto/`
- `tools/`, `examples/`, `web-docs/`: Scripts, examples, and developer docs

---

## Build, Test, and Development Commands

### Environment
```bash
uv venv && uv sync --all-extras --all-groups
```

### Build Python extensions
```bash
BUILD_CORE=1 BUILD_EXTENSION=1 uv run -vvv setup.py build_ext
```

### Lint & type check
```bash
uv run ruff check .
uv run mypy ./scstore
```

### Run tests
```bash
uv run pytest tests/python/**.py
bazel build //core:libscstore.so
bazel test //tests/cpp:all
```

### Protobufs
```bash
bash tools/build_proto_python.sh
```

### CLI
```bash
uv run scstore-cli --help
```

---

## Commit & Pull Request Guidelines
- **Commits**: Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `ci:`); imperative, concise subject
- **PRs**: clear description, linked issues, repro steps, before/after notes, and docs updates
- **Quality gates**: ensure ruff, mypy, pytest, and Bazel checks pass locally

---

## Security & Configuration Tips
- **No secrets in repo**: use environment variables or local config
- **Torch version** must match `pyproject.toml`
- **Regenerate stubs** after changing `proto/`

---

## Complexity Reduction Checklist
Use this at design and review time:
- Can this functionality be achieved with fewer public methods?
- Are error cases handled at the lowest appropriate level?
- Is the interface general enough for future use cases?
- Would a developer understand this without reading implementation?
- Are there any hidden dependencies between modules?
- Can Optional types be eliminated through better design?
