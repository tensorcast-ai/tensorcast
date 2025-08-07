# AGENTS.md

This document provides guidance to **AI coding agents** (e.g. OpenAI Codex, GPT-4) when working with the **StepCast Store** repository.

---

## 1. Purpose
StepCast Store is a production-grade, high-performance distributed model storage system. Automated code generation must meet the same quality bar as human contributions. This guide distills the in-repo engineering rules (see `.cursor/rules/`) into a concise reference for AI agents.

## 2. Recommended Workflow
1. **Understand the task**
   • Parse the user request (problem, constraints, acceptance criteria).
   • Locate relevant code via semantic search (`codebase_search`) or regex (`grep_search`).
   • Skim interfaces *before* implementations.

2. **Plan before coding**
   • Draft a high-level plan or todo list (`todo_write`).
   • Break large changes into atomic, independently testable edits.
   • Validate the design against the complexity checklist in `.cursor/rules/common.mdc`.

3. **Execute with tooling**
   • Use parallel tool calls (`multi_tool_use`) whenever possible.
   • Apply edits with `edit_file`; never paste full files in chat.
   • Include minimal diff context using `// ... existing code ...` markers.

4. **Validate**
   • **Python**: `uv run ruff check .`, `uv run mypy ./scstore`, `uv run pytest`.
   • **C++**: `bazel build //...`, `bazel test //tests/cpp:all`.
   • Add or update tests when fixing bugs or adding features.

5. **Iterate & finalise**
   • Fix linter/type errors (≤ 3 attempts).
   • Mark todos `completed` immediately after passing validation.
   • Produce a succinct human-readable summary of changes.

## 3. Repository-Specific Conventions
(derived from `.cursor/rules/`)

• **Languages**: C++20, Python 3.10+
• **Build systems**: Bazel (C++), setuptools + uv (Python)
• **Style**: clang-tidy & ruff; `snake_case` for functions/vars, `PascalCase` for types.
• **Error handling**: early returns; C++ `absl::Status`, Python exceptions.
• **Optionals**: avoid unless essential—design for presence.
• **Comment-first development**: write interface comments before code.
• **Layering**: higher layers may depend only on lower ones.
• **Documentation**: update docs alongside code; follow Docusaurus sidebar rules.

## 4. Prompt Patterns That Work Well
```text
# Add a unit test for scstore/global_store/registry.py::load_model()
<insert current function snippet here>
<describe test scenario and expected assertions>
```

```text
# Refactor: replace deprecated torch.load with torch.serialization.load
Constraints: keep public API stable, update docs accordingly, all tests must pass.
```

## 5. Common Pitfalls to Avoid
✗ Silencing linter errors instead of fixing root cause.
✗ Forgetting to regenerate protobufs (`bash tools/build_proto_python.sh`).
✗ Creating cyclic dependencies between architectural layers.
✗ Ignoring concurrency/thread-safety in C++ shared resources.

## 6. Reference Links
• Architecture details: `.cursor/rules/architecture.mdc`
• Build guide: `.cursor/rules/build.mdc`
• Coding standards: `.cursor/rules/common.mdc`, `cpp.mdc`, `python.mdc`
• Technical design template: `.cursor/rules/technical_design.mdc`

