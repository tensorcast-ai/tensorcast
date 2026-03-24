---
slug: 0001-docs-system-design
title: Human–AI Collaborative Documentation System (Design)
---

# Summary

Define the documentation model centered on `docs/designs/` as the durable architectural record. `docs/plans/` remain optional execution notes for work that benefits from phased tracking, but plans are not required to persist after completion if the final outcome is folded back into the design. The system standardizes directory semantics, required content, and metadata so design intent stays discoverable without forcing long-lived implementation checklists to accumulate in the repository.

This document specifies the directory semantics, required sections, and metadata schema for designs and plans.

```mermaid
flowchart LR
    A[Design
    why & what
    goals, constraints,
    APIs, schema] --> B[Plan
    how
    phases, tasks,
    tests, rollout]
    B --> C[Implement
    code changes]
    C --> D[Backstop
    tests, CI, checks]
    D --> E[Review
    human + agent]
    E --> F[Update & Refine
    docs, AGENTS.md,
    schema.sql]
```

# Goals (Why)

- Durable context: Make architectural intent, invariants, and interfaces discoverable and enforceable.
- Speed with safety: Enable agents to operate top‑down with human review, reducing rework.
- Clear separation of concerns: Designs (why/what) vs. optional Plans (temporary how) vs. Guides (how‑to/tutorials).
- Localized guidance: README.md and AGENTS.md provide precise, directory‑scoped instructions.
- Canonical data source: `schema.sql` is the single source of truth for data structures.
- Continuous correction: Treat each new design as a chance to detect where earlier accepted constraints no longer satisfy the real need, and adjust the documented rules so the system keeps moving in the globally correct direction.
- Guardrails: CI enforces metadata, cross‑linking, and the “doc sync” rule for code changes.

# Information Architecture (What)

## Directories and Semantics

- `docs/designs/` — Product requirements, goals, constraints, normative schemas and APIs, trade‑offs, risks, and acceptance criteria. Authoritative “why and what.”
- `docs/plans/` — Optional phased implementation notes with milestones, testing, rollout and backout. Use them when work benefits from active execution tracking; fold the final result back into the design when the work is complete.
- `docs/guides/` — Tutorials and how‑tos for APIs/tools, often drafted by agents after reading source/docs.
- `schema.sql` (repo root) — Canonical, versioned relational schema for the project’s persistent data; designs that change data must reference and modify this file.
- `README.md` and `AGENTS.md` — Placed at meaningful boundaries (module roots, service dirs, tools). These files give localized guidance to humans and agents. See “Localized Guidance” below.

## File Naming

- `docs/designs/` and `docs/plans/` filenames begin with a zero-padded sequence number: `0001-<slug>.md`, `0002-<slug>.md`, etc.
- The numeric prefix increments monotonically; do not reuse or skip numbers except when explicitly superseding a document (record the relationship in frontmatter).
- When a design has one or more companion plans, those plans should reuse the design sequence number or a clear derivative of it so the relationship is obvious (for example `docs/designs/0015-foo.md` and `docs/plans/0015-foo.md`, or `docs/plans/0040-01-foo.md`).
- The `<slug>` portion of the filename is a descriptive string reused in the frontmatter `slug` field without the numeric prefix.

## Document Types and Required Content

Design (docs/designs/<slug>.md)
- Problem statement, scope, goals/non‑goals
- Whole-system analysis of the problem and proposal, not only local-module impact
- Interfaces and schemas (include diffs against `schema.sql` when applicable)
- Invariants and error model
- Alternatives and rationale
- Constraint review: identify the prior designs, rules, or assumptions this design inherits; if an earlier accepted design no longer satisfies the need, say so explicitly, explain why strict adherence would produce poor system-level outcomes, and record what is kept, narrowed, revised, or superseded
- Any new interface/API proposed in a design (or its paired plan) must document how it adheres to the repository’s language-specific style guides (C++ rules in `AGENTS.md` and `core/` READMEs, Python rules in `tensorcast/` docs). For C++ APIs, include a short `Naming Compliance` call-out (table or bullet list) that explicitly proves each function/method is `snake_case`, each class/struct is `PascalCase`, and constants/macros are `ALL_CAPS`. Designs that skip this check or knowingly violate the conventions must be rejected. Do not introduce interfaces whose naming, error handling, or packaging violates those standards.
- Risks, success criteria, and compatibility
- Documentation impact: when the correct solution changes design principles, authoring rules, or governing constraints, update the relevant documentation-system guidance in the same change instead of leaving that logic implicit
- Cross‑links: owning code, related plans, and guides
 - Visualization: Prefer Mermaid diagrams for structured, graphical, flow, and hierarchical information (e.g., flowcharts, sequence diagrams, class diagrams, state diagrams, ER/graph diagrams).
  - Mermaid guidance:
    - You can use Mermaid to algorithmically express and emulate more complex cognitive processes (e.g., decision pipelines, concurrent branches, feedback loops).
    - For line breaks inside node labels, use `<br>` instead of `</n>`.
    - Avoid using `:` inside plain node text; colons can be interpreted as Mermaid syntax tokens in certain contexts.
    - Parentheses can also trigger syntax. When you need parentheses in text, wrap the entire label in quotes. Example: `A["Reclaimer<br>(idempotent cleanup)"]`.
    - Example (correct):
      ```mermaid
      flowchart LR
        A["Reclaimer<br>(idempotent cleanup)"] --> B["Done"]
      ```
    - Example (incorrect; will error due to `</n>` and unquoted parentheses):
      ```mermaid
      flowchart LR
        A[Reclaimer</n>(idempotent cleanup)] --> B[Done]
      ```

Plan (docs/plans/<slug>.md, optional)
- Grounding: summarize current state and link concrete code references; declare baseline
- Phases/milestones and tasks
- Acceptance checks, test plan, rollout/backout
- Risk tracking and owner checklist
 - Phases & Milestones MUST use Markdown checkboxes (`- [ ]`) for progress tracking

Guides (docs/guides/<topic>.md)
- Goal‑oriented, concise steps with code references
- Drafted/updated after code or design changes land

# Metadata Schema (Frontmatter)

Each document starts with YAML frontmatter. Minimal required keys:

```yaml
title: <Human readable title>
links:
  design: ../designs/<slug>.md        # for plans. (designs don't need a plan link)
```

Optional keys:
- `areas`: ["core", "daemon", "global_store", "sdk", "infra"]
- `related_code`: ["paths/glob"]

# Localized Guidance (README.md & AGENTS.md)

Purpose
- README.md: What this directory contains, how to build/test/run, quick links, local invariants, and owner contacts.
- AGENTS.md: Directory‑scoped rules and tips for agents, including code style and naming, build/test commands, layout, and precedence. Aligns with the root AGENTS.md spec and inherits its semantics.

Placement and Precedence
- Place at every module/service boundary and any directory where special rules apply.
- Precedence: deeper AGENTS.md overrides parent; direct human/developer/user instructions override AGENTS.md. Scope is limited to the directory subtree.
- Doc Sync Rule: Any code change that modifies behavior must update the nearest README/AGENTS.md, the relevant design, and any in-use companion plan.

Recommended Structure
- README.md: Overview, Setup, Commands, Layout, Key Links, Owner
- AGENTS.md: Scope, Coding Standards (local deltas only), Build/Run/Lint/Test, Do/Don’t, Common Pitfalls, Update Obligations

Required Philosophy Anchors (inherit from root AGENTS.md)
- Local `AGENTS.md` files must include and not contradict these anchors (see `../../AGENTS.md`, “Software Design Philosophy”).
- Complexity Reduction:
  - Strategic Programming; Deep Modules; Minimize Optional Types; Layer Architecture; Information Hiding
- Comment‑First Development:
  - Write the interface comment first; describe the “what” and “why,” not the “how”
- Error Handling Philosophy:
  - Define errors out of existence when possible; use exceptions for exceptional cases; make common errors impossible via API design; avoid partial failures—operations should be atomic

# Schema Governance (`schema.sql`)

- Acts as canonical ground truth for persistent data structures across components.
- Any design that changes data must include a “Schema Changes” section describing rationale and impact, and must propose a synchronized patch to `schema.sql`.
- Ownership: global_store owners maintain `schema.sql` with review from affected areas.
- Compatibility: migrations and backfills are tracked in plans when a plan exists; designs must specify compatibility strategy and versioning.

# Ownership, Reviews, and Status Lifecycle

- Status:
  - draft → proposed → accepted → (deprecated|superseded|retired)
  - Designs capture intent and compatibility commitment.
  - Plans, when present, track in-flight execution and may be removed after completion once the final status, decisions, and verification are captured in the design.
- Decision Records: When a design is accepted, capture the final decision and key rationale; link commits that implement it.

# Guardrails (What must be true)

- Frontmatter validation: required keys present; status is valid; cross‑links resolve.
- Linkage: every plan points to exactly one design; a design may have zero, one, or multiple plans over its lifetime.
- Link hygiene: CI link checker passes.
- Doc Sync: PRs that change code touching an owned area must update the relevant design and localized README/AGENTS.md; add or update a plan when phased execution tracking is useful.
- Constraint evolution is explicit: accepted designs are authoritative but not immutable; when new evidence shows an older constraint is misaligned with system goals, the newer design must document the adjustment or supersession instead of silently working around it.
- Schema discipline: designs referencing data models must link `schema.sql`; plans must include migration/testing steps.
- Local `AGENTS.md` include the repository’s Software Design Philosophy anchors (Complexity Reduction, Comment‑First Development, Error Handling Philosophy).

# Discoverability and Navigation

- Indexes: `docs/README.md` lists designs, plans, and guides by area and status with short descriptions.
- Stable slugs and IDs for deep links; filenames mirror `<slug>.md`.
- Cross-linking: designs link to owning code and related docs; plans, when present, link back to their design and tests.

# How to Write a Design

- Start from the whole system: evaluate user impact, operator feedback, architecture direction, and cross-module effects before optimizing a local component.
- Read the relevant accepted designs first, but do not treat them as fixed law. A new design may exist precisely because an earlier design did not satisfy the real need.
- Review inherited constraints one by one: keep them when they still help the system, and explicitly revise or supersede them when strict adherence would harm correctness, operability, simplicity, or long-term direction.
- If the new reasoning changes how future documents should be written, linked, or reviewed, update the governing documentation in the same change so the improved logic becomes part of the standard process.

# How to Write a Plan

- Use concise, outcome‑oriented titles for phases and milestones.
- Phases & Milestones MUST use Markdown checkboxes (`- [ ]`) so progress is visible in diffs and reviews.
- Link each milestone to owning code/tests where applicable.
- Keep tasks small, actionable, and verifiable.

- Plans must be deeply grounded in the project’s current state (code, data, infra). Do a brief discovery pass first: read owning README/AGENTS.md, key modules, and tests.
- Be targeted: map each phase and milestone to concrete code locations or tests; add code references using the repository’s code‑reference format.
- Capture a short "Current State" summary: constraints, feature flags, tech debt, active migrations, and relevant configurations with links to code/docs.
- Treat the plan as a living document: adjust during execution as you learn. Record meaningful changes, keep acceptance criteria aligned with the design, and update the design when intent materially changes.

# Templates (Authoring Aids)

Design (minimal)
```md
---
slug: <slug>
title: <Title>
links:
---

# Summary

# Goals / Non‑Goals

# Prior Constraints Reviewed

# Architecture & Interfaces

# Schema Changes (if any)

# Trade‑offs & Risks

# Compatibility & Acceptance Criteria

# References
```

Plan (minimal)
```md
---
slug: <slug>
title: <Title>
links:
  design: ../designs/<slug>.md
---

# Objective

# Current State & Grounding
- Key constraints observed in current code/data
- Code references to owning modules/tests/docs

# Phases & Milestones

- [ ] Phase 1: <name>
  - [ ] Milestone 1: <outcome>
  - [ ] Milestone 2: <outcome>
- [ ] Phase 2: <name>
  - [ ] Milestone 1: <outcome>

# Tasks

# Test / Rollout / Backout

# Risks & Tracking
```

# Open Questions

- Should we add an `internal: true|false` flag in frontmatter for possible external publishing later?
- Do we want a repo script to auto‑generate design/plan indexes and badges by status?
- What’s the exact ownership for `schema.sql` across teams (single team, or joint ownership with codeowners by table)?
