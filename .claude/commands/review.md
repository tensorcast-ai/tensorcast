---
allowed-tools: [Read, Write, Edit, MultiEdit, Bash, Glob, TodoWrite, Task]
description: "Comprehensive code and architecture review with optional RFC context"
---

# /review - Code Review Command

## Purpose
Perform a focused, high-quality review of code, architecture, or documentation changes.
If the `--rfc` flag is provided, the command augments the review by cross-checking the changes against the specified RFC’s goals and execution status.

## Usage
```
/review <target> [--rfc <rfc-number>] [--summary-only]
```

## Arguments
- `target` – (Optional) The path, module name, PR number, or commit hash to review. Can be empty if using `--rfc` to review an RFC. (Use gh cli to get the target)
  Examples: `backend/core/`, `123` (PR), `a1b2c3d` (commit).
- `--rfc` – (Optional) Numeric identifier of the RFC (e.g. `0001`).
  When supplied, the reviewer must:
  1. Parse the RFC located at `rfcs/<rfc-number>*`.
  2. Correlate the target changes with the RFC’s design and acceptance criteria.
  3. Highlight any deviations or alignment issues.
  4. Append the finalized review report to the corresponding RFC document under a *Review History* section. (The Reports to append should focus solely on shortcomings and areas for improvement; omit strengths or positive praise.)
- `--summary-only` – Output a condensed, executive-level summary (no detailed inline comments).

## Review Process
1. **Context Gathering**
   1. Load the target diff / codebase segment.
   2. If `--rfc` is present, also load the RFC and its *Execution Status* section.
2. **Structured Review**
   Apply the checklist from `code_review.mdc`:
   - Correctness
   - Code Quality
   - Performance
   - Readability
   - Maintainability
   - Best Practices
3. **RFC Alignment (Conditional)**
   - Validate that the implementation matches the RFC’s architecture, constraints, and rationale.
   - Flag gaps, over-scope, or under-delivery.
4. **Output**
   - Use the Review Format defined in `code_review.mdc`.
     For each issue provide: **Identify • Explain • Suggest • Priority**
   - If `--summary-only` is set, output a high-level list of critical findings and overall verdict.

## Intelligent Persona Activation
- **Architecture**: For system-wide impacts.
- **CoreStore / GlobalStore / StoreDaemon / Memory / Networking**: Auto-activate based on touched modules.
- **Security**: When changes involve authentication, authorization, or sensitive data.

## Example
```bash
/review 45                               # Review PR #45
/review --rfc 1                          # Review RFC 0001
/review backend/memory/ --summary-only   # Brief review of module
/review 112 --rfc 1                      # Review PR #112 in context of RFC 0001
```

## Principles
- Prioritize clarity and actionable feedback over volume.
- If the review uncovers design regressions, propose concrete improvement steps.
- When RFC alignment fails, clearly state whether to amend the code or the RFC.
- Focus the review on code that has **already been implemented**, not on TODO or planned sections.
- All evaluations must be grounded in the actual source code; do **not** rely solely on documentation when making judgments.