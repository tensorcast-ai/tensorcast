---
allowed-tools: [Read, Write, Edit, MultiEdit, Bash, Glob, TodoWrite, Task]
description: "Execute RFC proposal steps with synchronized documentation updates"
---

# /execute-rfc - RFC Execution & Synchronization

## Purpose
Execute the workflow proposed in an RFC document located under the `rfcs/` directory, perform the necessary codebase changes, and synchronously update the RFC to reflect execution progress and modifications.

## Usage
```
/execute-rfc <rfc-number> [--dry-run] [--with-tests]
```

## Arguments
- `rfc-number` - The numeric identifier of the RFC (e.g., `2` for `0002-unified-loader-architecture.md`)
- `--dry-run` - Perform an analysis and planning phase only, without modifying code or the RFC
- `--with-tests` - Generate or update accompanying tests while executing the RFC

## Execution
1. **Execution State Detection**
   - Determine the current progress of the target RFC execution (e.g., previously completed steps, merged code changes, existing *Execution Status* section).
   - Use this information to skip already completed tasks and resume from the correct point.
2. **Deep Analysis Phase**
   1. Parse and understand the target RFC. Identify proposed architecture changes, affected modules, and cross-module interactions.
   2. Think thoroughly about how the proposed change integrates with existing layers. Always strive for the optimal, most elegant solution—compatibility with legacy code can be relaxed if it simplifies complexity.
3. **Planning Output**
   - Produce a detailed execution plan (TodoWrite list) outlining all code edits, new files, deletions, and documentation updates required.
4. **Codebase Modification**
   - Apply the plan using `Write`, `Edit`, and `MultiEdit` tools.
   - Generate or adjust tests if `--with-tests` is supplied.
5. **RFC Synchronization**
   - Append an **Execution Status** section to the RFC summarizing completed steps, decisions, and deviations from the original proposal.
   - If the implementation reveals necessary RFC adjustments, modify the relevant sections directly within the RFC file.
6. **Validation & Review**
   - Run linting, type-checking, and tests.
   - Highlight any unresolved items or open questions for reviewer attention.

## Intelligent Persona Activation
- **Architecture**: Validate overall design alignment and cross-module consistency.
- **CoreStore**: Storage engine, loaders, memory management, and data integrity.
- **GlobalStore**: Metadata coordination layer and cluster-wide orchestration.
- **StoreDaemon**: Data-plane node handling local storage, P2P transfers, and CUDA IPC serving.
- **Memory**: Manage CPU/GPU allocations, memory pools, and zero-copy pathways.
- **Networking**: RDMA/TCP transfer optimisation and resilience.

## Auto-Activation Patterns
- Detect impacted modules via semantic search; auto-activate domain personas accordingly.
- Use `TodoWrite` to manage multi-step or cross-module executions.

## Example
```
/execute-rfc 2 --with-tests
/execute-rfc 1 --dry-run
```

## Notes
- Always prefer clarity and reduced complexity over backward compatibility.
- Document rationale for any divergence from the RFC.
- Ensure the RFC remains the single source of truth for its own execution state.