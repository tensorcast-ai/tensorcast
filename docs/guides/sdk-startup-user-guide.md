---
title: SDK Startup User Guide
description: Best practices for tensorcast.init, daemon lifecycle, and multi-process integration
sidebar_position: 5
---

# TensorCast Startup and Integration User Guide

This guide explains how users should bootstrap TensorCast in real deployments.
It focuses on:

- `tensorcast.init(...)` behavior and mode selection
- CLI-managed vs SDK-managed daemon lifecycle
- API/SDK integration patterns
- TP or other multi-process workloads sharing one daemon

## 1. Startup Model at a Glance

`tensorcast.init` is the main startup entry (`tensorcast.startup.init`).
It supports three modes:

| Mode | What it does | Daemon owner | Typical usage | Main risk |
|---|---|---|---|---|
| `connect` | Connects to an existing daemon only | External (CLI/operator/other process) | Production services, TP workers | Fails if no reachable daemon |
| `create` | Starts a local daemon and connects to it | Current process | Single-process tools, local dev | Process exit may stop daemon |
| `auto` | Singleflight connect-or-create under same runtime root | Leader process selected at runtime | Concurrent local workers booting together | Config mismatch across processes causes startup failure |

## 2. Recommended Decision Order

| Environment | Recommended pattern | Why |
|---|---|---|
| Production service / long-lived inference | Start daemon via CLI, app uses `init(mode="connect")` | Clean lifecycle boundaries, easier ops |
| Local development or notebook | `init(mode="create")` | Fast self-contained bootstrap |
| Many local processes start at same time | `init(mode="auto")` | Prevents duplicate daemon launches |
| TP/forked workers | Pre-start daemon, each worker `connect` | Most stable and predictable |

## 3. Configuration Resolution

### 3.1 Daemon config path (`create` / `auto`)

| Priority | Source |
|---|---|
| 1 | `daemon_config_path` parameter |
| 2 | `TENSORCAST_DAEMON_CONFIG` |
| 3 | `examples/config/store_daemon_config.yaml` (repo or packaged wheel) |

If none is found, startup fails.

### 3.2 Global Store orchestration

| `global_store_mode` | Behavior | When to use |
|---|---|---|
| `none` | No Global Store orchestration | Local-only or minimal setups |
| `connect` | Connect to an existing Global Store | Production clusters with managed GS |
| `start` | Reuse/start local Global Store first, then daemon | Local all-in-one workflows |

## 4. Integration Patterns

### Pattern A (Recommended): CLI-managed services + SDK connect

Use CLI to manage lifecycle, and keep app processes stateless regarding daemon ownership.

```bash
# 1) Start Global Store (if needed)
uv run tensorcast-cli global start --config=examples/config/global_store_config.yaml

# 2) Start Store Daemon
uv run tensorcast-cli daemon start \
  --config=examples/config/store_daemon_config.yaml \
  --global-store-mode connect \
  --global-store-address 127.0.0.1:50051
```

```python
import tensorcast as tc

tc.init(mode="connect", address="127.0.0.1:50052", show_daemon_logs=False)

artifact = tc.artifact(key="model:latest")
tensors = artifact.tensor_dict(device="cuda:0")

tc.shutdown()
```

### Pattern B: SDK self-managed launch (`create`)

Good for local dev and simple scripts.

```python
import tensorcast as tc

tc.init(
    mode="create",
    daemon_config_path="examples/config/store_daemon_config.yaml",
    global_store_mode="start",
    global_store_config_path="examples/config/global_store_config.yaml",
    show_daemon_logs=False,
)

# register / get / artifact operations...

tc.shutdown()
```

### Pattern C: Concurrent local startup (`auto`)

`auto` is useful when many local processes may start at once and should converge to one daemon.

```python
import tensorcast as tc

tc.init(
    mode="auto",
    daemon_config_path="examples/config/store_daemon_config.yaml",
    global_store_mode="connect",
    global_store_address="127.0.0.1:50051",
    show_daemon_logs=False,
)
```

Best practice for `auto`:

| Rule | Reason |
|---|---|
| Keep init parameters identical across participating processes | `auto` validates a config hash and rejects mismatches |
| Do not pass different explicit `session_id` values per process | Breaks process-group singleflight expectations |
| Prefer `connect` for long-lived production worker pools | Owner process semantics in `auto` are harder to operate |

## 5. API/SDK Usage Patterns

### 5.1 Module-level API (simple and common)

```python
import tensorcast as tc

tc.init(mode="connect", address="127.0.0.1:50052")
tc.register({"w": some_cuda_tensor}, key="model:v1")
art = tc.artifact(key="model:v1")
weights = art.tensor_dict(device="cuda:0")
```

### 5.2 Explicit `Store` object (advanced tuning)

```python
import tensorcast as tc
from tensorcast.api.store.types import RetryPolicy

tc.init(mode="connect", address="127.0.0.1:50052")

store = tc.store(
    opts=tc.StoreOptions(
        fallback="local",
        retry_overrides={"get": RetryPolicy(20.0, 2, 0.1, 2.0, 0.5)},
    )
)

art = store.artifact(key="model:latest")
```

## 6. TP / Multi-Process / Fork Best Practices

For tensor parallel or any multi-process setup, treat daemon as shared infrastructure.

### 6.1 Recommended architecture

| Step | Recommendation |
|---|---|
| 1 | Start daemon once (CLI/system service) |
| 2 | Spawn/fork worker processes |
| 3 | In each worker process, call `tc.init(mode="connect", address=...)` |
| 4 | Build per-rank views (`artifact.view(slices=...)`) and materialize locally |

### 6.2 Do / Don’t

| Do | Don’t |
|---|---|
| Initialize TensorCast inside each worker process | Call `tc.init()` in parent before `fork` |
| Use explicit daemon address in distributed launches | Rely on implicit local discovery across hosts |
| Use `connect` for stable long-running TP services | Use `create` in every rank |
| Keep one daemon endpoint per process | Attempt to switch daemon address after client is initialized |

### 6.3 Can forked workers call `auto` directly?

Yes. You do not need to pre-start daemon if each forked worker calls `tc.init(mode="auto")` after fork.
One worker will become leader and start daemon; others wait and connect.

Required conditions:

| Condition | Why |
|---|---|
| Call `auto` in child process (after fork) | Avoid inheriting parent-initialized runtime/client state |
| Keep startup args identical across workers | `auto` enforces config-hash consistency |
| Share the same runtime root (`TENSORCAST_HOME`) | Singleflight election happens under one runtime root |

Operational caveat:

| Caveat | Impact |
|---|---|
| Leader process is daemon owner | If owner exits early, daemon can be stopped and other workers are affected |

For long-running TP services, prefer a dedicated daemon process and worker `connect`.

### 6.4 Worker template

```python
def worker_main(rank: int, daemon_addr: str) -> None:
    import tensorcast as tc
    import torch

    torch.cuda.set_device(rank)
    tc.init(mode="connect", address=daemon_addr, show_daemon_logs=False)

    artifact = tc.artifact(key="model:latest").view(slices=build_rank_slices(rank))
    _ = artifact.tensor_dict(device=f"cuda:{rank}")

    tc.shutdown()
```

## 7. Troubleshooting

| Symptom | Likely cause | Action |
|---|---|---|
| `No local daemon session found` in `connect` | No running daemon, or no discovered local session | Start daemon via CLI or pass explicit `address` |
| `AUTO_CONFIG_MISMATCH` in `auto` | Different init/config params across processes | Make all `auto` startup args identical |
| `Materialization requires a Global Store connection` | Daemon not connected to Global Store for requested operation | Start/attach Global Store (`global_store_mode="start"` or `connect`) |
| `client already initialized for address ... refusing second client` | Same process tried to bind to another daemon address | Use one daemon endpoint per process; restart process if switching is needed |

## 8. Production Checklist

| Item | Status |
|---|---|
| Daemon lifecycle managed outside app process (CLI/system) | ☐ |
| App uses `init(mode="connect", address=...)` | ☐ |
| Global Store mode selected intentionally (`none`/`connect`/`start`) | ☐ |
| TP workers initialize TensorCast after process start | ☐ |
| All startup configs are deterministic and version-controlled | ☐ |
