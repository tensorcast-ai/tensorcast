---
title: Model Loading Workflow
description: Complete model loading workflow and component interactions in StepCast Store
sidebar_position: 1
---

# Model Loading Workflow

This diagram shows the complete model loading workflow in StepCast Store, including the interaction between different components.

## System Components

- **InferenceInstance**: Python + CXX EXT
  - Entrypoint: `torch_util.py::load_dict`
  - CLI class: `client.py::DaemonCtl`
  - CXX: `checkpoint_py.cc`

- **LocalStoreDaemon**: Python + CXX EXT
  - Python entrypoint: `store_daemon.py::StoreDaemonServicer`
  - CXX: `store_engine_py.cc`

- **GlobalStore**: Python
  - Entrypoint: `global_store.py::GlobalModelStoreServicer`

- **RemoteStoreDaemon**: Same as LocalStoreDaemon

## Model Loading Sequence

```mermaid
sequenceDiagram
    participant InferenceInstance
    participant LocalStoreDaemon
    participant GlobalStore
    participant RemoteStoreDaemon
    participant JuiceFS

    InferenceInstance->>LocalStoreDaemon: 0. Malloc CUDA Memory
    Note right of InferenceInstance: Local: store_engine.py::allocate_cuda_memory

    InferenceInstance->>LocalStoreDaemon: 1. Request ModelWeight<br/>with CUDA IPC
    Note left of LocalStoreDaemon: RPC: LoadModel

    LocalStoreDaemon->>GlobalStore: 2. Request Model MetaInfo
    Note left of GlobalStore: RPC: GetModelInfo

    LocalStoreDaemon->>GlobalStore: 3. If not in local,<br/>request a remote replica
    Note left of GlobalStore: RPC: RequestModelReplicaTransport

    GlobalStore-->>LocalStoreDaemon: 4. Return A remote replica or NOT
    Note left of GlobalStore: RPC Resp: RequestModelReplicaTransport Resp

    alt Have remote replica
        LocalStoreDaemon-->>RemoteStoreDaemon: 5.1 load_model_from_remote (via P2P comm_engine.read_tensor)
    else NOT have remote replica
        LocalStoreDaemon->>JuiceFS:
        JuiceFS-->>LocalStoreDaemon: 5.2 If not have remote replica,<br/>load_model_from_disk
    end

    InferenceInstance->>LocalStoreDaemon: 6. Finish loading
    Note left of LocalStoreDaemon: RPC: ConfirmModel

    alt If have Global Store
        LocalStoreDaemon->>GlobalStore: 7. Complete P2P transport and register replica
        Note left of GlobalStore: RPC: CompleteModelReplicaTransport
    end

    InferenceInstance->>LocalStoreDaemon: 8. Exit, unregister
    Note left of LocalStoreDaemon: RPC: UnloadModel
```

## Key Steps Explained

1. **Memory Allocation**: InferenceInstance allocates CUDA memory for model storage
2. **Model Request**: Request model weights using CUDA IPC
3. **Metadata Lookup**: LocalStoreDaemon queries GlobalStore for model metadata
4. **Replica Location**: Request remote replica location if model not available locally
5. **Model Loading**: Load model either via P2P from remote daemon or from disk
6. **GPU Transfer**: Copy model to GPU memory
7. **Confirmation**: Confirm model loading completion
8. **Registration**: Register replica with GlobalStore (if using distributed setup)
9. **Cleanup**: Unregister when inference instance exits