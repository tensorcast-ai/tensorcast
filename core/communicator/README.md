---
id: developer-guides/core/communicator/README
title: Communicator (C++)
sidebar_label: Communicator
---

# Communicator (C++)

The **Communicator** module is STEP AI's low-level, high-performance data-movement engine. It powers peer-to-peer tensor exchange between **Store Daemon**, **Global Store**, and user processes.

This document explains its internal architecture, threading replica, key abstractions, data-flow, and extension points.

> The source lives under `core/communicator/`. All code samples and line numbers below refer to that directory unless stated otherwise.

---

## 1. High-level Architecture

```mermaid
flowchart TB
  subgraph Application["Application Layer (Multi-threaded)"]
    T1[Thread 1]
    T2[Thread 2]
    T3[Thread N]
  end

  subgraph Communicator["Communicator Module"]
    CE[Communicator<br/>Thread-safe API]

    subgraph "Control Path"
      CH[Channel Manager]
      PS[PartitionTensorStore]
    end

    subgraph "Data Path"
      TR[Transport Layer]
      MTCP[MTcpTransport]
      RDMA[RdmaTransport]
      STAGER[GpuNetStager]
    end

    subgraph "Worker Threads"
      RT[Request Thread]
      GT[GC Thread]
      TCPThreads[TCP I/O Threads]
      RDMAThreads[RDMA I/O Threads]
    end
  end

  T1 & T2 & T3 --> CE
  CE --> CH
  CE --> PS
  CH --> TR
  TR --> MTCP & RDMA
  MTCP --> STAGER
  CE --> RT & GT
  TR --> TCPThreads & RDMAThreads
```

### Key Components

* **Communicator** — Central orchestrator providing thread-safe public API
* **Channel** — Manages logical connections (control + data) to remote peers
* **Transport Layer** — Pluggable I/O mechanisms: TCP, Multi-TCP (MTCP), RDMA
* **PartitionTensorStore** — Thread-safe registry for local tensors
* **GpuNetStager** — GPU→CPU staging for TCP transport (when RDMA disabled)

---

## 2. Threading Replica

The Communicator is designed for **concurrent access from multiple application threads**. Here's the complete threading architecture:

```mermaid
flowchart LR
  subgraph "Application Threads"
    App1[App Thread 1]
    App2[App Thread 2]
    AppN[App Thread N]
  end

  subgraph "Communicator"
    API[Thread-safe API<br/>- read_tensor<br/>- register_tensor<br/>- unregister_tensor]
    RQ[Request Queue<br/>Thread-safe]
    CM[Channel Map<br/>Thread-safe]
    PS[Tensor Store<br/>Thread-safe]
  end

  subgraph "Engine Worker Threads"
    RT[request_thread_<br/>Processes outgoing reads]
    GT[gc_thread_<br/>Cleans expired channels]
  end

  subgraph "TCP Subsystem"
    TC[TcpContext]
    LT[listen_thread_<br/>Accepts connections]
    TRT[recv_thread_<br/>Epoll event loop]
  end

  subgraph "MTCP Workers"
    MT1[MTcpTransport 1]
    MT1S[send_thread_]
    MT1R[recv_thread_]
    MTN[MTcpTransport N]
  end

  subgraph "RDMA Subsystem"
    RC[RdmaContext]
    RTH1[RdmaThread 1<br/>Per NIC]
    RS1[send_loop]
    RP1[poll_loop]
    RR1[recv_loop]
  end

  App1 & App2 & AppN --> API
  API --> RQ & CM & PS
  RQ --> RT
  CM --> GT
  TC --> LT & TRT
  MT1 --> MT1S & MT1R
  RTH1 --> RS1 & RP1 & RR1
```

### Thread Responsibilities

#### Application Threads
- Can call any public API method concurrently
- All API methods are thread-safe through internal synchronization

#### Communicator Threads
1. **request_thread_** — Dequeues read requests and initiates remote operations
2. **gc_thread_** — Periodically scans and closes idle channels

#### TCP Infrastructure Threads
1. **TcpContext::listen_thread_** — Accepts incoming TCP connections
2. **TcpContext::recv_thread_** — Epoll-based event processing for all TCP sockets

#### MTCP Transport Threads (per connection)
1. **MTcpTransport::send_thread_** — Distributes write operations across multiple sockets
2. **MTcpTransport::recv_thread_** — Handles incoming data assembly
3. **MTcpTransportTask threads** — Per-socket send/recv workers

#### RDMA Infrastructure Threads (per NIC)
1. **RdmaThread::send_loop** — Posts RDMA READ operations
2. **RdmaThread::poll_loop** — Polls completion queues
3. **RdmaThread::recv_loop** — (Reserved for future RDMA RECV operations)

### Thread Safety Mechanisms

```cpp
// All public APIs use thread-safe containers:
Map<std::string, channel_t> channels_;        // Thread-safe map
Queue<read_request_t> request_queue_;          // Thread-safe queue
PartitionTensorStore store_;                   // Thread-safe store

// Example thread-safe implementation in Map<K,V>:
template <class K, class V>
void Map<K,V>::put(K key, V value) {
    std::unique_lock<std::mutex> lock(mu_);   // Synchronized access
    map_.insert(std::make_pair(key, value));
}
```

---

## 3. Core Components Deep Dive

### 3.1 Communicator

Location: `engine/engine.{h,cc}`

The engine maintains thread-safe state and coordinates all operations:

```mermaid
classDiagram
  class Communicator {
    -bool enable_rdma_
    -Map channels_
    -Queue request_queue_
    -PartitionTensorStore store_
    -thread request_thread_
    -thread gc_thread_
    -GpuNetStager gpu_memory_stager_

    +read_tensor() thread_safe
    +register_tensor() thread_safe
    +unregister_tensor() thread_safe
    +close_connection() thread_safe
    -do_read_request_loop()
    -do_channel_gc_loop()
  }
```

Key design points:
- All public methods are thread-safe
- Internal worker threads handle async operations
- GPU staging automatically enabled when RDMA disabled

### 3.2 Channel

Location: `engine/channel.{h,cc}`

Channels encapsulate the connection state to a remote peer:

```mermaid
stateDiagram-v2
  [*] --> Created: new Channel()
  Created --> Connected: TCP handshake
  Connected --> MTCPSetup: MTCP mode
  Connected --> RDMASetup: RDMA mode
  MTCPSetup --> Active: MTCP connected
  RDMASetup --> Active: QP connected
  Active --> Expired: idle timeout
  Expired --> Closed: GC thread
  Closed --> [*]
```

### 3.3 Transport Layer

The transport abstraction allows pluggable implementations:

| Transport       | Use-case              | Parallelism               | GPU Support |
| --------------- | --------------------- | ------------------------- | ----------- |
| `TcpTransport`  | Control channel       | Single socket             | N/A         |
| `MTcpTransport` | Large CPU/GPU tensors | Multi-socket (default: 8) | Via staging |
| `RdmaTransport` | Zero-copy GPU/CPU     | Per-QP                    | Direct      |

### 3.4 GPU→CPU Staging (TCP Mode)

When RDMA is disabled, GPU tensors are transparently staged through pinned memory:

```mermaid
sequenceDiagram
  participant App
  participant Engine
  participant Stager
  participant CUDA
  participant TCP

  App->>Engine: read_tensor(GPU)
  Engine->>Stager: stage_scoped(tensor, offset, bytes)
  Stager->>CUDA: cudaMemcpyAsync(D2H)
  CUDA-->>Stager: pinned buffer
  Stager-->>Engine: ScopedStagedBuffer
  Engine->>TCP: send(staged->data(), staged->size())
  TCP-->>Remote: data
  Note over Engine: ScopedStagedBuffer auto-releases
```

Key features:
- Double-buffered for pipelining
- Configurable chunk size (default: 64 MiB)
- Automatic buffer pool management
- RAII resource management with ScopedStagedBuffer

### 3.5 TCP Mode Transfer Support Matrix

With the complete implementation, TCP mode now supports all transfer combinations:

| Source | Target | Mechanism                                          | Status |
| ------ | ------ | -------------------------------------------------- | ------ |
| CPU    | CPU    | Direct transfer                                    | ✅      |
| CPU    | GPU    | Network→StreamingPinnedBuffer→GPU                  | ✅      |
| GPU    | CPU    | GPU→GpuNetStager→Network                           | ✅      |
| GPU    | GPU    | GPU→GpuNetStager→Network→StreamingPinnedBuffer→GPU | ✅      |

### 3.6 Complete Data Transfer Architecture for TCP Mode

The following diagram shows all four data transfer paths supported in both TCP and RDMA modes:

```mermaid
flowchart TB
    subgraph "Source Node"
        SrcCPU["CPU Tensor"]
        SrcGPU["GPU Tensor"]
        SrcStager["GpuNetStager<br/>(GPU→CPU staging)"]
        SrcMTCP["MTcpTransport<br/>(send)"]
    end

    subgraph "Network"
        TCP[("TCP Sockets<br/>(Multiple connections)")]
    end

    subgraph "Target Node"
        TgtMTCP["MTcpTransport<br/>(recv)"]
        TgtBuffer["StreamingPinnedBuffer<br/>(Pinned memory pool)"]
        TgtCPU["CPU Tensor"]
        TgtGPU["GPU Tensor"]
    end

    %% Path 1: CPU → CPU (Direct)
    SrcCPU -->|"Direct send"| SrcMTCP
    SrcMTCP --> TCP
    TCP --> TgtMTCP
    TgtMTCP -->|"Direct recv"| TgtCPU

    %% Path 2: GPU → CPU (Staged send, direct recv)
    SrcGPU -->|"cudaMemcpyAsync<br/>(D2H)"| SrcStager
    SrcStager -->|"Pinned buffer"| SrcMTCP
    SrcMTCP --> TCP
    TCP --> TgtMTCP
    TgtMTCP -->|"Direct recv"| TgtCPU

    %% Path 3: CPU → GPU (Direct send, staged recv)
    SrcCPU -->|"Direct send"| SrcMTCP
    SrcMTCP --> TCP
    TCP --> TgtMTCP
    TgtMTCP -->|"Recv to buffer"| TgtBuffer
    TgtBuffer -->|"cudaMemcpyAsync<br/>(H2D)"| TgtGPU

    %% Path 4: GPU → GPU (Staged send, staged recv)
    SrcGPU -->|"cudaMemcpyAsync<br/>(D2H)"| SrcStager
    SrcStager -->|"Pinned buffer"| SrcMTCP
    SrcMTCP --> TCP
    TCP --> TgtMTCP
    TgtMTCP -->|"Recv to buffer"| TgtBuffer
    TgtBuffer -->|"cudaMemcpyAsync<br/>(H2D)"| TgtGPU

    style SrcStager fill:#ffd700
    style TgtBuffer fill:#ffd700
    style TCP fill:#87ceeb
```


### 3.7 GPU Transfer Optimizations

#### Staging Buffer Configuration (typed)

Use `CommunicatorConfig` fields instead of env vars:

- stager.stage_chunk_mb_gpu: size of each staging chunk (MB). Default: 16.
- stager.stage_chunk_mb_cpu: CPU chunk size (MB). Default: 4.
- stager.buffers_per_flow: number of buffers per flow. Default: 4.

#### Performance Characteristics

1. **Memory Usage**: Only requires `chunk_size × (send_buffers + recv_buffers)` of pinned memory
2. **Pipelining**: GPU copies overlap with network transfers
3. **Zero-copy**: Direct transfer for CPU tensors
4. **RAII Safety**: Automatic resource cleanup prevents leaks

### 3.8 RDMA

Location: `engine/rdma.{h,cc}`

RDMA encapsulates the connection state to a remote peer:

```mermaid
stateDiagram-v2
  [*] --> Created: new RdmaContext()
  Created --> Connected: RDMA QP connected
  Connected --> Expired: idle timeout
  Expired --> Closed: GC thread
  Closed --> [*]
```

---

## 4. Data-flow Scenarios

### 4.1 Remote Read Request Flow

```mermaid
sequenceDiagram
  participant App as Application Thread
  participant CE as Communicator
  participant RQ as Request Queue
  participant RT as Request Thread
  participant CH as Channel
  participant TCP as TCP Transport
  participant RDMA as RDMA Transport
  participant Remote as Remote Peer

  App->>CE: read_tensor(key, addr, bytes)
  CE->>RQ: push(ReadRequest)
  CE-->>App: future<ReadResult>

  Note over RT: Async processing
  RT->>RQ: pop()
  RT->>CH: get_or_create_channel()
  RT->>TCP: send(ProtoReadRequest)

  Remote->>Remote: lookup tensor
  Remote->>TCP: send(ProtoReadResponse)

  alt RDMA Path
    TCP->>RDMA: create QP if needed
    RDMA->>RDMA: RDMA READ
    RDMA-->>App: complete future
  else TCP Path
    TCP->>TCP: recv data chunks
    TCP-->>App: complete future
  end
```

### 4.2 Tensor Registration Flow

```mermaid
sequenceDiagram
  participant App as App Thread 1
  participant App2 as App Thread 2
  participant CE as Communicator
  participant Store as PartitionTensorStore
  participant NetDev as NetDev
  participant RegThread as Register Thread

  par Concurrent Registration
    App->>CE: register_tensor("key1", addr1)
    App2->>CE: register_tensor("key2", addr2)
  end

  CE->>Store: store.register_tensor()
  Note over Store: Thread-safe insert

  alt RDMA Enabled
    CE->>NetDev: reg_async(tensor)
    NetDev->>RegThread: queue MR registration
    RegThread->>IBV: ibv_reg_mr()
    IBV-->>RegThread: MR handle
  end

  CE-->>App: absl::OkStatus()
  CE-->>App2: absl::OkStatus()
```

### 4.3 GPU→GPU Transfer Flow (TCP Mode)

```mermaid
sequenceDiagram
  participant App
  participant Engine
  participant ReqThread as Request Thread
  participant SrcStager as Source GpuNetStager
  participant Network as MTCP Network
  participant TgtBuffer as Target StreamingBuffer
  participant GPU as Target GPU

  App->>Engine: read_tensor(GPU→GPU)
  Engine->>ReqThread: queue request
  Engine-->>App: future<ReadResult>

  Note over ReqThread: Source side processing
  ReqThread->>SrcStager: stage_scoped(offset, size)
  SrcStager->>SrcStager: cudaMemcpyAsync(D2H)
  SrcStager-->>ReqThread: ScopedStagedBuffer
  ReqThread->>Network: send staged data

  Note over Network: TCP transfer
  Network-->>Network: Multi-socket transfer

  Note over TgtBuffer: Target side processing
  Network->>TgtBuffer: recv to pinned buffer
  TgtBuffer->>GPU: cudaMemcpyAsync(H2D)
  GPU-->>App: complete future

  Note over SrcStager: Auto-release buffer
```

This flow demonstrates:
- Automatic GPU→CPU staging at source
- Efficient network transfer using multiple TCP sockets
- Automatic CPU→GPU staging at target
- RAII-based resource management throughout

---

## 5. Configuration (typed)

Communicator is configured via `CommunicatorConfig` (C++ type, mirrored in Python). See the migration guide
"CommunicatorConfig Migration" for YAML examples. Key fields:

- enable_rdma: enable RDMA transport.
- channel_expire_sec: channel idle timeout (0=never).
- transport.tcp_conn_count: parallel TCP sockets for MTCP.
- rdma.ack_ttl_ms: RDMA staged-segment ACK TTL.
- stager.stage_chunk_mb_{gpu,cpu}, stager.buffers_per_flow: staging parameters.

---

## 6. Performance Considerations

### Thread Affinity
For optimal performance, consider:
- Pin RDMA polling threads to cores near the NIC
- Keep application threads on NUMA node with target memory
- Isolate GC thread to avoid interference

### Concurrency Tuning
Set `transport.tcp_conn_count` and `channel_expire_sec` in `CommunicatorConfig`.

### Memory Registration
- RDMA memory registration happens asynchronously
- First access may incur registration latency
- Pre-register frequently used tensors

### GPU Transfer Optimization
Tune `stager.stage_chunk_mb_gpu`, `stager.buffers_per_flow`, and `transport.tcp_conn_count`.

**GPU-Specific Tips:**
1. **Chunk Size**: Match staging chunk size to typical tensor dimensions
2. **Buffer Count**: Increase for concurrent transfers (e.g., replica parallel loading)
3. **CUDA Streams**: Each stager uses its own CUDA stream for overlap
4. **Device Selection**: Ensure correct `device_id` to avoid cross-GPU copies

---

### Common Issues

1. **High CPU in poll_loop**
   - Normal for RDMA polling mode
   - Consider interrupt mode for low-traffic scenarios

2. **Blocked application threads**
   - Check request queue size
   - Verify network connectivity
   - Examine GC thread activity

3. **Memory growth**
   - Ensure tensors are unregistered
   - Check channel expiration settings
   - Monitor staging buffer usage

---

## 7. Message Protocol (engine/protocol.h)

The Communicator uses a compact binary – **EngineMessage** – to exchange control information between peers.  Each message starts with a fixed-size `ProtoHeader` followed by a type-specific payload.  The header contains the **op-code** (`uint16_t op`) which maps 1-to-1 to the `ENGINE_OP_*` enum in `engine/protocol.h`.

### 7.1 Op-code Reference

| Op-code                           | Direction       | Purpose                                                   | When Triggered                                                          |
| --------------------------------- | --------------- | --------------------------------------------------------- | ----------------------------------------------------------------------- |
| `ENGINE_OP_READ_REQUEST`          | Client ➜ Server | Request a tensor slice (offset + bytes)                   | `read_tensor()` sends request from **request_thread_**.                 |
| `ENGINE_OP_READ_RESPONSE_EX`      | Server ➜ Client | Multi-segment response and transport indicator (MTCP/RDMA) | Server side of `on_receive_request()` ↠ client `on_receive_response()`. |
| `ENGINE_OP_READ_FAILED`           | Server ➜ Client | Read cannot be served (tensor missing / overflow)         | Validation failure inside `on_receive_request()`.                       |
| `ENGINE_OP_RDMA_CONNECT_REQUEST`  | Client ➜ Server | Propose an RDMA QP handshake for a NIC pair               | Issued lazily when first RDMA READ is required.                         |
| `ENGINE_OP_RDMA_CONNECT_RESPONSE` | Server ➜ Client | Return remote QP info so client can **RTR/RTS**           | `channel->get_rdma()` handshake path.                                   |
| `ENGINE_OP_RDMA_CONNECT_FAILED`   | Server ➜ Client | RDMA handshake rejected (e.g. no NIC)                     | Transport creation or `ibv_modify_qp()` failure.                        |
| `ENGINE_OP_MTCP_CONNECT_REQUEST`  | Client ➜ Server | Ask peer to open a multi-TCP (MTCP) listener              | First time a large tensor must flow over TCP.                           |
| `ENGINE_OP_MTCP_CONNECT_RESPONSE` | Server ➜ Client | Provides listener IP/port & agreed socket count           | Client then dials `MTcpTransport::connect()`.                           |
| `ENGINE_OP_MTCP_CONNECT_FAILED`   | Server ➜ Client | Peer could not create listener                            | Port exhaustion or internal error.                                      |
| `ENGINE_OP_CLOSE`                 | Either side     | Graceful channel shutdown                                 | `close_connection()` or GC thread expiry.                               |

> **Tip** – The op-codes are arranged so that **request / response / failure** triplets are numerically adjacent, simplifying switch-case handling.

### 7.2 Connection Handshake State-machine

```mermaid
stateDiagram-v2
    [*] --> Idle
    Idle --> TcpConnected: TCP Control Channel Ready
    TcpConnected --> MTCP_Request: RDMA disabled
    MTCP_Request --> MTCP_Ready: ENGINE_OP_MTCP_CONNECT_RESPONSE
    TcpConnected --> RDMA_Request: RDMA enabled
    RDMA_Request --> RDMA_Ready: ENGINE_OP_RDMA_CONNECT_RESPONSE
    MTCP_Request --> Failure: ENGINE_OP_MTCP_CONNECT_FAILED
    RDMA_Request --> Failure: ENGINE_OP_RDMA_CONNECT_FAILED
    MTCP_Ready --> Active
    RDMA_Ready --> Active
    Active --> Closed: ENGINE_OP_CLOSE / GC Expire
    Failure --> Closed
    Closed --> [*]
```

### 7.3 Read Request Life-cycle

```mermaid
sequenceDiagram
    participant Client as Request Thread
    participant Channel
    participant Server as Peer Engine
    participant Transport

    Client->>Channel: ENGINE_OP_READ_REQUEST
    note over Channel: store in pending_requests_
    Channel->>Server: READ_REQUEST (TCP)
    Server->>Server: Validate & Locate Tensor
    alt Success
        Server-->>Channel: ENGINE_OP_READ_RESPONSE_EX
        Channel-->>Client: on_receive_response()
        alt RDMA Path
            Client->>Transport: rdma.read()
            Transport-->>Client: Future fulfilled
        else MTCP Path
            Client->>Transport: mtcp.recv()
            Transport-->>Client: Future fulfilled
        end
    else Failure
        Server-->>Channel: ENGINE_OP_READ_FAILED
        Channel-->>Client: set_result(error)
    end
```

### 7.4 Request Object States (`ReadRequest`)

```mermaid
stateDiagram-v2
    [*] --> Created
    Created --> Queued: request_queue_.push()
    Queued --> Sent: READ_REQUEST dispatched
    Sent --> WaitingResponse
    WaitingResponse --> Transporting: READ_RESPONSE_EX ok
    Transporting --> Completed: Data copied (MTCP) / RDMA done
    WaitingResponse --> Failed: READ_FAILED or disconnect
    Transporting --> Failed: IO Error
    Completed --> [*]
```
