<p align="center">
  <img src="docs/assets/contributors/tensorcast-contributors.png" alt="TensorCast" width="960">
</p>

<p align="center">
  <a href="docs/README.md">Docs</a> |
  <a href="docs/architecture/architecture-overview.md">Architecture</a> |
  <a href="#quickstart">Quickstart</a> |
  <a href="docs/development/build-from-source.md">Build from source</a> |
  <a href="docs/development/testing.md">Testing</a> |
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <a href="https://pypi.org/project/tensorcast/"><img src="https://img.shields.io/pypi/v/tensorcast.svg?label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/tensorcast/"><img src="https://img.shields.io/pypi/pyversions/tensorcast.svg" alt="Python versions"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT%20AND%20Apache--2.0-blue.svg" alt="License"></a>
  <a href="https://tensorcast.ai"><img src="https://img.shields.io/badge/docs-tensorcast.ai-0b65c2.svg" alt="Docs"></a>
  <img src="https://img.shields.io/badge/platform-Linux-blue.svg" alt="Platform: Linux">
</p>

## About

Modern AI workloads are increasingly constrained by state movement rather than
only GPU compute. Large model weights and dynamic KV cache move repeatedly
across storage tiers, CPU memory, GPU memory, process boundaries, and network
links. Inefficient state management leads to slow cold starts, weak elasticity,
fragmented GPU pools, repeated prefill work, and topology-blind network
hotspots.

TensorCast is a tensor state infrastructure layer that extracts model weights,
KV cache, checkpoints, RL parameters, and other tensor state from application
processes and manages them as distributed artifacts. It separates control-plane
scheduling from data-plane transfer: local daemons expose CUDA IPC for
zero-copy GPU sharing on the same node, while cross-node artifact movement runs
over RDMA or TCP P2P paths. Artifact metadata drives where state should live,
how it should move, and which tensor view a consumer needs.

## Project Status

> **⚠️ Warning:** This software is under active development. Production deployments
> require strong distributed-systems and network operations experience.

## Features

- **Tensor-native state management**: model weights, KV cache, checkpoints,
  RL parameters, activations, and tensor dictionaries are represented as
  artifacts with tensor-aware metadata, not opaque byte blobs. This gives the
  system enough structure to reason about shape, dtype, layout, views, and
  replica placement.
- **State lifecycle decoupled from workers**: tensor state can outlive the
  process that produced or consumed it. Serving workers can restart, scale out,
  or switch models by attaching to daemon-managed artifacts instead of
  rebuilding the same state from remote storage.
- **High-performance zero-copy data path**: the Store Daemon owns local tensor
  memory and exposes CUDA IPC handles to clients, allowing multiple processes on
  the same node to share VRAM-resident tensors without an additional model load
  or process-local copy.
- **Unified materialization pipeline**: disk loads, checkpoint restore, memory
  staging, network transfer, and tensor materialization are executed through a
  shared asynchronous data path, making DISK -> DRAM -> VRAM movement explicit,
  bounded, and reusable across workflows.
- **Topology-aware P2P distribution**: TensorCast treats existing workers as a
  distributed replica pool. A node that already holds an artifact can serve
  downstream nodes over TCP or RDMA-capable paths, while the Global Store plans
  fanout with replica location, media tier, load, and network topology in mind.
- **Tensor views and in-flight transformation**: consumers can request the
  tensor view they need, such as slices, tensor-parallel shards, transposes, or
  layout-specific materializations, instead of forcing every workload to fetch
  and reshape the full artifact.
- **Global scheduling over artifact metadata**: placement and routing policies
  are expressed over artifact records, replica state, node load, media priority,
  and topology distance, allowing scheduling behavior to evolve without baking
  one fixed strategy into each framework integration.
- **Artifact-first Python SDK**: a small set of Python APIs (`register`, `put`,
  `artifact`, `from_disk`) creates durable tensor artifacts, while artifact
  methods such as `tensor_dict`, `tensor_dict_into`, `bind`, and `prefetch`
  realize those artifacts into the target a workload actually needs.

## Install

Install TensorCast with [`uv`](https://docs.astral.sh/uv/) (recommended) or
`pip`:

```bash
uv pip install tensorcast
```

The wheel pulls in the matching `torch==2.11.0` CUDA 12.8 build automatically.
If your environment already has a different torch version, build from source
instead: [Build from source](docs/development/build-from-source.md).

| Axis | Supported |
|---|---|
| Python | 3.10 / 3.11 / 3.12 |
| OS | Linux only, kernel >= 5.10 |
| glibc | >= 2.28 (RHEL 8, Ubuntu 20.04+, Debian 10+) |
| torch | 2.11.0 + CUDA 12.8 (exact pin; ABI-checked at import) |
| CUDA | 12.8 driver + runtime |

## Quickstart

TensorCast starts from an **artifact**. An artifact is durable tensor state plus
metadata: tensor names, shapes, dtypes, views, source locations, replicas, and
routing hints. Creating an artifact handle is lazy; bytes move only when the
artifact is realized into a target such as a tensor dict, caller-owned tensors,
a binding, or a prefetch operation.

The usual workflow is:

- Create or discover an artifact from tensors (`tc.put(...)`) or disk
  (`tc.from_disk(...)`).
- Keep the artifact handle or key as the stable identity your workers share.
- Optionally derive the exact view a worker needs with `artifact.view(...)`.
- Realize the artifact into the target form for that worker with
  `artifact.tensor_dict(...)`, `artifact.bind(...)`, `artifact.prefetch(...)`,
  or `artifact.tensor_dict_into(...)`.

The recommended startup pattern is **CLI-managed services + SDK connect**:
operators or launch scripts start the Global Store and Store Daemon, while each
Python worker only connects to its node-local daemon. This keeps service
lifecycle explicit and avoids each worker owning infrastructure processes.

### Minimal path

Start the services first:

```bash
# 1. Start the Global Store.
tensorcast-cli global start --config=examples/config/global_store_config.yaml

# 2. Start the Store Daemon and connect it to the Global Store.
tensorcast-cli daemon start \
  --config=examples/config/store_daemon_config.yaml \
  --global-store-mode connect \
  --global-store-address 127.0.0.1:50051

# 3. Verify both services are up.
tensorcast-cli global status
tensorcast-cli daemon status
```

If you are running from a source checkout with `uv`, prefix those commands with
`uv run`. By default the services run in the background; use
`tensorcast-cli daemon logs -f` to follow daemon logs.

Then connect from Python. This is the smallest useful SDK loop: publish tensors
as an artifact, resolve the artifact by key, and realize it back as CUDA
tensors.

```python
import torch
import tensorcast as tc

tc.init(mode="connect", address="127.0.0.1:50052")

state_dict = {
    "layers.0.weight": torch.randn(4096, 4096, device="cuda:0"),
    "layers.0.bias": torch.randn(4096, device="cuda:0"),
}

# policy declares artifact placement and durability. "cache" favors a fast
# local path and allows eviction; use "durable" or "ha" when the artifact must
# survive daemon restarts or be reused across nodes.
registered = tc.put(state_dict, key="demo:model:v1", policy="cache")

print(registered.artifact_id)

# Artifact handles are lazy. The data moves when the artifact is realized.
artifact = tc.artifact("demo:model:v1")
weights = artifact.tensor_dict(device="cuda:0")
print(weights["layers.0.weight"].shape)

tc.shutdown()  # Closes the SDK client context; CLI-managed services keep running.
```

Stop the services when you are done:

```bash
tensorcast-cli daemon stop
tensorcast-cli global stop
```

For notebooks or small local scripts, SDK-managed startup is also available with
`tc.init(mode="create", global_store_mode="start")`, but production and
multi-worker runs should prefer the explicit CLI-managed pattern above.

Policy presets:

- `cache`: fast local stable memory, best-effort, evictable.
- `durable`: must persist to shared disk, and should keep a local stable copy.
- `ha`: durable storage plus local and remote stable replicas when possible.
- `cold`: shared disk required, with temporary local stable memory by TTL.
- `warm`: local stable memory preferred; reject instead of evicting/spilling.
- `pinned`: local stable memory required and pinned; reject on overflow.

`policy` is the artifact placement and durability contract. Retrieval source
choices such as local, disk, or P2P are selected later with
`GetArtifactOptions(source=...)` when the artifact is realized.

### Artifact capability tour

The snippets below assume a process has already called `tc.init(...)` and has
an artifact handle such as `artifact = tc.artifact("demo:model:v1")`.

#### Create an artifact from disk

Disk is also an artifact source. `tc.from_disk(...)` resolves a local directory
into an artifact handle; reads still go through the same realization path as
in-memory artifacts. The primary supported format is safetensors: TensorCast
loads all `*.safetensors` files in the directory, including HuggingFace-style
sharded folders such as `model.safetensors.index.json` plus
`model-00001-of-000XX.safetensors`. TensorCast disk artifacts can also use the
native `tensor.data*` layout.

```python
disk_artifact = tc.from_disk("/shared/tensorcast/models/demo-model")

weights = disk_artifact.tensor_dict(
    device="cuda:0",
    options=tc.GetArtifactOptions(source="disk_first"),
)
```

Use `tc.import_from_disk(..., key="demo:model:v1")` when you want to explicitly
import a disk source into managed artifact storage and publish a key in one
step.

#### Slice or transform before data moves

Views are artifact transformations. A view is still lazy: it records the tensor
selection, and TensorCast applies that selection during realization instead of
forcing every consumer to fetch the full artifact.

```python
rank0_view = artifact.view(
    slices={
        "layers.0.weight": [(0, slice(0, 2048))],
        "layers.0.bias": [(0, slice(0, 2048))],
    },
)

rank0_weights = rank0_view.tensor_dict(device="cuda:0")
```

This is the basic form of in-flight transform: the artifact identity remains
stable, while each consumer asks for the representation it needs. Transposes and
layout-specific materializations use the same view-centered model.

#### Prefer local, disk, or P2P sources

Source policy belongs to artifact realization, not to a separate loader API.
With the default policy, TensorCast prefers available local replicas and can use
disk or P2P when the daemon topology allows it. In a multi-daemon deployment,
P2P is a source preference, not a direct transport call:

```python
p2p_first = tc.GetArtifactOptions(
    source={
        "preference": "prefer_p2p",
        "allow_p2p": True,
        "allow_disk": True,
    }
)

weights = artifact.tensor_dict(
    device="cuda:0",
    options=p2p_first,
)
```

Applications still connect only to their node-local Store Daemon. The Global
Store coordinates artifact metadata and replica routing; daemon-to-daemon P2P
performs the data transfer over the configured TCP or RDMA-capable path. Use
`tensor_dict_with_diagnostics(...)` when you want to confirm which source was
chosen.

#### Use the same artifact for DP and TP

Data parallel workers usually realize the same artifact key on each node. Each
worker connects to its local daemon, and TensorCast decides whether the best
source is already local, on disk, or available from a peer.

```python
local_rank = 0

# Run this in each worker process after that node's daemon is already running.
tc.init(mode="connect", address="127.0.0.1:50052")
artifact = tc.artifact("demo:model:v1")
dp_weights = artifact.tensor_dict(device=f"cuda:{local_rank}")
```

Tensor parallel workers usually realize rank-local views of the same artifact.
The shard is expressed as artifact metadata, so source selection, P2P routing,
verification, and materialization still use the same pipeline.

```python
tp_rank = 0
tp_world_size = 2
rows_per_rank = 4096 // tp_world_size
start = tp_rank * rows_per_rank

tp_view = artifact.view(
    slices={"layers.0.weight": [(0, slice(start, start + rows_per_rank))]}
)

tp_weights = tp_view.tensor_dict(device=f"cuda:{tp_rank}")
```

For coordinated multi-rank startup, WeightPublisher, and group realization
flows, see the [Weight Publisher deployment guide](docs/deployment/weight-publisher.md).

#### Keep runtime memory stable with bindings

For serving and long-lived workers, a binding lets TensorCast allocate a stable
daemon-owned CUDA layout and refill it from artifacts. Consumers keep using the
same tensor addresses while TensorCast swaps the underlying artifact version.

```python
binding = artifact.bind("cuda:0", publish=True)

next_artifact = tc.artifact("demo:model:v2")
binding.swap(next_artifact, publish=True)
```

Use `artifact.prefetch(device="cuda:0")` when you want to warm an artifact before
the request path needs it. Prefetch, tensor dict materialization, in-place fills,
and bindings all start from the same artifact handle.

For production startup patterns, see the
[SDK Startup User Guide](docs/guides/sdk-startup-user-guide.md). For the full
artifact API surface, views, policies, prefetch, bindings, and realization
internals, see [API Architecture](docs/architecture/api/README.md) and the
[Store SDK reference](tensorcast/api/store/README.md).

## Documentation

- [Developer Guides](docs/README.md) - architecture map and developer docs
- [Architecture Overview](docs/architecture/architecture-overview.md) - system overview
- [API Architecture](docs/architecture/api/README.md) - SDK surface and flows
- [SDK Startup User Guide](docs/guides/sdk-startup-user-guide.md) - SDK and daemon startup
- [Build from Source](docs/development/build-from-source.md) - local development build, troubleshooting, and test entry points
- [Testing Guide](docs/development/testing.md) - Python, C++, P2P, and RDMA tests
- [Store Daemon Deployment](docs/deployment/store-daemon.md) - daemon deployment and config
- [Global Store Deployment](docs/deployment/global-store-deployment.md) - Global Store deployment
- [Release Guide](RELEASE.md) - how releases are cut
- [Contributing](CONTRIBUTING.md) - contribution workflow and DCO sign-off
- [Security Policy](SECURITY.md) - private vulnerability reporting
- [Code of Conduct](CODE_OF_CONDUCT.md) - community standards
- [Repo Automation Rules](AGENTS.md) - automation and sandbox rules

## License

TensorCast uses mixed licensing. TensorCast-owned code is licensed under MIT
unless otherwise noted. Portions derived from ServerlessLLM remain licensed
under Apache-2.0. Third-party dependencies are licensed under their respective
licenses.

See [LICENSE](LICENSE), [NOTICE](NOTICE), and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
