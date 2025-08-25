---
title: Device Registry
description: Centralised mapping between physical devices and logical DeviceKey identifiers
sidebar_position: 3
---

# Device Registry

The **Device Registry** is a lightweight singleton utility that provides a *canonical* mapping between physical compute devices (currently GPUs) and the logical `DeviceKey` abstraction used throughout the Core Store.

## Motivation

Historically, StepCast components referred to GPUs only by *ordinal* (`0 … N-1`). This becomes problematic when:

1. PCIe topology changes and GPU enumeration order shifts.
2. GPUs are hot-plugged, replaced, or removed.
3. We need to refer to remote GPUs in a cluster-wide context.

By pairing the ordinal with the *stable* device UUID we obtain an identifier that survives restarts and driver updates, yet is still efficient to compare/hash.

## Header Location

```text
core/store/device_registry.h
```

## Key Data Structures

### DeviceKey

```cpp
struct DeviceKey {
  DeviceType type;   // CPU / GPU / REMOTE
  int32_t     ordinal; // -1 for CPU, >=0 for GPU/remote ranking
  std::string uuid;    // Optional – stable across restarts
};
```

* Hashable via `DeviceKeyHash` for use in `absl::flat_hash_map`/`set`.
* String conversion helpers ease logging & debugging.

### DeviceRegistry

```cpp
class DeviceRegistry {
 public:
  static DeviceRegistry& instance();          // Thread-safe singleton

  void register_gpu(int ordinal, std::string uuid);
  DeviceKey gpu_key(int ordinal) const;       // Ordinal → canonical key
  DeviceKey normalize(const DeviceKey& key) const; // Complete missing fields
};
```

* Internally stores **bidirectional** maps ordinal ⇆ UUID so that either field can be resolved.
* `absl::Mutex` guards maps; read operations are fast thanks to Abseil's flat hash map.

## Typical Usage

```cpp
// At start-up, enumerate GPUs once and register mapping.
for (int ordinal = 0; ordinal < cuda_device_count; ++ordinal) {
  std::string uuid = query_cuda_uuid(ordinal);
  DeviceRegistry::instance().register_gpu(ordinal, uuid);
}

// Later in the code – obtain canonical key for comparisons.
DeviceKey key = DeviceRegistry::instance().gpu_key(target_ordinal);
LOG(INFO) << "Using device " << key.to_string();
```

## Thread-Safety

* All public methods are thread-safe.
* Reads are lock-free after the initial registration phase because look-ups are performed on an *immutable* snapshot of the maps.

## Relation to Other Components

* **Memory Manager** – resolves GPU ordinals from `MemoryLocation` to allocate CUDA buffers.
* **Store Engine** – normalises user-supplied `DeviceKey`s in API calls.
* **Global Store** – transports include `device_uuid` so that replicas can be relocated unambiguously.

## Future Extensions

1. **Dynamic hot-plug**: Listen to NVML events and update maps at runtime.
2. **Remote discovery**: Populate remote GPUs for cross-node scheduling in the future.
3. **Persistent cache**: Serialize registry state for even faster start-up.