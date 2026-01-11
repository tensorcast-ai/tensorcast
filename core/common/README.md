# `core/common`

Shared C++ utilities used across TensorCast core and services.

CUDA-specific backends, device/stream/event RAII helpers, shared IPC handle bytes, and error handling live in
`core/cuda`.

Daemon/global config IO helpers live in `core/common/config`, including duration normalization for HA settings
(heartbeat/state sync/full sync RPC timeouts).
