# core/testing

Shared C++ utilities that back TensorCast's integration and concurrency tests.

## TempArtifactFixture

- Creates RFC-0007 compliant artifacts for disk-based StoreEngine tests.
- Fixture roots now include a unique hex suffix derived from time, thread id, and a counter (e.g., `/tmp/store_engine_multi_gpu_b2_<suffix>`) so parallel runs never collide or wipe each other's artifacts.
- Paths are automatically cleaned up when the fixture goes out of scope.
