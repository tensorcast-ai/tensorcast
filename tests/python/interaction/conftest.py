#  Copyright (c) 2025, TensorCast Team.

import pytest
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import grpc
from tensorcast.proto import global_store_pb2_grpc
from tensorcast.global_store.grpc_service import GlobalStoreServicer
from tensorcast.store_daemon import health_check as _health_check  # pylint: disable=import-error
from typing import Iterator


@dataclass
class InProcessGlobalStore:
    """Wrapper for an in-process Global Store service with its gRPC server."""

    service: GlobalStoreServicer
    address: str
    grpc_server: grpc.Server

    # ------------------------------------------------------------------
    # Convenience helpers – make the wrapper behave like the underlying
    # *GlobalStoreServicer* so that existing tests can transparently
    # call RPC handler methods directly on the fixture without having to
    # drill down into the ``.service`` attribute.  This preserves backwards
    # compatibility with older test-suites which expected the fixture to
    # expose the servicer interface directly.
    # ------------------------------------------------------------------

    def __getattr__(self, item):  # noqa: D401 – simple delegation helper
        """Delegate attribute access to the wrapped *service* instance.

        This allows test helpers to call methods like ``RegisterWorker`` or
        ``RequestReplicaTransport`` directly on the fixture object as
        if it were the *GlobalStoreServicer* itself.
        """
        try:
            return getattr(self.service, item)
        except AttributeError as exc:
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{item}'") from exc

    # Older tests referenced a private ``_address`` attribute.  Provide a
    # read-only proxy for compatibility so we avoid touching every call-site.
    @property
    def _address(self) -> str:  # pylint: disable=invalid-name
        """Backwards-compatibility alias for :pyattr:`address`."""
        return self.address


# -----------------------------------------------------------------------------
# Global Store setup – start an in-process gRPC server backed by FakeGlobalStore
# -----------------------------------------------------------------------------


@pytest.fixture(scope="session")
def global_store_service() -> Iterator[InProcessGlobalStore]:
    """Start an in-memory GlobalStoreServicer behind a real gRPC server.

    Returns:
        InProcessGlobalStore wrapper containing the service, address, and server.
    """

    # Instantiate service (in-memory DuckDB) and server
    service = GlobalStoreServicer()
    server = grpc.server(ThreadPoolExecutor(max_workers=8))
    global_store_pb2_grpc.add_GlobalStoreServicer_to_server(service, server)

    # Bind to a random free port on localhost (port 0 asks the OS to pick)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()

    # Create wrapper with all the necessary information
    wrapper = InProcessGlobalStore(
        service=service, address=f"127.0.0.1:{port}", grpc_server=server
    )

    # ------------------------------------------------------------------
    # Prevent port-collision issues when the daemon's internal HealthCheckServer
    # attempts to start – just turn its start() into a no-op so it never binds.
    # ------------------------------------------------------------------
    _health_check.HealthCheckServer.start = lambda self: None # type: ignore[method-assign]

    yield wrapper

    # Teardown – stop gRPC server cleanly
    server.stop(0)


# -----------------------------------------------------------------------------
# Test isolation – reset Global Store state between individual tests
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_global_store_before_test(global_store_service: InProcessGlobalStore):
    """Ensure each test starts with a pristine Global Store database."""
    global_store_service.service.reset_state()
    yield
    # No teardown needed – per-test cleanup happens on *next* setup call
