# gRPC Client Tests for Web UI Backend

This directory contains comprehensive tests for the Web UI's gRPC client that communicates with the Global Store service.

## Test Files

### 1. `test_grpc_client.py`
Main test suite with comprehensive unit tests including:
- Connection management
- All gRPC operations (list workers, replicas, model info, etc.)
- Retry logic and error handling
- Concurrent request handling
- Edge cases and error conditions

### 2. `test_grpc_client_integration.py`
Full integration test that:
- Starts a test server
- Runs through all client functionality
- Provides visual output with rich formatting
- Tests real-world usage patterns

### 3. `test_grpc_client_performance.py`
Performance benchmarking including:
- Latency measurements
- Throughput testing
- Concurrent load testing
- Large response handling

### 4. `demo_grpc_client_usage.py`
Practical examples showing:
- Basic client usage
- Error handling patterns
- Concurrent request patterns
- Proper setup and teardown

### 5. `test_grpc_server_simple.py`
Simple server verification to ensure the test infrastructure works.

## Running Tests

### Run all tests
```bash
python -m pytest tests/python/global_store/test_grpc_client.py -v
```

### Run integration test
```bash
python tests/python/global_store/test_grpc_client_integration.py
```

### Run performance test
```bash
python tests/python/global_store/test_grpc_client_performance.py
```

### Run usage demo
```bash
python tests/python/global_store/demo_grpc_client_usage.py
```

## Key Features Tested

1. **Connection Management**
   - Successful connections
   - Connection failures and timeouts
   - Reconnection after disconnect

2. **All gRPC Operations**
   - `list_active_workers()` - with filtering
   - `list_model_replicas()` - with various filters
   - `get_model_info()` - for existing and non-existing models
   - `get_summary_stats()` - aggregated statistics

3. **Retry Logic**
   - Automatic retry on transient failures
   - Configurable retry count and delays
   - Proper error propagation after retries exhausted

4. **Concurrent Usage**
   - Multiple simultaneous requests
   - Thread-safe operations
   - Performance under load

5. **Error Handling**
   - gRPC errors
   - Network timeouts
   - Invalid responses
   - Server unavailable

## Test Infrastructure

The tests use a `TestGlobalModelStoreServicer` that implements the Global Store gRPC service with:
- Mock data for workers and model replicas
- Configurable failure modes for testing retry logic
- Call tracking for verification
- Support for all required RPCs

## Implementation Notes

1. The gRPC client uses an async implementation for better performance
2. Retry logic is implemented without async context managers to avoid generator issues
3. All methods return strongly-typed proto messages
4. The client supports singleton pattern for connection reuse
5. Comprehensive logging for debugging

## Known Limitations

1. The `get_active_transports()` method is a placeholder - the proto needs extension
2. Timeout testing can be flaky due to async/sync interactions in gRPC
3. Some proto fields (like replica IDs) are generated client-side