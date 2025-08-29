#  Copyright (c) 2025, TensorCast Team.

"""Global Store test suite.

This package contains modular tests for the Global Store component:

- test_grpc_service.py: gRPC service interface tests
- test_artifacts.py: Domain artifact tests
- test_repositories.py: Repository layer tests
- test_services.py: Service layer tests
- test_configuration.py: Configuration tests
- test_integration.py: Full stack integration tests
- conftest.py: Shared fixtures and utilities

Run all tests:
    pytest tests/python/global_store/

Run specific test module:
    pytest tests/python/global_store/test_artifacts.py

Run with coverage:
    pytest tests/python/global_store/ --cov=scstore.global_store
"""