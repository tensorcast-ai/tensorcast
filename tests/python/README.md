# Python Tests for TensorCast

This directory contains comprehensive test suites for the TensorCast system, covering both the Global Store service and the Store Daemon functionality, as well as auto-daemon management and artifact transport capabilities.


## Running Tests

### Prerequisites
Ensure you have the Python virtual environment activated:
```bash
source .venv/bin/activate
```

### Running All Tests
```bash
# Run all Python tests
uv run pytest tests/python

# Run with verbose output
uv run pytest tests/python -v

# Run with detailed output and capture print statements
uv run pytest tests/python -v -s
```
