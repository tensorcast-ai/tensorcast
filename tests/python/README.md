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
python -m pytest tests/python

# Run with verbose output
python -m pytest tests/python -v

# Run with detailed output and capture print statements
python -m pytest tests/python -v -s
```
