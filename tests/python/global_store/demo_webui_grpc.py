#!/usr/bin/env python3
#  Copyright (c) 2025, TensorCast Team.

"""Demo script showing the refactored Web UI architecture."""
import asyncio
import subprocess
import time
import sys
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_webui_endpoints():
    """Test Web UI endpoints after services are running."""
    base_url = "http://localhost:9000/api"

    endpoints = [
        "/summary",
        "/workers",
        "/models",
        "/nodes",
        "/replicas?page_size=10",
        "/transports",
    ]

    print("\nTesting Web UI endpoints:")
    print("-" * 50)

    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if "data" in data:
                    if isinstance(data["data"], list):
                        count = len(data["data"])
                        print(f"✓ {endpoint}: {count} items")
                    else:
                        print(f"✓ {endpoint}: OK")
                else:
                    print(f"✓ {endpoint}: {response.status_code}")
            else:
                print(f"✗ {endpoint}: {response.status_code}")
        except Exception as e:
            print(f"✗ {endpoint}: {type(e).__name__}: {e}")

    print("-" * 50)


def main():
    """Demo the refactored Web UI with gRPC backend."""
    print("=" * 70)
    print("Web UI gRPC Refactoring Demo")
    print("=" * 70)
    print("\nThis demo shows the refactored Web UI architecture where:")
    print("1. Global Store runs with gRPC service and DuckDB")
    print("2. Web UI runs as a separate process using only gRPC")
    print("3. No direct database access from Web UI = No lock conflicts!")
    print("\n" + "=" * 70)

    # Check if services are already running
    try:
        response = requests.get("http://localhost:50051", timeout=1)
    except:
        print("\n⚠️  Global Store not running on port 50051")
        print("Please start it with: python -m scstore.global_store --port 50051")
        sys.exit(1)

    try:
        response = requests.get("http://localhost:9000/api/health", timeout=1)
        print("\n✓ Web UI already running on port 9000")
        test_webui_endpoints()
        return
    except:
        pass

    print("\nStarting Web UI with gRPC backend...")
    print("Command: python -m scstore.global_store.webui_backend.main \\")
    print("           --grpc-host 127.0.0.1 --grpc-port 50051 \\")
    print("           --ui-host 0.0.0.0 --ui-port 9000")

    # Start Web UI process
    webui_proc = subprocess.Popen([
        sys.executable, "-m", "scstore.global_store.webui_backend.main",
        "--grpc-host", "127.0.0.1",
        "--grpc-port", "50051",
        "--ui-host", "0.0.0.0",
        "--ui-port", "9000"
    ])

    print("\n⏳ Waiting for Web UI to start...")

    # Wait for Web UI to be ready
    for i in range(10):
        try:
            response = requests.get("http://localhost:9000/api/health", timeout=1)
            if response.status_code == 200:
                print("✓ Web UI is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("✗ Web UI failed to start")
        webui_proc.terminate()
        sys.exit(1)

    # Test endpoints
    test_webui_endpoints()

    print("\nKey observations:")
    print("- Web UI communicates only via gRPC (no DuckDB access)")
    print("- No database lock warnings in logs")
    print("- Can run on separate machines")
    print("- WebSocket updates via polling (5s interval)")

    print("\n✓ Demo complete! Press Ctrl+C to stop the Web UI.")

    try:
        webui_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down Web UI...")
        webui_proc.terminate()
        webui_proc.wait()
        print("✓ Shutdown complete")


if __name__ == "__main__":
    main()