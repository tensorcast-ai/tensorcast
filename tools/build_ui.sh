#!/usr/bin/env bash
# Build script for Global Store Web UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WEBUI_DIR="${PROJECT_ROOT}/scstore/global_store/webui_frontend"
BUILD_OUTPUT="${PROJECT_ROOT}/scstore/global_store/webui_backend/build"

echo "Building Global Store Web UI..."

# Check if pnpm is installed
if ! command -v pnpm &> /dev/null; then
    echo "pnpm is not installed. Installing pnpm..."
    npm install -g pnpm
fi

# Navigate to webui directory
cd "${WEBUI_DIR}"

# Install dependencies with frozen lockfile
echo "Installing dependencies..."
pnpm install --frozen-lockfile

# Run TypeScript check
echo "Running TypeScript check..."
pnpm tsc --noEmit

# Build the application
echo "Building application..."
pnpm build

# Ensure the build was successful
if [ ! -d "${BUILD_OUTPUT}" ]; then
    echo "Error: Build output directory not found at ${BUILD_OUTPUT}"
    exit 1
fi

echo "Build completed successfully!"
echo "Output directory: ${BUILD_OUTPUT}"

# List build artifacts
echo ""
echo "Build artifacts:"
ls -la "${BUILD_OUTPUT}"