#!/bin/bash
################################################################################
# Unified entry point: tools/release.sh
# -------------------------------------
# This script is now the single entry point for Torch/PyPI release operations
# AND day-to-day torch version management tasks.  If the first argument matches
# a management sub-command, we transparently forward all arguments to
# `tools/manage_torch_version.py`, then exit. Otherwise, the original build
# flow continues unchanged.
################################################################################

# Directory of this script (absolute path)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# If called with a management sub-command, forward to Python helper and exit
case "$1" in
    status|sync-venv|update-pyproject|cache)
        python "${SCRIPT_DIR}/manage_torch_version.py" "$@"
        exit $?
        ;;
esac
# Default values
export BUILD_VERSION=0.0.2
TORCH_VERSION="2.6.0"
CUDA_VERSION=""
SKIP_UV_SYNC=false
CACHE_UV_LOCK=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --torch-version)
            TORCH_VERSION="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --build-version)
            BUILD_VERSION="$2"
            shift 2
            ;;
        --skip-uv-sync)
            SKIP_UV_SYNC=true
            shift
            ;;
        --cache-uv-lock)
            CACHE_UV_LOCK=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--torch-version VERSION] [--cuda-version VERSION] [--build-version VERSION] [--skip-uv-sync] [--cache-uv-lock]"
            echo "Example: $0 --torch-version 2.5.0 --cuda-version cu118"
            exit 1
            ;;
    esac
done

echo "Building with:"
echo "  BUILD_VERSION: $BUILD_VERSION"
echo "  TORCH_VERSION: $TORCH_VERSION"
echo "  CUDA_VERSION: $CUDA_VERSION"

# (SCRIPT_DIR already defined at top)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Backup original pyproject.toml
cp pyproject.toml pyproject.toml.bak

# Update torch version in pyproject.toml using Python script
echo "Updating torch version to ${TORCH_VERSION} in pyproject.toml..."
uv run "${SCRIPT_DIR}/update_torch_version.py" "${TORCH_VERSION}"

# Set environment variables for the build
if [[ -n "$CUDA_VERSION" ]]; then
    # For CUDA builds, set the PyTorch index URL
    echo "Configuring PyTorch index for CUDA build..."
    export PIP_INDEX_URL="https://download.pytorch.org/whl/${CUDA_VERSION}"
    export PIP_EXTRA_INDEX_URL="https://pypi.org/simple"
    export UV_INDEX_URL="https://download.pytorch.org/whl/${CUDA_VERSION}"
    export UV_EXTRA_INDEX_URL="https://pypi.org/simple"
fi

# Try to restore cached uv.lock if available
echo "Checking for cached uv.lock..."
if python -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from torch_version_manager import restore_uv_lock
result = restore_uv_lock('${TORCH_VERSION}', '${CUDA_VERSION}' if '${CUDA_VERSION}' else None)
sys.exit(0 if result else 1)
" 2>/dev/null; then
    echo "Using cached uv.lock file"
else
    echo "No cached uv.lock found for torch ${TORCH_VERSION} ${CUDA_VERSION}"
fi

# Run uv sync to ensure environment is correct unless skipped
if [ "$SKIP_UV_SYNC" = false ]; then
    echo "Running uv sync to ensure environment matches pyproject.toml..."
    if ! uv sync; then
        echo "Error: uv sync failed. Please check your dependencies."
        mv pyproject.toml.bak pyproject.toml
        exit 1
    fi

    # Cache the uv.lock if requested
    if [ "$CACHE_UV_LOCK" = true ]; then
        echo "Caching uv.lock for future use..."
        python -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from torch_version_manager import cache_uv_lock
cache_uv_lock('${TORCH_VERSION}', '${CUDA_VERSION}' if '${CUDA_VERSION}' else None)
"
    fi
else
    echo "Skipping uv sync (--skip-uv-sync flag was set)"
fi

# Validate torch versions before building
echo "Validating torch versions..."
if ! python - <<EOF
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from torch_version_manager import validate_torch_versions

# validate_torch_versions returns (is_consistent, versions)
# Exit with 0 if consistent, 1 otherwise
is_consistent, _ = validate_torch_versions(raise_on_error=False)
sys.exit(0 if is_consistent else 1)
EOF
 then
    echo "Error: Torch version validation failed!"
    echo "Please ensure all torch versions are consistent across:"
    echo "  - pyproject.toml (build-system and dependencies)"
    echo "  - .venv installation"
    echo "  - MODULE.bazel references"
    mv pyproject.toml.bak pyproject.toml
    exit 1
fi

# Build the wheel using uv
echo "Building wheel with torch ${TORCH_VERSION}..."
RELEASE=1 BUILD_EXTENSION=1 BUILD_CORE=1 uv build --wheel

# The built wheel should now have the correct torch version in metadata
echo "Build completed! Wheel files are in dist/"

# Show the generated wheel files
echo ""
echo "Generated wheel files:"
ls -la dist/*.whl 2>/dev/null || echo "No wheel files found in dist/"

# Restore original pyproject.toml
echo "Restoring original pyproject.toml..."
mv pyproject.toml.bak pyproject.toml

# Create a manifest file with build information
echo "Creating build manifest..."
cat > "dist/build_manifest_${BUILD_VERSION}.txt" << EOF
Build Information:
==================
Build Version: ${BUILD_VERSION}
Torch Version: ${TORCH_VERSION}
CUDA Version: ${CUDA_VERSION:-CPU}
Build Date: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Git Commit: $(git rev-parse --short HEAD)
Git Branch: $(git rev-parse --abbrev-ref HEAD)

Wheel Files:
============
$(ls -1 dist/*.whl 2>/dev/null || echo "No wheel files generated")
EOF

echo ""
echo "Build manifest saved to dist/build_manifest_${BUILD_VERSION}.txt"
