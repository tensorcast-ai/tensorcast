#! /bin/bash

set -euo pipefail

# Generate Python protobuf files using grpc_tools (recursive)
echo "Generating Python protobuf files using grpc_tools..."

# Collect all proto files under proto/ (absolute paths)
mapfile -t PROTOS < <(find proto -type f -name "*.proto" | sort)
if [ ${#PROTOS[@]} -eq 0 ]; then
  echo "No .proto files found under ./proto" >&2
  exit 1
fi

# Clean and recreate output root to avoid stale nested paths
rm -rf tensorcast/proto
mkdir -p tensorcast/proto

# Build relative paths w.r.t. proto/ for protoc inputs
REL_PROTOS=()
for p in "${PROTOS[@]}"; do
  REL_PROTOS+=("${p#proto/}")
done

# Generate Python and gRPC Python stubs
uv run python -m grpc_tools.protoc \
  --python_out=tensorcast/proto \
  --pyi_out=tensorcast/proto \
  --grpc_python_out=tensorcast/proto \
  --proto_path=proto \
  "${REL_PROTOS[@]}"

echo "Python protobuf files generated successfully!"

# Normalize imports to be package-relative in both *_pb2.py and *_pb2_grpc.py
while IFS= read -r -d '' file; do
  sed -i -E 's/^import ([A-Za-z0-9_]+)_pb2 as /from . import \1_pb2 as /' "$file"
done < <(find tensorcast/proto -type f \( -name "*_pb2_grpc.py" -o -name "*_pb2.py" \) -print0)

# Normalize cross-module imports in *_pb2.py to be relative to tensorcast/proto
while IFS= read -r -d '' file; do
  # Convert e.g., `from tensorcast.common.v1 import common_pb2 as ...` to
  # `from .tensorcast.common.v1 import common_pb2 as ...`
  sed -i -E 's/^from (tensorcast\.[A-Za-z0-9_\.]+) import ([A-Za-z0-9_]+)_pb2 as /from .\1 import \2_pb2 as /' "$file"
done < <(find tensorcast/proto -type f -name "*_pb2.py" -print0)

# Ensure every directory under tensorcast/proto is a Python package
while IFS= read -r -d '' dir; do
  if [ ! -f "$dir/__init__.py" ]; then
    echo "# Package marker" > "$dir/__init__.py"
  fi
done < <(find tensorcast/proto -type d -print0)

# get directory of this script
current_dir=$(dirname "$0")
root_dir=$(dirname "$current_dir")

# Build C++ proto/grpc headers for both services (if present)
echo "Building C++ proto targets..."
bazel build //proto:global_store_grpc //proto:store_daemon_grpc //proto:communicator_config_cc \
  //proto:common_grpc || {
  echo "Error: Failed to build proto targets" >&2
  exit 1
}

# Copy generated headers into tensorcast/csrc/proto
mkdir -p "$root_dir/tensorcast/csrc/proto"
for tgt in global_store_grpc store_daemon_grpc common_grpc; do
  if ls "$root_dir/bazel-bin/proto/$tgt/proto"/*.pb.h >/dev/null 2>&1; then
    cp -f "$root_dir/bazel-bin/proto/$tgt/proto"/*.pb.h "$root_dir/tensorcast/csrc/proto"
  fi
done

# Format generated Python files (recursively)
uv run ruff format tensorcast/proto
