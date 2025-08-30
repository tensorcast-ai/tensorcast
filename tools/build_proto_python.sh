#! /bin/bash

set -euo pipefail

# Generate Python protobuf files using grpc_tools
echo "Generating Python protobuf files using grpc_tools..."

# Generate store_daemon protobuf files
uv run python -m grpc_tools.protoc \
    --python_out=tensorcast/proto \
    --pyi_out=tensorcast/proto \
    --grpc_python_out=tensorcast/proto \
    --proto_path=proto \
    proto/store_daemon.proto

# Generate global_store protobuf files
uv run python -m grpc_tools.protoc \
    --python_out=tensorcast/proto \
    --pyi_out=tensorcast/proto \
    --grpc_python_out=tensorcast/proto \
    --proto_path=proto \
    proto/global_store.proto

# Generate communicator_config protobuf files (messages only)
uv run python -m grpc_tools.protoc \
    --python_out=tensorcast/proto \
    --pyi_out=tensorcast/proto \
    --proto_path=proto \
    proto/communicator_config.proto

echo "Python protobuf files generated successfully!"
# Fix import paths in generated *_pb2_grpc.py files
sed -i 's/^import store_daemon_pb2/import tensorcast.proto.store_daemon_pb2/' tensorcast/proto/store_daemon_pb2_grpc.py
sed -i 's/^import global_store_pb2/import tensorcast.proto.global_store_pb2/' tensorcast/proto/global_store_pb2_grpc.py

# Fix relative imports in generated non-gRPC modules if necessary
if [ -f tensorcast/proto/communicator_config_pb2.py ]; then
  sed -i '1i# mypy: ignore-errors' tensorcast/proto/communicator_config_pb2.py
fi

# get directory of this script
current_dir=$(dirname "$0")
root_dir=$(dirname "$current_dir")
# build proto
# build proto
if ! bazel build //proto:global_store_grpc; then
    echo "Error: Failed to build proto target" >&2
    exit 1
fi

# copy proto files to tensorcast/proto
if ! cp -f $root_dir/bazel-bin/proto/global_store_grpc/proto/*.pb.h $root_dir/tensorcast/csrc/proto; then
    echo "Error: Failed to copy proto header files" >&2
    exit 1
fi
cp -f $root_dir/bazel-bin/proto/global_store_grpc/proto/*.pb.h $root_dir/tensorcast/csrc/proto

uv run ruff format tensorcast/proto/*
