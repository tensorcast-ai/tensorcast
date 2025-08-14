#! /bin/bash

# Generate Python protobuf files using grpc_tools
echo "Generating Python protobuf files using grpc_tools..."

# Generate store_daemon protobuf files
uv run python -m grpc_tools.protoc \
    --python_out=scstore/proto \
    --pyi_out=scstore/proto \
    --grpc_python_out=scstore/proto \
    --proto_path=scstore/proto \
    proto/store_daemon.proto

# Generate global_store protobuf files
uv run python -m grpc_tools.protoc \
    --python_out=scstore/proto \
    --pyi_out=scstore/proto \
    --grpc_python_out=scstore/proto \
    --proto_path=scstore/proto \
    proto/global_store.proto

echo "Python protobuf files generated successfully!"
# Fix import paths in generated *_pb2_grpc.py files
sed -i 's/^import store_daemon_pb2/import scstore.proto.store_daemon_pb2/' scstore/proto/store_daemon_pb2_grpc.py
sed -i 's/^import global_store_pb2/import scstore.proto.global_store_pb2/' scstore/proto/global_store_pb2_grpc.py
