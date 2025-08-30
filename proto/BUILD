load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_grpc_compile", "cpp_grpc_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "global_store_proto",
    srcs = ["global_store.proto"],
)

cpp_grpc_compile(
    name = "global_store_grpc",
    protos = [":global_store_proto"],
)

cpp_grpc_library(
    name = "global_store_grpc_cpp",
    protos = [":global_store_proto"],
)

# Store Daemon service protos (C++ targets)
proto_library(
    name = "store_daemon_proto",
    srcs = ["store_daemon.proto"],
)

cpp_grpc_compile(
    name = "store_daemon_grpc",
    protos = [":store_daemon_proto"],
)

cpp_grpc_library(
    name = "store_daemon_grpc_cpp",
    protos = [":store_daemon_proto"],
)
# Communicator config (messages only)
proto_library(
    name = "communicator_config_proto",
    srcs = ["communicator_config.proto"],
)
