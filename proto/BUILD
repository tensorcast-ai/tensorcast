load("@protobuf//bazel:proto_library.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_grpc_compile", "cpp_grpc_library", "cpp_proto_compile", "cpp_proto_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "global_store_proto",
    srcs = ["tensorcast/global/global_store.proto"],
    strip_import_prefix = "",
    deps = [
        ":common_proto",
        "@protobuf//:duration_proto",
        "@protobuf//:timestamp_proto",
    ],
)

cpp_grpc_compile(
    name = "global_store_grpc",
    protos = [":global_store_proto"],
)

cpp_grpc_library(
    name = "global_store_grpc_cpp",
    protos = [":global_store_proto"],
    deps = [
        ":common_grpc_cpp",
    ],
)

# Store Daemon service protos (C++ targets)
proto_library(
    name = "store_daemon_proto",
    srcs = ["tensorcast/daemon/store_daemon.proto"],
    strip_import_prefix = "",
    deps = [
        ":common_proto",
        "@protobuf//:timestamp_proto",
    ],
)

cpp_grpc_compile(
    name = "store_daemon_grpc",
    protos = [":store_daemon_proto"],
)

cpp_grpc_library(
    name = "store_daemon_grpc_cpp",
    protos = [":store_daemon_proto"],
    deps = [
        ":common_grpc_cpp",
    ],
)

# Communicator config (messages only)
proto_library(
    name = "communicator_config_proto",
    srcs = ["tensorcast/communicator/communicator_config.proto"],
    strip_import_prefix = "",
)

proto_library(
    name = "common_proto",
    srcs = [
        "tensorcast/common/common.proto",
    ],
    strip_import_prefix = "",
    deps = [
        "@protobuf//:duration_proto",
        "@protobuf//:timestamp_proto",
    ],
)

cpp_grpc_compile(
    name = "common_grpc",
    protos = [":common_proto"],
)

cpp_grpc_library(
    name = "common_grpc_cpp",
    protos = [":common_proto"],
)

## artifact/replica merged into common.proto

## Transport module removed (no added value for simple string IDs)
# Generate C++ protobuf for communicator config (no gRPC services)
cpp_proto_compile(
    name = "communicator_config_cc",
    protos = [":communicator_config_proto"],
)

cpp_proto_library(
    name = "communicator_config_cc_lib",
    protos = [":communicator_config_proto"],
)
