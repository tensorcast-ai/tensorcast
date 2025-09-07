load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "config_proto_lib",
    srcs = [
        "client_config.proto",
        "common.proto",
        "daemon_config.proto",
        "global_store_config.proto",
    ],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/communicator/v1:communicator_proto",
        "@protobuf//:duration_proto",
    ],
)

cpp_proto_library(
    name = "config_cc",
    protos = [":config_proto_lib"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/communicator/v1:communicator_cc",
    ],
)

buf_lint_test(
    name = "config_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":config_proto_lib"],
)
