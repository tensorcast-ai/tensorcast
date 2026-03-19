load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_grpc_library")

proto_library(
    name = "daemon_proto_lib",
    srcs = ["store_daemon.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_proto",
        "//proto/tensorcast/operation/v1:operation_proto",
        "//proto/tensorcast/plan/v1:plan_proto",
        "@protobuf//:timestamp_proto",
    ],
)

cpp_grpc_library(
    name = "daemon_grpc_cc",
    protos = [":daemon_proto_lib"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/operation/v1:operation_cc",
        "//proto/tensorcast/plan/v1:plan_cc",
    ],
)

buf_lint_test(
    name = "daemon_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":daemon_proto_lib"],
)
