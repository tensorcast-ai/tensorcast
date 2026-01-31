load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "operation_proto",
    srcs = ["operation.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "@protobuf//:any_proto",
        "@protobuf//:timestamp_proto",
    ],
)

cpp_proto_library(
    name = "operation_cc",
    protos = [":operation_proto"],
    visibility = ["//visibility:public"],
)

buf_lint_test(
    name = "operation_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":operation_proto"],
)
