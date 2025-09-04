load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "communicator_proto",
    srcs = ["communicator_config.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
)

buf_lint_test(
    name = "communicator_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":communicator_proto"],
)

cpp_proto_library(
    name = "communicator_cc",
    protos = [":communicator_proto"],
    visibility = ["//visibility:public"],
)
