load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_grpc_library", "cpp_proto_library")

proto_library(
    name = "memory_tier_proto_lib",
    srcs = ["memory_tier.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "@protobuf//:timestamp_proto",
    ],
)

cpp_proto_library(
    name = "memory_tier_cc",
    protos = [":memory_tier_proto_lib"],
    visibility = ["//visibility:public"],
)

cpp_grpc_library(
    name = "memory_tier_grpc_cc",
    protos = [":memory_tier_proto_lib"],
    visibility = ["//visibility:public"],
    deps = [
        ":memory_tier_cc",
    ],
)

buf_lint_test(
    name = "memory_tier_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":memory_tier_proto_lib"],
)
