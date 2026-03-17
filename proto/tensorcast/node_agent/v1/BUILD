load("@rules_buf//buf:defs.bzl", "buf_lint_test")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_proto_grpc_cpp//:defs.bzl", "cpp_proto_library")

proto_library(
    name = "node_agent_proto",
    srcs = ["node_agent.proto"],
    strip_import_prefix = "/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_proto",
        "//proto/tensorcast/plan/v1:plan_proto",
        "@protobuf//:timestamp_proto",
    ],
)

cpp_proto_library(
    name = "node_agent_cc",
    protos = [":node_agent_proto"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/plan/v1:plan_cc",
    ],
)

buf_lint_test(
    name = "node_agent_proto_lint",
    config = "//proto:buf.yaml",
    targets = [":node_agent_proto"],
)
