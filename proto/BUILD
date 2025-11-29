load("@rules_buf//buf:defs.bzl", "buf_format")

package(default_visibility = ["//visibility:public"])

exports_files(["buf.yaml"])

# Format all .proto files in this package using Buf
buf_format(
    name = "buf_format",
)

# ---------------------------------------------------------------------------
# Backward-compatibility aliases for refactored proto targets
# The repo previously depended on top-level //proto:XXX targets. After moving
# protos under //proto/tensorcast/**, we re-export expected names here.
# ---------------------------------------------------------------------------

alias(
    name = "communicator_config_cc_lib",
    actual = "//proto/tensorcast/communicator/v1:communicator_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "communicator_config_cc",
    actual = "//proto/tensorcast/communicator/v1:communicator_cc",
    visibility = ["//visibility:public"],
)

# Common doesn't define services; map both names to the C++ proto lib
alias(
    name = "common_grpc_cpp",
    actual = "//proto/tensorcast/common/v1:common_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "common_grpc",
    actual = "//proto/tensorcast/common/v1:common_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "global_store_grpc_cpp",
    actual = "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "global_store_grpc",
    actual = "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "store_daemon_grpc_cpp",
    actual = "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "store_daemon_grpc",
    actual = "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "memory_tier_grpc_cpp",
    actual = "//proto/tensorcast/memory_tier/v1:memory_tier_grpc_cc",
    visibility = ["//visibility:public"],
)

alias(
    name = "memory_tier_grpc",
    actual = "//proto/tensorcast/memory_tier/v1:memory_tier_grpc_cc",
    visibility = ["//visibility:public"],
)
