load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")

# Aggregate library that includes all core components
cc_library(
    name = "core",
    visibility = ["//visibility:public"],
    deps = [
        "//core/checkpoint",
        "//core/common",
        "//core/communicator:config_io_lib",
        "//core/communicator:engine_lib",
        "//core/store:store_engine",
    ],
    alwayslink = True,
)

# Shared library with the minimal surface needed by tensorcast._C.
cc_binary(
    name = "libcheckpoint_ext.so",
    linkshared = True,
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = [
        "//core/checkpoint",
        "//core/common:artifact_hash_lib",
        "//core/common:artifact_verification_lib",
        "//core/common:logging_init_lib",
        "//core/store/loader:canonical_index",
        "//core/store/loader:file_partition_source",
        "//core/store/loader:safetensors_util",
        "//core/store/loader:source_hash",
    ],
)
