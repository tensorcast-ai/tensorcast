load("//core:BUILD.bzl", "sc_cc_library")

package(default_visibility = ["//visibility:public"])

sc_cc_library(
    name = "chunk_aware_strategy",
    srcs = ["chunk_aware_strategy.cc"],
    hdrs = ["chunk_aware_strategy.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:async_copy_manager_lib",
        "//core/common:async_runtime_lib",
        "//core/common:memory_location_lib",
        "//core/store:communication_types",
        "//core/store:device_registry",
        "//core/store:global_store_client",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/materialization/dataplane:disk_loader",
        "//core/store/replica:replica_load_controller",
        "//core/store/replica:unified_memory_authority",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)
