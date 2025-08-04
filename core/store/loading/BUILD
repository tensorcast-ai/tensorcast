load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "loading_spec",
    hdrs = ["loading_spec.h"],
    deps = [
        "//core/store:communication_types",
        "//core/store:device_types",
        "//core/store/model:memory_state",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "chunk_aware_loading_strategy",
    srcs = ["chunk_aware_loading_strategy.cc"],
    hdrs = ["chunk_aware_loading_strategy.h"],
    deps = [
        ":loading_spec",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:communication_types",
        "//core/store:global_store_client",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
        "//core/store/model:memory_manager",
        "//core/store/model:model_location",
        "//core/store/model:unified_model_memory",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings:str_format",
    ],
)
