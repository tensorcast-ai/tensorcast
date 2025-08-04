load("@rules_cc//cc:defs.bzl", "cc_library")

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "chunk_meta",
    hdrs = ["chunk_meta.h"],
    deps = [],
)

cc_library(
    name = "model_location",
    hdrs = ["model_location.h"],
    deps = [],
)

cc_library(
    name = "memory_state",
    hdrs = ["memory_state.h"],
    deps = [],
)

cc_library(
    name = "unified_model_memory",
    srcs = ["unified_model_memory.cc"],
    hdrs = ["unified_model_memory.h"],
    deps = [
        ":chunk_meta",
        ":model_location",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:device_types",
        "//core/store/loading:loading_spec",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
    ],
)

cc_library(
    name = "memory_manager",
    srcs = ["memory_manager.cc"],
    hdrs = ["memory_manager.h"],
    deps = [
        ":chunk_meta",
        ":memory_state",
        ":model_location",
        ":unified_model_memory",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/communicator",
        "//core/store:device_types",
        "//core/store:memory_types",
        "//core/store/loading:loading_spec",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_library(
    name = "model",
    srcs = [
        "model.cc",
    ],
    hdrs = [
        "chunk_directory.h",
        "model.h",
        "model_config.h",
    ],
    includes = [
    ],
    deps = [
        ":chunk_meta",
        ":memory_manager",
        ":memory_state",
        ":model_location",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/communicator",
        "//core/store:communication_types",
        "//core/store:device_types",
        "//core/store:memory_types",
        "//core/store/loader",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
        "//core/store/loading:loading_spec",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/functional:overload",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/types:span",
    ],
)
