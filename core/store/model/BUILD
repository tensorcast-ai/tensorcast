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
    name = "model_memory_coordinator",
    srcs = ["model_memory_coordinator.cc"],
    hdrs = ["model_memory_coordinator.h"],
    deps = [
        ":chunk_meta",
        ":model_location",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:device_registry",
        "//core/store:device_types",
        "//core/store:direct_write",
        "//core/store:loading_spec",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

cc_library(
    name = "transfer_helpers",
    srcs = ["transfer_helpers.cc"],
    hdrs = ["transfer_helpers.h"],
    deps = [
        ":chunk_meta",
        ":model_location",
        ":model_memory_coordinator",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:device_types",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
    ],
)

cc_library(
    name = "transfer_service",
    srcs = ["transfer_service.cc"],
    hdrs = ["transfer_service.h"],
    deps = [
        ":model_memory_coordinator",
        ":transfer_helpers",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/store:device_types",
        "//core/store:loading_spec",
        "//core/store/loader:dvmp_region_sink",
        "//core/store/loader:gpu_memory_sink",
        "//core/store/loader:pump",
        "//core/store/loader:streaming_buffer_adapter",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

cc_library(
    name = "chunk_export_service",
    srcs = ["chunk_export_service.cc"],
    hdrs = ["chunk_export_service.h"],
    deps = [
        ":model_memory_coordinator",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/communicator",
        "//core/store:communication_types",
        "@abseil-cpp//absl/hash",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/types:span",
        "@gsl",
    ],
)

cc_library(
    name = "memory_manager",
    srcs = ["memory_manager.cc"],
    hdrs = ["memory_manager.h"],
    deps = [
        ":chunk_export_service",
        ":chunk_meta",
        ":memory_state",
        ":model_location",
        ":model_memory_coordinator",
        ":transfer_helpers",
        ":transfer_service",
        "//core/common",
        "//core/common:distributed_memory_pool",
        "//core/communicator",
        "//core/store:device_types",
        "//core/store:direct_write",
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader:dvmp_region_sink",
        "//core/store/loader:gpu_memory_sink",
        "//core/store/loader:pump",
        "//core/store/loader:streaming_buffer_adapter",
        "@abseil-cpp//absl/base:core_headers",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@gsl",
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
        "//core/store:loading_spec",
        "//core/store:memory_types",
        "//core/store/loader",
        "//core/store/loader:disk_loader",
        "//core/store/loader:p2p_loader",
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
