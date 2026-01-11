load("@rules_cc//cc:defs.bzl", "cc_binary")

cc_binary(
    name = "safetensors_load_strategy_benchmark",
    srcs = [
        "safetensors_load_strategy_benchmark_main.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_memory_lib",
        "//core/common:logging_init_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/common:streaming_pinned_buffer_lib",
        "//core/cuda:cuda_api_lib",
        "//core/store/materialization/dataplane:file_partition_source",
        "//core/store/materialization/dataplane:gpu_memory_sink",
        "//core/store/materialization/dataplane:index_reader",
        "//core/store/materialization/dataplane:multi_safetensors_source",
        "//core/store/materialization/dataplane:pump",
        "//core/store/materialization/dataplane:safetensors_source",
        "//core/store/materialization/dataplane:streaming_buffer_adapter",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@nccl",
        "@nlohmann_json//:json",
    ],
)
