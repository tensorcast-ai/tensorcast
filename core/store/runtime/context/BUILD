load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library")

sc_cc_library(
    name = "runtime_context_events",
    srcs = ["runtime_context_events.cc"],
    hdrs = ["runtime_context_events.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:device_types",
        "//core/store/materialization/contracts:loading_spec",
        "//core/store/runtime:ingestion_events",
        "@abseil-cpp//absl/functional:any_invocable",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time",
        "@folly//folly:mpmc_queue",
    ],
)

sc_cc_library(
    name = "runtime_context",
    srcs = ["runtime_context.cc"],
    hdrs = ["runtime_context.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":runtime_context_events",
        "//core/common:async_copy_manager_lib",
        "//core/common:host_memory_lib",
        "//core/common:pinned_buffer_pool_lib",
        "//core/store:communication_manager",
        "//core/store:device_manager",
        "//core/store:global_store_client",
        "//core/store:metrics_collector",
        "//core/store:replica_registry",
        "//core/store:stable_dram_cache_manager",
        "//core/store:store_engine_options",
        "//core/store:worker_identity",
        "//core/store/materialization/common:view_hash_utils",
        "//core/store/runtime/ingestion:ingestion_event_hub",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time",
    ],
)

cc_test(
    name = "runtime_context_test",
    srcs = ["runtime_context_test.cc"],
    deps = [
        ":runtime_context",
        "//core/store:memory_tier_budget",
        "//core/store:store_engine_options",
        "//core/testing:catch_main",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
        "@abseil-cpp//absl/synchronization",
        "@catch2//:catch2_main",
    ],
)
