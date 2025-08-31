load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_cc_library(
    name = "status_utils",
    hdrs = ["status_utils.h"],
)

sc_cc_library(
    name = "grpc_service_impl",
    srcs = ["grpc_service_impl.cc"],
    hdrs = ["grpc_service_impl.h"],
    copts = [
        # Enable OTel C++ API instrumentation in RPC handlers
        "-DTC_ENABLE_OTEL_CXX=1",
    ],
    deps = [
        ":ref_tracker_hdr",
        ":session_manager_hdr",
        ":status_utils",
        ":transport_lock_manager_hdr",
        "//core/common:otel_config_lib",
        "//core/common:otel_grpc_propagation_lib",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

cc_binary(
    name = "tensorcast_daemon",
    srcs = ["server_main.cc"],
    deps = [
        ":grpc_service_impl",
        ":worker_lifecycle",
        "//core/common:otel_init_lib",
        "//core/common:otel_link_deps_lib",
        "//core/common:otel_log_sink_lib",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
    ],
)

# Header-only alias for the session manager to avoid leaking absl deps to binaries unintentionally.
sc_header_only_library(
    name = "session_manager_hdr",
    hdrs = ["replica_session_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_header_only_library(
    name = "transport_lock_manager_hdr",
    hdrs = ["transport_lock_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "worker_lifecycle",
    srcs = ["worker_lifecycle_manager.cc"],
    hdrs = ["worker_lifecycle_manager.h"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto:global_store_grpc_cpp",
    ],
)

sc_header_only_library(
    name = "ref_tracker_hdr",
    hdrs = ["ref_tracker.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

cc_test(
    name = "transport_lock_manager_test",
    srcs = ["transport_lock_manager_test.cc"],
    deps = [
        ":transport_lock_manager_hdr",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "status_utils_test",
    srcs = ["status_utils_test.cc"],
    deps = [
        ":status_utils",
        "//proto:store_daemon_grpc_cpp",
        "@abseil-cpp//absl/status",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_registration_test",
    srcs = ["grpc_service_impl_registration_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@abseil-cpp//absl/log",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_parity_test",
    srcs = ["grpc_service_impl_parity_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_test",
    srcs = ["grpc_service_impl_status_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_multigpu_test",
    srcs = ["grpc_service_impl_status_multigpu_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_mixed_residency_test",
    srcs = ["grpc_service_impl_status_mixed_residency_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/testing:common",
        "//proto:store_daemon_grpc_cpp",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_compat_test",
    srcs = ["grpc_service_impl_compat_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "worker_lifecycle_manager_sync_test",
    srcs = ["worker_lifecycle_manager_sync_test.cc"],
    deps = [
        ":grpc_service_impl",
        ":worker_lifecycle",
        "//core/store:store_engine",
        "//core/testing:common",
        "//proto:global_store_grpc_cpp",
        "@abseil-cpp//absl/log",
        "@catch2//:catch2_main",
    ],
)
