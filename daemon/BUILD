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
    deps = [
        ":metrics_exporter",
        ":ref_tracker_hdr",
        ":session_manager_hdr",
        ":status_utils",
        ":transport_lock_manager_hdr",
        "//core/store:store_engine",
        "//proto:store_daemon_grpc_cpp",
        "@abseil-cpp//absl/strings",
    ],
)

cc_binary(
    name = "tensorcast_daemon",
    srcs = ["server_main.cc"],
    deps = [
        ":grpc_service_impl",
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
    name = "metrics_exporter",
    srcs = ["metrics_exporter.cc"],
    hdrs = ["metrics_exporter.h"],
    deps = [
        "//core/store:store_engine",
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
