load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//core:BUILD.bzl", "sc_cc_library", "sc_header_only_library")

package(default_visibility = ["//visibility:public"])

sc_header_only_library(
    name = "safe_sys_hdr",
    hdrs = ["common/safe_sys.h"],
    deps = [
        "@abseil-cpp//absl/status",
    ],
)

sc_cc_library(
    name = "status_utils",
    hdrs = ["status_utils.h"],
)

sc_cc_library(
    name = "grpc_service_impl",
    srcs = ["grpc_service_impl.cc"],
    hdrs = ["grpc_service_impl.h"],
    copts = [],
    deps = [
        ":background_scheduler_hdr",
        ":cuda_ipc_raii_hdr",
        ":device_resolver_hdr",
        ":grpc_metrics_hdr",
        ":grpc_span_hdr",
        ":ipc_region_registry_lib",
        ":lip_bridge_lib",
        ":lip_manager_lib",
        ":materialization_controller_lib",
        ":ref_tracker_hdr",
        ":registration_controller_lib",
        ":registration_manager_lib",
        ":replica_listing_hdr",
        ":rpc_context_hdr",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        ":sessions_service_hdr",
        ":status_controller_hdr",
        ":status_utils",
        ":sweep_tasks_hdr",
        ":transport_controller_lib",
        ":transport_lock_manager_hdr",
        ":types_hdr",
        ":verification_tracker_hdr",
        "//core/common:artifact_hash_lib",
        "//core/common:otel_grpc_propagation_lib",
        "//core/store:global_store_client",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:segment_plan_source",
        "//core/store/materialization/dataplane:source_hash",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "replica_listing_hdr",
    hdrs = ["replica_listing.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":ref_tracker_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@nlohmann_json//:json",
    ],
)

sc_header_only_library(
    name = "verification_tracker_hdr",
    hdrs = ["verification_tracker.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
    ],
)

sc_cc_library(
    name = "registration_manager_lib",
    hdrs = ["registration_manager.h"],
    deps = [
        ":types_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_cc_library(
    name = "lip_manager_lib",
    srcs = [
        "lip_manager.cc",
        "lip_metadata_utils.cc",
    ],
    hdrs = [
        "lip_manager.h",
        "lip_metadata_utils.h",
    ],
    deps = [
        ":cuda_ipc_raii_hdr",
        ":ipc_region_registry_lib",
        ":types_hdr",
        "//core/common:artifact_hash_lib",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:canonical_index",
        "//core/store/materialization/dataplane:segment_plan_source",
        "//core/store/materialization/dataplane:source_hash",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
        "@nlohmann_json//:json",
    ],
)

sc_cc_library(
    name = "ipc_region_registry_lib",
    srcs = ["ipc_region_registry.cc"],
    hdrs = ["ipc_region_registry.h"],
    deps = [
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/random:random",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@abseil-cpp//absl/time:time",
    ],
)

cc_test(
    name = "ipc_region_registry_test",
    srcs = ["ipc_region_registry_test.cc"],
    deps = [
        ":ipc_region_registry_lib",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_binary(
    name = "tensorcast_daemon",
    srcs = ["server_main.cc"],
    deps = [
        ":grpc_service_impl",
        ":worker_lifecycle",
        "//core/common:daemon_config_io_lib",
        "//core/common:logging_init_lib",
        "//core/common:otel_init_lib",
        "//core/common:otel_log_sink_lib",
        "//core/common:trace_manager_lib",
        "//core/communicator:config_io_lib",
        "//proto/tensorcast/config/v1:config_cc",
        "@abseil-cpp//absl/flags:flag",
        "@abseil-cpp//absl/flags:parse",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@gsl",
    ],
)

cc_test(
    name = "lip_metadata_utils_test",
    srcs = ["lip_metadata_utils_test.cc"],
    deps = [
        ":lip_manager_lib",
        "@catch2//:catch2_main",
    ],
)

# Header-only alias for the session manager to avoid leaking absl deps to binaries unintentionally.
sc_header_only_library(
    name = "session_manager_hdr",
    hdrs = ["replica_session_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_header_only_library(
    name = "transport_lock_manager_hdr",
    hdrs = ["transport_lock_manager.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:loading_spec",
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
        "//core/communicator:utils_lib",
        "//core/store:store_engine",
        "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
        "@abseil-cpp//absl/strings:str_format",
        "@abseil-cpp//absl/time:time",
        "@gsl",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "ref_tracker_hdr",
    hdrs = ["ref_tracker.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/store/materialization/contracts:loading_spec",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/synchronization",
    ],
)

sc_header_only_library(
    name = "cuda_ipc_raii_hdr",
    hdrs = ["cuda_ipc_raii.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:cuda_api_lib",
        "@abseil-cpp//absl/status",
    ],
)

sc_header_only_library(
    name = "types_hdr",
    hdrs = ["types.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":cuda_ipc_raii_hdr",
        "//core/common:artifact_identity_lib",
    ],
)

sc_header_only_library(
    name = "background_scheduler_hdr",
    hdrs = ["background_scheduler.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@abseil-cpp//absl/base:core_headers",
    ],
)

sc_header_only_library(
    name = "grpc_span_hdr",
    hdrs = ["grpc_span.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//core/common:otel_grpc_propagation_lib",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "grpc_metrics_hdr",
    hdrs = ["grpc_metrics.h"],
    visibility = ["//visibility:public"],
    deps = [
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "sweep_tasks_hdr",
    hdrs = ["sweep_tasks.h"],
    deps = [
        ":background_scheduler_hdr",
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        ":transport_lock_manager_hdr",
        ":verification_tracker_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/synchronization",
        "@nlohmann_json//:json",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "session_lifecycle_hdr",
    hdrs = ["session_lifecycle.h"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":safe_sys_hdr",
        ":session_manager_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/strings",
        "@abseil-cpp//absl/time",
        "@opentelemetry-cpp//api:api",
    ],
)

# RFC-0016 helpers and controllers
sc_header_only_library(
    name = "rpc_context_hdr",
    hdrs = ["rpc_context.h"],
    deps = [
        ":grpc_metrics_hdr",
        ":grpc_span_hdr",
    ],
)

sc_header_only_library(
    name = "deadline_utils_hdr",
    hdrs = ["deadline_utils.h"],
    deps = [],
)

sc_header_only_library(
    name = "device_resolver_hdr",
    hdrs = ["device_resolver.h"],
    deps = [
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
    ],
)

sc_header_only_library(
    name = "sessions_service_hdr",
    hdrs = ["sessions_service.h"],
    deps = [
        ":background_scheduler_hdr",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        ":verification_tracker_hdr",
        "@abseil-cpp//absl/log",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_header_only_library(
    name = "status_controller_hdr",
    hdrs = [
        "service/controllers/status_assembler.h",
        "service/controllers/status_controller.h",
    ],
    deps = [
        ":ref_tracker_hdr",
        ":replica_listing_hdr",
        ":rpc_context_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_map",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "lip_bridge_lib",
    srcs = ["lip_bridge.cc"],
    hdrs = ["lip_bridge.h"],
    deps = [
        ":lip_manager_lib",
        # no direct StoreEngine dependency in header; uses loading_spec
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
    ],
)

sc_cc_library(
    name = "materialization_controller_lib",
    srcs = ["service/controllers/materialization_controller.cc"],
    hdrs = ["service/controllers/materialization_controller.h"],
    deps = [
        ":deadline_utils_hdr",
        ":device_resolver_hdr",
        ":lip_bridge_lib",
        ":ref_tracker_hdr",
        ":rpc_context_hdr",
        ":session_lifecycle_hdr",
        ":sessions_service_hdr",
        ":status_utils",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:view_planner",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/container:flat_hash_set",
        "@abseil-cpp//absl/strings",
        "@opentelemetry-cpp//api:api",
        "@protobuf",
    ],
)

sc_cc_library(
    name = "registration_controller_lib",
    srcs = ["service/controllers/registration_controller.cc"],
    hdrs = ["service/controllers/registration_controller.h"],
    deps = [
        ":cuda_ipc_raii_hdr",
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":rpc_context_hdr",
        ":session_lifecycle_hdr",
        ":status_utils",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:segment_plan_source",
        "//proto/tensorcast/common/v1:common_cc",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
        "@opentelemetry-cpp//api:api",
    ],
)

sc_cc_library(
    name = "transport_controller_lib",
    srcs = ["service/controllers/transport_controller.cc"],
    hdrs = ["service/controllers/transport_controller.h"],
    deps = [
        ":lip_manager_lib",
        ":rpc_context_hdr",
        ":status_utils",
        ":transport_lock_manager_hdr",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/strings",
    ],
)

cc_test(
    name = "session_lifecycle_test",
    srcs = ["session_lifecycle_test.cc"],
    copts = ["-DTC_ENABLE_TEST_HOOKS"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_immediate_unload_test",
    srcs = ["session_lifecycle_immediate_unload_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_pid_unwatch_test",
    srcs = ["session_lifecycle_pid_unwatch_test.cc"],
    copts = ["-DTC_ENABLE_TEST_HOOKS"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_session_guard_test",
    srcs = ["session_lifecycle_session_guard_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "pid_monitor_unwatch_integration_test",
    srcs = ["pid_monitor_unwatch_integration_test.cc"],
    copts = ["-DTC_ENABLE_TEST_HOOKS"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "eviction_task_behavior_test",
    srcs = ["eviction_task_behavior_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        ":sweep_tasks_hdr",
        "//core/store:store_engine",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_eviction_notify_test",
    srcs = ["session_lifecycle_eviction_notify_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        ":session_manager_hdr",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "session_lifecycle_load_test",
    srcs = ["session_lifecycle_load_test.cc"],
    deps = [
        ":lip_manager_lib",
        ":ref_tracker_hdr",
        ":registration_manager_lib",
        ":session_lifecycle_hdr",
        "//core/store:store_engine",
        "@abseil-cpp//absl/time:time",
        "@catch2//:catch2_main",
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
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
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
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@abseil-cpp//absl/log",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_keepalive_revoke_test",
    srcs = ["grpc_service_impl_keepalive_revoke_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_lease_copy_test",
    srcs = ["grpc_service_impl_lease_copy_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/store/materialization/dataplane:source_hash",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)

cc_test(
    name = "grpc_service_impl_lip_same_device_denial_test",
    srcs = ["grpc_service_impl_lip_same_device_denial_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_lip_ttl_expiry_test",
    srcs = ["grpc_service_impl_lip_ttl_expiry_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_parity_test",
    srcs = ["grpc_service_impl_parity_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_disk_index_test",
    srcs = ["grpc_service_impl_disk_index_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "resolve_artifact_from_disk_test",
    srcs = ["resolve_artifact_from_disk_test.cc"],
    deps = [
        ":grpc_service_impl",
        ":ipc_region_registry_lib",
        ":lip_manager_lib",
        ":materialization_controller_lib",
        "//core/common:artifact_hash_lib",
        "//core/store:store_engine",
        "//core/testing:common",
        "//proto/tensorcast/daemon/v2:daemon_grpc_cc",
        "@abseil-cpp//absl/time:time",
        "@abseil-cpp//absl/types:span",
        "@catch2//:catch2_main",
        "@nlohmann_json//:json",
    ],
)

cc_test(
    name = "grpc_service_impl_status_test",
    srcs = ["grpc_service_impl_status_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
        "@catch2//:catch2_main",
    ],
)

cc_test(
    name = "grpc_service_impl_status_multigpu_test",
    srcs = ["grpc_service_impl_status_multigpu_test.cc"],
    deps = [
        ":grpc_service_impl",
        "//core/store:store_engine",
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
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
        "//proto/tensorcast/daemon/v1:daemon_grpc_cc",
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
        "//proto/tensorcast/global_store/v1:global_store_grpc_cc",
        "//proto/tensorcast/memory_tier/v1:memory_tier_grpc_cc",
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/synchronization",
        "@catch2//:catch2_main",
        "@gsl",
    ],
)
