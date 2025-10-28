load("@rules_cc//cc:defs.bzl", "cc_binary")

package(default_visibility = ["//visibility:public"])

# Benchmark 配置设置：用于高性能基准测试
config_setting(
    name = "benchmark_mode",
    define_values = {"benchmark_mode": "true"},
)

cc_binary(
    name = "data_chunk_benchmark",
    srcs = ["data_chunk_benchmark.cc"],
    # Benchmark 特定的构建选项
    copts = [
        "-O3",  # 最高优化级别
        "-march=native",  # 针对本地架构优化
        "-mtune=native",  # 针对本地 CPU 调优
    ],
    linkopts = [
        "-lrt",  # 链接 clock_gettime 等函数
    ],
    deps = [
        "//core/local/chunk:chunk_lib",
        "//core/local/loader:disk_chunk_loader",
        "@abseil-cpp//absl/log:absl_log",
        "@abseil-cpp//absl/log:initialize",
        "@abseil-cpp//absl/time:time",
    ],
)
