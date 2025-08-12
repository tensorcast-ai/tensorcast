"""
Module extension for CUDA.
"""

load("@bazel_tools//tools/build_defs/repo:local.bzl", "new_local_repository")

def _impl(module_ctx):
    env = module_ctx.os.environ
    cuda_path = env.get("CUDA_PATH") or env.get("CUDA_HOME")
    if not cuda_path:
        fail(
            "CUDA path not found. Please set CUDA_PATH or CUDA_HOME, or pass via --repo_env=CUDA_PATH=/path or --repo_env=CUDA_HOME=/path.",
        )

    new_local_repository(
        name = "cuda",
        path = cuda_path,
        build_file = "@//third_party/cuda:BUILD",
    )

cuda_env = module_extension(
    implementation = _impl,
    environ = ["CUDA_PATH", "CUDA_HOME"],
)
