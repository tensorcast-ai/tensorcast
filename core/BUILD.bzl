"""Helper macros for TensorCast BUILD files.

This file provides minimal abstractions to reduce repetition while keeping
the build system simple and natural to read.
"""

load("@rules_cc//cc:defs.bzl", "cc_library")

def sc_header_only_library(
        name,
        hdrs,
        deps = None,
        visibility = None,
        **kwargs):
    """Creates a header-only library target.

    This is a simple wrapper for header-only libraries that don't have source files.
    It reduces boilerplate for the many header-only targets in the codebase.

    Args:
        name: Target name
        hdrs: List of header files
        deps: Optional dependencies
        visibility: Optional visibility, defaults to private
        **kwargs: Additional arguments passed to cc_library
    """
    cc_library(
        name = name,
        hdrs = hdrs,
        deps = deps or [],
        visibility = visibility or ["//visibility:private"],
        **kwargs
    )

def sc_cc_library(
        name,
        srcs = None,
        hdrs = None,
        deps = None,
        visibility = None,
        alwayslink = None,
        **kwargs):
    """Creates a cc_library with common defaults and standard dependencies.

    This wrapper automatically includes commonly used Abseil libraries
    (absl/log, absl/status, absl/status:statusor) that are used throughout
    the codebase.

    Args:
        name: Target name
        srcs: Optional source files
        hdrs: Optional header files
        deps: Optional dependencies (standard deps are added automatically)
        visibility: Optional visibility, defaults to private
        alwayslink: Optional alwayslink flag, defaults to True if srcs are provided
        **kwargs: Additional arguments passed to cc_library
    """

    # Standard dependencies that almost all libraries need
    standard_deps = [
        "@abseil-cpp//absl/log",
        "@abseil-cpp//absl/log:absl_check",
        "@abseil-cpp//absl/status",
        "@abseil-cpp//absl/status:statusor",
    ]

    # Merge provided deps with standard deps
    all_deps = standard_deps + (deps or [])

    # Default alwayslink to True if sources are provided (to ensure symbols are included in shared libraries)
    if alwayslink == None:
        alwayslink = bool(srcs)

    cc_library(
        name = name,
        srcs = srcs or [],
        hdrs = hdrs or [],
        deps = all_deps,
        visibility = visibility or ["//visibility:private"],
        alwayslink = alwayslink,
        **kwargs
    )
