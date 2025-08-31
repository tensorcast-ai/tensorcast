// Copyright (c) 2025, TensorCast Team.

#pragma once

// Build-time switch for enabling OpenTelemetry C++ API usage in C++ targets.
// This provides a single point to flip on/off OTel instrumentation in places
// where the environment/toolchain may not yet be fully wired.
//
// Override at compile via -DTC_ENABLE_OTEL_CXX=1 or Bazel copts.
#ifndef TC_ENABLE_OTEL_CXX
#define TC_ENABLE_OTEL_CXX 0
#endif
