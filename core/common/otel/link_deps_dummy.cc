// Copyright (c) 2025, TensorCast Team.

// Intentionally empty compilation unit used to aggregate linkable OpenTelemetry
// dependencies through a Bazel library that is marked alwayslink. This ensures
// that, when linked into a binary, any non-header OTel symbols are retained.

namespace tensorcast::obs {
void tc_otel_link_deps_keep() {}
} // namespace tensorcast::obs
