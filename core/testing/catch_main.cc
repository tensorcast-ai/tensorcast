
// Copyright (c) 2025, TensorCast Team.

#define CATCH_CONFIG_MAIN // This tells Catch to provide a main() - only do this in one cpp file

#include <catch2/catch_all.hpp>

#include "absl/debugging/failure_signal_handler.h"
#include "core/common/logging_init.h"

// Custom listener to initialize logging before tests
struct LoggingInitializer : Catch::EventListenerBase {
  using EventListenerBase::EventListenerBase; // inherit constructor

  void testRunStarting(Catch::TestRunInfo const& /*testRunInfo*/) override {
    static bool initialized = false;
    if (!initialized) {
      // Use ensure_logging_initialized which respects TENSORCAST_VLOG_LEVEL env var
      tensorcast::store::ensure_logging_initialized();
      absl::FailureSignalHandlerOptions options;
      absl::InstallFailureSignalHandler(options);
      initialized = true;
    }
  }
};

CATCH_REGISTER_LISTENER(LoggingInitializer)