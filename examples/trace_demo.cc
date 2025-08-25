// Copyright (c) 2025, StepCast Team. All rights reserved.

// Example demonstrating the Chrome Trace functionality

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>

#include "core/common/trace/trace_macros.h"
#include "core/common/trace/trace_manager.h"

// Simulate some work
void simulate_work(int ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

int main() {
  // Set output directory for Chrome traces
  const char* trace_dir = std::getenv("SC_TRACE_OUTPUT_DIR");
  if (!trace_dir) {
    std::cout << "Hint: Set SC_TRACE_OUTPUT_DIR environment variable to save Chrome traces\n";
    std::cout << "Example: export SC_TRACE_OUTPUT_DIR=/tmp/traces\n\n";
  } else {
    std::cout << "Chrome traces will be saved to: " << trace_dir << "\n\n";
  }

  // Simulate loading a artifact
  const std::string artifact_id = "llama-7b";
  const std::string request_id = "demo_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());

  {
    // This guard will output summary and save Chrome trace on destruction
    SC_TRACE_INIT_GUARD(request_id, artifact_id, "load_model_demo");

    {
      SC_TRACE_SCOPE("allocate_memory");
      simulate_work(50);
    }

    {
      SC_TRACE_SCOPE("load_from_disk");
      simulate_work(200);
    }

    // Simulate parallel loading
    {
      SC_TRACE_SCOPE("parallel_load");

      auto task1 = std::async(std::launch::async, [&] {
        stepcast::store::TraceManager::RequestIdGuard guard(request_id);
        stepcast::store::TraceManager::ArtifactIdGuard artifact(artifact_id);
        SC_TRACE_SCOPE("load_partition_1");
        simulate_work(100);
      });

      auto task2 = std::async(std::launch::async, [&] {
        stepcast::store::TraceManager::RequestIdGuard guard(request_id);
        stepcast::store::TraceManager::ArtifactIdGuard artifact(artifact_id);
        SC_TRACE_SCOPE("load_partition_2");
        simulate_work(150);
      });

      task1.wait();
      task2.wait();
    }

    {
      SC_TRACE_SCOPE("finalize");
      simulate_work(30);
    }
  }

  std::cout << "\nDemo completed!\n";
  if (trace_dir) {
    std::cout << "Check " << trace_dir << "/" << artifact_id << "+" << request_id << ".json\n";
    std::cout << "Open chrome://tracing and load the JSON file to visualize\n";
  }

  return 0;
}