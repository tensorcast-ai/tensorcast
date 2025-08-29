// Copyright (c) 2025, TensorCast Team.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//
//   You may obtain a copy of the License at
//
//                   http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
//  ----------------------------------------------------------------------------
#pragma once

#include <memory>
#include <string>

#include "aligned_buffer.h"
namespace tensorcast::store {

[[maybe_unused]] constexpr size_t kPartitionMaxSize = 10L << 30; // 10GB

// A tensor writer that writes the raw tensor data to a file in raw binary.
class TensorWriter final {
 public:
  explicit TensorWriter(std::string filename);
  ~TensorWriter();

  uint64_t write_record(const char* data, size_t size);

  /**
   * @brief Calculate the byte size of a record after applying 8-byte alignment padding.
   *
   * This helper is shared by both the producer (StreamingTensorWriter) and the
   * consumer (TensorWriter) so that the two components use a single definition
   * for alignment logic.
   *
   * @param size Raw data size in bytes (without padding)
   * @return Size after padding so that the total is a multiple of 8 bytes.
   */
  static constexpr size_t aligned_size(size_t size) {
    const size_t remainder = size & 7ULL; // Equivalent to size % 8
    return remainder ? (size + (8ULL - remainder)) : size;
  }

 private:
  size_t offset_ = 0;
  int partition_idx_ = -1;
  size_t partition_size_ = 0;
  std::string filename_;
  std::unique_ptr<AlignedBuffer> buffer_;
};

} // namespace tensorcast::store
