// Copyright (c) 2025, TensorCast Team.

//  ServerlessLLM
//  Copyright (c) ServerlessLLM Team 2024
//  Modified by TensorCast Team, 2025-2026.
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
#include "tensor_writer.h"

namespace tensorcast::checkpoint {

TensorWriter::TensorWriter(std::string filename) : filename_(std::move(filename)) {}

TensorWriter::~TensorWriter() = default;

uint64_t TensorWriter::write_record(const char* data, size_t size) {
  const size_t padded_size = TensorWriter::aligned_size(size);

  if (partition_idx_ == -1 || partition_size_ + padded_size > kPartitionMaxSize) {
    // create a new partition
    partition_idx_++;
    partition_size_ = 0;
    const std::string partition_filename = filename_ + "_" + std::to_string(partition_idx_);
    buffer_ = std::make_unique<AlignedBuffer>(partition_filename);
  }

  const uint64_t start_offset = offset_;

  // Write raw tensor bytes
  size_t written = buffer_->write_data(data, size);

  // Append up-to-7 bytes of padding if needed to maintain 8-byte alignment.
  if (padded_size != size) {
    const size_t pad_bytes = padded_size - size;
    written += buffer_->write_padding(pad_bytes);
  }
  offset_ += written;
  partition_size_ += written;

  return start_offset;
}

} // namespace tensorcast::checkpoint
