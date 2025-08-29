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
#include "pinned_memory.h"

#include "absl/log/log.h"
#include "pinned_memory_pool.h"

namespace tensorcast::store {

PinnedMemory::~PinnedMemory() {
  LOG(INFO) << "Deallocating " << buffers_.size() << " memory chunks";
  const int ret = mempool_->deallocate(buffers_);

  if (ret != 0) {
    LOG(ERROR) << "Error deallocating CPU memory";
  }
}

int PinnedMemory::allocate(
    size_t size,
    std::shared_ptr<PinnedMemoryPool> mempool,
    const std::chrono::milliseconds& timeout) {
  if (!buffers_.empty()) {
    LOG(ERROR) << "Memory already allocated";
    return 1;
  }

  mempool_ = std::move(mempool);
  int result = mempool_->allocate(size, buffers_, timeout);

  if (result == 0) {
    // Verify all buffers are aligned for DIRECT_IO
    for (size_t i = 0; i < buffers_.size(); ++i) {
      if (reinterpret_cast<uintptr_t>(buffers_[i]) % PinnedMemoryPool::kDirectIOAlignment != 0) {
        LOG(WARNING) << "PinnedMemory: Buffer " << i << " is not aligned for DIRECT_IO";
      }
    }
  }

  return result;
}

std::vector<char*>& PinnedMemory::get() {
  return buffers_;
}

} // namespace tensorcast::store
