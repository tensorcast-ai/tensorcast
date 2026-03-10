
// Copyright (c) 2025-2026, TensorCast Team.

#include <utility>

#include "core/communicator/engine/store.h"

namespace tensorcast::communicator::engine {

PartitionTensorStore::PartitionTensorStore() = default;

transport::tensor_t PartitionTensorStore::get_tensor(std::string tensor_key) {
  return tensors_.get(std::move(tensor_key));
}

void PartitionTensorStore::register_tensor(const transport::tensor_t& t) {
  const std::string key = t->get_key();
  // Registration keys are reused across promotion cycles; replace stale entries
  // so byte-size/address metadata always reflects the newest export.
  (void)tensors_.erase_if_present(key);
  tensors_.put(key, t);
}

void PartitionTensorStore::unregister_tensor(std::string tensor_key) {
  tensors_.del(std::move(tensor_key));
}

void PartitionTensorStore::clear() {
  tensors_.clear();
}

} // namespace tensorcast::communicator::engine
