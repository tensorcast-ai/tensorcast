
// Copyright (c) 2025, TensorCast Team.

#include <utility>

#include "core/communicator/engine/store.h"

namespace tensorcast::communicator {

PartitionTensorStore::PartitionTensorStore() = default;

tensor_t PartitionTensorStore::get_tensor(std::string tensor_key) {
  return tensors_.get(std::move(tensor_key));
}

void PartitionTensorStore::register_tensor(const tensor_t& t) {
  tensors_.put(t->get_key(), t);
}

void PartitionTensorStore::unregister_tensor(std::string tensor_key) {
  tensors_.del(std::move(tensor_key));
}

void PartitionTensorStore::clear() {
  tensors_.clear();
}

} // namespace tensorcast::communicator
