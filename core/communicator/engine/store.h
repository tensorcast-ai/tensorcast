// Copyright (c) 2025, TensorCast Team.

#ifndef CORE_COMMUNICATOR_ENGINE_STORE_H_
#define CORE_COMMUNICATOR_ENGINE_STORE_H_

#include <string>
#include <vector>

#include "core/communicator/misc/map.h"
#include "core/communicator/transport/partition_tensor.h"

namespace tensorcast::communicator {

class PartitionTensorStore {
 public:
  PartitionTensorStore();

  tensor_t get_tensor(std::string tensor_key);

  void register_tensor(const tensor_t& t);

  void unregister_tensor(std::string tensor_key);

  void clear();

 private:
  Map<std::string, tensor_t> tensors_;
};

} // namespace tensorcast::communicator

#endif // CORE_COMMUNICATOR_ENGINE_STORE_H_
