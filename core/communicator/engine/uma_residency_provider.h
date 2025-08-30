// Copyright (c) 2025, TensorCast Team.

#pragma once

#include <memory>
#include <string>

#include "core/communicator/engine/engine.h"
#include "core/store/components/uma_lease_provider.h"
#include "gsl/pointers"

namespace tensorcast::communicator {

class UmaResidencyProvider : public CommunicateEngine::ResidencyProvider {
 public:
  UmaResidencyProvider()
      : uma_(gsl::not_null<std::shared_ptr<tensorcast::store::UmaLeaseProvider>>{
            tensorcast::store::UmaLeaseProvider::instance()}) {}
  bool is_hot(const std::string& tensor_key, uint64_t offset, uint64_t bytes) override {
    return uma_->is_range_hot(tensor_key, offset, bytes);
  }

 private:
  gsl::not_null<std::shared_ptr<tensorcast::store::UmaLeaseProvider>> uma_;
};

} // namespace tensorcast::communicator

