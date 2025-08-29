// Copyright (c) 2025, TensorCast Team.

#ifndef COMMUNICATOR_MISC_ENVS_H_
#define COMMUNICATOR_MISC_ENVS_H_

extern "C" {
#include <stdint.h>
}

namespace tensorcast::communicator {

void init_envs();

const char* get_env(const char* name, const char* default_val);

int get_env(const char* name, int default_val);

#define ENV_PARAM(ENV_NAME, DEFAULT) static auto ENV_NAME = get_env("TENSORCAST_COMM_" #ENV_NAME, DEFAULT)

#define ENV_PARAM_STR(ENV_NAME, DEFAULT) static std::string ENV_NAME = get_env("TENSORCAST_COMM_" #ENV_NAME, DEFAULT)

} // namespace tensorcast::communicator

#endif // COMMUNICATOR_MISC_ENVS_H_
