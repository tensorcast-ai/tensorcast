
// Copyright (c) 2025, TensorCast Team.

extern "C" {
#include <pthread.h>
#include <pwd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
}

#include "misc/envs.h"

namespace tensorcast::communicator {

static void init_envs_func() {}

void init_envs() {
  static pthread_once_t once = PTHREAD_ONCE_INIT;
  pthread_once(&once, init_envs_func);
}

const char* get_env(const char* name, const char* default_val) {
  init_envs();
  auto env_val = getenv(name);
  if (env_val) {
    return env_val;
  }
  return default_val;
}

int get_env(const char* name, int default_val) {
  init_envs();
  auto env_val = getenv(name);
  if (env_val) {
    return strtoll(env_val, nullptr, 0);
  }
  return default_val;
}

} // namespace tensorcast::communicator
