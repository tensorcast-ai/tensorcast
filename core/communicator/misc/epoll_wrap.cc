// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_MISC_EPOLL_WRAP_H_
#define CORE_COMMUNICATOR_MISC_EPOLL_WRAP_H_

#include "misc/epoll_wrap.h"

namespace stepcast::communicator {

#ifdef __APPLE__

int wrap_epoll_create(int) {
  return 0;
}

int wrap_epoll_ctl(int, int, int, struct epoll_event*) {
  return 0;
}

int wrap_epoll_wait(int, struct epoll_event*, int maxevents, int timeout) {
  return 0;
}

#endif

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_MISC_EPOLL_WRAP_H_
