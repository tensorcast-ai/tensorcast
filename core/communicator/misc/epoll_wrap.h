// Copyright (c) 2025, StepCast Team. All rights reserved.

#ifndef CORE_COMMUNICATOR_MISC_EPOLL_WRAP_H_
#define CORE_COMMUNICATOR_MISC_EPOLL_WRAP_H_

#include <cstdint>

namespace stepcast::communicator {

#ifdef __APPLE__
enum EPOLL_EVENTS {
  EPOLLIN = 0x001,
#define EPOLLIN EPOLLIN
  EPOLLPRI = 0x002,
#define EPOLLPRI EPOLLPRI
  EPOLLOUT = 0x004,
#define EPOLLOUT EPOLLOUT
  EPOLLRDNORM = 0x040,
#define EPOLLRDNORM EPOLLRDNORM
  EPOLLRDBAND = 0x080,
#define EPOLLRDBAND EPOLLRDBAND
  EPOLLWRNORM = 0x100,
#define EPOLLWRNORM EPOLLWRNORM
  EPOLLWRBAND = 0x200,
#define EPOLLWRBAND EPOLLWRBAND
  EPOLLMSG = 0x400,
#define EPOLLMSG EPOLLMSG
  EPOLLERR = 0x008,
#define EPOLLERR EPOLLERR
  EPOLLHUP = 0x010,
#define EPOLLHUP EPOLLHUP
  EPOLLRDHUP = 0x2000,
#define EPOLLRDHUP EPOLLRDHUP
  EPOLLONESHOT = (1 << 30),
#define EPOLLONESHOT EPOLLONESHOT
  EPOLLET = (1 << 31)
#define EPOLLET EPOLLET
};

/* Valid opcodes ( "op" parameter ) to issue to epoll_ctl().  */
#define EPOLL_CTL_ADD 1 /* Add a file descriptor to the interface.  */
#define EPOLL_CTL_DEL 2 /* Remove a file descriptor from the interface. */
#define EPOLL_CTL_MOD 3 /* Change file descriptor epoll_event structure.  */

typedef union epoll_data {
  void* ptr;
  int fd;
  uint32_t u32;
  uint64_t u64;
} epoll_data_t;

struct epoll_event {
  uint32_t events; /* Epoll events */
  epoll_data_t data; /* User data variable */
} __attribute__((__packed__));

int wrap_epoll_create(int);
int wrap_epoll_ctl(int, int, int, struct epoll_event*);
int wrap_epoll_wait(int, struct epoll_event*, int maxevents, int timeout);

#else
extern "C" {
#include <stdint.h>
#include <sys/epoll.h>
#include <sys/types.h>
}

#define wrap_epoll_create epoll_create
#define wrap_epoll_ctl epoll_ctl
#define wrap_epoll_wait epoll_wait

#endif

} // namespace stepcast::communicator

#endif // CORE_COMMUNICATOR_MISC_EPOLL_WRAP_H_
