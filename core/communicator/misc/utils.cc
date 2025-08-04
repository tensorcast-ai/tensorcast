
// Copyright (c) 2025, StepCast Team. All rights reserved.

extern "C" {
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <sys/fcntl.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
}

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <utility>

#include "core/communicator/misc/envs.h"
#include "core/communicator/misc/utils.h"

namespace stepcast::communicator {

#define MAX_IFS 32

std::string ip2str(uint32_t ip) {
  in_addr addr{};
  addr.s_addr = ip;
  std::string ip_str = inet_ntoa(addr);
  return ip_str;
}

std::string gid2str(uint8_t* gid) {
  std::stringstream ss;
  for (uint32_t i = 0; i < 16; i += 2) {
    if (i == 0) {
      ss << std::hex << std::setw(2) << std::setfill('0') << int(gid[i]) << std::setw(2) << std::setfill('0')
         << int(gid[i + 1]);
    } else {
      ss << ":" << std::hex << std::setw(2) << std::setfill('0') << int(gid[i]) << std::setw(2) << std::setfill('0')
         << int(gid[i + 1]);
    }
  }

  return ss.str();
}

ENV_PARAM_STR(IFNAME, "eth0");

uint64_t get_us() {
  auto now = std::chrono::system_clock::now();
  auto now_us = std::chrono::time_point_cast<std::chrono::microseconds>(now);
  return now_us.time_since_epoch().count();
}

std::string get_default_ip() {
  struct ifaddrs* if_addr = nullptr;
  if (getifaddrs(&if_addr) < 0) {
    LOG(ERROR) << "failed to get interface addresses.";
    return "";
  }

  std::string ip;

  for (auto ifa = if_addr; ifa != nullptr; ifa = ifa->ifa_next) {
    if (ifa->ifa_addr == nullptr)
      continue;
    if (ifa->ifa_addr->sa_family == AF_INET) {
      void* tmp = &((struct sockaddr_in*)ifa->ifa_addr)->sin_addr;
      char address[INET_ADDRSTRLEN];
      inet_ntop(AF_INET, tmp, address, INET_ADDRSTRLEN);
      if ((ifa->ifa_flags & IFF_UP) && IFNAME == ifa->ifa_name) {
        ip = address;
        break;
      }
    }
  }

  freeifaddrs(if_addr);
  return ip;
}

} // namespace stepcast::communicator
