// Copyright (c) 2025, StepCast Team. All rights reserved.

#include "core/testing/test_helpers.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/log/log.h"

namespace stepcast::communicator::test {

std::vector<uint8_t> create_test_pattern(std::size_t size, uint8_t seed) {
  std::vector<uint8_t> data(size);
  for (std::size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>((i + seed) % 256);
  }
  return data;
}

bool verify_pattern(const void* data, std::size_t size, uint8_t seed) {
  const uint8_t* bytes = static_cast<const uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    if (bytes[i] != static_cast<uint8_t>((i + seed) % 256)) {
      LOG(ERROR) << "Mismatch at offset " << i << ": expected " << static_cast<int>((i + seed) % 256) << ", got "
                 << static_cast<int>(bytes[i]);
      return false;
    }
  }
  return true;
}

int find_available_port(int base_port, int max_attempts) {
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    int port = base_port + attempt;

    // Try to bind to the port
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
      LOG(ERROR) << "Failed to create socket";
      return -1;
    }

    // Allow reuse of address
    int optval = 1;
    setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    struct sockaddr_in addr;
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");
    addr.sin_port = htons(port);

    if (bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
      // Port is available
      close(sock);
      LOG(INFO) << "Found available port: " << port;
      return port;
    }

    close(sock);
  }

  LOG(ERROR) << "Failed to find available port after " << max_attempts << " attempts";
  return -1;
}

} // namespace stepcast::communicator::test
