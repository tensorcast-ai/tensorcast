// Copyright (c) 2025, TensorCast Team.

#include "daemon/metrics_exporter.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstring>
#include <sstream>

namespace tensorcast::daemon {

void MetricsExporter::start() {
  stop_.store(false);
  th_ = std::thread(&MetricsExporter::run, this);
}

void MetricsExporter::stop() {
  stop_.store(true);
  if (th_.joinable())
    th_.join();
}

std::string MetricsExporter::collect_metrics() {
  // Minimal set of metrics for now
  std::ostringstream os;
  os << "# HELP store_daemon_memory_pool_total_bytes Total memory pool bytes\n";
  os << "# TYPE store_daemon_memory_pool_total_bytes gauge\n";
  os << "store_daemon_memory_pool_total_bytes " << engine_->get_mem_pool_size() << "\n";
  os << "# HELP store_daemon_memory_pool_available_bytes Available memory bytes\n";
  os << "# TYPE store_daemon_memory_pool_available_bytes gauge\n";
  os << "store_daemon_memory_pool_available_bytes " << engine_->get_available_memory() << "\n";
  if (extra_metrics_provider_) {
    os << extra_metrics_provider_();
  }
  return os.str();
}

void MetricsExporter::run() {
  int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0)
    return;
  int opt = 1;
  setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port_);
  if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    close(server_fd);
    return;
  }
  listen(server_fd, 8);

  while (!stop_.load()) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(server_fd, &fds);
    timeval tv{.tv_sec = 1, .tv_usec = 0};
    int rv = select(server_fd + 1, &fds, nullptr, nullptr, &tv);
    if (rv <= 0)
      continue;
    int client = accept(server_fd, nullptr, nullptr);
    if (client < 0)
      continue;
    // Read minimal request
    char buf[1024];
    ssize_t n = ::recv(client, buf, sizeof(buf) - 1, 0);
    if (n < 0) {
      ::close(client);
      continue;
    }
    buf[n] = '\0';
    std::string req(buf);
    // Parse the first line: METHOD SP PATH SP HTTP/...
    std::string path = "/";
    size_t sp1 = req.find(' ');
    if (sp1 != std::string::npos) {
      size_t sp2 = req.find(' ', sp1 + 1);
      if (sp2 != std::string::npos) {
        path = req.substr(sp1 + 1, sp2 - (sp1 + 1));
      }
    }
    std::string body;
    std::ostringstream resp;
    if (path == "/health" || path == "/ready") {
      body = "ok\n";
      resp << "HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: " << body.size()
           << "\r\nConnection: close\r\n\r\n"
           << body;
    } else {
      body = collect_metrics();
      resp << "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: " << body.size()
           << "\r\nConnection: close\r\n\r\n"
           << body;
    }
    std::string out = resp.str();
    ::send(client, out.data(), out.size(), 0);
    ::close(client);
  }
  ::close(server_fd);
}

} // namespace tensorcast::daemon
