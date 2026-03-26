// Copyright (c) 2026, TensorCast Team.

#include "daemon/util/grpc_daemon_transport.h"

#include <string>

#include "daemon/util/grpc_peer_utils.h"
#include "grpcpp/create_channel.h"
#include "grpcpp/security/credentials.h"
#include "grpcpp/support/channel_arguments.h"

namespace tensorcast::daemon {
namespace {

constexpr int kInterDaemonGrpcMaxMessageLength = 64 * 1024 * 1024;

} // namespace

std::shared_ptr<grpc::ChannelCredentials> make_inter_daemon_channel_credentials(
    const DaemonOptions::InterDaemonGrpcSecurity& security) {
  if (!security.tls_enabled) {
    return grpc::InsecureChannelCredentials();
  }
  grpc::SslCredentialsOptions ssl_options;
  ssl_options.pem_root_certs = security.root_cert_pem;
  ssl_options.pem_private_key = security.private_key_pem;
  ssl_options.pem_cert_chain = security.cert_chain_pem;
  return grpc::SslCredentials(ssl_options);
}

std::shared_ptr<grpc::Channel> create_inter_daemon_channel(
    std::string_view address,
    const std::shared_ptr<grpc::ChannelCredentials>& credentials) {
  grpc::ChannelArguments args;
  // Keep inter-daemon client limits aligned with daemon server defaults so
  // payload transport chunk sizing is not silently capped by gRPC defaults.
  args.SetMaxSendMessageSize(kInterDaemonGrpcMaxMessageLength);
  args.SetMaxReceiveMessageSize(kInterDaemonGrpcMaxMessageLength);
  return grpc::CreateCustomChannel(
      std::string(address), credentials != nullptr ? credentials : grpc::InsecureChannelCredentials(), args);
}

DaemonHopAuthClass inter_daemon_hop_auth_class(
    std::string_view address,
    const DaemonOptions::InterDaemonGrpcSecurity& security) {
  if (security.mutual_auth_enabled) {
    return DaemonHopAuthClass::kDaemonMutualAuth;
  }
  if (grpc_peer_matches_address("ipv4:127.0.0.1:0", address) || grpc_peer_matches_address("ipv6:[::1]:0", address)) {
    return DaemonHopAuthClass::kDeploymentTrustedChannel;
  }
  if (address.rfind("127.", 0) == 0 || address.rfind("localhost:", 0) == 0 || address.rfind("[::1]:", 0) == 0) {
    return DaemonHopAuthClass::kDeploymentTrustedChannel;
  }
  return DaemonHopAuthClass::kLegacyUnauthenticated;
}

} // namespace tensorcast::daemon
