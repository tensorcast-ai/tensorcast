// Copyright (c) 2026, TensorCast Team.

#pragma once

#include <memory>
#include <string_view>

#include "daemon/state/daemon_options.h"
#include "daemon/state/routed_authority_protocol.h"
#include "grpcpp/channel.h"
#include "grpcpp/security/credentials.h"

namespace tensorcast::daemon {

[[nodiscard]] std::shared_ptr<grpc::ChannelCredentials> make_inter_daemon_channel_credentials(
    const DaemonOptions::InterDaemonGrpcSecurity& security);

[[nodiscard]] std::shared_ptr<grpc::Channel> create_inter_daemon_channel(
    std::string_view address,
    const std::shared_ptr<grpc::ChannelCredentials>& credentials);

[[nodiscard]] DaemonHopAuthClass inter_daemon_hop_auth_class(
    std::string_view address,
    const DaemonOptions::InterDaemonGrpcSecurity& security);

} // namespace tensorcast::daemon
