// Copyright (c) 2026, TensorCast Team.

#include "daemon/util/grpc_daemon_transport.h"

#include <chrono>
#include <memory>
#include <optional>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "daemon/state/distributed_security_kernel.h"
#include "grpcpp/client_context.h"
#include "grpcpp/security/server_credentials.h"
#include "grpcpp/server.h"
#include "grpcpp/server_builder.h"
#include "tensorcast/daemon/v2/store_daemon.grpc.pb.h"

namespace tensorcast::daemon {
namespace {

constexpr std::size_t kLargePayloadBytes = (4U << 20U) + 1024U;

constexpr char kCaCertPem[] = R"(-----BEGIN CERTIFICATE-----
MIIDGzCCAgOgAwIBAgIUXeqZ4PVXCLRPgyJkup3nWR2HNHowDQYJKoZIhvcNAQEL
BQAwHTEbMBkGA1UEAwwSdGVuc29yY2FzdC10ZXN0LWNhMB4XDTI2MDMxMTE1Mjk1
M1oXDTM2MDMwODE1Mjk1M1owHTEbMBkGA1UEAwwSdGVuc29yY2FzdC10ZXN0LWNh
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwRttmQv8saHytKSwaJ2S
N/QVyEE5QPkP0djcwSIdQNyeI0qHnZ5aDXdAmb/6mQon+JnIzc+IJVdf9cURG6/z
RUfNE+AB5HeYHgZ+zwrAcCTF172vsJ/1YTPfrdnLyinazTi+wuUR3ge7reJlpvK4
EvObEzJpGmtegnOgNIE3RRDbqfTm2T/6Vj+CqsWkRR3cfxOq56DZL1ti0p3keT9R
NPvAs+823jSDdvE5feAtT0rOiVlU8mNfLKvgx6EgdI1oNIYrK3m6SpjNkUyuQTh4
3pgh8pweEHSNzDtZ3ubmZI5qNEYfUqiXnZjVhfG4FJBzjimCLUX2M/HU1y28rNrt
IwIDAQABo1MwUTAdBgNVHQ4EFgQUE5j5UwHwin+y8rRSE8xbc/Z/qAwwHwYDVR0j
BBgwFoAUE5j5UwHwin+y8rRSE8xbc/Z/qAwwDwYDVR0TAQH/BAUwAwEB/zANBgkq
hkiG9w0BAQsFAAOCAQEAM8221QmyCLBRvGIHBSGK5ElIScfo6qShHD9v6lzwkB0A
aXD944f6vGs9BDh5fSdVTD349gTmO/Ne4OPLPbQ+2g4Bbkkqj4n/xbrYqMNr+rNo
uMq1LCQMmxaFDtTT9pHmZCK4e57NjkZPFCsYWvU++1IEDvHyv+nCQT0Na0m1rZ9H
JnXZKqCxKd1G2365Jp/ZZBK9rqP2yXT6+qykhItev9bU8pULW5WlwSnBLnu5+ALP
xEkZ5OAwTNZnfQz6FvG3jzyCDJ/HfuIy7cMe+VnVu38sIbBQAAagP2B+7L7D4mVL
6/pPn09UZSiQ1JibAohqsOkZHotfWN1PTzhXQkschA==
-----END CERTIFICATE-----
)";

constexpr char kServerCertPem[] = R"(-----BEGIN CERTIFICATE-----
MIIDWzCCAkOgAwIBAgIUEFdmw9OD6e/uTbgiEmC97DQJDtkwDQYJKoZIhvcNAQEL
BQAwHTEbMBkGA1UEAwwSdGVuc29yY2FzdC10ZXN0LWNhMB4XDTI2MDMxMTE1Mjk1
NFoXDTM2MDMwODE1Mjk1NFowEzERMA8GA1UEAwwIZGFlbW9uLWEwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQCWlKnn+BZnm0LjYhPmppT4EV7SLW/X9JWZ
g2rcu54DJO3Qpun2cjM8vdR7kh7oNHmkLgQShPpJPXga2z3MemlMWt0kxQEd04Ii
OsAhPaooVOJvw8q8XfMiVqJOzhlxyl4bxX7R1dcJ+UonHdrF71bJdQEtV9KRQTgr
0IOHCazu53h5v9jRlrXGCCthTU680rbhsw1wQ6lzTRk29nNQ9P4nW/nHRVGhpegy
uAsD/QrnmMWhsIXo4BuUlDVdyAvB12n3vyj14GKAbbagXndODNOij39ioeeg6LN2
pMu9ybJ3Kjd/RQxeNfjeCuSBWAMMHTIukAkHzSS5z34MfQPcOxyTAgMBAAGjgZww
gZkwQgYDVR0RBDswOYIJbG9jYWxob3N0hwR/AAABhiZzcGlmZmU6Ly90ZW5zb3Jj
YXN0L2F1dGhvcml0eS9kYWVtb24tYTATBgNVHSUEDDAKBggrBgEFBQcDATAdBgNV
HQ4EFgQUsCcZvZegHYPo+2mkslo0gkVj1P4wHwYDVR0jBBgwFoAUE5j5UwHwin+y
8rRSE8xbc/Z/qAwwDQYJKoZIhvcNAQELBQADggEBAF2V5K9cFzwqAp5uwGpFgrRV
XFRN66AZivpH3Y1brZSfDP4d2A4x5I4nUhuWAVqLRQPl0Q5VL8WKsEQz+eQI4uKg
6Cpr/EsK75BimfYPJ5sePOz36Ue1LuoZV3hfgCY6BHBVUAK/gj2Fvy7OLqg/NacW
35KP8qG/JAlA8W9sVKtGQqqJC/ntnEvLAzwAxfZbZHXnh9tqlm1kl31gk6FF5HaS
CGS9FfEAsuF5e15fRx47yP24uTMturhsKJl99I/5ObAFpIJzas2lP16aCFNq2//d
VDBbm4F2X9eIuo5Dgdz9/U/CmjEuRBU6mKRX4IIkPAaqRr2tZXXfLxqx2Lz7nhc=
-----END CERTIFICATE-----
)";

constexpr char kServerKeyPem[] = R"(-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCWlKnn+BZnm0Lj
YhPmppT4EV7SLW/X9JWZg2rcu54DJO3Qpun2cjM8vdR7kh7oNHmkLgQShPpJPXga
2z3MemlMWt0kxQEd04IiOsAhPaooVOJvw8q8XfMiVqJOzhlxyl4bxX7R1dcJ+Uon
HdrF71bJdQEtV9KRQTgr0IOHCazu53h5v9jRlrXGCCthTU680rbhsw1wQ6lzTRk2
9nNQ9P4nW/nHRVGhpegyuAsD/QrnmMWhsIXo4BuUlDVdyAvB12n3vyj14GKAbbag
XndODNOij39ioeeg6LN2pMu9ybJ3Kjd/RQxeNfjeCuSBWAMMHTIukAkHzSS5z34M
fQPcOxyTAgMBAAECggEADxGwkSjIITUt8/BoOnBYPx0J5rGqXCu7XuICazWRYhIr
hHJ3vnR+QSlkHruoQSejNrVuLeZF39d5cgKXxCNJ71y2fcwDSRdWEhPcD1eK9D9O
a83lt4UVZE3UL1u7uyH72f6hKuGiA/sBGzoHH0DM6Bs/82gzB9tFgkj7KKPkvQGg
uLnTR8JzLVOGyRIh/4cO1I+fwcAZak9n6HyI2pZojBexm2t3WIlKRgBNv3QJOpnK
4FnLlCWpHlDz3oNJRcNe4aEGVQ3n2qcvd7JuwKM3N9qMHdoptl0Nc8D68avl0Txb
hq9kYIeyjYeVBvPiwSnPTmiswAgWCY4JujLDRCYSQQKBgQDO0Msj0sx/fT29NKmL
Z2ZG9Smuglo/cXWiYN2fCzSx4UYnIe7BfBI9jkXa83Fgns7pVkoOo+5i8cYp5dEt
RMhzh2S6bMjkZHpmPRogFvNxjjZzOMwP+VR7OwGMWkBhUfamFrO/sXJqJadeIXtA
5dilCtzXCvMYrcB0MLkJjFOdQQKBgQC6ZDdlKzhn04vWCclAd95IjeHxzA5VSLav
7T74kZUpLsre/tay9uM+UESHUEagrUJqcsI3ab5I03AGnD1fI/CWzqlR0swBeTgR
BZJFzbrenNFFcuARDVQYWa0sgqHyivlW71VSiGvEsclwTL1KGcFNkK6GSqbLKxFZ
mCQFSLCA0wKBgD42LIUwKffHssSxjLa0ed2qbcliyMcA2EPqI4BuHIHNpA6tdGab
bdk6bOT6CgbofpFONaTFxzXYSKXdzdhyMmIePjyd8KhTWUZb5vn0LXLhNpveX/QG
KlWPYF3Z4DfmMe2wMo3dUO+BOCFmrO7OtSagZ+IAFzQ7QAMzjNXEcnqBAoGAbkjH
whONn1k7HxzROFT0b+eLbd84B9wrQ/LTVI4HPMPUA5ezf1a3ZHJn6cxkzcNWTdey
9Mibb0mzuiWN5pJEU2baJnDchm6r7t5mCulMjVUqObZbv5ssS/Abl/9/RRxXpcmc
qWXQpiQvtUOiqIqOpo9ZxFm9UO8dp+5uObYlwUMCgYBrcnaxHtT/J8Xc5a9Iw0XK
WBtnWt0VuFHSnBUzgPvMy4JjpnBwvYnBgikCQYjOMZy62eJQovbNgizOagGveTyd
BKlhK4eEk65c59hmvlPXZgMBnIhg8wCLVuO2Qe4qbfvTrz0sVskKbRnbvYqdBecN
YQvO02LJ6tHDUNajIMA06Q==
-----END PRIVATE KEY-----
)";

constexpr char kClientCertPem[] = R"(-----BEGIN CERTIFICATE-----
MIIDTTCCAjWgAwIBAgIUXG7fl7p4i8/hGhyy7I3FNxDAUJcwDQYJKoZIhvcNAQEL
BQAwHTEbMBkGA1UEAwwSdGVuc29yY2FzdC10ZXN0LWNhMB4XDTI2MDMxMTE1Mjk1
NVoXDTM2MDMwODE1Mjk1NVowEzERMA8GA1UEAwwIY2xpZW50LWEwggEiMA0GCSqG
SIb3DQEBAQUAA4IBDwAwggEKAoIBAQCA7hJCxMFatZTDuigUmY0yMBPPbGrRGczs
sG8ELbQIRcMIleCGToaZgQTf0s1RLOEwSmyH3NOwgQhkfAo3gVINQUezFMcOK+HF
hekKrRg/naGWMc09k6iaHLHXeD979vSuXLdxm/KXM7kEJA4gHwKUfbIwGKVAmk5i
usn6+ZEeS7I51CjCHDNYnQbOUxz+fsWhcOzrcRKxJCjdHjHtsqaG1Nu7c6m2AuY2
8CXzJu5q7G8/jejb9F4dBf2cXw9NBhinG59Cq4tWMGtMBMnoEhbU3PcnzkqeN7M+
ZAAa+7JWms52RsF2Cj2Vk15Dccz1lZQNG2t3NB9cegsQz1jo7GD1AgMBAAGjgY4w
gYswNAYDVR0RBC0wK4Ypc3BpZmZlOi8vdGVuc29yY2FzdC9hdXRob3JpdHkvYXV0
aG9yaXR5LWEwEwYDVR0lBAwwCgYIKwYBBQUHAwIwHQYDVR0OBBYEFKC2laUh5wMc
1+1JFbI//EtIKb2VMB8GA1UdIwQYMBaAFBOY+VMB8Ip/svK0UhPMW3P2f6gMMA0G
CSqGSIb3DQEBCwUAA4IBAQCN40Bvp1nGmAj1LWSJtQ5JuLk7eVXXxv6cf92YLHB5
RLUVV3YZe1D0ohUxkKfxqd6nP/pHOSrZN/7Zr005MsJPOb7VhaV5lx6NOpHlUmoL
bOIdG4Gz1ZiCPx+mTFvrViCbffHoMPUrGOhPRa0VDsPpwXriHC1dS+vsHSGDT+nI
wkRijoTC1Hl9Q7y7B5nsfGqbL3D1NjfQ/sNmp8Gy6izkYHXcat2lwZ2uLl3GQp/E
htcvfIghE0M24a2zDguG4cpDGT5eyS5+dn8EhxrVWbBu6chhN5KMT2k+MF2ay+GN
sXJfyCtj/IPVZRIhkkKgvdsPDub4meASgciapgpOKU64
-----END CERTIFICATE-----
)";

constexpr char kClientKeyPem[] = R"(-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCA7hJCxMFatZTD
uigUmY0yMBPPbGrRGczssG8ELbQIRcMIleCGToaZgQTf0s1RLOEwSmyH3NOwgQhk
fAo3gVINQUezFMcOK+HFhekKrRg/naGWMc09k6iaHLHXeD979vSuXLdxm/KXM7kE
JA4gHwKUfbIwGKVAmk5iusn6+ZEeS7I51CjCHDNYnQbOUxz+fsWhcOzrcRKxJCjd
HjHtsqaG1Nu7c6m2AuY28CXzJu5q7G8/jejb9F4dBf2cXw9NBhinG59Cq4tWMGtM
BMnoEhbU3PcnzkqeN7M+ZAAa+7JWms52RsF2Cj2Vk15Dccz1lZQNG2t3NB9cegsQ
z1jo7GD1AgMBAAECggEAHmemdEsAuzLTuqW5woPktad7YID1nsq6Fj5Ua/SYPpQv
FqT7vkih+uzfeKY2p7RRBcmjVxX3cSo4z7Ol2Cmd70TMExotCDVGiMWX693euE/k
9a3YdDNQgUmPkhwIQqR+VulEFJ19g/VdZjHXh/EwM1MCNM/3FqldH3Dj5ZU5yR4k
oTBft027+0hQz4a6xmH7lIMsXIMUD2wXnCgXX7+cTerBxfHjA9JKt10ISKJB83ir
Q1o/eEnKjZdNtQasD/EGW4xoUUua4EyMu5e6pOaeYnIwyTfTZsimlRGeORFWE0sT
LRfUaviXH3GHvEm6s0GmI39Ext7aGg0yekj9i/aqgQKBgQC1kJrQzzuPa60OngCQ
xykOyIwZJdwlxGTgkXqfifQK1MKkkqz8MvSA2C7TV9mBw0/exg2K8EZykTXuPtba
71Q5I+sbo4yDZncDOCMuKiuqUQ9z+B/oFrD98kXXZVZEeEGloZIemJQVaX7diy8x
EVkhxlWnSIE4K0tCzrVDO8kmgQKBgQC1yWMR2N93xRidqk2IOf0zhSFBLpG3C+Bp
9jLh8/zjRJlXTWvSL2pmWmYcYK6TIhVcgk4T2JXI8A5V5cGH6rTYc2ud1bWA3PZw
prrIXfhCL0JWmgyEDZLAnXsS5VlyraCHqS35qMrNIcukr+/4uoSmgAs+UYwZLfkA
tVzYJWnIdQKBgF8xvwoF7UtoACcuzksaMLuwiEvTHtaqXt2jSPCGyu422QqiYJIm
QS2gqwRiBgdUGPdLTeRvz+/XlLgiOFI3syf2XhlyqYRnX7TPZRqaP6SftYNvL4Nn
CktLEDU7y3xAtOKbkNn704BafIq5o/eNCfd8XoJDsIR7po0ThdQHb5KBAoGBAIAx
rHOBhNVpYJqO5m4StsQGNhVJSejTr0YKIIfHD6cVUS2Ho2ltlpLnXOrWI0YO2xGJ
spW8PqSc5P8eLwQyN6YMfu+nLX/aUs/ORBnYaqIBwb5glELrb3n1lD0XD6UXXAVP
AOT2a02Nb5aLm6bDoZfo5ATmbO20xcwCGZ8zgw2hAoGBAIbsrBcE/WhpsP099T0u
YMTbc6AXhen4BfScIQXugYWSrwMalYNr4jelJfvLQJ3joguGhOfnPyIbNkp0/bAj
mwMd5Yx1HpJv3NXLGcNFV0b1Y39CsmK+uiHIzIxdqYOJHFwnjN7NNIs/2/qRFIcC
vx9GTN36CWzePsoNCJN6a9eX
-----END PRIVATE KEY-----
)";

class AuthCaptureService final : public v2::StoreDaemonService::Service {
 public:
  grpc::Status RouteAuthorityStage(
      grpc::ServerContext* server_context,
      const v2::RouteAuthorityStageRequest*,
      v2::RouteAuthorityStageResponse* response) override {
    const auto transport_security_context =
        DistributedSecurityKernel::transport_security_context_from_server_context(*server_context);
    const auto authenticated_peer_identity =
        DistributedSecurityKernel::derive_authenticated_peer_identity(transport_security_context);
    {
      absl::MutexLock lock(&mu_);
      captured_peer_identity_ = authenticated_peer_identity;
    }
    response->set_status(v2::BATCH_ITEM_STATUS_OK);
    return grpc::Status::OK;
  }

  [[nodiscard]] std::optional<AuthenticatedPeerIdentity> captured_peer_identity() const {
    absl::MutexLock lock(&mu_);
    return captured_peer_identity_;
  }

 private:
  mutable absl::Mutex mu_;
  std::optional<AuthenticatedPeerIdentity> captured_peer_identity_ ABSL_GUARDED_BY(mu_);
};

class LargePayloadService final : public v2::StoreDaemonService::Service {
 public:
  LargePayloadService() : payload_(kLargePayloadBytes, 'x') {}

  grpc::Status FetchPayloadRefChunk(
      grpc::ServerContext*,
      const v2::FetchPayloadRefChunkRequest*,
      v2::FetchPayloadRefChunkResponse* response) override {
    response->set_status(v2::BATCH_ITEM_STATUS_OK);
    response->set_total_size(payload_.size());
    response->set_eof(true);
    response->set_chunk(payload_);
    return grpc::Status::OK;
  }

 private:
  std::string payload_;
};

TEST_CASE(
    "inter-daemon transport helper establishes mTLS and authority presentation",
    "[daemon][grpc_transport][mtls]") {
  grpc::SslServerCredentialsOptions server_ssl_options;
  server_ssl_options.pem_root_certs = kCaCertPem;
  server_ssl_options.client_certificate_request = GRPC_SSL_REQUEST_AND_REQUIRE_CLIENT_CERTIFICATE_AND_VERIFY;
  server_ssl_options.pem_key_cert_pairs.push_back(
      grpc::SslServerCredentialsOptions::PemKeyCertPair{
          .private_key = kServerKeyPem,
          .cert_chain = kServerCertPem,
      });

  AuthCaptureService service;
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::SslServerCredentials(server_ssl_options), &selected_port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);

  const DaemonOptions::InterDaemonGrpcSecurity security{
      .tls_enabled = true,
      .mutual_auth_enabled = true,
      .cert_chain_pem = kClientCertPem,
      .private_key_pem = kClientKeyPem,
      .root_cert_pem = kCaCertPem,
  };
  auto channel = create_inter_daemon_channel(
      absl::StrCat("127.0.0.1:", selected_port), make_inter_daemon_channel_credentials(security));
  auto stub = v2::StoreDaemonService::NewStub(channel);

  grpc::ClientContext client_context;
  client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  v2::RouteAuthorityStageRequest request;
  v2::RouteAuthorityStageResponse response;
  const auto status = stub->RouteAuthorityStage(&client_context, request, &response);
  REQUIRE(status.ok());
  CHECK(response.status() == v2::BATCH_ITEM_STATUS_OK);

  const auto client_transport_security_context =
      DistributedSecurityKernel::transport_security_context_from_client_context(client_context);
  const auto client_peer_identity =
      DistributedSecurityKernel::derive_authenticated_peer_identity(client_transport_security_context);
  CHECK(client_peer_identity.auth_class == DaemonHopAuthClass::kDaemonMutualAuth);
  REQUIRE(client_peer_identity.presented_authority_ref.has_value());
  CHECK(client_peer_identity.presented_authority_ref->authority_id == "daemon-a");

  const auto server_peer_identity = service.captured_peer_identity();
  REQUIRE(server_peer_identity.has_value());
  CHECK(server_peer_identity->auth_class == DaemonHopAuthClass::kDaemonMutualAuth);
  REQUIRE(server_peer_identity->presented_authority_ref.has_value());
  CHECK(server_peer_identity->presented_authority_ref->authority_id == "authority-a");

  server->Shutdown();
}

TEST_CASE(
    "inter-daemon transport helper accepts payloads larger than grpc defaults",
    "[daemon][grpc_transport][message_size]") {
  LargePayloadService service;
  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort("127.0.0.1:0", grpc::InsecureServerCredentials(), &selected_port);
  builder.RegisterService(&service);
  auto server = builder.BuildAndStart();
  REQUIRE(server != nullptr);
  REQUIRE(selected_port != 0);

  auto channel =
      create_inter_daemon_channel(absl::StrCat("127.0.0.1:", selected_port), grpc::InsecureChannelCredentials());
  auto stub = v2::StoreDaemonService::NewStub(channel);

  grpc::ClientContext client_context;
  client_context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));
  v2::FetchPayloadRefChunkRequest request;
  request.set_payload_ref("payload-ref");
  request.set_artifact_id("artifact-id");
  request.set_offset(0);
  request.set_max_bytes(kLargePayloadBytes);
  v2::FetchPayloadRefChunkResponse response;
  const auto status = stub->FetchPayloadRefChunk(&client_context, request, &response);
  REQUIRE(status.ok());
  CHECK(response.status() == v2::BATCH_ITEM_STATUS_OK);
  CHECK(response.chunk().size() == kLargePayloadBytes);
  CHECK(response.eof());

  server->Shutdown();
}

} // namespace
} // namespace tensorcast::daemon
