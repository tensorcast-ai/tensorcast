from typing import Iterable, Optional, Sequence, Tuple, Any

import grpc
from grpc import Channel, Server, ServicerContext

from tensorcast.proto.daemon.v1 import store_daemon_pb2 as pb2


class StoreDaemonServiceStub:
    def __init__(self, channel: Channel) -> None: ...

    MaterializeReplica: grpc.UnaryUnaryMultiCallable
    ConfirmReplica: grpc.UnaryUnaryMultiCallable
    UnloadReplica: grpc.UnaryUnaryMultiCallable
    ClearMem: grpc.UnaryUnaryMultiCallable
    GetServerConfig: grpc.UnaryUnaryMultiCallable
    GetWorkerStatus: grpc.UnaryUnaryMultiCallable
    WaitReplicaVerification: grpc.UnaryUnaryMultiCallable
    GetDetailedStatus: grpc.UnaryUnaryMultiCallable
    GetLoadedReplicasV2: grpc.UnaryUnaryMultiCallable
    LockTransportChunks: grpc.UnaryUnaryMultiCallable
    UnlockTransportChunks: grpc.UnaryUnaryMultiCallable
    BeginRegisterArtifact: grpc.UnaryUnaryMultiCallable
    FeedRegisterArtifact: grpc.UnaryUnaryMultiCallable
    FeedRegisterArtifactStream: grpc.StreamUnaryMultiCallable
    KeepAliveRegisterArtifact: grpc.UnaryUnaryMultiCallable
    CommitRegisteredArtifact: grpc.UnaryUnaryMultiCallable
    AbortRegisteredArtifact: grpc.UnaryUnaryMultiCallable
    RevokeRegisteredArtifact: grpc.UnaryUnaryMultiCallable


class StoreDaemonServiceServicer:
    def MaterializeReplica(self, request: pb2.MaterializeReplicaRequest, context: ServicerContext) -> pb2.MaterializeReplicaResponse: ...
    def ConfirmReplica(self, request: pb2.ConfirmReplicaRequest, context: ServicerContext) -> pb2.ConfirmReplicaResponse: ...
    def UnloadReplica(self, request: pb2.UnloadReplicaRequest, context: ServicerContext) -> pb2.UnloadReplicaResponse: ...
    def ClearMem(self, request: pb2.ClearMemRequest, context: ServicerContext) -> pb2.ClearMemResponse: ...
    def GetServerConfig(self, request: pb2.GetServerConfigRequest, context: ServicerContext) -> pb2.GetServerConfigResponse: ...
    def GetWorkerStatus(self, request: pb2.GetWorkerStatusRequest, context: ServicerContext) -> pb2.GetWorkerStatusResponse: ...
    def WaitReplicaVerification(self, request: pb2.WaitReplicaVerificationRequest, context: ServicerContext) -> pb2.WaitReplicaVerificationResponse: ...
    def GetDetailedStatus(self, request: pb2.GetDetailedStatusRequest, context: ServicerContext) -> pb2.GetDetailedStatusResponse: ...
    def GetLoadedReplicasV2(self, request: pb2.GetLoadedReplicasV2Request, context: ServicerContext) -> pb2.GetLoadedReplicasV2Response: ...
    def LockTransportChunks(self, request: pb2.LockTransportChunksRequest, context: ServicerContext) -> pb2.LockTransportChunksResponse: ...
    def UnlockTransportChunks(self, request: pb2.UnlockTransportChunksRequest, context: ServicerContext) -> pb2.UnlockTransportChunksResponse: ...
    def BeginRegisterArtifact(self, request: pb2.BeginRegisterArtifactRequest, context: ServicerContext) -> pb2.BeginRegisterArtifactResponse: ...
    def FeedRegisterArtifact(self, request: pb2.FeedRegisterArtifactRequest, context: ServicerContext) -> pb2.FeedRegisterArtifactResponse: ...
    def FeedRegisterArtifactStream(self, request_iterator: Iterable[pb2.FeedRegisterArtifactStreamRequest], context: ServicerContext) -> pb2.FeedRegisterArtifactStreamResponse: ...
    def KeepAliveRegisterArtifact(self, request: pb2.KeepAliveRegisterArtifactRequest, context: ServicerContext) -> pb2.KeepAliveRegisterArtifactResponse: ...
    def CommitRegisteredArtifact(self, request: pb2.CommitRegisteredArtifactRequest, context: ServicerContext) -> pb2.CommitRegisteredArtifactResponse: ...
    def AbortRegisteredArtifact(self, request: pb2.AbortRegisteredArtifactRequest, context: ServicerContext) -> pb2.AbortRegisteredArtifactResponse: ...
    def RevokeRegisteredArtifact(self, request: pb2.RevokeRegisteredArtifactRequest, context: ServicerContext) -> pb2.RevokeRegisteredArtifactResponse: ...


def add_StoreDaemonServiceServicer_to_server(servicer: StoreDaemonServiceServicer, server: Server) -> None: ...


class StoreDaemonService:
    @staticmethod
    def MaterializeReplica(
        request: pb2.MaterializeReplicaRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.MaterializeReplicaResponse: ...

    @staticmethod
    def ConfirmReplica(
        request: pb2.ConfirmReplicaRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.ConfirmReplicaResponse: ...

    @staticmethod
    def UnloadReplica(
        request: pb2.UnloadReplicaRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.UnloadReplicaResponse: ...

    @staticmethod
    def ClearMem(
        request: pb2.ClearMemRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.ClearMemResponse: ...

    @staticmethod
    def GetServerConfig(
        request: pb2.GetServerConfigRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.GetServerConfigResponse: ...

    @staticmethod
    def GetWorkerStatus(
        request: pb2.GetWorkerStatusRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.GetWorkerStatusResponse: ...

    @staticmethod
    def WaitReplicaVerification(
        request: pb2.WaitReplicaVerificationRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.WaitReplicaVerificationResponse: ...

    @staticmethod
    def GetDetailedStatus(
        request: pb2.GetDetailedStatusRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.GetDetailedStatusResponse: ...

    @staticmethod
    def GetLoadedReplicasV2(
        request: pb2.GetLoadedReplicasV2Request,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.GetLoadedReplicasV2Response: ...

    @staticmethod
    def LockTransportChunks(
        request: pb2.LockTransportChunksRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.LockTransportChunksResponse: ...

    @staticmethod
    def UnlockTransportChunks(
        request: pb2.UnlockTransportChunksRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.UnlockTransportChunksResponse: ...

    @staticmethod
    def BeginRegisterArtifact(
        request: pb2.BeginRegisterArtifactRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.BeginRegisterArtifactResponse: ...

    @staticmethod
    def FeedRegisterArtifact(
        request: pb2.FeedRegisterArtifactRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.FeedRegisterArtifactResponse: ...

    @staticmethod
    def FeedRegisterArtifactStream(
        request_iterator: Iterable[pb2.FeedRegisterArtifactStreamRequest],
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.FeedRegisterArtifactStreamResponse: ...

    @staticmethod
    def KeepAliveRegisterArtifact(
        request: pb2.KeepAliveRegisterArtifactRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.KeepAliveRegisterArtifactResponse: ...

    @staticmethod
    def CommitRegisteredArtifact(
        request: pb2.CommitRegisteredArtifactRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.CommitRegisteredArtifactResponse: ...

    @staticmethod
    def AbortRegisteredArtifact(
        request: pb2.AbortRegisteredArtifactRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.AbortRegisteredArtifactResponse: ...

    @staticmethod
    def RevokeRegisteredArtifact(
        request: pb2.RevokeRegisteredArtifactRequest,
        target: str,
        options: Sequence[Tuple[str, Any]] = ...,
        channel_credentials: Optional[grpc.ChannelCredentials] = ...,
        call_credentials: Optional[grpc.CallCredentials] = ...,
        insecure: bool = ...,
        compression: Optional[grpc.Compression] = ...,
        wait_for_ready: Optional[bool] = ...,
        timeout: Optional[float] = ...,
        metadata: Optional[Sequence[Tuple[str, str]]] = ...,
    ) -> pb2.RevokeRegisteredArtifactResponse: ...


