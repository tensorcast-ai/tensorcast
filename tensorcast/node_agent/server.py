#  Copyright (c) 2026, TensorCast Team.

from __future__ import annotations

import grpc
from google.protobuf import timestamp_pb2

from tensorcast.api.operation import OperationStatus
from tensorcast.node_agent.executor import NodeAgentExecutor
from tensorcast.proto.node_agent.v1 import node_agent_pb2, node_agent_pb2_grpc

_STATE_MAP = {
    "pending": node_agent_pb2.OPERATION_STATE_PENDING,
    "running": node_agent_pb2.OPERATION_STATE_RUNNING,
    "success": node_agent_pb2.OPERATION_STATE_SUCCESS,
    "failed": node_agent_pb2.OPERATION_STATE_FAILED,
    "cancelled": node_agent_pb2.OPERATION_STATE_CANCELLED,
    "degraded": node_agent_pb2.OPERATION_STATE_DEGRADED,
}


def _status_to_proto(status: OperationStatus) -> node_agent_pb2.OperationStatus:
    state_value = _STATE_MAP.get(
        status.state, node_agent_pb2.OPERATION_STATE_UNSPECIFIED
    )
    msg = node_agent_pb2.OperationStatus(state=state_value)
    if status.message:
        msg.message = status.message
    if status.progress is not None:
        msg.progress = float(status.progress)
    if status.as_of_ms is not None:
        ts = timestamp_pb2.Timestamp()
        ts.FromMilliseconds(int(status.as_of_ms))
        msg.as_of.CopyFrom(ts)
    if status.error is not None:
        msg.error.CopyFrom(
            node_agent_pb2.OperationError(
                status_code=str(status.error.status_code),
                message=str(status.error.message),
                retryable=bool(status.error.retryable),
            )
        )
    return msg


class NodeAgentServicer(node_agent_pb2_grpc.NodeAgentServiceServicer):
    def __init__(self, executor: NodeAgentExecutor) -> None:
        self._executor = executor

    def ExecutePlan(
        self,
        request: node_agent_pb2.ExecutePlanRequest,
        context: grpc.ServicerContext,
    ) -> node_agent_pb2.ExecutePlanResponse:
        del context
        result = self._executor.execute_plan(request.plan, dry_run=request.dry_run)
        response = node_agent_pb2.ExecutePlanResponse(
            request_id=result.request_id,
            ok=bool(result.ok),
        )
        for step in result.steps.values():
            entry = response.steps.add()
            entry.step_id = step.step_id
            entry.target_id = step.target_id
            entry.action = step.action
            entry.status.CopyFrom(_status_to_proto(step.status))
        return response

    def GetAgentInfo(
        self,
        request: node_agent_pb2.GetAgentInfoRequest,
        context: grpc.ServicerContext,
    ) -> node_agent_pb2.GetAgentInfoResponse:
        del request
        del context
        return node_agent_pb2.GetAgentInfoResponse(
            agent_id=self._executor.agent_id,
            daemon_id=self._executor.daemon_id,
            instance_id=self._executor.instance_id or "",
            version=self._executor.version,
        )


def add_servicer_to_server(
    server: grpc.Server, executor: NodeAgentExecutor
) -> NodeAgentServicer:
    servicer = NodeAgentServicer(executor)
    node_agent_pb2_grpc.add_NodeAgentServiceServicer_to_server(servicer, server)
    return servicer


__all__ = [
    "NodeAgentServicer",
    "add_servicer_to_server",
]
