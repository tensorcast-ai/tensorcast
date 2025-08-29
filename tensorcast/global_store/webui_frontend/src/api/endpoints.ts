import { apiClient, ApiResponse, GlobalMetrics, WorkerOut, ReplicaOut, ArtifactSummary, NodeSummary, TransportOut } from './client'

export const api = {
  // Summary
  getSummary: () =>
    apiClient.get<ApiResponse<GlobalMetrics>>('/summary'),

  // Workers
  getWorkers: (params: { include_unavailable?: boolean; page?: number; page_size?: number }) =>
    apiClient.get<ApiResponse<WorkerOut[]>>('/workers', { params }),

  getWorker: (workerId: string) =>
    apiClient.get<ApiResponse<WorkerOut>>(`/workers/${workerId}`),

  // Replicas
  getReplicas: (params: { artifact_id?: string; node_id?: string; memory_type?: string; worker_id?: string; page?: number; page_size?: number }) =>
    apiClient.get<ApiResponse<ReplicaOut[]>>('/replicas', { params }),

  getReplica: (replicaId: string) =>
    apiClient.get<ApiResponse<ReplicaOut>>(`/replicas/${replicaId}`),

  // Artifacts
  getArtifacts: () =>
    apiClient.get<ApiResponse<ArtifactSummary[]>>('/artifacts'),

  getArtifact: (artifactId: string) =>
    apiClient.get<ApiResponse<ArtifactSummary>>(`/artifacts/${artifactId}`),

  // Nodes
  getNodes: () =>
    apiClient.get<ApiResponse<NodeSummary[]>>('/nodes'),

  // Transports
  getTransports: (params: { status?: string; artifact_id?: string; page?: number; page_size?: number }) =>
    apiClient.get<ApiResponse<TransportOut[]>>('/transports', { params }),
}