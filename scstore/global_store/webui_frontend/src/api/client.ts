import axios from 'axios'

export const apiClient = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error)
    return Promise.reject(error)
  }
)

export interface ApiResponse<T> {
  data: T
  meta?: Record<string, any>
}

export interface PaginationMeta {
  page: number
  page_size: number
  total_count: number
  total_pages: number
}

export interface GlobalMetrics {
  total_workers: number
  active_workers: number
  total_replicas: number
  available_replicas: number
  total_models: number
  active_transports: number
  total_memory_bytes: number
  available_memory_bytes: number
}

export interface WorkerOut {
  worker_id: string
  node_id: string
  node_address: string
  grpc_port: number
  p2p_port: number
  mem_pool_total_size: number
  mem_pool_available_size: number
  accepting_new_requests: boolean
  registered_at: string | null
  last_heartbeat: string
  status: 'healthy' | 'warning' | 'critical' | 'dead'
  replica_count: number
}

export interface ReplicaOut {
  replica_id: string
  model_name: string
  node_id: string
  node_address: string
  node_port: number
  memory_size: number
  memory_type: 'GPU' | 'RAM' | 'DISK'
  device_id: number
  max_concurrency: number
  is_available: boolean
  worker_id: string | null
  created_at: string
  updated_at: string
  current_requests: number
  worker_accepting: boolean
}

export interface ModelSummary {
  model_name: string
  total_replicas: number
  available_replicas: number
  gpu_replicas: number
  ram_replicas: number
  disk_replicas: number
  total_memory_size: number
  avg_load_ratio: number
}

export interface NodeSummary {
  node_id: string
  total_replicas: number
  total_memory: number
  gpu_memory: number
  ram_memory: number
  disk_memory: number
  active_workers: number
}

export interface TransportOut {
  transport_id: string
  replica_id: string
  model_name: string
  source_node_id: string
  source_address: string
  source_port: number
  created_at: string
  completed_at: string | null
  status: string
  wait_duration_seconds: number | null
}