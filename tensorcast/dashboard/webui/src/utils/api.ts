export type Health = { status: string };

export type WorkersResponse = {
  workers: Array<{
    worker_id: string;
    node_id: string;
    node_address: string;
    grpc_port: number;
    p2p_port: number;
    mem_pool_total: number;
    mem_pool_available: number;
    accepting_new_requests: boolean;
    last_heartbeat_ts: string; // RFC3339
    state_version: number;
    status: string; // ACTIVE|UNAVAILABLE|...
    memory_tier?: MemoryTierSnapshot | null;
  }>;
};

export type MemoryTierSnapshot = {
  stable_total_bytes: number;
  stable_used_bytes: number;
  preemptible_total_bytes: number;
  preemptible_marked_bytes: number;
  faults_per_sec: number;
  rehydrate_p99_ns: number;
  enable_preemptible: boolean;
  memory_tier_config_json: string;
  snapshot_epoch_ns: number;
};

export type ReplicaRow = {
  artifact_id: string;
  node_id: string;
  node_address: string;
  device_id: number | null;
  memory_type: 'RAM' | 'GPU' | 'DISK' | string;
  bytes: number;
  state?: string | null;
  created_ts?: string | null;
  expires_at?: string | null;
  id_kind?: string | null;
};

export type ReplicasResponse = {
  replicas: Array<ReplicaRow>;
  page_info?: { next_page_token?: string | null } | null;
};

export type ArtifactDetailResponse = {
  artifact_id: string;
  artifact_kind?: string | null;
  descriptor?: {
    artifact_id: string;
    id_kind?: string | null;
    index_multihash?: string | null;
    data_multihash?: string | null;
    schema_version?: string | null;
    encoding?: string | null;
    total_size?: number | null;
  } | null;
  replicas?: Array<{
    node_id: string;
    node_address: string;
    device_id: number | null;
    memory_type: 'RAM' | 'GPU' | 'DISK' | string;
    bytes: number;
    created_ts?: string | null;
    expires_at?: string | null;
  }>;
  view_meta?: {
    view_spec_json: string;
    view_size: number;
    view_data_hash: string;
    verified_at?: string | null;
  } | null;
  leaves?: Array<{ index: number; digest_b64: string }> | null;
  partial_coverage?: Array<{
    space_kind: string;
    space_id: string;
    missing: Array<{ offset: number; length: number }>;
  }> | null;
};

export type MemoryTierLease = {
  lease_id: string;
  node_id: string;
  kind: string;
  artifact_id: string;
  chunk_range: { offset: number; length: number };
  chunk_ids: Array<number>;
  ledger_version: number;
  bytes: number;
  workload_id: string;
  state: string;
  request_id: string;
  ack_epoch_ns?: number | null;
  issued_at_ns: number;
  expires_at_ns?: number | null;
};

export type MemoryTierLeasesResponse = {
  leases: Array<MemoryTierLease>;
};

function withApiBase(path: string): string {
  // 使用独立的 API 前缀，避免把静态资源前缀（/static/）带到 API。
  const appBase = (import.meta.env.VITE_API_BASE_PATH as string | undefined) || '';
  const normalized = appBase === '/' ? '' : appBase;
  return `${normalized.replace(/\/$/, '')}${path.startsWith('/') ? '' : '/'}${path}`;
}

async function getJson<T>(path: string, init?: RequestInit): Promise<T> {
  const url = withApiBase(path);
  const res = await fetch(url, {
    ...init,
    headers: { 'accept': 'application/json', ...(init?.headers || {}) },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`HTTP ${res.status} ${res.statusText}: ${text}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health(): Promise<Health> {
    return getJson('/api/health');
  },
  workers(includeUnavailable: boolean): Promise<WorkersResponse> {
    const q = includeUnavailable ? '?include_unavailable=1' : '';
    return getJson(`/api/workers${q}`);
  },
  replicas(params: {
    artifact_id?: string;
    node_id?: string;
    node_address?: string;
    memory_type?: 'RAM' | 'GPU' | 'DISK' | '';
    device_id?: number | '' | null;
    page_token?: string | null;
    page_size?: number;
  }): Promise<ReplicasResponse> {
    const search = new URLSearchParams();
    if (params.artifact_id) search.set('artifact_id', params.artifact_id);
    if (params.node_id) search.set('node_id', params.node_id);
    if (params.node_address) search.set('node_address', params.node_address);
    if (params.memory_type) search.set('memory_type', params.memory_type);
    if (params.device_id !== undefined && params.device_id !== null && params.device_id !== '')
      search.set('device_id', String(params.device_id));
    if (params.page_token) search.set('page_token', params.page_token);
    if (params.page_size) search.set('page_size', String(params.page_size));
    const qs = search.toString();
    return getJson(`/api/replicas${qs ? `?${qs}` : ''}`);
  },
  artifactDetail(artifactId: string, include: Array<'replicas' | 'view' | 'leaves'> = ['replicas', 'view']): Promise<ArtifactDetailResponse> {
    const search = new URLSearchParams();
    if (include.length) search.set('include', include.join(','));
    return getJson(`/api/artifacts/${encodeURIComponent(artifactId)}?${search.toString()}`);
  },
  memoryTierLeases(params: { node_id?: string; states?: Array<string> }): Promise<MemoryTierLeasesResponse> {
    const search = new URLSearchParams();
    if (params.node_id) search.set('node_id', params.node_id);
    if (params.states && params.states.length > 0) search.set('states', params.states.join(','));
    const qs = search.toString();
    return getJson(`/api/memory_tier/leases${qs ? `?${qs}` : ''}`);
  },
};
