import { rest, RestRequest, ResponseComposition, RestContext, DefaultBodyType } from 'msw'
import { summary, workers, replicas, artifacts, nodes, transports } from './data'

// Helper function to wrap data in standard response format
const envelope = <T>(data: T) => ({ data })

export const handlers = [
  // Summary
  rest.get('/api/summary', (_req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    return res(ctx.status(200), ctx.json(envelope(summary)))
  }),

  // Workers list
  rest.get('/api/workers', (_req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    return res(ctx.status(200), ctx.json(envelope(workers)))
  }),

  // Single Worker (basic lookup)
  rest.get('/api/workers/:workerId', (req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    const { workerId } = req.params as { workerId: string }
    const worker = workers.find((w) => w.worker_id === workerId)
    if (!worker) return res(ctx.status(404))
    return res(ctx.status(200), ctx.json(envelope(worker)))
  }),

  // Replicas list with filtering
  rest.get('/api/replicas', (req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    const url = new URL(req.url)
    const workerId = url.searchParams.get('worker_id')
    const modelId = url.searchParams.get('artifact_id')
    const nodeId = url.searchParams.get('node_id')
    const memoryType = url.searchParams.get('memory_type')

    let filteredReplicas = [...replicas]

    if (workerId) {
      filteredReplicas = filteredReplicas.filter(r => r.worker_id === workerId)
    }
    if (modelId) {
      filteredReplicas = filteredReplicas.filter(r =>
        r.artifact_id.toLowerCase().includes(modelId.toLowerCase())
      )
    }
    if (nodeId) {
      filteredReplicas = filteredReplicas.filter(r => r.node_id === nodeId)
    }
    if (memoryType) {
      filteredReplicas = filteredReplicas.filter(r => r.memory_type === memoryType)
    }

    return res(ctx.status(200), ctx.json(envelope(filteredReplicas)))
  }),

  rest.get('/api/replicas/:replicaId', (req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    const { replicaId } = req.params as { replicaId: string }
    const replica = replicas.find((r) => r.replica_id === replicaId)
    if (!replica) return res(ctx.status(404))
    return res(ctx.status(200), ctx.json(envelope(replica)))
  }),

  // Artifacts list
  rest.get('/api/artifacts', (_req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    return res(ctx.status(200), ctx.json(envelope(artifacts)))
  }),

  rest.get('/api/artifacts/:artifactId', (req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    const { artifactId } = req.params as { artifactId: string }
    const artifact = artifacts.find((m) => m.artifact_id === artifactId)
    if (!artifact) return res(ctx.status(404))
    return res(ctx.status(200), ctx.json(envelope(artifact)))
  }),

  // Nodes summary
  rest.get('/api/nodes', (_req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    return res(ctx.status(200), ctx.json(envelope(nodes)))
  }),

  // Transports history
  rest.get('/api/transports', (req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    const url = new URL(req.url)
    const page = parseInt(url.searchParams.get('page') || '1', 10)
    const pageSize = parseInt(url.searchParams.get('page_size') || '50', 10)
    const status = url.searchParams.get('status')
    const modelId = url.searchParams.get('artifact_id')

    // Filter transports based on query params
    let filteredTransports = [...transports]
    if (status) {
      filteredTransports = filteredTransports.filter(t => t.status === status)
    }
    if (modelId) {
      filteredTransports = filteredTransports.filter(t =>
        t.artifact_id.toLowerCase().includes(modelId.toLowerCase())
      )
    }

    // Calculate pagination
    const totalCount = filteredTransports.length
    const totalPages = Math.ceil(totalCount / pageSize)
    const startIndex = (page - 1) * pageSize
    const endIndex = startIndex + pageSize
    const paginatedTransports = filteredTransports.slice(startIndex, endIndex)

    return res(ctx.status(200), ctx.json({
      data: paginatedTransports,
      meta: {
        page,
        page_size: pageSize,
        total_count: totalCount,
        total_pages: totalPages,
      }
    }))
  }),
]