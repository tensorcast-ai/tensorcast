import { rest, RestRequest, ResponseComposition, RestContext, DefaultBodyType } from 'msw'
import { summary, workers, replicas, models, nodes, transports } from './data'

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
    const modelName = url.searchParams.get('model_name')
    const nodeId = url.searchParams.get('node_id')
    const memoryType = url.searchParams.get('memory_type')

    let filteredReplicas = [...replicas]

    if (workerId) {
      filteredReplicas = filteredReplicas.filter(r => r.worker_id === workerId)
    }
    if (modelName) {
      filteredReplicas = filteredReplicas.filter(r =>
        r.model_name.toLowerCase().includes(modelName.toLowerCase())
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

  // Models list
  rest.get('/api/models', (_req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    return res(ctx.status(200), ctx.json(envelope(models)))
  }),

  rest.get('/api/models/:modelName', (req: RestRequest, res: ResponseComposition<DefaultBodyType>, ctx: RestContext) => {
    const { modelName } = req.params as { modelName: string }
    const model = models.find((m) => m.model_name === modelName)
    if (!model) return res(ctx.status(404))
    return res(ctx.status(200), ctx.json(envelope(model)))
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
    const modelName = url.searchParams.get('model_name')

    // Filter transports based on query params
    let filteredTransports = [...transports]
    if (status) {
      filteredTransports = filteredTransports.filter(t => t.status === status)
    }
    if (modelName) {
      filteredTransports = filteredTransports.filter(t =>
        t.model_name.toLowerCase().includes(modelName.toLowerCase())
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