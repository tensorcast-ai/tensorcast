# Global Store Web UI

A real-time monitoring dashboard for the Global Store distributed artifact storage system.

## Features

- **Real-time Monitoring**: Live updates via WebSocket for worker heartbeats, replica changes, and transport operations
- **Dashboard Overview**: Quick view of system health with key metrics
- **Worker Management**: Monitor worker nodes, memory usage, and health status
- **Replica Tracking**: View all artifact replicas with filtering by artifact, node, and memory type
- **Artifact analytics**: Detailed artifact statistics and distribution across nodes
- **Performance Visualization**: Charts for memory distribution, replica allocation, and load balancing

## Technology Stack

- **Frontend**: React 18 + TypeScript + Vite
- **UI Framework**: Ant Design v5
- **State Management**: TanStack Query (React Query)
- **Charts**: Recharts
- **WebSocket**: Native WebSocket API for real-time updates
- **Virtual Scrolling**: react-window for large datasets

## Development

### Prerequisites

- Node.js 18+ and pnpm
- Python 3.10+ with Global Store dependencies

### Setup

1. Install frontend dependencies:
```bash
pnpm install
```

2. Start development server:
```bash
pnpm dev
```

The development server runs on http://localhost:3000 with proxy to the backend API.

### Build

To build the production bundle:
```bash
pnpm build
```

This creates optimized assets in `../scstore/global_store/webui/build/`.

## Backend Integration

The Web UI is served by the Global Store Python backend when enabled:

```bash
# Install with UI dependencies
pip install scstore[ui]

# Start Global Store with Web UI
python -m scstore.global_store
```

Access the UI at http://localhost:9000

### Configuration

Environment variables:
- `GLOBAL_STORE_UI_PORT`: Web UI port (default: 9000)
- `GLOBAL_STORE_UI_HOST`: Web UI host (default: 0.0.0.0)
- `GLOBAL_STORE_UI_ENABLED`: Enable/disable UI (default: true)

## API Endpoints

### REST API
- `GET /api/summary` - Global metrics summary
- `GET /api/workers` - List workers with pagination
- `GET /api/replicas` - List replicas with filters
- `GET /api/artifacts` - List artifacts with statistics
- `GET /api/nodes` - Node aggregated data
- `GET /api/transports` - Transport history

### WebSocket
- `WS /ws/stream` - Real-time updates stream

## Testing

### Unit Tests
```bash
pnpm test
```

### E2E Tests
```bash
pnpm test:e2e
```

## Production Deployment

The Web UI is automatically included in the Global Store Docker image:

```dockerfile
EXPOSE 9000
ENV GLOBAL_STORE_UI_PORT=9000
```

For custom static file serving:
```bash
GLOBAL_STORE_UI_STATIC_DIR=/path/to/custom/build python -m scstore.global_store
```