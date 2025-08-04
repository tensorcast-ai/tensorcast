import { Row, Col, Card } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts'
import MetricCard from '@/components/MetricCard'
import { api } from '@/api/endpoints'
import { ModelSummary, NodeSummary } from '@/api/client'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useEffect, useState } from 'react'

const Dashboard = () => {
  // Capture transport.started events via WebSocket callback
  const [qpsData, setQpsData] = useState<{ time: string; qps: number }[]>([])
  const [counterRef] = useState({ value: 0 })

  useWebSocket((msg) => {
    if (msg.type === 'updates') {
      msg.updates.forEach((u) => {
        if (u.topic === 'transport' && u.payload.action === 'started') {
          counterRef.value += 1
        }
      })
    }
  })

  useEffect(() => {
    const id = setInterval(() => {
      setQpsData((prev) => {
        const next = [...prev, { time: new Date().toLocaleTimeString(), qps: counterRef.value }]
        if (next.length > 60) next.shift()
        return next
      })
      counterRef.value = 0
    }, 1000)
    return () => clearInterval(id)
  }, [])

  const { data: summary, isLoading } = useQuery({
    queryKey: ['summary'],
    queryFn: async () => {
      const res = await api.getSummary()
      return res.data.data
    },
  })

  const { data: models } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const res = await api.getModels()
      return res.data.data
    },
  })

  const { data: nodes } = useQuery({
    queryKey: ['nodes'],
    queryFn: async () => {
      const res = await api.getNodes()
      return res.data.data
    },
  })

  const metrics = summary

  // Format bytes to human readable
  const formatBytes = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
  }

  // Prepare chart data
  const replicaDistribution = models?.slice(0, 10).map((model: ModelSummary) => ({
    name: model.model_name,
    gpu: model.gpu_replicas,
    ram: model.ram_replicas,
    disk: model.disk_replicas,
  })) || []

  const nodeMemoryData = nodes?.map((node: NodeSummary) => ({
    name: node.node_id,
    gpu: node.gpu_memory / (1024 * 1024 * 1024), // Convert to GB
    ram: node.ram_memory / (1024 * 1024 * 1024),
    disk: node.disk_memory / (1024 * 1024 * 1024),
  })) || []

  return (
    <div>
      <div className="page-header">
        <h1>Dashboard</h1>
      </div>

      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={6}>
          <MetricCard
            title="Active Workers"
            value={metrics?.active_workers || 0}
            suffix={`/ ${metrics?.total_workers || 0}`}
            loading={isLoading}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <MetricCard
            title="Available Replicas"
            value={metrics?.available_replicas || 0}
            suffix={`/ ${metrics?.total_replicas || 0}`}
            loading={isLoading}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <MetricCard
            title="Total Models"
            value={metrics?.total_models || 0}
            loading={isLoading}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <MetricCard
            title="Active Transports"
            value={metrics?.active_transports || 0}
            loading={isLoading}
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} sm={12} md={12}>
          <MetricCard
            title="Total Memory"
            value={formatBytes(metrics?.total_memory_bytes || 0)}
            loading={isLoading}
          />
        </Col>
        <Col xs={24} sm={12} md={12}>
          <MetricCard
            title="Available Memory"
            value={formatBytes(metrics?.available_memory_bytes || 0)}
            loading={isLoading}
          />
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginTop: 24 }}>
        <Col xs={24} lg={12}>
          <Card title="Replica Distribution (Top 10 Models)">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={replicaDistribution}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="gpu" stackId="a" fill="#8884d8" name="GPU" />
                  <Bar dataKey="ram" stackId="a" fill="#82ca9d" name="RAM" />
                  <Bar dataKey="disk" stackId="a" fill="#ffc658" name="DISK" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Node Memory Distribution (GB)">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={nodeMemoryData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="gpu" fill="#8884d8" name="GPU" />
                  <Bar dataKey="ram" fill="#82ca9d" name="RAM" />
                  <Bar dataKey="disk" fill="#ffc658" name="DISK" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={12}>
          <Card title="Transport QPS (last 60s)">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={qpsData}>
                  <XAxis dataKey="time" interval={10} minTickGap={15} />
                  <YAxis allowDecimals={false} />
                  <Tooltip />
                  <Line type="monotone" dataKey="qps" stroke="#8884d8" dot={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard