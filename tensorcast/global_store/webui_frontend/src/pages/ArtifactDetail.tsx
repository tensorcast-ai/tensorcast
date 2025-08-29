import { useParams } from 'react-router-dom'
import { Card, Row, Col, Statistic, Table, Tag } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import dayjs from 'dayjs'
import { api } from '@/api/endpoints'
import { ReplicaOut } from '@/api/client'
import { useWebSocket } from '@/hooks/useWebSocket'

const ArtifactDetail = () => {
  useWebSocket()
  const { artifactId } = useParams<{ artifactId: string }>()

  const { data: artifact } = useQuery({
    queryKey: ['artifact', artifactId],
    queryFn: async () => {
      const res = await api.getArtifact(artifactId!)
      return res.data.data
    },
    enabled: !!artifactId,
  })

  const { data: replicasData } = useQuery({
    queryKey: ['replicas', { artifact_id: artifactId }],
    queryFn: async () => {
      const res = await api.getReplicas({ artifact_id: artifactId, page: 1, page_size: 100 })
      return res.data
    },
    enabled: !!artifactId,
  })

  const replicas = replicasData?.data || []

  const formatBytes = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
  }

  // Pie chart data
  const pieData = artifact
    ? [
        { name: 'GPU', value: artifact.gpu_replicas, fill: '#8884d8' },
        { name: 'RAM', value: artifact.ram_replicas, fill: '#82ca9d' },
        { name: 'DISK', value: artifact.disk_replicas, fill: '#ffc658' },
      ].filter(item => item.value > 0)
    : []

  // Group replicas by node
  const replicasByNode = replicas.reduce((acc: Record<string, ReplicaOut[]>, replica: ReplicaOut) => {
    if (!acc[replica.node_id]) {
      acc[replica.node_id] = []
    }
    acc[replica.node_id].push(replica)
    return acc
  }, {} as Record<string, ReplicaOut[]>)

  const columns = [
    {
      title: 'Replica ID',
      dataIndex: 'replica_id',
      key: 'replica_id',
      ellipsis: true,
    },
    {
      title: 'Memory Type',
      dataIndex: 'memory_type',
      key: 'memory_type',
      render: (value: string) => {
        const colors = { GPU: 'purple', RAM: 'blue', DISK: 'orange' }
        return <Tag color={colors[value as keyof typeof colors]}>{value}</Tag>
      },
    },
    {
      title: 'Device ID',
      dataIndex: 'device_id',
      key: 'device_id',
    },
    {
      title: 'Status',
      key: 'status',
      render: (_: any, record: ReplicaOut) => (
        <Tag color={record.is_available ? 'green' : 'red'}>
          {record.is_available ? 'Available' : 'Unavailable'}
        </Tag>
      ),
    },
    {
      title: 'Load',
      key: 'load',
      render: (_: any, record: ReplicaOut) =>
        `${record.current_requests} / ${record.max_concurrency}`,
    },
    {
      title: 'Updated',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (value: string) => dayjs(value).format('MM-DD HH:mm:ss'),
    },
  ]

  if (!artifact) {
    return <div>Loading...</div>
  }

  return (
    <div>
      <div className="page-header">
        <h1>{artifactId}</h1>
      </div>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <Row gutter={16}>
              <Col span={6}>
                <Statistic title="Total Replicas" value={artifact.total_replicas} />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Available Replicas"
                  value={artifact.available_replicas}
                  valueStyle={{
                    color: artifact.available_replicas < artifact.total_replicas ? '#faad14' : '#3f8600'
                  }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Total Size"
                  value={formatBytes(artifact.total_memory_size)}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="Average Load"
                  value={`${Math.round(artifact.avg_load_ratio * 100)}%`}
                  valueStyle={{
                    color: artifact.avg_load_ratio > 0.8 ? '#cf1322' : '#3f8600'
                  }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="Memory Type Distribution">
            <div style={{ height: 300 }}>
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card title="Replica Distribution by Node">
            {Object.entries(replicasByNode).map(([nodeId, nodeReplicas]: [string, ReplicaOut[]]) => (
              <Card
                key={nodeId}
                type="inner"
                title={`Node: ${nodeId}`}
                style={{ marginBottom: 16 }}
              >
                <Table
                  columns={columns}
                  dataSource={nodeReplicas}
                  rowKey="replica_id"
                  pagination={false}
                  size="small"
                />
              </Card>
            ))}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default ArtifactDetail