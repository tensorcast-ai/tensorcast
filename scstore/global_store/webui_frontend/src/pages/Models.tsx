import { Table, Tag, Progress, Button } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { ColumnType } from 'antd/es/table'
import { api } from '@/api/endpoints'
import { ModelSummary } from '@/api/client'
import { useWebSocket } from '@/hooks/useWebSocket'

const Models = () => {
  useWebSocket()
  const navigate = useNavigate()

  const { data: models, isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const res = await api.getModels()
      return res.data.data
    },
  })

  const formatBytes = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
  }

  const columns: ColumnType<ModelSummary>[] = [
    {
      title: 'Model Name',
      dataIndex: 'model_name',
      key: 'model_name',
      sorter: (a, b) => a.model_name.localeCompare(b.model_name),
    },
    {
      title: 'Total Replicas',
      dataIndex: 'total_replicas',
      key: 'total_replicas',
      sorter: (a, b) => a.total_replicas - b.total_replicas,
      render: (value) => <Tag>{value}</Tag>,
    },
    {
      title: 'Available',
      dataIndex: 'available_replicas',
      key: 'available_replicas',
      render: (value, record) => (
        <span style={{ color: value < record.total_replicas ? '#faad14' : '#52c41a' }}>
          {value} / {record.total_replicas}
        </span>
      ),
    },
    {
      title: 'Distribution',
      key: 'distribution',
      render: (_, record) => (
        <span>
          <Tag color="purple">GPU: {record.gpu_replicas}</Tag>
          <Tag color="blue">RAM: {record.ram_replicas}</Tag>
          <Tag color="orange">DISK: {record.disk_replicas}</Tag>
        </span>
      ),
    },
    {
      title: 'Total Size',
      dataIndex: 'total_memory_size',
      key: 'total_memory_size',
      sorter: (a, b) => a.total_memory_size - b.total_memory_size,
      render: (value) => formatBytes(value),
    },
    {
      title: 'Avg Load',
      dataIndex: 'avg_load_ratio',
      key: 'avg_load_ratio',
      width: 150,
      sorter: (a, b) => a.avg_load_ratio - b.avg_load_ratio,
      render: (value) => (
        <Progress
          percent={Math.round(value * 100)}
          size="small"
          status={value > 0.8 ? 'exception' : 'normal'}
        />
      ),
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button
          type="link"
          onClick={() => navigate(`/models/${encodeURIComponent(record.model_name)}`)}
        >
          View Details
        </Button>
      ),
    },
  ]

  return (
    <div>
      <div className="page-header">
        <h1>Models</h1>
      </div>

      <Table
        columns={columns}
        dataSource={models || []}
        rowKey="model_name"
        loading={isLoading}
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showTotal: (total) => `Total ${total} models`,
        }}
      />
    </div>
  )
}

export default Models