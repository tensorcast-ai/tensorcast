import { useState, useEffect } from 'react'
import { Table, Tag, Space, Button, Drawer, Descriptions, Progress, Card } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { ColumnType } from 'antd/es/table'
import dayjs from 'dayjs'
import relativeTime from 'dayjs/plugin/relativeTime'
import { api } from '@/api/endpoints'
import { WorkerOut } from '@/api/client'
import { useWebSocket } from '@/hooks/useWebSocket'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

dayjs.extend(relativeTime)

const Workers = () => {
  const [selectedWorker, setSelectedWorker] = useState<WorkerOut | null>(null)
  const [drawerVisible, setDrawerVisible] = useState(false)

  // Local state for heartbeat timestamps per selected worker
  const [heartbeatHistory, setHeartbeatHistory] = useState<{ timestamp: number }[]>([])

  const { data, isLoading } = useQuery({
    queryKey: ['workers'],
    queryFn: async () => {
      const res = await api.getWorkers({ include_unavailable: true, page: 1, page_size: 100 })
      return res.data.data
    },
  })

  const workers = data || []

  useWebSocket((msg) => {
    if (msg.type === 'updates') {
      msg.updates.forEach((u) => {
        if (u.topic === 'heartbeat' && selectedWorker && u.payload.worker_id === selectedWorker.worker_id) {
          setHeartbeatHistory(prev => {
            const next = [...prev, { timestamp: Date.now() }]
            if (next.length > 120) next.shift()
            return next
          })
        }
      })
    }
  })

  // Reset heartbeat history when worker changes
  useEffect(() => {
    setHeartbeatHistory([])
  }, [selectedWorker])

  const { data: replicasData } = useQuery({
    enabled: !!selectedWorker,
    queryKey: ['replicas', { worker: selectedWorker?.worker_id }],
    queryFn: async () => {
      const res = await api.getReplicas({ worker_id: selectedWorker!.worker_id, page: 1, page_size: 100 })
      return res.data.data
    },
  })

  const replicas = replicasData || []

  const getStatusTag = (status: string) => {
    const statusConfig = {
      healthy: { color: 'green', text: 'Healthy' },
      warning: { color: 'yellow', text: 'Warning' },
      critical: { color: 'red', text: 'Critical' },
      dead: { color: 'default', text: 'Dead' },
    }

    const config = statusConfig[status as keyof typeof statusConfig] || statusConfig.dead
    return <Tag color={config.color}>{config.text}</Tag>
  }

  const formatBytes = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
  }

  const columns: ColumnType<WorkerOut>[] = [
    {
      title: 'Worker ID',
      dataIndex: 'worker_id',
      key: 'worker_id',
      width: 200,
      ellipsis: true,
    },
    {
      title: 'Node',
      dataIndex: 'node_id',
      key: 'node_id',
    },
    {
      title: 'Address',
      key: 'address',
      render: (_, record) => `${record.node_address}:${record.grpc_port}`,
    },
    {
      title: 'Memory Pool',
      key: 'memory',
      render: (_, record) => {
        const used = record.mem_pool_total_size - record.mem_pool_available_size
        const percent = (used / record.mem_pool_total_size) * 100
        return (
          <Space direction="vertical" size="small" style={{ width: 150 }}>
            <Progress
              percent={percent}
              size="small"
              status={percent > 90 ? 'exception' : 'normal'}
            />
            <span style={{ fontSize: 12 }}>
              {formatBytes(used)} / {formatBytes(record.mem_pool_total_size)}
            </span>
          </Space>
        )
      },
    },
    {
      title: 'Status',
      key: 'status',
      render: (_, record) => (
        <Space>
          {getStatusTag(record.status)}
          {!record.accepting_new_requests && <Tag color="orange">Not Accepting</Tag>}
        </Space>
      ),
    },
    {
      title: 'Replicas',
      dataIndex: 'replica_count',
      key: 'replica_count',
      width: 100,
    },
    {
      title: 'Last Heartbeat',
      dataIndex: 'last_heartbeat',
      key: 'last_heartbeat',
      render: (value) => dayjs(value).fromNow(),
    },
    {
      title: 'Action',
      key: 'action',
      render: (_, record) => (
        <Button
          type="link"
          onClick={() => {
            setSelectedWorker(record)
            setDrawerVisible(true)
          }}
        >
          Details
        </Button>
      ),
    },
  ]

  return (
    <div>
      <div className="page-header">
        <h1>Workers</h1>
      </div>

      <Table
        columns={columns}
        dataSource={workers}
        rowKey="worker_id"
        loading={isLoading}
        pagination={{
          pageSize: 20,
          showSizeChanger: true,
          showTotal: (total) => `Total ${total} workers`,
        }}
      />

      <Drawer
        title="Worker Details"
        placement="right"
        width={600}
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
      >
        {selectedWorker && (
          <>
            <Descriptions column={1} bordered>
              <Descriptions.Item label="Worker ID">
                {selectedWorker.worker_id}
              </Descriptions.Item>
              <Descriptions.Item label="Node ID">
                {selectedWorker.node_id}
              </Descriptions.Item>
              <Descriptions.Item label="Address">
                {selectedWorker.node_address}
              </Descriptions.Item>
              <Descriptions.Item label="gRPC Port">
                {selectedWorker.grpc_port}
              </Descriptions.Item>
              <Descriptions.Item label="Comm Port">
                {selectedWorker.p2p_port}
              </Descriptions.Item>
              <Descriptions.Item label="Status">
                {getStatusTag(selectedWorker.status)}
              </Descriptions.Item>
              <Descriptions.Item label="Accepting Requests">
                {selectedWorker.accepting_new_requests ? 'Yes' : 'No'}
              </Descriptions.Item>
              <Descriptions.Item label="Registered At">
                {selectedWorker.registered_at
                  ? dayjs(selectedWorker.registered_at).format('YYYY-MM-DD HH:mm:ss')
                  : 'Not available'}
              </Descriptions.Item>
              <Descriptions.Item label="Last Heartbeat">
                {dayjs(selectedWorker.last_heartbeat).format('YYYY-MM-DD HH:mm:ss')}
                <br />
                ({dayjs(selectedWorker.last_heartbeat).fromNow()})
              </Descriptions.Item>
            </Descriptions>

            <Card title="Memory Pool" style={{ marginTop: 16 }}>
              <Progress
                percent={
                  ((selectedWorker.mem_pool_total_size - selectedWorker.mem_pool_available_size) /
                    selectedWorker.mem_pool_total_size) *
                  100
                }
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
              <Descriptions column={2} style={{ marginTop: 16 }}>
                <Descriptions.Item label="Total">
                  {formatBytes(selectedWorker.mem_pool_total_size)}
                </Descriptions.Item>
                <Descriptions.Item label="Available">
                  {formatBytes(selectedWorker.mem_pool_available_size)}
                </Descriptions.Item>
                <Descriptions.Item label="Used">
                  {formatBytes(
                    selectedWorker.mem_pool_total_size - selectedWorker.mem_pool_available_size
                  )}
                </Descriptions.Item>
                <Descriptions.Item label="Replicas">
                  {selectedWorker.replica_count}
                </Descriptions.Item>
              </Descriptions>
            </Card>

            {/* Heartbeat history line chart */}
            <Card title="Heartbeat History (last ~1min)" style={{ marginTop: 16 }}>
              <div style={{ width: '100%', height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={heartbeatHistory.map(h => ({ time: new Date(h.timestamp).toLocaleTimeString() }))}>
                    <XAxis dataKey="time" interval={5} minTickGap={15} />
                    <YAxis hide={true} domain={[0, 1]} />
                    <Tooltip />
                    <Line type="monotone" dataKey={() => 1} stroke="#82ca9d" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </Card>

            {/* Replica sub table */}
            <Card title="Replicas on Worker" style={{ marginTop: 16 }}>
              <Table
                dataSource={replicas}
                size="small"
                rowKey="replica_id"
                pagination={false}
                columns={[
                  { title: 'Model', dataIndex: 'model_name', key: 'model' },
                  { title: 'Memory', dataIndex: 'memory_type', key: 'mem' },
                  { title: 'Device', dataIndex: 'device_id', key: 'device', width: 80 },
                  { title: 'Requests', key: 'req', render: (_: any, r: any) => `${r.current_requests}/${r.max_concurrency}` },
                  { title: 'Available', dataIndex: 'is_available', key: 'avail', render: (v:boolean) => (v ? 'Yes' : 'No') },
                ]}
              />
            </Card>
          </>
        )}
      </Drawer>
    </div>
  )
}

export default Workers