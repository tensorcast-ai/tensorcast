import { useState } from 'react'
import { Table, Tag, Select, Input, Space, Row, Col } from 'antd'
import { useQuery } from '@tanstack/react-query'
import { ColumnType } from 'antd/es/table'
import { FixedSizeList as List } from 'react-window'
import dayjs from 'dayjs'
import { api } from '@/api/endpoints'
import { ReplicaOut } from '@/api/client'
import { useWebSocket } from '@/hooks/useWebSocket'

const { Search } = Input

const Replicas = () => {
  useWebSocket()

  const [filters, setFilters] = useState({
    model_id: undefined as string | undefined,
    node_id: undefined as string | undefined,
    memory_type: undefined as string | undefined,
  })

  const [pagination, setPagination] = useState({
    page: 1,
    page_size: 100,
  })

  const { data, isLoading } = useQuery({
    queryKey: ['replicas', filters, pagination],
    queryFn: async () => {
      const res = await api.getReplicas({ ...filters, ...pagination })
      return res.data
    },
  })

  const replicas = data?.data || []
  const totalCount = data?.meta?.total_count || 0

  const getMemoryTypeTag = (type: string) => {
    const colors = {
      GPU: 'purple',
      RAM: 'blue',
      DISK: 'orange',
    }
    return <Tag color={colors[type as keyof typeof colors] || 'default'}>{type}</Tag>
  }

  const formatBytes = (bytes: number) => {
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    if (bytes === 0) return '0 B'
    const i = Math.floor(Math.log(bytes) / Math.log(1024))
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`
  }

  const columns: ColumnType<ReplicaOut>[] = [
    {
      title: 'Model ID',
      dataIndex: 'model_id',
      key: 'model_id',
      fixed: 'left',
      width: 200,
      ellipsis: true,
    },
    {
      title: 'Replica ID',
      dataIndex: 'replica_id',
      key: 'replica_id',
      width: 200,
      ellipsis: true,
    },
    {
      title: 'Memory Type',
      dataIndex: 'memory_type',
      key: 'memory_type',
      width: 120,
      render: (value) => getMemoryTypeTag(value),
      filters: [
        { text: 'GPU', value: 'GPU' },
        { text: 'RAM', value: 'RAM' },
        { text: 'DISK', value: 'DISK' },
      ],
      onFilter: (value, record) => record.memory_type === value,
    },
    {
      title: 'Device ID',
      dataIndex: 'device_id',
      key: 'device_id',
      width: 100,
    },
    {
      title: 'Node',
      dataIndex: 'node_id',
      key: 'node_id',
      width: 150,
      ellipsis: true,
    },
    {
      title: 'Memory Size',
      dataIndex: 'memory_size',
      key: 'memory_size',
      width: 120,
      render: (value) => formatBytes(value),
    },
    {
      title: 'Concurrency',
      key: 'concurrency',
      width: 120,
      render: (_, record) => `${record.current_requests} / ${record.max_concurrency}`,
    },
    {
      title: 'Status',
      key: 'status',
      width: 200,
      render: (_, record) => (
        <Space>
          <Tag color={record.is_available ? 'green' : 'red'}>
            {record.is_available ? 'Available' : 'Unavailable'}
          </Tag>
          {!record.worker_accepting && (
            <Tag color="orange">Worker Not Accepting</Tag>
          )}
        </Space>
      ),
    },
    {
      title: 'Updated',
      dataIndex: 'updated_at',
      key: 'updated_at',
      width: 150,
      render: (value) => dayjs(value).format('MM-DD HH:mm:ss'),
    },
  ]

  // Get unique values for filters
  const uniqueNodes = Array.from(new Set(replicas.map((r: ReplicaOut) => r.node_id)))

  return (
    <div>
      <div className="page-header">
        <h1>Replicas</h1>
      </div>

      <div className="filter-bar">
        <Row gutter={16}>
          <Col span={8}>
            <Search
              placeholder="Search model id"
              allowClear
              onSearch={(value) => setFilters({ ...filters, model_id: value || undefined })}
            />
          </Col>
          <Col span={8}>
            <Select
              style={{ width: '100%' }}
              placeholder="Filter by node"
              allowClear
              onChange={(value) => setFilters({ ...filters, node_id: value })}
              options={uniqueNodes.map(node => ({ label: node, value: node }))}
            />
          </Col>
          <Col span={8}>
            <Select
              style={{ width: '100%' }}
              placeholder="Filter by memory type"
              allowClear
              onChange={(value) => setFilters({ ...filters, memory_type: value })}
              options={[
                { label: 'GPU', value: 'GPU' },
                { label: 'RAM', value: 'RAM' },
                { label: 'DISK', value: 'DISK' },
              ]}
            />
          </Col>
        </Row>
      </div>

      <Table
        columns={columns}
        dataSource={replicas}
        rowKey="replica_id"
        loading={isLoading}
        scroll={{ x: 1500 }}
        pagination={{
          current: pagination.page,
          pageSize: pagination.page_size,
          total: totalCount,
          showSizeChanger: true,
          showTotal: (total) => `Total ${total} replicas`,
          pageSizeOptions: ['50', '100', '200', '500'],
          onChange: (page, pageSize) => {
            setPagination({ page, page_size: pageSize || 100 })
          },
        }}
        // Enable virtual scrolling for better performance with large datasets
        components={{
          body: totalCount > 1000 ? {
            wrapper: ({ children, ...restProps }: any) => (
              <List
                height={600}
                itemCount={replicas.length}
                itemSize={54}
                width="100%"
                {...restProps}
              >
                {({ index, style }) => {
                  const record = replicas[index]
                  return (
                    <div style={style} key={record.replica_id}>
                      {children[1].props.children[0].props.children[index]}
                    </div>
                  )
                }}
              </List>
            ),
          } : undefined,
        }}
      />
    </div>
  )
}

export default Replicas