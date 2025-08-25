import React, { useMemo } from 'react';
import { Table, Tag, Space, Card, Statistic, Row, Col, Select, Input } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { ClockCircleOutlined, CheckCircleOutlined, SyncOutlined } from '@ant-design/icons';
import { apiClient } from '../api/client';
import type { ColumnsType } from 'antd/es/table';

interface Transport {
  transport_id: string;
  replica_id: string;
  artifact_id: string;
  source_node_id: string;
  source_address: string;
  source_port: number;
  created_at: string;
  completed_at?: string;
  status: string;
  wait_duration_seconds?: number;
}

interface TransportResponse {
  data: Transport[];
  meta: {
    page: number;
    page_size: number;
    total_count: number;
    total_pages: number;
  };
}

const Transports: React.FC = () => {
  const [filters, setFilters] = React.useState({
    status: undefined as string | undefined,
    artifact_id: undefined as string | undefined,
    page: 1,
    page_size: 50,
  });

  const { data, isLoading } = useQuery({
    queryKey: ['transports', filters],
    queryFn: async () => {
      const res = await apiClient.get<TransportResponse>('/transports', { params: filters })
      return res.data
    },
    refetchInterval: 5000,
  });

  const transports = data?.data || [];
  const meta = data?.meta;

  // Calculate transport statistics
  const stats = useMemo(() => {
    const inProgress = transports.filter(t => t.status === 'in_progress').length;
    const completed = transports.filter(t => t.status === 'completed').length;
    const failed = transports.filter(t => t.status === 'failed').length;

    const avgDuration = transports
      .filter(t => t.wait_duration_seconds)
      .reduce((sum, t) => sum + (t.wait_duration_seconds || 0), 0) /
      (transports.filter(t => t.wait_duration_seconds).length || 1);

    return { inProgress, completed, failed, avgDuration };
  }, [transports]);

  const columns: ColumnsType<Transport> = [
    {
      title: 'Transport ID',
      dataIndex: 'transport_id',
      key: 'transport_id',
      width: 200,
      render: (id: string) => (
        <span style={{ fontFamily: 'monospace', fontSize: '12px' }}>
          {id.substring(0, 8)}...
        </span>
      ),
    },
    {
      title: 'Artifact ID',
      dataIndex: 'artifact_id',
      key: 'artifact_id',
      width: 200,
    },
    {
      title: 'Source Node',
      dataIndex: 'source_node_id',
      key: 'source_node_id',
      width: 150,
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (status: string) => {
        const config = {
          in_progress: { color: 'processing', icon: <SyncOutlined spin /> },
          completed: { color: 'success', icon: <CheckCircleOutlined /> },
          failed: { color: 'error', icon: <ClockCircleOutlined /> },
        };
        const { color, icon } = config[status as keyof typeof config] || { color: 'default', icon: null };
        return (
          <Tag color={color} icon={icon}>
            {status.toUpperCase()}
          </Tag>
        );
      },
    },
    {
      title: 'Duration',
      key: 'duration',
      width: 120,
      render: (_, record) => {
        if (record.wait_duration_seconds) {
          return `${record.wait_duration_seconds.toFixed(2)}s`;
        }
        if (record.status === 'in_progress') {
          const startTime = new Date(record.created_at).getTime();
          const now = new Date().getTime();
          const duration = (now - startTime) / 1000;
          return `${duration.toFixed(0)}s`;
        }
        return '-';
      },
    },
    {
      title: 'Created At',
      dataIndex: 'created_at',
      key: 'created_at',
      width: 180,
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: 'Completed At',
      dataIndex: 'completed_at',
      key: 'completed_at',
      width: 180,
      render: (date?: string) => date ? new Date(date).toLocaleString() : '-',
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <h1>Artifact Transports</h1>

      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="In Progress"
              value={stats.inProgress}
              prefix={<SyncOutlined spin />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Completed"
              value={stats.completed}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Failed"
              value={stats.failed}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Avg Duration"
              value={stats.avgDuration}
              precision={2}
              suffix="s"
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Space style={{ marginBottom: '16px' }}>
          <Select
            style={{ width: 150 }}
            placeholder="Filter by status"
            allowClear
            value={filters.status}
            onChange={(value) => setFilters({ ...filters, status: value, page: 1 })}
          >
            <Select.Option value="in_progress">In Progress</Select.Option>
            <Select.Option value="completed">Completed</Select.Option>
            <Select.Option value="failed">Failed</Select.Option>
          </Select>

          <Input.Search
            style={{ width: 200 }}
            placeholder="Filter by artifact id"
            allowClear
            value={filters.artifact_id}
            onChange={(e) => setFilters({ ...filters, artifact_id: e.target.value || undefined, page: 1 })}
          />
        </Space>

        <Table
          columns={columns}
          dataSource={transports}
          rowKey="transport_id"
          loading={isLoading}
          pagination={{
            current: filters.page,
            pageSize: filters.page_size,
            total: meta?.total_count || 0,
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} transports`,
            onChange: (page, pageSize) => {
              setFilters({ ...filters, page, page_size: pageSize || 50 });
            },
          }}
        />
      </Card>
    </div>
  );
};

export default Transports;