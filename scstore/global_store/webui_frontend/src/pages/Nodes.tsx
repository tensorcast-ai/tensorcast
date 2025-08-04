import * as React from 'react';
import { Card, Table, Progress, Tag, Row, Col, Statistic } from 'antd';
import { useQuery } from '@tanstack/react-query';
import { CloudServerOutlined, DatabaseOutlined, ThunderboltOutlined } from '@ant-design/icons';
import { apiClient } from '../api/client';
import type { ColumnsType } from 'antd/es/table';

interface NodeSummary {
  node_id: string;
  total_replicas: number;
  total_memory: number;
  gpu_memory: number;
  ram_memory: number;
  disk_memory: number;
  active_workers: number;
}

const Nodes: React.FC = () => {
  const { data, isLoading } = useQuery<NodeSummary[]>({
    queryKey: ['nodes'],
    queryFn: () => apiClient.get('/nodes').then((res: any) => res.data.data),
    refetchInterval: 10000,
  });

  const nodes = data || [];

  // Calculate aggregate statistics
  const totalStats = nodes.reduce((acc, node) => ({
    nodes: acc.nodes + 1,
    replicas: acc.replicas + node.total_replicas,
    memory: acc.memory + node.total_memory,
    workers: acc.workers + node.active_workers,
  }), { nodes: 0, replicas: 0, memory: 0, workers: 0 });

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const columns: ColumnsType<NodeSummary> = [
    {
      title: 'Node ID',
      dataIndex: 'node_id',
      key: 'node_id',
      width: 200,
      render: (id: string) => (
        <Tag icon={<CloudServerOutlined />} color="blue">
          {id}
        </Tag>
      ),
    },
    {
      title: 'Active Workers',
      dataIndex: 'active_workers',
      key: 'active_workers',
      width: 120,
      sorter: (a, b) => a.active_workers - b.active_workers,
    },
    {
      title: 'Total Replicas',
      dataIndex: 'total_replicas',
      key: 'total_replicas',
      width: 120,
      sorter: (a, b) => a.total_replicas - b.total_replicas,
    },
    {
      title: 'Memory Distribution',
      key: 'memory',
      width: 300,
      render: (_, record) => {
        const total = record.total_memory || 1;
        const gpuPercent = (record.gpu_memory / total) * 100;
        const ramPercent = (record.ram_memory / total) * 100;
        const diskPercent = (record.disk_memory / total) * 100;

        return (
          <div>
            <div style={{ marginBottom: '8px' }}>
              <Progress
                percent={100}
                success={{
                  percent: gpuPercent,
                  strokeColor: '#52c41a',
                }}
                strokeColor="#1890ff"
                trailColor="#f0f0f0"
                format={() => `GPU: ${gpuPercent.toFixed(1)}%`}
              />
            </div>
            <div style={{ marginBottom: '8px' }}>
              <Progress
                percent={ramPercent}
                strokeColor="#faad14"
                format={() => `RAM: ${ramPercent.toFixed(1)}%`}
              />
            </div>
            <div>
              <Progress
                percent={diskPercent}
                strokeColor="#722ed1"
                format={() => `Disk: ${diskPercent.toFixed(1)}%`}
              />
            </div>
          </div>
        );
      },
    },
    {
      title: 'Total Memory',
      dataIndex: 'total_memory',
      key: 'total_memory',
      width: 150,
      render: formatBytes,
      sorter: (a, b) => a.total_memory - b.total_memory,
    },
    {
      title: 'GPU Memory',
      dataIndex: 'gpu_memory',
      key: 'gpu_memory',
      width: 150,
      render: (bytes: number) => (
        <span style={{ color: '#52c41a' }}>
          <ThunderboltOutlined /> {formatBytes(bytes)}
        </span>
      ),
      sorter: (a, b) => a.gpu_memory - b.gpu_memory,
    },
    {
      title: 'RAM Memory',
      dataIndex: 'ram_memory',
      key: 'ram_memory',
      width: 150,
      render: (bytes: number) => (
        <span style={{ color: '#faad14' }}>
          <DatabaseOutlined /> {formatBytes(bytes)}
        </span>
      ),
      sorter: (a, b) => a.ram_memory - b.ram_memory,
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <h1>Node Overview</h1>

      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Nodes"
              value={totalStats.nodes}
              prefix={<CloudServerOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Workers"
              value={totalStats.workers}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Replicas"
              value={totalStats.replicas}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Total Memory"
              value={formatBytes(totalStats.memory)}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Table
          columns={columns}
          dataSource={nodes}
          rowKey="node_id"
          loading={isLoading}
          pagination={{
            showSizeChanger: true,
            showTotal: (total) => `Total ${total} nodes`,
          }}
        />
      </Card>
    </div>
  );
};

export default Nodes;