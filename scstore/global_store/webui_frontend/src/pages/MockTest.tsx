import React, { useState } from 'react';
import { Card, Button, Table, Tabs, Alert, Space, Tag, Spin } from 'antd';
import { CheckCircleOutlined, CloseCircleOutlined, ApiOutlined } from '@ant-design/icons';

interface TestResult {
  endpoint: string;
  status: 'success' | 'error' | 'pending';
  responseTime?: number;
  data?: any;
  error?: string;
}

const MockTest: React.FC = () => {
  const [testResults, setTestResults] = useState<TestResult[]>([]);
  const [testing, setTesting] = useState(false);

  const endpoints = [
    { path: '/api/summary', name: 'Summary' },
    { path: '/api/workers', name: 'Workers List' },
    { path: '/api/nodes', name: 'Nodes List' },
    { path: '/api/artifacts', name: 'Artifacts List' },
    { path: '/api/replicas', name: 'Replicas List' },
    { path: '/api/transports?page=1&page_size=5', name: 'Transports (Paginated)' },
    { path: '/api/workers/worker-1', name: 'Single Worker' },
    { path: '/api/artifacts/text-classifier', name: 'Single Artifact' },
    { path: '/api/replicas?worker_id=worker-1', name: 'Filtered Replicas' },
  ];

  const testEndpoint = async (endpoint: { path: string; name: string }) => {
    const startTime = performance.now();
    try {
      const response = await fetch(endpoint.path);
      const data = await response.json();
      const responseTime = performance.now() - startTime;

      return {
        endpoint: endpoint.name,
        status: 'success' as const,
        responseTime,
        data,
      };
    } catch (error) {
      return {
        endpoint: endpoint.name,
        status: 'error' as const,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  };

  const runAllTests = async () => {
    setTesting(true);
    setTestResults([]);

    const results: TestResult[] = [];
    for (const endpoint of endpoints) {
      const result = await testEndpoint(endpoint);
      results.push(result);
      setTestResults([...results]);
      await new Promise(resolve => setTimeout(resolve, 100)); // Small delay for visual effect
    }

    setTesting(false);
  };

  const columns = [
    {
      title: 'Endpoint',
      dataIndex: 'endpoint',
      key: 'endpoint',
    },
    {
      title: 'Status',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag
          icon={status === 'success' ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
          color={status === 'success' ? 'success' : 'error'}
        >
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: 'Response Time',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time?: number) => time ? `${time.toFixed(2)}ms` : '-',
    },
    {
      title: 'Data Structure',
      key: 'structure',
      render: (_: any, record: TestResult) => {
        if (record.status === 'error') return <span style={{ color: 'red' }}>{record.error}</span>;
        if (!record.data) return '-';

        const hasData = 'data' in record.data;
        const hasMeta = 'meta' in record.data;
        const dataType = hasData && Array.isArray(record.data.data) ? 'array' : 'object';

        return (
          <Space>
            {hasData && <Tag color="blue">data: {dataType}</Tag>}
            {hasMeta && <Tag color="green">meta</Tag>}
            {!hasData && !hasMeta && <Tag>direct response</Tag>}
          </Space>
        );
      },
    },
  ];

  const successCount = testResults.filter(r => r.status === 'success').length;
  const errorCount = testResults.filter(r => r.status === 'error').length;

  return (
    <div style={{ padding: '24px' }}>
      <h1>Mock API Test Dashboard</h1>

      {!import.meta.env.VITE_MOCK_API && (
        <Alert
          message="Mock API is not enabled"
          description="Run 'pnpm run dev:mock' to enable the mock API server"
          type="warning"
          showIcon
          style={{ marginBottom: '24px' }}
        />
      )}

      <Card style={{ marginBottom: '24px' }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Button
            type="primary"
            onClick={runAllTests}
            loading={testing}
            icon={<ApiOutlined />}
            size="large"
          >
            {testing ? 'Testing...' : 'Run All Tests'}
          </Button>

          {testResults.length > 0 && (
            <Space>
              <Tag color="success">Success: {successCount}</Tag>
              <Tag color="error">Errors: {errorCount}</Tag>
            </Space>
          )}
        </Space>
      </Card>

      {testing && testResults.length < endpoints.length && (
        <Card style={{ marginBottom: '24px', textAlign: 'center' }}>
          <Spin tip={`Testing endpoint ${testResults.length + 1} of ${endpoints.length}...`} />
        </Card>
      )}

      {testResults.length > 0 && (
        <Card>
          <Tabs defaultActiveKey="summary">
            <Tabs.TabPane tab="Test Summary" key="summary">
              <Table
                columns={columns}
                dataSource={testResults}
                rowKey="endpoint"
                pagination={false}
              />
            </Tabs.TabPane>

            <Tabs.TabPane tab="Response Details" key="details">
              {testResults.map((result, index) => (
                <Card key={index} title={result.endpoint} style={{ marginBottom: '16px' }}>
                  {result.status === 'success' ? (
                    <pre style={{
                      backgroundColor: '#f5f5f5',
                      padding: '12px',
                      borderRadius: '4px',
                      overflow: 'auto',
                      maxHeight: '300px'
                    }}>
                      {JSON.stringify(result.data, null, 2)}
                    </pre>
                  ) : (
                    <Alert type="error" message={result.error} />
                  )}
                </Card>
              ))}
            </Tabs.TabPane>
          </Tabs>
        </Card>
      )}
    </div>
  );
};

export default MockTest;