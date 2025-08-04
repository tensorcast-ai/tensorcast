import { Card, Statistic } from 'antd'
import { ArrowUpOutlined, ArrowDownOutlined } from '@ant-design/icons'

interface MetricCardProps {
  title: string
  value: number | string
  suffix?: string
  prefix?: React.ReactNode
  precision?: number
  trend?: number
  loading?: boolean
}

const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  suffix,
  prefix,
  precision = 0,
  trend,
  loading = false,
}) => {
  return (
    <Card loading={loading} className="metric-card">
      <Statistic
        title={title}
        value={value}
        precision={precision}
        suffix={suffix}
        prefix={prefix}
        valueStyle={
          trend !== undefined
            ? { color: trend > 0 ? '#3f8600' : '#cf1322' }
            : undefined
        }
      />
      {trend !== undefined && (
        <div style={{ marginTop: 8 }}>
          {trend > 0 ? (
            <ArrowUpOutlined style={{ color: '#3f8600' }} />
          ) : (
            <ArrowDownOutlined style={{ color: '#cf1322' }} />
          )}
          <span style={{ marginLeft: 4 }}>{Math.abs(trend)}%</span>
        </div>
      )}
    </Card>
  )
}

export default MetricCard