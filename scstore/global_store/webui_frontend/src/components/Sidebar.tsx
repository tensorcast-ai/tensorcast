import { Layout, Menu } from 'antd'
import { useNavigate, useLocation } from 'react-router-dom'
import {
  DashboardOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  AppstoreOutlined,
  SwapOutlined,
  ClusterOutlined,
  BugOutlined,
} from '@ant-design/icons'

const { Sider } = Layout

const Sidebar = () => {
  const navigate = useNavigate()
  const location = useLocation()

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/workers',
      icon: <CloudServerOutlined />,
      label: 'Workers',
    },
    {
      key: '/replicas',
      icon: <DatabaseOutlined />,
      label: 'Replicas',
    },
    {
      key: '/artifacts',
      icon: <AppstoreOutlined />,
      label: 'Artifacts',
    },
    {
      key: '/transports',
      icon: <SwapOutlined />,
      label: 'Transports',
    },
    {
      key: '/nodes',
      icon: <ClusterOutlined />,
      label: 'Nodes',
    },
    ...(import.meta.env.DEV && import.meta.env.VITE_MOCK_API === 'true' ? [{
      key: '/mock-test',
      icon: <BugOutlined />,
      label: 'Mock API Test',
    }] : []),
  ]

  return (
    <Sider width={200} style={{ background: '#fff' }}>
      <div style={{ height: 64, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <h2 style={{ margin: 0 }}>Global Store</h2>
      </div>
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        style={{ height: '100%', borderRight: 0 }}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
      />
    </Sider>
  )
}

export default Sidebar