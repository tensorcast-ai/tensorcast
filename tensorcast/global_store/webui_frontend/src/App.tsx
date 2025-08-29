import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Layout } from 'antd'
import Sidebar from './components/Sidebar'
import Dashboard from './pages/Dashboard'
import Workers from './pages/Workers'
import Replicas from './pages/Replicas'
import Artifacts from './pages/Artifacts'
import ArtifactDetail from './pages/ArtifactDetail'
import Transports from './pages/Transports'
import Nodes from './pages/Nodes'
import MockTest from './pages/MockTest'
import './App.css'

const { Content } = Layout

function App() {
  return (
    <Router>
      <Layout style={{ minHeight: '100vh' }}>
        <Sidebar />
        <Layout>
          <Content style={{ padding: '24px' }}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/workers" element={<Workers />} />
              <Route path="/replicas" element={<Replicas />} />
              <Route path="/artifacts" element={<Artifacts />} />
              <Route path="/artifacts/:artifactId" element={<ArtifactDetail />} />
              <Route path="/transports" element={<Transports />} />
              <Route path="/nodes" element={<Nodes />} />
              {import.meta.env.DEV && <Route path="/mock-test" element={<MockTest />} />}
            </Routes>
          </Content>
        </Layout>
      </Layout>
    </Router>
  )
}

export default App