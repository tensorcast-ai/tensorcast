import { createBrowserRouter } from 'react-router-dom';
import Layout from './components/Layout';
import Overview from './pages/Overview';
import Workers from './pages/Workers';
import Replicas from './pages/Replicas';
import ArtifactDetail from './pages/ArtifactDetail';
import Metrics from './pages/Metrics';
import MemoryTier from './pages/MemoryTier';

export const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <Layout />,
      children: [
        { index: true, element: <Overview /> },
        { path: 'workers', element: <Workers /> },
        { path: 'memory-tier', element: <MemoryTier /> },
        { path: 'replicas', element: <Replicas /> },
        { path: 'artifacts/:artifactId', element: <ArtifactDetail /> },
        { path: 'metrics', element: <Metrics /> },
      ],
    },
  ],
  { basename: import.meta.env.BASE_URL },
);
