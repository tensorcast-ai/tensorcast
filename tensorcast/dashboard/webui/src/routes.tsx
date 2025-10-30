import { createBrowserRouter } from 'react-router-dom';
import App from './App';
import Overview from './pages/Overview';
import Workers from './pages/Workers';
import Replicas from './pages/Replicas';
import ArtifactDetail from './pages/ArtifactDetail';
import Metrics from './pages/Metrics';

export const router = createBrowserRouter(
  [
    {
      path: '/',
      element: <App />,
      children: [
        { index: true, element: <Overview /> },
        { path: 'workers', element: <Workers /> },
        { path: 'replicas', element: <Replicas /> },
        { path: 'artifacts/:artifactId', element: <ArtifactDetail /> },
        { path: 'metrics', element: <Metrics /> },
      ],
    },
  ],
  { basename: import.meta.env.BASE_URL },
);
