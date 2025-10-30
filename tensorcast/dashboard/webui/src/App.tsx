import { NavLink, Outlet } from 'react-router-dom';
import { metricsConfig } from '@/lib/config';

export default function App() {
  const cfg = metricsConfig();
  return (
    <div className="p-4">
      <nav className="flex gap-4 mb-6">
        <NavLink to="/" end className="hover:underline">
          Overview
        </NavLink>
        <NavLink to="/workers" className="hover:underline">
          Workers
        </NavLink>
        <NavLink to="/replicas" className="hover:underline">
          Replicas
        </NavLink>
        {cfg.enabled ? (
          <NavLink to="/metrics" className="hover:underline">
            Metrics
          </NavLink>
        ) : null}
      </nav>
      <Outlet />
    </div>
  );
}
