import { metricsConfig } from '@/lib/config';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { NavLink } from 'react-router-dom';

export default function Metrics() {
  const cfg = metricsConfig();
  if (!cfg.enabled) {
    return (
      <div className="space-y-4">
        <h1 className="text-2xl font-semibold">Metrics</h1>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">未配置</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-sm text-muted-foreground">未检测到 Grafana 嵌入配置（VITE_GRAFANA_HOST、VITE_GRAFANA_DASHBOARD_UID、VITE_GRAFANA_PANEL_IDS）。</div>
            <div className="mt-3">
              <Button asChild size="sm" variant="outline">
                <NavLink to="/">返回概览</NavLink>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Metrics</h1>
      <div className="grid gap-4 md:grid-cols-2">
        {cfg.panelIds.map((id) => (
          <iframe
            key={id}
            src={`${cfg.host}/d/${cfg.dashboardUid}?viewPanel=${id}`}
            className="w-full h-[480px] border rounded"
            title={`grafana-panel-${id}`}
          />
        ))}
      </div>
    </div>
  );
}
