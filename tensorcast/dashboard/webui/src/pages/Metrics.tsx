import { metricsConfig } from '@/lib/config';

export default function Metrics() {
  const cfg = metricsConfig();
  if (!cfg.enabled) {
    return (
      <div className="space-y-2">
        <h1 className="text-2xl font-semibold">Metrics</h1>
        <div className="text-muted-foreground">未配置 GRAFANA_ 参数，Metrics 页面不可用。</div>
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


