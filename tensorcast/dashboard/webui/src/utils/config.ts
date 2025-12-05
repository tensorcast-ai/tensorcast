export function metricsConfig() {
  const host = import.meta.env.VITE_GRAFANA_HOST as string | undefined;
  const dashboardUid = import.meta.env.VITE_GRAFANA_DASHBOARD_UID as string | undefined;
  const panelIdsRaw = import.meta.env.VITE_GRAFANA_PANEL_IDS as string | undefined;
  const enabled = Boolean(host && dashboardUid && panelIdsRaw);
  const panelIds = (panelIdsRaw ?? '')
    .split(',')
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return { enabled, host: host ?? '', dashboardUid: dashboardUid ?? '', panelIds };
}
