## TensorCast Dashboard Web UI

前端基于 React + TypeScript + Vite + Tailwind（shadcn/ui）。对应设计文档：`docs/designs/0021-global-store-dashboard.md`。

### 路由结构（MVP）

- `/`：Overview
- `/workers`：Workers 列表
- `/replicas`：Replicas 列表
- `/artifacts/:artifactId`：Artifact 详情
- `/metrics`：Metrics（仅在配置了 Grafana 相关变量时显示入口）

Router 使用 `createBrowserRouter` 并通过 `basename` 与 Vite 的 `base` 对齐，实现子路径部署。

### 环境变量

前端使用 Vite 约定的 `VITE_` 前缀变量：

- `VITE_BASE_PATH`：静态站点子路径前缀（如 `/tensorcast-dashboard/`）。默认 `'/'`。
- `VITE_GRAFANA_HOST`：Grafana 基础地址（可选）。
- `VITE_GRAFANA_DASHBOARD_UID`：Grafana Dashboard UID（可选）。
- `VITE_GRAFANA_PANEL_IDS`：以逗号分隔的 Panel ID 列表（可选）。

当三者（HOST/UID/PANEL_IDS）均存在时，前端显示 Metrics 页面并以内嵌 iframe 的方式展示指定面板；否则隐藏菜单并在直接访问 `/metrics` 时提示未配置。

安全提示：不要在前端注入任何 Grafana Token。若需要鉴权，请在反向代理侧处理或启用匿名只读模式。

### 本地开发

```bash
pnpm install
pnpm dev
```

可在项目根目录创建 `.env.local` 注入前端变量，例如：

```
VITE_BASE_PATH=/tensorcast-dashboard/
VITE_GRAFANA_HOST=https://grafana.example.com
VITE_GRAFANA_DASHBOARD_UID=abcd1234
VITE_GRAFANA_PANEL_IDS=1,2,3
```

### 构建与输出

```bash
pnpm build
```

构建产物输出至 `tensorcast/dashboard/static/`，用于被后端打包与服务（见设计文档 Packaging 部分）。

### 子路径部署说明

- 前端：设置 `VITE_BASE_PATH`（例如 `/tensorcast-dashboard/`），Vite 会在构建时写入静态资源路径，Router 使用 `basename=import.meta.env.BASE_URL` 保持一致。
- 后端：通过 `BASE_PATH` 在 ASGI 层挂载相同前缀以正确分发静态与 API 路径（见设计文档）。

### 后续接入

- Overview/Workers/Replicas/ArtifactDetail 页面已就绪占位，将通过 Dashboard 后端 REST API 接入数据。
- Metrics 页面只做外部面板嵌入，不直接读取 Prometheus。


