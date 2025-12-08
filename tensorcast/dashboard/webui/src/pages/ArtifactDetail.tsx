import { useEffect, useState } from 'react';
import { NavLink, useParams } from 'react-router-dom';
import { api, type ArtifactDetailResponse } from '@/utils/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { formatBytes, formatDateTime } from '@/utils/utils';

export default function ArtifactDetail() {
  const { artifactId = '' } = useParams();
  const [data, setData] = useState<ArtifactDetailResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const d = await api.artifactDetail(artifactId, ['replicas', 'view']);
        if (cancelled) return;
        setData(d);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }
    if (artifactId) load();
    return () => {
      cancelled = true;
    };
  }, [artifactId]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h1 className="text-2xl font-semibold">Artifact: {artifactId}</h1>
          {data?.artifact_kind ? (
            <div className="text-xs text-muted-foreground">Kind: {data.artifact_kind}</div>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          <Button asChild size="sm">
            <NavLink to={`/replicas?artifact_id=${encodeURIComponent(artifactId)}`}>查看关联 Replicas</NavLink>
          </Button>
        </div>
      </div>

      {error ? <div className="text-sm text-red-600">{error}</div> : null}

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Artifact Descriptor</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {loading ? (
              <div className="text-muted-foreground">加载中...</div>
            ) : data?.descriptor ? (
              <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
                <dt className="text-muted-foreground">ID Kind</dt>
                <dd>{data.descriptor.id_kind ?? '—'}</dd>
                <dt className="text-muted-foreground">Index Hash</dt>
                <dd className="font-mono break-all">{data.descriptor.index_multihash ?? '—'}</dd>
                <dt className="text-muted-foreground">Data Hash</dt>
                <dd className="font-mono break-all">{data.descriptor.data_multihash ?? '—'}</dd>
                <dt className="text-muted-foreground">Schema</dt>
                <dd>{data.descriptor.schema_version ?? '—'}</dd>
                <dt className="text-muted-foreground">Encoding</dt>
                <dd>{data.descriptor.encoding ?? '—'}</dd>
                <dt className="text-muted-foreground">Total Size</dt>
                <dd>{data.descriptor.total_size ? formatBytes(data.descriptor.total_size) : '—'}</dd>
              </dl>
            ) : (
              <div className="text-muted-foreground">无描述信息</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">View 元数据</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {loading ? (
              <div className="text-muted-foreground">加载中...</div>
            ) : data?.view_meta ? (
              <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
                <dt className="text-muted-foreground">View Size</dt>
                <dd>{formatBytes(data.view_meta.view_size)}</dd>
                <dt className="text-muted-foreground">Data Hash</dt>
                <dd className="font-mono break-all">{data.view_meta.view_data_hash}</dd>
                <dt className="text-muted-foreground">Verified At</dt>
                <dd>{data.view_meta.verified_at ? formatDateTime(data.view_meta.verified_at) : '—'}</dd>
              </dl>
            ) : (
              <div className="text-muted-foreground">无视图元数据</div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Replicas</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {loading ? (
              <div className="text-muted-foreground">加载中...</div>
            ) : !data?.replicas || data.replicas.length === 0 ? (
              <div className="text-muted-foreground">无副本</div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Node</TableHead>
                    <TableHead>Memory</TableHead>
                    <TableHead>Bytes</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Expires</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.replicas.map((r, idx) => (
                    <TableRow key={`${r.node_id}-${r.device_id ?? idx}`}>
                      <TableCell>
                        {r.node_address} <span className="text-muted-foreground">({r.node_id})</span>
                      </TableCell>
                      <TableCell>{r.memory_type}</TableCell>
                      <TableCell>{formatBytes(r.bytes)}</TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {r.created_ts ? formatDateTime(r.created_ts) : '—'}
                      </TableCell>
                      <TableCell className="font-mono text-xs text-muted-foreground">
                        {r.expires_at ? formatDateTime(r.expires_at) : '—'}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
