import { useEffect, useState } from 'react';
import { NavLink, useParams } from 'react-router-dom';
import { api, type ArtifactDetailResponse } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';

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
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Artifact: {artifactId}</h1>
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
            <CardTitle className="text-sm">View 元数据</CardTitle>
          </CardHeader>
          <CardContent className="text-sm">
            {loading ? (
              <div className="text-muted-foreground">加载中...</div>
            ) : data?.view_meta ? (
              <dl className="grid grid-cols-2 gap-x-3 gap-y-2">
                <dt className="text-muted-foreground">View ID</dt>
                <dd className="font-mono">{data.view_meta.view_id}</dd>
                <dt className="text-muted-foreground">Total Leaves</dt>
                <dd>{data.view_meta.total_leaves}</dd>
                <dt className="text-muted-foreground">Schema Version</dt>
                <dd>{data.view_meta.schema_version}</dd>
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
                    <TableHead>Replica</TableHead>
                    <TableHead>Node</TableHead>
                    <TableHead>Memory</TableHead>
                    <TableHead>State</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {data.replicas.map((r) => (
                    <TableRow key={r.replica_id}>
                      <TableCell className="font-mono">{r.replica_id}</TableCell>
                      <TableCell>{r.node_id}</TableCell>
                      <TableCell>{r.memory_type}</TableCell>
                      <TableCell><Badge>{r.state}</Badge></TableCell>
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
