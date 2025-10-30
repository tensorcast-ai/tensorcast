import { useEffect, useMemo, useState } from 'react';
import { api, type Health, type WorkersResponse } from '@/lib/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes } from '@/lib/utils';
import { NavLink } from 'react-router-dom';

export default function Overview() {
  const [health, setHealth] = useState<Health | null>(null);
  const [workers, setWorkers] = useState<WorkersResponse['workers']>([]);
  const [replicasRAM, setReplicasRAM] = useState<number | null>(null);
  const [replicasGPU, setReplicasGPU] = useState<number | null>(null);
  const [replicasDISK, setReplicasDISK] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const [h, w, rRAM, rGPU, rDISK] = await Promise.all([
          api.health(),
          api.workers(true),
          api.replicas({ memory_type: 'RAM', page_size: 100 }),
          api.replicas({ memory_type: 'GPU', page_size: 100 }),
          api.replicas({ memory_type: 'DISK', page_size: 100 }),
        ]);
        if (cancelled) return;
        setHealth(h);
        setWorkers(w.workers);
        setReplicasRAM(rRAM.replicas.length);
        setReplicasGPU(rGPU.replicas.length);
        setReplicasDISK(rDISK.replicas.length);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      }
    }
    load();
    const t = setInterval(load, 15000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const memTotals = useMemo(() => {
    const total = workers.reduce((acc, w) => acc + w.mem_pool_total, 0);
    const avail = workers.reduce((acc, w) => acc + w.mem_pool_available, 0);
    return { total, avail };
  }, [workers]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Overview</h1>
        <Button asChild variant="outline" size="sm">
          <NavLink to="/workers">查看 Workers</NavLink>
        </Button>
      </div>

      {error ? (
        <div className="text-sm text-red-600">加载失败：{error}</div>
      ) : null}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Health</CardTitle>
          </CardHeader>
          <CardContent>
            {health ? (
              <div className="text-2xl font-semibold">{health.status}</div>
            ) : (
              <Skeleton className="h-8 w-24" />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Active Workers</CardTitle>
          </CardHeader>
          <CardContent>
            {workers ? (
              <div className="text-2xl font-semibold">{workers.length}</div>
            ) : (
              <Skeleton className="h-8 w-24" />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Pool Memory (Total)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{formatBytes(memTotals.total)}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Pool Memory (Available)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{formatBytes(memTotals.avail)}</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Replicas (RAM, first page)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{replicasRAM ?? '—'}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Replicas (GPU, first page)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{replicasGPU ?? '—'}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm text-muted-foreground">Replicas (DISK, first page)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-semibold">{replicasDISK ?? '—'}</div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm text-muted-foreground">快速入口</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            <Button asChild size="sm">
              <NavLink to="/replicas">浏览 Replicas</NavLink>
            </Button>
            <Button asChild size="sm" variant="outline">
              <NavLink to="/metrics">查看 Metrics</NavLink>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
