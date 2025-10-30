import { useEffect, useMemo, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { api, type WorkersResponse } from '@/utils/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes, timeAgo } from '@/utils/utils';

type Row = WorkersResponse['workers'][number];

export default function Workers() {
  const [searchParams, setSearchParams] = useSearchParams();
  const spInclude = (searchParams.get('include_unavailable') ?? '0') === '1';
  const spQuery = searchParams.get('q') ?? '';
  const [includeUnavailable, setIncludeUnavailable] = useState(spInclude);
  const [query, setQuery] = useState(spQuery);
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (r) =>
        r.worker_id.toLowerCase().includes(q) ||
        r.node_id.toLowerCase().includes(q) ||
        r.node_address.toLowerCase().includes(q),
    );
  }, [rows, query]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const data = await api.workers(includeUnavailable);
        if (cancelled) return;
        setRows(data.workers);
        setError(null);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setLoading(false);
      }
    }
    load();
    const t = setInterval(load, 10000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [includeUnavailable]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold">Workers</h1>
        <div className="flex items-center gap-3">
          <Label className="gap-2">
            <input
              type="checkbox"
              className="size-4 accent-foreground"
              checked={includeUnavailable}
              onChange={(e) => {
                const v = e.target.checked; setIncludeUnavailable(v);
                setSearchParams((prev) => {
                  const next = new URLSearchParams(prev);
                  if (v) next.set('include_unavailable', '1'); else next.delete('include_unavailable');
                  return next;
                });
              }}
            />
            显示不可用
          </Label>
          <div className="w-[220px]">
            <Input
              value={query}
              onChange={(e) => { const v = e.target.value; setQuery(v); setSearchParams((prev) => {
                const next = new URLSearchParams(prev);
                if (v) next.set('q', v); else next.delete('q');
                return next;
              }); }}
              placeholder="搜索 worker/node"
            />
          </div>
          <Button onClick={() => setQuery('')} variant="outline" size="sm">
            清空
          </Button>
        </div>
      </div>

      {error ? <div className="text-sm text-red-600">{error}</div> : null}

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Worker</TableHead>
            <TableHead>Node</TableHead>
            <TableHead>Address</TableHead>
            <TableHead>Ports</TableHead>
            <TableHead>Pool</TableHead>
            <TableHead>Accept</TableHead>
            <TableHead>Heartbeat</TableHead>
            <TableHead>Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <>
              {Array.from({ length: 3 }).map((_, i) => (
                <TableRow key={i}>
                  {Array.from({ length: 8 }).map((__, j) => (
                    <TableCell key={j}><Skeleton className="h-4 w-full max-w-[160px]" /></TableCell>
                  ))}
                </TableRow>
              ))}
            </>
          ) : filtered.length === 0 ? (
            <TableRow>
              <TableCell colSpan={8} className="p-3 text-muted-foreground">无数据</TableCell>
            </TableRow>
          ) : (
            filtered.map((r) => (
              <TableRow key={r.worker_id}>
                <TableCell className="font-mono">{r.worker_id}</TableCell>
                <TableCell className="font-mono">{r.node_id}</TableCell>
                <TableCell>{r.node_address}</TableCell>
                <TableCell>
                  <span className="font-mono">gRPC {r.grpc_port}</span>
                  <span className="mx-2 text-muted-foreground">/</span>
                  <span className="font-mono">P2P {r.p2p_port}</span>
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span>{formatBytes(r.mem_pool_available)}</span>
                    <span className="text-muted-foreground">/</span>
                    <span>{formatBytes(r.mem_pool_total)}</span>
                  </div>
                </TableCell>
                <TableCell>
                  {r.accepting_new_requests ? (
                    <Badge variant="secondary">YES</Badge>
                  ) : (
                    <Badge variant="destructive">NO</Badge>
                  )}
                </TableCell>
                <TableCell className="text-muted-foreground">{timeAgo(r.last_heartbeat_ts)}</TableCell>
                <TableCell>
                  <Badge>{r.status}</Badge>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
