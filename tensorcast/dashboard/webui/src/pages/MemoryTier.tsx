import { useEffect, useMemo, useState } from 'react';
import { api, type MemoryTierLease } from '@/utils/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes, formatEpochNs, timeAgoNs } from '@/utils/utils';

const ALL_STATES = ['pending', 'active', 'revoking', 'expired'] as const;
type LeaseState = (typeof ALL_STATES)[number];

export default function MemoryTier() {
  const [nodeId, setNodeId] = useState('');
  const [stateFilters, setStateFilters] = useState<Set<LeaseState>>(new Set(['pending', 'active', 'revoking']));
  const [leases, setLeases] = useState<MemoryTierLease[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectedStates = useMemo(() => Array.from(stateFilters), [stateFilters]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      try {
        const resp = await api.memoryTierLeases({
          node_id: nodeId.trim() || undefined,
          states: selectedStates,
        });
        if (cancelled) return;
        setLeases(resp.leases);
        setError(null);
      } catch (e) {
        if (!cancelled) setError((e as Error).message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    const t = setInterval(load, 8000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, [nodeId, selectedStates]);

  const toggleState = (state: LeaseState) => {
    setStateFilters((prev) => {
      const next = new Set(prev);
      if (next.has(state)) {
        next.delete(state);
      } else {
        next.add(state);
      }
      if (next.size === 0) {
        return new Set(prev); // keep at least one
      }
      return next;
    });
  };

  const renderState = (state: string) => {
    const variant = state === 'active' ? 'secondary' : state === 'revoking' ? 'outline' : state === 'pending' ? 'default' : 'destructive';
    return <Badge variant={variant}>{state}</Badge>;
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Memory Tier</h1>
          <p className="text-sm text-muted-foreground">Latest UMA lease state (pending/active/revoking/expired) per node.</p>
        </div>
        <Button onClick={() => setStateFilters(new Set(['pending', 'active', 'revoking']))} variant="outline" size="sm">
          重置筛选
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <div className="space-y-1">
          <Label htmlFor="nodeId">按 node_id 过滤</Label>
          <Input
            id="nodeId"
            value={nodeId}
            onChange={(e) => setNodeId(e.target.value)}
            className="w-[220px]"
            placeholder="node-1"
          />
        </div>
        <div className="space-y-1">
          <Label>状态</Label>
          <div className="flex flex-wrap gap-2">
            {ALL_STATES.map((s) => (
              <label key={s} className="flex items-center gap-1 text-sm">
                <input
                  type="checkbox"
                  checked={stateFilters.has(s)}
                  onChange={() => toggleState(s)}
                  className="size-4 accent-foreground"
                />
                {s}
              </label>
            ))}
          </div>
        </div>
      </div>

      {error ? <div className="text-sm text-red-600">{error}</div> : null}

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Lease</TableHead>
            <TableHead>Node</TableHead>
            <TableHead>Kind</TableHead>
            <TableHead>Artifact</TableHead>
            <TableHead>Chunks</TableHead>
            <TableHead>Bytes</TableHead>
            <TableHead>Ledger</TableHead>
            <TableHead>State</TableHead>
            <TableHead>Times</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <TableRow key={i}>
                {Array.from({ length: 9 }).map((__, j) => (
                  <TableCell key={j}>
                    <Skeleton className="h-4 w-full max-w-[160px]" />
                  </TableCell>
                ))}
              </TableRow>
            ))
          ) : leases.length === 0 ? (
            <TableRow>
              <TableCell colSpan={9} className="p-3 text-muted-foreground">
                无数据
              </TableCell>
            </TableRow>
          ) : (
            leases.map((lease) => (
              <TableRow key={lease.lease_id}>
                <TableCell className="font-mono text-xs">{lease.lease_id}</TableCell>
                <TableCell>
                  <div className="font-mono text-xs">{lease.node_id}</div>
                </TableCell>
                <TableCell>
                  <Badge variant={lease.kind === 'preemptible' ? 'secondary' : 'outline'}>{lease.kind}</Badge>
                </TableCell>
                <TableCell className="font-mono text-xs">{lease.artifact_id}</TableCell>
                <TableCell className="text-xs">
                  <div className="font-mono">
                    range: [{lease.chunk_range.offset}, +{lease.chunk_range.length}]
                  </div>
                  {lease.chunk_ids.length ? (
                    <div className="text-muted-foreground">ids: {lease.chunk_ids.join(', ')}</div>
                  ) : null}
                </TableCell>
                <TableCell className="font-mono text-xs">{formatBytes(lease.bytes)}</TableCell>
                <TableCell className="font-mono text-xs">v{lease.ledger_version}</TableCell>
                <TableCell>{renderState(lease.state)}</TableCell>
                <TableCell className="text-xs">
                  <div>issued: {formatEpochNs(lease.issued_at_ns)}</div>
                  <div>ack: {lease.ack_epoch_ns ? formatEpochNs(lease.ack_epoch_ns) : '—'}</div>
                  <div className="text-muted-foreground">{timeAgoNs(lease.issued_at_ns)} </div>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}
