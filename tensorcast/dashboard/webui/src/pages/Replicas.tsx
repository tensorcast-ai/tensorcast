import { useEffect, useMemo, useState } from 'react';
import { api, type ReplicasResponse } from '@/utils/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Skeleton } from '@/components/ui/skeleton';
import { formatBytes } from '@/utils/utils';
import { NavLink, useSearchParams } from 'react-router-dom';

type Row = ReplicasResponse['replicas'][number];

export default function Replicas() {
  const [searchParams, setSearchParams] = useSearchParams();
  const spArtifact = searchParams.get('artifact_id') ?? '';
  const spNodeAddress = searchParams.get('node_address') ?? '';
  const spMemory = (searchParams.get('memory_type') as 'RAM' | 'GPU' | 'DISK' | null) ?? null;
  const spDevice = searchParams.get('device_id') ?? '';
  const spPageSize = Number(searchParams.get('page_size') ?? '100') || 100;
  const spPageToken = searchParams.get('page_token');

  const [artifactId, setArtifactId] = useState(spArtifact);
  const [nodeAddress, setNodeAddress] = useState(spNodeAddress);
  const [memoryType, setMemoryType] = useState<'ALL' | 'RAM' | 'GPU' | 'DISK'>(spMemory ?? 'ALL');
  const [deviceId, setDeviceId] = useState<string>(spDevice);
  const [pageSize, setPageSize] = useState<number>(spPageSize);
  const [pageToken, setPageToken] = useState<string | null>(spPageToken);
  const [prevTokens, setPrevTokens] = useState<string[]>([]);

  const [rows, setRows] = useState<Row[]>([]);
  const [nextToken, setNextToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const canPrev = prevTokens.length > 0;
  const canNext = !!nextToken;

  const params = useMemo(
    () => {
      const memTypeParam = memoryType === 'ALL' ? undefined : memoryType;
      return {
        artifact_id: artifactId || undefined,
        node_address: nodeAddress || undefined,
        memory_type: memTypeParam,
        device_id: deviceId === '' ? undefined : Number(deviceId),
        page_token: pageToken,
        page_size: pageSize,
      };
    },
    [artifactId, nodeAddress, memoryType, deviceId, pageToken, pageSize],
  );

  async function load(currentParams = params) {
    setLoading(true);
    try {
      const data = await api.replicas(currentParams);
      setRows(data.replicas);
      setNextToken(data.page_info?.next_page_token ?? null);
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [artifactId, nodeAddress, memoryType, deviceId, pageSize, pageToken]);

  function onApplyFilters() {
    setPrevTokens([]);
    setPageToken(null);
    // 同步 URL 查询参数（不包含 page_token）
    const next = new URLSearchParams();
    if (artifactId) next.set('artifact_id', artifactId);
    if (nodeAddress) next.set('node_address', nodeAddress);
    if (memoryType !== 'ALL') next.set('memory_type', memoryType);
    if (deviceId !== '') next.set('device_id', deviceId);
    if (pageSize && pageSize !== 100) next.set('page_size', String(pageSize));
    setSearchParams(next, { replace: false });
  }

  function onNext() {
    if (!nextToken) return;
    setPrevTokens((s) => [...s, pageToken ?? '']);
    setPageToken(nextToken);
    // 更新 URL 的 page_token，保留现有筛选
    setSearchParams((prev) => {
      const next = new URLSearchParams(prev);
      if (nextToken) next.set('page_token', nextToken);
      return next;
    });
  }

  function onPrev() {
    if (prevTokens.length === 0) return;
    const copy = [...prevTokens];
    const prev = copy.pop()!;
    setPrevTokens(copy);
    setPageToken(prev || null);
    setSearchParams((sp) => {
      const next = new URLSearchParams(sp);
      if (prev) next.set('page_token', prev);
      else next.delete('page_token');
      return next;
    });
  }

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Replicas</h1>

      <div className="rounded-md border p-3">
        <div className="grid gap-3 md:grid-cols-3">
          <Label className="flex items-center gap-2">
            <span className="w-28 text-right text-muted-foreground">Artifact ID</span>
            <Input
              value={artifactId}
              onChange={(e) => setArtifactId(e.target.value)}
              placeholder="mi2:... 或 af-..."
            />
          </Label>

          <Label className="flex items-center gap-2">
            <span className="w-28 text-right text-muted-foreground">Node Address</span>
            <Input
              value={nodeAddress}
              onChange={(e) => setNodeAddress(e.target.value)}
              placeholder="10.0.0.x"
            />
          </Label>

          <Label className="flex items-center gap-2">
            <span className="w-28 text-right text-muted-foreground">Memory Type</span>
            <Select value={memoryType} onValueChange={(v) => setMemoryType(v as typeof memoryType)}>
              <SelectTrigger className="flex-1"><SelectValue placeholder="All" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="ALL">All</SelectItem>
                <SelectItem value="RAM">RAM</SelectItem>
                <SelectItem value="GPU">GPU</SelectItem>
                <SelectItem value="DISK">DISK</SelectItem>
              </SelectContent>
            </Select>
          </Label>

          <Label className="flex items-center gap-2">
            <span className="w-28 text-right text-muted-foreground">Device ID</span>
            <Input
              value={deviceId}
              onChange={(e) => setDeviceId(e.target.value)}
              type="number"
              placeholder="可选"
            />
          </Label>

          <Label className="flex items-center gap-2">
            <span className="w-28 text-right text-muted-foreground">Page Size</span>
            <Input
              value={pageSize}
              onChange={(e) => setPageSize(Number(e.target.value) || 100)}
              type="number"
              min={1}
              max={500}
              className="w-28"
            />
          </Label>

          <div className="flex items-center justify-end gap-2 md:col-span-2">
            <Button size="sm" variant="outline" onClick={() => {
              setArtifactId(''); setNodeAddress(''); setMemoryType('ALL'); setDeviceId('');
            }}>重置</Button>
            <Button size="sm" onClick={onApplyFilters}>应用筛选</Button>
          </div>
        </div>
      </div>

      {error ? <div className="text-sm text-red-600">{error}</div> : null}

      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Artifact</TableHead>
            <TableHead>Replica</TableHead>
            <TableHead>Node</TableHead>
            <TableHead>Device</TableHead>
            <TableHead>Memory</TableHead>
            <TableHead>Bytes</TableHead>
            <TableHead>State</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {loading ? (
            <>
              {Array.from({ length: 3 }).map((_, i) => (
                <TableRow key={i}>
                  {Array.from({ length: 7 }).map((__, j) => (
                    <TableCell key={j}><Skeleton className="h-4 w-full max-w-[160px]" /></TableCell>
                  ))}
                </TableRow>
              ))}
            </>
          ) : rows.length === 0 ? (
            <TableRow>
              <TableCell className="p-3 text-muted-foreground" colSpan={7}>无数据</TableCell>
            </TableRow>
          ) : (
            rows.map((r) => (
              <TableRow key={`${r.artifact_id}|${r.node_id}|${r.device_id ?? '-'}|${r.memory_type}`}>
                <TableCell className="font-mono">
                  <NavLink className="underline-offset-2 hover:underline" to={`/artifacts/${encodeURIComponent(r.artifact_id)}`}>
                    {r.artifact_id}
                  </NavLink>
                </TableCell>
                <TableCell className="font-mono">{`${r.node_id}:${r.memory_type}:${r.device_id ?? '-'}`}</TableCell>
                <TableCell>
                  {r.node_address} <span className="text-muted-foreground">({r.node_id})</span>
                </TableCell>
                <TableCell>{r.device_id ?? '—'}</TableCell>
                <TableCell>{r.memory_type}</TableCell>
                <TableCell>{formatBytes(r.bytes)}</TableCell>
                <TableCell>
                  <Badge>{r.state ?? '—'}</Badge>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>

      <div className="flex items-center justify-between">
        <div className="text-xs text-muted-foreground">
          {rows.length} items {nextToken ? '(has next)' : ''}
        </div>
        <div className="flex items-center gap-2">
          <Button size="sm" variant="outline" onClick={onPrev} disabled={!canPrev}>上一页</Button>
          <Button size="sm" onClick={onNext} disabled={!canNext}>下一页</Button>
        </div>
      </div>
    </div>
  );
}
