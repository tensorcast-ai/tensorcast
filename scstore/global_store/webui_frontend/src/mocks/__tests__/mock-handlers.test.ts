import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { setupServer } from 'msw/node';
import { handlers } from '../handlers';

const server = setupServer(...handlers);

beforeAll(() => server.listen());
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

describe('Mock API Handlers', () => {
  it('should return summary data with correct structure', async () => {
    const response = await fetch('/api/summary');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data).toHaveProperty('data');
    expect(data.data).toHaveProperty('total_workers');
    expect(data.data).toHaveProperty('active_workers');
    expect(data.data).toHaveProperty('total_replicas');
  });

  it('should return workers list with data wrapper', async () => {
    const response = await fetch('/api/workers');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data).toHaveProperty('data');
    expect(Array.isArray(data.data)).toBe(true);
    expect(data.data.length).toBeGreaterThan(0);
  });

  it('should return nodes list with data wrapper', async () => {
    const response = await fetch('/api/nodes');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data).toHaveProperty('data');
    expect(Array.isArray(data.data)).toBe(true);
    expect(data.data.length).toBeGreaterThan(0);

    // Check node structure
    const node = data.data[0];
    expect(node).toHaveProperty('node_id');
    expect(node).toHaveProperty('total_replicas');
    expect(node).toHaveProperty('total_memory');
  });

  it('should return transports with pagination metadata', async () => {
    const response = await fetch('/api/transports?page=1&page_size=2');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data).toHaveProperty('data');
    expect(data).toHaveProperty('meta');
    expect(Array.isArray(data.data)).toBe(true);

    // Check meta structure
    expect(data.meta).toHaveProperty('page');
    expect(data.meta).toHaveProperty('page_size');
    expect(data.meta).toHaveProperty('total_count');
    expect(data.meta).toHaveProperty('total_pages');

    // Check pagination
    expect(data.meta.page).toBe(1);
    expect(data.meta.page_size).toBe(2);
    expect(data.data.length).toBeLessThanOrEqual(2);
  });

  it('should filter transports by status', async () => {
    const response = await fetch('/api/transports?status=completed');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data.data.every((t: any) => t.status === 'completed')).toBe(true);
  });

  it('should filter replicas by worker_id', async () => {
    const response = await fetch('/api/replicas?worker_id=worker-1');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data.data.every((r: any) => r.worker_id === 'worker-1')).toBe(true);
  });

  it('should return 404 for non-existent worker', async () => {
    const response = await fetch('/api/workers/non-existent');

    expect(response.status).toBe(404);
  });

  it('should return single artifact with data wrapper', async () => {
    const response = await fetch('/api/artifacts/mi2:inx1:data1');
    const data = await response.json();

    expect(response.ok).toBe(true);
    expect(data).toHaveProperty('data');
    expect(data.data).toHaveProperty('artifact_id', 'mi2:inx1:data1');
  });
});