/**
 * Test script to verify mock API responses
 * Run this in browser console when mock server is active
 */

// Base URL for API requests
const API_BASE = '/api';

// Helper to make API calls and log results
async function testEndpoint(path: string, params?: Record<string, string>) {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, value);
    });
  }

  try {
    const response = await fetch(url.toString());
    const data = await response.json();
    console.log(`✅ ${path}:`, data);
    return { success: true, data };
  } catch (error) {
    console.error(`❌ ${path}:`, error);
    return { success: false, error };
  }
}

// Test all endpoints
export async function testAllMockEndpoints() {
  console.log('🧪 Testing Mock API Endpoints...\n');

  const tests: Array<{ name: string; path: string; params?: Record<string, string> }> = [
    // Basic endpoints
    { name: 'Summary', path: '/summary' },
    { name: 'Workers', path: '/workers' },
    { name: 'Nodes', path: '/nodes' },
    { name: 'Models', path: '/models' },

    // Paginated endpoint
    { name: 'Transports (page 1)', path: '/transports', params: { page: '1', page_size: '2' } },
    { name: 'Transports (filtered)', path: '/transports', params: { status: 'completed' } },

    // Filtered endpoints
    { name: 'Replicas (all)', path: '/replicas' },
    { name: 'Replicas (by worker)', path: '/replicas', params: { worker_id: 'worker-1' } },
    { name: 'Replicas (by model)', path: '/replicas', params: { model_name: 'text-classifier' } },

    // Single item endpoints
    { name: 'Single Worker', path: '/workers/worker-1' },
    { name: 'Single Model', path: '/models/text-classifier' },
    { name: 'Single Replica', path: '/replicas/replica-1' },
  ];

  for (const test of tests) {
    console.log(`\n📍 Testing: ${test.name}`);
    const result = await testEndpoint(test.path, test.params);

    // Validate response structure
    if (result.success && result.data) {
      if (test.path === '/transports') {
        if (!result.data.data || !result.data.meta) {
          console.error(`❌ Missing data or meta in transports response`);
        } else {
          console.log(`✅ Transports response has correct structure (data + meta)`);
        }
      } else if (result.data.data !== undefined) {
        console.log(`✅ Response has data field with ${Array.isArray(result.data.data) ? result.data.data.length : '1'} items`);
      }
    }
  }

  console.log('\n✅ Mock API testing complete!');
}

// Export for use in browser console
(window as any).testMockAPI = testAllMockEndpoints;