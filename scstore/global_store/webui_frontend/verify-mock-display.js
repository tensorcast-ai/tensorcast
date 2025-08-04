#!/usr/bin/env node

/**
 * Simple script to verify mock data endpoints are returning correct format
 * Run this after starting the mock server: pnpm run dev:mock
 */

const http = require('http');

const endpoints = [
  { path: '/api/summary', name: 'Summary' },
  { path: '/api/workers', name: 'Workers' },
  { path: '/api/nodes', name: 'Nodes' },
  { path: '/api/transports?page=1&page_size=5', name: 'Transports' },
  { path: '/api/replicas', name: 'Replicas' },
];

function testEndpoint(endpoint) {
  return new Promise((resolve) => {
    const port = process.env.VITE_PORT || 5173;
    const options = {
      hostname: 'localhost',
      port: port,
      path: endpoint.path,
      method: 'GET',
    };

    const req = http.request(options, (res) => {
      let data = '';
      res.on('data', (chunk) => {
        data += chunk;
      });
      res.on('end', () => {
        try {
          const json = JSON.parse(data);
          const hasData = 'data' in json;
          const hasMeta = 'meta' in json;
          const dataLength = hasData && Array.isArray(json.data) ? json.data.length : 'N/A';

          resolve({
            endpoint: endpoint.name,
            path: endpoint.path,
            success: true,
            hasData,
            hasMeta,
            dataLength,
            sample: hasData && Array.isArray(json.data) && json.data[0] ?
              Object.keys(json.data[0]).slice(0, 3).join(', ') + '...' :
              'N/A'
          });
        } catch (e) {
          resolve({
            endpoint: endpoint.name,
            path: endpoint.path,
            success: false,
            error: e.message
          });
        }
      });
    });

    req.on('error', (e) => {
      resolve({
        endpoint: endpoint.name,
        path: endpoint.path,
        success: false,
        error: e.message
      });
    });

    req.end();
  });
}

async function runTests() {
  console.log('🧪 Testing Mock API Endpoints...\n');
  console.log('Make sure the dev server is running: pnpm run dev:mock\n');

  const results = [];
  for (const endpoint of endpoints) {
    const result = await testEndpoint(endpoint);
    results.push(result);
  }

  // Display results in a table
  console.log('Results:');
  console.log('========================================');
  console.log('Endpoint      | Data | Meta | Items | Sample Fields');
  console.log('--------------|------|------|-------|---------------');

  results.forEach(r => {
    if (r.success) {
      console.log(
        `${r.endpoint.padEnd(13)} | ${r.hasData ? '✅' : '❌'}   | ${r.hasMeta ? '✅' : '  '}   | ${
          String(r.dataLength).padEnd(5)
        } | ${r.sample}`
      );
    } else {
      console.log(`${r.endpoint.padEnd(13)} | ❌ Error: ${r.error}`);
    }
  });

  console.log('\nSpecial checks:');
  const transportsResult = results.find(r => r.endpoint === 'Transports');
  if (transportsResult && transportsResult.success && transportsResult.hasMeta) {
    console.log('✅ Transports endpoint has pagination metadata (meta field)');
  } else {
    console.log('❌ Transports endpoint missing pagination metadata');
  }

  const nodesResult = results.find(r => r.endpoint === 'Nodes');
  if (nodesResult && nodesResult.success && nodesResult.hasData) {
    console.log('✅ Nodes endpoint returns data in correct format');
  } else {
    console.log('❌ Nodes endpoint data format issue');
  }

  console.log('\n✅ Test complete!');
}

runTests().catch(console.error);