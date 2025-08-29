import { test, expect } from '@playwright/test';

// Start dev server with mock API before running tests
test.beforeAll(async () => {
  console.log('Make sure to run "pnpm run dev:mock" before running these tests');
});

test.describe('Mock Data Display Tests', () => {
  test.beforeEach(async ({ page }) => {
    // Set up console message listener to verify mock API is enabled
    page.on('console', msg => {
      if (msg.text().includes('[Mock API] MSW enabled')) {
        console.log('✅ Mock API is active');
      }
    });
  });

  test('Transports page displays mock data correctly', async ({ page }) => {
    // Navigate to transports page
    await page.goto('http://localhost:5173/transports');

    // Wait for the page to load
    await page.waitForLoadState('networkidle');

    // Check page title
    await expect(page.locator('h1')).toContainText('Artifact Transports');

    // Check statistics cards are displayed
    await expect(page.locator('.ant-statistic-title:has-text("In Progress")')).toBeVisible();
    await expect(page.locator('.ant-statistic-title:has-text("Completed")')).toBeVisible();
    await expect(page.locator('.ant-statistic-title:has-text("Failed")')).toBeVisible();
    await expect(page.locator('.ant-statistic-title:has-text("Avg Duration")')).toBeVisible();

    // Verify statistics values (based on mock data)
    const inProgressStat = await page.locator('.ant-statistic-content').first().textContent();
    expect(inProgressStat).toBe('1'); // 1 in-progress transport in mock data

    // Check table is displayed with data
    await expect(page.locator('.ant-table')).toBeVisible();

    // Verify table has rows (mock data has 5 transports)
    const rows = page.locator('.ant-table-tbody tr');
    await expect(rows).toHaveCount(5);

    // Verify first row contains expected data
    const firstRow = rows.first();
    await expect(firstRow).toContainText('trans-1');
    await expect(firstRow).toContainText('text-classifier');
    await expect(firstRow).toContainText('node-a');
    await expect(firstRow).toContainText('IN PROGRESS');

    // Test filtering by status
    await page.locator('.ant-select').first().click();
    await page.locator('.ant-select-item-option[title="Completed"]').click();

    // Wait for filter to apply
    await page.waitForTimeout(500);

    // Verify only completed transports are shown
    const filteredRows = page.locator('.ant-table-tbody tr');
    const filteredCount = await filteredRows.count();
    expect(filteredCount).toBe(3); // 3 completed transports in mock data

    // Verify all visible rows show "COMPLETED" status
    for (let i = 0; i < filteredCount; i++) {
      await expect(filteredRows.nth(i)).toContainText('COMPLETED');
    }

    console.log('✅ Transports page test passed');
  });

  test('Nodes page displays mock data correctly', async ({ page }) => {
    // Navigate to nodes page
    await page.goto('http://localhost:5173/nodes');

    // Wait for the page to load
    await page.waitForLoadState('networkidle');

    // Check page title
    await expect(page.locator('h1')).toContainText('Node Overview');

    // Check statistics cards
    await expect(page.locator('.ant-statistic-title:has-text("Total Nodes")')).toBeVisible();
    await expect(page.locator('.ant-statistic-title:has-text("Total Workers")')).toBeVisible();
    await expect(page.locator('.ant-statistic-title:has-text("Total Replicas")')).toBeVisible();
    await expect(page.locator('.ant-statistic-title:has-text("Total Memory")')).toBeVisible();

    // Verify statistics values (based on mock data: 2 nodes)
    const totalNodesContent = await page.locator('.ant-statistic-content').first().textContent();
    expect(totalNodesContent?.trim()).toMatch(/2/); // 2 nodes in mock data

    // Check table is displayed
    await expect(page.locator('.ant-table')).toBeVisible();

    // Verify table has correct number of rows
    const rows = page.locator('.ant-table-tbody tr');
    await expect(rows).toHaveCount(2); // 2 nodes in mock data

    // Verify node IDs
    await expect(rows.first()).toContainText('node-a');
    await expect(rows.nth(1)).toContainText('node-b');

    // Verify memory distribution progress bars are displayed
    const progressBars = page.locator('.ant-progress');
    const progressCount = await progressBars.count();
    expect(progressCount).toBeGreaterThan(0); // Should have progress bars for memory distribution

    // Verify memory values are formatted (e.g., "48 GB")
    const memoryCell = await page.locator('.ant-table-cell').filter({ hasText: 'GB' }).first();
    await expect(memoryCell).toBeVisible();

    // Test sorting by clicking column header
    await page.locator('.ant-table-column-sorters').first().click();
    await page.waitForTimeout(500);

    // Verify table still has data after sorting
    await expect(rows).toHaveCount(2);

    console.log('✅ Nodes page test passed');
  });

  test('Mock API test page works correctly', async ({ page }) => {
    // Navigate to mock test page
    await page.goto('http://localhost:5173/mock-test');

    // Wait for the page to load
    await page.waitForLoadState('networkidle');

    // Check page title
    await expect(page.locator('h1')).toContainText('Mock API Test Dashboard');

    // Click "Run All Tests" button
    await page.locator('button:has-text("Run All Tests")').click();

    // Wait for tests to complete (with timeout)
    await page.waitForSelector('.ant-tag:has-text("Success:")', { timeout: 10000 });

    // Verify test results
    const successTag = await page.locator('.ant-tag:has-text("Success:")').textContent();
    const successCount = parseInt(successTag?.match(/\d+/)?.[0] || '0');
    expect(successCount).toBeGreaterThan(0);

    // Check that test summary table is displayed
    await expect(page.locator('.ant-table')).toBeVisible();

    // Verify some endpoints passed
    const successStatuses = page.locator('.ant-tag:has-text("SUCCESS")');
    const successCount2 = await successStatuses.count();
    expect(successCount2).toBeGreaterThan(0);

    console.log('✅ Mock API test page works correctly');
  });

  test('Navigation between pages works', async ({ page }) => {
    // Start at home page
    await page.goto('http://localhost:5173/');

    // Navigate to Transports
    await page.locator('.ant-menu-item:has-text("Transports")').click();
    await expect(page).toHaveURL('http://localhost:5173/transports');
    await expect(page.locator('h1')).toContainText('Artifact Transports');

    // Navigate to Nodes
    await page.locator('.ant-menu-item:has-text("Nodes")').click();
    await expect(page).toHaveURL('http://localhost:5173/nodes');
    await expect(page.locator('h1')).toContainText('Node Overview');

    // Navigate to Mock Test (should be visible in dev mode)
    await page.locator('.ant-menu-item:has-text("Mock API Test")').click();
    await expect(page).toHaveURL('http://localhost:5173/mock-test');
    await expect(page.locator('h1')).toContainText('Mock API Test Dashboard');

    console.log('✅ Navigation test passed');
  });
});

test.describe('Data Consistency Tests', () => {
  test('Transports pagination works correctly', async ({ page }) => {
    await page.goto('http://localhost:5173/transports');
    await page.waitForLoadState('networkidle');

    // Check pagination info
    const paginationInfo = await page.locator('.ant-pagination-total-text').textContent();
    expect(paginationInfo).toContain('Total 5 transports');

    // Change page size
    await page.locator('.ant-pagination-options-size-changer').click();
    await page.locator('.ant-select-item-option[title="10 / page"]').click();

    // Wait for table to update
    await page.waitForTimeout(500);

    // Verify all items are shown on one page now
    const rows = page.locator('.ant-table-tbody tr');
    await expect(rows).toHaveCount(5);

    console.log('✅ Pagination test passed');
  });

  test('Mock data matches expected schema', async ({ page }) => {
    // Use the mock test page to verify data schema
    await page.goto('http://localhost:5173/mock-test');
    await page.waitForLoadState('networkidle');

    // Run all tests
    await page.locator('button:has-text("Run All Tests")').click();

    // Wait for tests to complete
    await page.waitForSelector('.ant-tabs-tab:has-text("Response Details")', { timeout: 10000 });

    // Switch to Response Details tab
    await page.locator('.ant-tabs-tab:has-text("Response Details")').click();

    // Verify transports response structure
    const transportCard = await page.locator('.ant-card-head-title:has-text("Transports (Paginated)")').locator('..').locator('..');
    const transportJson = await transportCard.locator('pre').textContent();
    expect(transportJson).toContain('"data"');
    expect(transportJson).toContain('"meta"');
    expect(transportJson).toContain('"page"');
    expect(transportJson).toContain('"page_size"');

    // Verify nodes response structure
    const nodesCard = await page.locator('.ant-card-head-title:has-text("Nodes List")').locator('..').locator('..');
    const nodesJson = await nodesCard.locator('pre').textContent();
    expect(nodesJson).toContain('"data"');
    expect(nodesJson).toContain('"node_id"');
    expect(nodesJson).toContain('"total_memory"');

    console.log('✅ Schema validation test passed');
  });
});