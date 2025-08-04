import { test, expect } from '@playwright/test'

test('dashboard loads', async ({ page }) => {
  await page.goto('http://localhost:9000')
  await expect(page.getByRole('heading', { name: 'Dashboard' })).toBeVisible()
})