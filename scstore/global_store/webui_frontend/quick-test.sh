#!/bin/bash

# 快速测试脚本 - 验证 mock 数据显示

echo "🧪 快速测试 Mock 数据显示"
echo "========================="

# 1. 首先检查服务器是否已经在运行
PORT=""
for p in 3000 3001 3002 5173; do
    if curl -s http://localhost:$p > /dev/null 2>&1; then
        PORT=$p
        echo "✓ 发现服务器运行在端口 $PORT"
        break
    fi
done

if [ -z "$PORT" ]; then
    echo "❌ 未发现运行的开发服务器"
    echo "请先运行: pnpm run dev:mock"
    exit 1
fi

# 2. 测试关键端点
echo -e "\n测试 API 端点..."

# 测试 transports
echo -n "Testing /api/transports... "
TRANSPORTS=$(curl -s http://localhost:$PORT/api/transports?page=1\&page_size=5)
if echo "$TRANSPORTS" | grep -q '"data"' && echo "$TRANSPORTS" | grep -q '"meta"'; then
    echo "✅ OK (包含 data 和 meta)"
else
    echo "❌ 失败"
    echo "响应: $TRANSPORTS"
fi

# 测试 nodes
echo -n "Testing /api/nodes... "
NODES=$(curl -s http://localhost:$PORT/api/nodes)
if echo "$NODES" | grep -q '"data"' && echo "$NODES" | grep -q '"node_id"'; then
    echo "✅ OK (包含 data 和 node_id)"
else
    echo "❌ 失败"
    echo "响应: $NODES"
fi

# 3. 使用 puppeteer 进行简单的 UI 检查
echo -e "\n运行 UI 检查..."

# 创建一个简单的 puppeteer 脚本
cat > /tmp/ui-test.js << 'EOF'
const puppeteer = require('puppeteer');

(async () => {
  const port = process.argv[2] || 3001;
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();

  try {
    // 测试 Transports 页面
    console.log('检查 /transports 页面...');
    await page.goto(`http://localhost:${port}/transports`, { waitUntil: 'networkidle2' });

    // 检查标题
    const title = await page.$eval('h1', el => el.textContent);
    console.log('  标题:', title);

    // 检查统计卡片
    const stats = await page.$$eval('.ant-statistic-title', els => els.map(el => el.textContent));
    console.log('  统计卡片:', stats.join(', '));

    // 检查表格行数
    const rows = await page.$$('.ant-table-tbody tr');
    console.log('  表格行数:', rows.length);

    // 测试 Nodes 页面
    console.log('\n检查 /nodes 页面...');
    await page.goto(`http://localhost:${port}/nodes`, { waitUntil: 'networkidle2' });

    // 检查标题
    const nodesTitle = await page.$eval('h1', el => el.textContent);
    console.log('  标题:', nodesTitle);

    // 检查表格行数
    const nodeRows = await page.$$('.ant-table-tbody tr');
    console.log('  表格行数:', nodeRows.length);

    console.log('\n✅ UI 测试通过!');
  } catch (error) {
    console.error('❌ 测试失败:', error.message);
  }

  await browser.close();
})();
EOF

# 检查是否有 puppeteer
if command -v npx &> /dev/null && [ -d "node_modules/puppeteer" ]; then
    node /tmp/ui-test.js $PORT
else
    echo "跳过 UI 测试 (需要 puppeteer)"
fi

echo -e "\n测试完成!"
echo "你可以手动访问以下页面检查："
echo "  - http://localhost:$PORT/transports"
echo "  - http://localhost:$PORT/nodes"
echo "  - http://localhost:$PORT/mock-test"