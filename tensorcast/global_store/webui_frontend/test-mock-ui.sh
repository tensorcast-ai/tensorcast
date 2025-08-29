#!/bin/bash

# 自动化测试脚本 - 一键验证 mock 数据显示
# 使用方法: ./test-mock-ui.sh

set -e  # 遇到错误立即退出

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}     Mock UI 自动化测试 - 一键运行${NC}"
echo -e "${BLUE}================================================${NC}\n"

# 切换到正确的目录
cd "$(dirname "$0")"

# 1. 安装依赖
echo -e "${YELLOW}[1/5] 检查并安装依赖...${NC}"
if [ ! -d "node_modules" ]; then
    pnpm install --silent
    echo -e "${GREEN}✓ 依赖安装完成${NC}"
else
    echo -e "${GREEN}✓ 依赖已存在${NC}"
fi

# 2. 清理旧进程
echo -e "\n${YELLOW}[2/5] 清理旧进程...${NC}"
pkill -f "vite" 2>/dev/null || true
sleep 1
echo -e "${GREEN}✓ 清理完成${NC}"

# 3. 启动 mock 服务器
echo -e "\n${YELLOW}[3/5] 启动 Mock 开发服务器...${NC}"
pnpm run dev:mock > dev-server.log 2>&1 &
SERVER_PID=$!

# 等待服务器启动
echo -n "等待服务器启动"
PORT=5173
for i in {1..30}; do
    # 尝试检测实际端口
    if [ $i -eq 5 ]; then
        ACTUAL_PORT=$(grep -o "http://localhost:[0-9]*" dev-server.log | head -1 | grep -o "[0-9]*$" || echo "5173")
        if [ -n "$ACTUAL_PORT" ]; then
            PORT=$ACTUAL_PORT
        fi
    fi

    if curl -s http://localhost:$PORT > /dev/null 2>&1; then
        echo -e "\n${GREEN}✓ 服务器启动成功 (PID: $SERVER_PID, Port: $PORT)${NC}"
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "\n${RED}✗ 服务器启动失败${NC}"
        cat dev-server.log
        exit 1
    fi
done

# 导出端口给后续使用
export VITE_PORT=$PORT

# 4. 运行 Mock API 验证
echo -e "\n${YELLOW}[4/5] 验证 Mock API 响应格式...${NC}"
sleep 2  # 等待 MSW 初始化

# 使用 Node.js 验证 API
node verify-mock-display.js

# 5. 运行 Playwright UI 测试
echo -e "\n${YELLOW}[5/5] 运行 Playwright UI 测试...${NC}"

# 安装 Playwright 浏览器（如果需要）
if [ ! -d "node_modules/.cache/ms-playwright" ]; then
    echo "首次运行，安装 Playwright 浏览器..."
    npx playwright install chromium
fi

# 运行测试
PLAYWRIGHT_HEADED=0 npx playwright test tests/mock-pages.spec.ts --reporter=list

# 保存测试结果
TEST_EXIT_CODE=$?

# 清理
echo -e "\n${YELLOW}清理测试环境...${NC}"
kill $SERVER_PID 2>/dev/null || true
rm -f dev-server.log

# 显示结果
echo -e "\n${BLUE}================================================${NC}"
echo -e "${BLUE}              测试结果总结${NC}"
echo -e "${BLUE}================================================${NC}"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 所有测试通过！${NC}\n"
    echo -e "已验证的功能："
    echo -e "  ${GREEN}✓${NC} /transports 页面正确显示 mock 数据"
    echo -e "  ${GREEN}✓${NC} /transports 分页和过滤功能正常"
    echo -e "  ${GREEN}✓${NC} /nodes 页面正确显示节点信息"
    echo -e "  ${GREEN}✓${NC} /nodes 内存分布图表正常渲染"
    echo -e "  ${GREEN}✓${NC} Mock API 数据格式符合预期"
    echo -e "  ${GREEN}✓${NC} 页面导航功能正常"
else
    echo -e "${RED}❌ 测试失败${NC}"
    echo -e "请查看上方的错误信息"
    exit 1
fi

echo -e "\n${BLUE}提示：${NC}"
echo -e "  - 查看详细测试报告: ${YELLOW}npx playwright show-report${NC}"
echo -e "  - 手动测试: ${YELLOW}pnpm run dev:mock${NC} 然后访问 http://localhost:5173"
echo -e "  - 查看手动测试步骤: ${YELLOW}cat manual-test.md${NC}"

exit $TEST_EXIT_CODE