#!/bin/bash

echo "🚀 Starting UI Test for Mock Data Display"
echo "==========================================="

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pnpm install
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :5173 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Kill any existing dev server
if check_port; then
    echo -e "${YELLOW}Stopping existing dev server on port 5173...${NC}"
    lsof -ti:5173 | xargs kill -9 2>/dev/null
    sleep 2
fi

# Start the mock dev server in background
echo -e "${GREEN}Starting mock dev server...${NC}"
pnpm run dev:mock > /dev/null 2>&1 &
DEV_PID=$!

# Wait for dev server to start
echo "Waiting for dev server to start..."
for i in {1..30}; do
    if check_port; then
        echo -e "${GREEN}Dev server started successfully!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}Failed to start dev server${NC}"
        exit 1
    fi
    sleep 1
done

# Give it a bit more time to fully initialize
sleep 3

# Run the unit tests for mock handlers
echo -e "\n${GREEN}Running mock handler unit tests...${NC}"
pnpm test:mock

# Run Playwright tests
echo -e "\n${GREEN}Running Playwright UI tests...${NC}"
npx playwright test tests/mock-pages.spec.ts --reporter=list

# Capture test result
TEST_RESULT=$?

# Stop the dev server
echo -e "\n${YELLOW}Stopping dev server...${NC}"
kill $DEV_PID 2>/dev/null

# Summary
echo -e "\n==========================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ All UI tests passed!${NC}"
    echo -e "\nThe following pages were tested:"
    echo "  - /transports - Model transport history with pagination"
    echo "  - /nodes - Node overview with statistics"
    echo "  - /mock-test - Mock API testing dashboard"
    echo -e "\n${GREEN}Mock data is displaying correctly on all tested pages!${NC}"
else
    echo -e "${RED}❌ Some tests failed${NC}"
    echo -e "Check the test output above for details"
    exit 1
fi

echo -e "\n${YELLOW}Tip: To manually test, run:${NC}"
echo "  pnpm run dev:mock"
echo "  Then open http://localhost:5173 in your browser"
echo "  Check manual-test.md for detailed manual testing steps"