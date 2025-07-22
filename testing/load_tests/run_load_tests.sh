#!/bin/bash
# Load Testing Script for LLM Retrieval System

echo "🏋️ Starting Load Tests..."

# Configuration
KB_SERVICE="http://knowledge-base-service:8002"
CONV_SERVICE="http://conversation-service:8001"
ANALYTICS_SERVICE="http://analytics-service:8005"

# Number of concurrent requests
CONCURRENT_REQUESTS=10
TOTAL_REQUESTS=100

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}$1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Test 1: Knowledge Base Search Load Test
log_info "Test 1: Knowledge Base Search Load Test"
echo "Sending $TOTAL_REQUESTS search requests with $CONCURRENT_REQUESTS concurrent connections..."

start_time=$(date +%s)

# Create background jobs for concurrent requests
for ((i=1; i<=CONCURRENT_REQUESTS; i++)); do
    {
        for ((j=1; j<=10; j++)); do  # 10 requests per connection
            curl -s -X GET "$KB_SERVICE/api/v1/search?q=test%20query%20$i$j&limit=5" > /dev/null
            if [ $? -eq 0 ]; then
                echo "Search $i-$j: OK"
            else
                echo "Search $i-$j: FAILED"
            fi
        done
    } &
done

# Wait for all background jobs to complete
wait

end_time=$(date +%s)
duration=$((end_time - start_time))
rps=$((TOTAL_REQUESTS / duration))

log_success "Search load test completed in ${duration}s (${rps} requests/second)"

# Test 2: Conversation Service Load Test
log_info "Test 2: Conversation Service Load Test"
echo "Sending chat requests..."

start_time=$(date +%s)

for ((i=1; i<=20; i++)); do
    {
        curl -s -X POST "$CONV_SERVICE/api/v1/chat" \
            -H "Content-Type: application/json" \
            -d "{
                \"message\": \"Load test message $i\",
                \"conversation_id\": \"load-test-$i\"
            }" > /dev/null
        
        if [ $? -eq 0 ]; then
            echo "Chat $i: OK"
        else
            echo "Chat $i: FAILED"
        fi
    } &
    
    # Limit concurrent requests to avoid overwhelming
    if [ $((i % 5)) -eq 0 ]; then
        wait
    fi
done

wait

end_time=$(date +%s)
duration=$((end_time - start_time))

log_success "Chat load test completed in ${duration}s"

# Test 3: Analytics Service Load Test
log_info "Test 3: Analytics Service Load Test"
echo "Sending evaluation requests..."

start_time=$(date +%s)

for ((i=1; i<=15; i++)); do
    {
        curl -s -X POST "$ANALYTICS_SERVICE/api/v1/evaluate" \
            -H "Content-Type: application/json" \
            -d "{
                \"query\": \"Load test query $i\",
                \"context\": \"This is test context for load testing purposes\",
                \"response\": \"This is a test response for evaluation\",
                \"conversation_id\": \"load-eval-$i\"
            }" > /dev/null
        
        if [ $? -eq 0 ]; then
            echo "Evaluation $i: OK"
        else
            echo "Evaluation $i: FAILED"
        fi
    } &
    
    # Limit concurrent requests
    if [ $((i % 3)) -eq 0 ]; then
        wait
    fi
done

wait

end_time=$(date +%s)
duration=$((end_time - start_time))

log_success "Analytics load test completed in ${duration}s"

# Test 4: Mixed Workload Test
log_info "Test 4: Mixed Workload Test"
echo "Running mixed workload simulation..."

start_time=$(date +%s)

# Simulate realistic usage patterns
for ((i=1; i<=10; i++)); do
    {
        # Search for documents
        curl -s -X GET "$KB_SERVICE/api/v1/search?q=mixed%20test%20$i" > /dev/null
        sleep 0.5
        
        # Have a conversation
        curl -s -X POST "$CONV_SERVICE/api/v1/chat" \
            -H "Content-Type: application/json" \
            -d "{
                \"message\": \"Mixed workload test $i\",
                \"conversation_id\": \"mixed-$i\"
            }" > /dev/null
        sleep 0.5
        
        # Evaluate response
        curl -s -X POST "$ANALYTICS_SERVICE/api/v1/evaluate" \
            -H "Content-Type: application/json" \
            -d "{
                \"query\": \"Mixed test $i\",
                \"context\": \"Mixed test context\",
                \"response\": \"Mixed test response\",
                \"conversation_id\": \"mixed-$i\"
            }" > /dev/null
        
        echo "Mixed workflow $i: Complete"
    } &
done

wait

end_time=$(date +%s)
duration=$((end_time - start_time))

log_success "Mixed workload test completed in ${duration}s"

# Final system health check
log_info "Final Health Check"

# Check if all services are still responsive
if curl -s -f "$KB_SERVICE/health" > /dev/null; then
    log_success "Knowledge Base Service: Healthy after load test"
else
    log_error "Knowledge Base Service: Not responsive after load test"
fi

if curl -s -f "$CONV_SERVICE/health" > /dev/null; then
    log_success "Conversation Service: Healthy after load test"
else
    log_error "Conversation Service: Not responsive after load test"
fi

if curl -s -f "$ANALYTICS_SERVICE/health" > /dev/null; then
    log_success "Analytics Service: Healthy after load test"
else
    log_error "Analytics Service: Not responsive after load test"
fi

echo ""
log_success "🎉 Load Testing Complete!"
echo "Summary:"
echo "- Knowledge Base searches: $TOTAL_REQUESTS requests"
echo "- Conversation messages: 20 requests"  
echo "- Analytics evaluations: 15 requests"
echo "- Mixed workload: 10 full workflows"
echo ""
echo "All services should remain healthy and responsive under load."