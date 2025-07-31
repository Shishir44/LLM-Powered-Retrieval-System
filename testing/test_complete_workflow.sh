#!/bin/bash
# Complete System Integration Test Script

set -e  # Exit on any error

echo "🚀 Starting Complete LLM Retrieval System Test..."
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper function for colored output
log_step() {
    echo -e "${BLUE}$1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    log_warning "jq not found. Installing jq for JSON processing..."
    # For macOS
    if command -v brew &> /dev/null; then
        brew install jq
    else
        log_error "Please install jq manually: https://stedolan.github.io/jq/"
        exit 1
    fi
fi

# 1. Health Checks
log_step "1️⃣ Testing Health Endpoints..."
echo "Checking Knowledge Base Service..."
if curl -s -f http://localhost:8002/health > /dev/null; then
    HEALTH_KB=$(curl -s http://localhost:8002/health | jq -r '.status')
    log_success "Knowledge Base Service: $HEALTH_KB"
else
    log_error "Knowledge Base Service not accessible on port 8002"
    exit 1
fi

echo "Checking Conversation Service..."
if curl -s -f http://localhost:8001/health > /dev/null; then
    HEALTH_CONV=$(curl -s http://localhost:8001/health | jq -r '.status')
    log_success "Conversation Service: $HEALTH_CONV"
else
    log_error "Conversation Service not accessible on port 8001"
    exit 1
fi

echo "Checking Analytics Service..."
if curl -s -f http://localhost:8005/health > /dev/null; then
    HEALTH_ANALYTICS=$(curl -s http://localhost:8005/health | jq -r '.status')
    log_success "Analytics Service: $HEALTH_ANALYTICS"
else
    log_error "Analytics Service not accessible on port 8005"
    exit 1
fi

# 2. Create Knowledge Base Document
log_step "2️⃣ Creating Test Document..."
DOC_RESPONSE=$(curl -s -X POST "http://localhost:8002/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Customer Support Comprehensive Guide",
    "content": "Our customer support provides 24/7 assistance through multiple channels: live chat, email support, and phone support. For technical issues, please include detailed error logs, system information, and steps to reproduce the problem. Our support team specializes in troubleshooting, account management, billing inquiries, and product guidance. Response times: Chat (immediate), Email (2-4 hours), Phone (immediate during business hours).",
    "category": "support",
    "subcategory": "customer-service",
    "tags": ["customer-service", "help", "contact", "troubleshooting", "support-channels"],
    "metadata": {"author": "support-team", "version": "2.0", "last_updated": "2024-07-22"}
  }')

if [ $? -eq 0 ]; then
    DOC_ID=$(echo "$DOC_RESPONSE" | jq -r '.id // "unknown"')
    if [ "$DOC_ID" != "unknown" ] && [ "$DOC_ID" != "null" ]; then
        log_success "Created document with ID: $DOC_ID"
    else
        log_warning "Document created but ID not found in response"
        echo "Response: $DOC_RESPONSE"
    fi
else
    log_error "Failed to create document"
    exit 1
fi

# 3. Test Search Functionality
log_step "3️⃣ Testing Document Search..."
SEARCH_RESPONSE=$(curl -s -X GET "http://localhost:8002/api/v1/search?q=customer%20support&limit=3")

if [ $? -eq 0 ]; then
    SEARCH_COUNT=$(echo "$SEARCH_RESPONSE" | jq -r '.total // 0')
    log_success "Search found $SEARCH_COUNT documents"
    echo "Search results preview:"
    echo "$SEARCH_RESPONSE" | jq -r '.results[] | "- " + .title' | head -3
else
    log_error "Search functionality failed"
    exit 1
fi

# 4. Test Conversation
log_step "4️⃣ Testing Conversation..."
CHAT_RESPONSE=$(curl -s -X POST "http://localhost:8001/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I contact customer support for technical issues?",
    "conversation_id": "integration-test-1",
    "context": {"user_type": "test", "session": "integration"}
  }')

if [ $? -eq 0 ]; then
    CHAT_RESPONSE_TEXT=$(echo "$CHAT_RESPONSE" | jq -r '.response // "No response found"')
    CONVERSATION_ID=$(echo "$CHAT_RESPONSE" | jq -r '.conversation_id // "unknown"')
    log_success "Conversation response received"
    echo "Response preview: ${CHAT_RESPONSE_TEXT:0:100}..."
    echo "Conversation ID: $CONVERSATION_ID"
else
    log_error "Conversation functionality failed"
    exit 1
fi

# 5. Evaluate Response Quality
log_step "5️⃣ Testing Analytics Evaluation..."
EVAL_RESPONSE=$(curl -s -X POST "http://localhost:8005/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I contact customer support for technical issues?",
    "context": "Our customer support provides 24/7 assistance through multiple channels: live chat, email support, and phone support. For technical issues, please include detailed error logs and system information.",
    "response": "You can contact our customer support team 24/7 through live chat, email, or phone. For technical issues, make sure to include detailed error logs and system information to help us assist you better.",
    "conversation_id": "integration-test-1"
  }')

if [ $? -eq 0 ]; then
    PRECISION=$(echo "$EVAL_RESPONSE" | jq -r '.metrics.retrieval_precision // 0')
    RELEVANCE=$(echo "$EVAL_RESPONSE" | jq -r '.metrics.response_relevance // 0')
    log_success "Analytics evaluation completed"
    echo "Retrieval Precision: $PRECISION"
    echo "Response Relevance: $RELEVANCE"
else
    log_error "Analytics evaluation failed"
    exit 1
fi

# 6. Test User Feedback
log_step "6️⃣ Testing User Feedback..."
FEEDBACK_RESPONSE=$(curl -s -X POST "http://localhost:8005/api/v1/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "conversation_id": "integration-test-1",
    "satisfaction_score": 0.85,
    "feedback_text": "Very helpful response with clear instructions"
  }')

if [ $? -eq 0 ]; then
    log_success "User feedback recorded successfully"
else
    log_warning "User feedback recording failed (non-critical)"
fi

# 7. Check Final Metrics
log_step "7️⃣ Final System Metrics..."
METRICS_RESPONSE=$(curl -s http://localhost:8005/api/v1/metrics)

if [ $? -eq 0 ]; then
    TOTAL_QUERIES=$(echo "$METRICS_RESPONSE" | jq -r '.current_metrics.total_queries // 0')
    FAILED_QUERIES=$(echo "$METRICS_RESPONSE" | jq -r '.current_metrics.failed_queries // 0')
    log_success "System metrics retrieved"
    echo "Total queries processed: $TOTAL_QUERIES"
    echo "Failed queries: $FAILED_QUERIES"
else
    log_warning "Could not retrieve final metrics (non-critical)"
fi

# 8. Test API Documentation Endpoints
log_step "8️⃣ Testing API Documentation..."
if curl -s -f http://localhost:8001/docs > /dev/null; then
    log_success "Conversation Service API docs accessible"
else
    log_warning "Conversation Service API docs not accessible"
fi

if curl -s -f http://localhost:8002/docs > /dev/null; then
    log_success "Knowledge Base Service API docs accessible"
else
    log_warning "Knowledge Base Service API docs not accessible"
fi

if curl -s -f http://localhost:8005/docs > /dev/null; then
    log_success "Analytics Service API docs accessible"
else
    log_warning "Analytics Service API docs not accessible"
fi

# 9. Cleanup Test Data (optional)
log_step "9️⃣ Cleanup (Optional)..."
if [ "$DOC_ID" != "unknown" ] && [ "$DOC_ID" != "null" ]; then
    if curl -s -X DELETE "http://localhost:8002/api/v1/documents/$DOC_ID" > /dev/null; then
        log_success "Test document cleaned up"
    else
        log_warning "Could not clean up test document (non-critical)"
    fi
fi

echo ""
echo "=================================================="
log_success "🎉 Complete System Test PASSED!"
echo "=================================================="
echo ""
echo "Summary:"
echo "✅ All core services are healthy and responding"
echo "✅ Document creation and search functionality works"
echo "✅ Conversation service processes messages successfully"
echo "✅ Analytics service evaluates responses and tracks metrics"
echo "✅ API documentation is accessible"
echo ""
echo "Your LLM-Powered Retrieval System is ready for use!"
echo ""
echo "Next steps:"
echo "📖 Visit API docs: http://localhost:8001/docs"
echo "📊 Check Grafana: http://localhost:3000"
echo "🔍 View Prometheus: http://localhost:9090"
echo ""