# 🧪 Complete System Testing Guide

## 🚀 Quick Start Testing

### Prerequisites
```bash
# Required environment variables
export OPENAI_API_KEY="your_openai_api_key"
export PINECONE_API_KEY="your_pinecone_api_key"  # or other vector DB
export PINECONE_INDEX_NAME="llm-retrieval-kb"
```

### 1. **Environment Setup**
```bash
# Clone and navigate
cd /Users/shishirkafle/Desktop/ChatBoq/LLM-Powered-Retrieval-System

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start all services
docker-compose up -d

# Verify all services are running
docker-compose ps
```

### 2. **Health Check Verification**
```bash
# Check all services are healthy
curl http://localhost:8001/health  # Conversation Service
curl http://localhost:8002/health  # Knowledge Base Service  
curl http://localhost:8005/health  # Analytics Service

# Expected response: {"status": "healthy", "service": "...", "version": "1.0.0"}
```

## 🔧 Individual Service Testing

### **Knowledge Base Service (Port 8002)**

#### Test Document Creation
```bash
# Create a test document
curl -X POST "http://localhost:8002/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "API Documentation",
    "content": "This is a comprehensive guide to using our REST API. It includes authentication, endpoints, and examples.",
    "category": "documentation", 
    "subcategory": "api",
    "tags": ["api", "guide", "documentation"],
    "metadata": {"author": "tech-team", "version": "1.0"}
  }'
```

#### Test Document Search
```bash
# Search for documents
curl -X GET "http://localhost:8002/api/v1/search?q=API%20guide&limit=5"

# Search with filters
curl -X GET "http://localhost:8002/api/v1/search?q=documentation&category=documentation&tags=api"
```

#### Test Document Retrieval
```bash
# Get specific document (use ID from creation response)
curl -X GET "http://localhost:8002/api/v1/documents/{document_id}"
```

### **Conversation Service (Port 8001)**

#### Test Chat Functionality
```bash
# Send a chat message
curl -X POST "http://localhost:8001/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I use the API?",
    "conversation_id": "test-conversation-1"
  }'
```

#### Test Streaming Chat
```bash
# Test streaming response
curl -X POST "http://localhost:8001/api/v1/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Tell me about the documentation",
    "conversation_id": "test-stream-1",
    "stream": true
  }'
```

### **Analytics Service (Port 8005)**

#### Test Response Evaluation
```bash
# Evaluate a RAG response
curl -X POST "http://localhost:8005/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I use the API?",
    "context": "This is a comprehensive guide to using our REST API. It includes authentication, endpoints, and examples.",
    "response": "To use the API, you need to authenticate first and then make requests to the documented endpoints.",
    "conversation_id": "test-conversation-1"
  }'
```

#### Test Metrics Retrieval
```bash
# Get system metrics
curl -X GET "http://localhost:8005/api/v1/metrics"

# Get health metrics
curl -X GET "http://localhost:8005/api/v1/health-metrics"
```

## 🔗 End-to-End Integration Testing

### **Complete Workflow Test**

```bash
#!/bin/bash
# Save as: test_complete_workflow.sh

echo "🚀 Starting Complete System Test..."

# 1. Health Checks
echo "1️⃣ Testing Health Endpoints..."
curl -s http://localhost:8001/health | jq .
curl -s http://localhost:8002/health | jq .
curl -s http://localhost:8005/health | jq .

# 2. Create Knowledge Base Document
echo "2️⃣ Creating Test Document..."
DOC_RESPONSE=$(curl -s -X POST "http://localhost:8002/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Customer Support Guide",
    "content": "Our customer support provides 24/7 assistance through chat, email, and phone. For technical issues, please include error logs and system details.",
    "category": "support",
    "tags": ["customer-service", "help", "contact"]
  }')

DOC_ID=$(echo $DOC_RESPONSE | jq -r '.id')
echo "Created document: $DOC_ID"

# 3. Test Search Functionality
echo "3️⃣ Testing Document Search..."
curl -s -X GET "http://localhost:8002/api/v1/search?q=customer%20support&limit=3" | jq .

# 4. Test Conversation
echo "4️⃣ Testing Conversation..."
CHAT_RESPONSE=$(curl -s -X POST "http://localhost:8001/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can I contact customer support?",
    "conversation_id": "integration-test-1"
  }')

echo $CHAT_RESPONSE | jq .

# 5. Evaluate Response Quality
echo "5️⃣ Testing Analytics Evaluation..."
curl -s -X POST "http://localhost:8005/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How can I contact customer support?",
    "context": "Our customer support provides 24/7 assistance through chat, email, and phone.",
    "response": "You can contact our customer support team 24/7 through chat, email, or phone.",
    "conversation_id": "integration-test-1"
  }' | jq .

# 6. Check Final Metrics
echo "6️⃣ Final System Metrics..."
curl -s http://localhost:8005/api/v1/metrics | jq .

echo "✅ Complete System Test Finished!"
```

```bash
# Make executable and run
chmod +x test_complete_workflow.sh
./test_complete_workflow.sh
```

## 📊 Monitoring & Observability Testing

### **Prometheus Metrics**
```bash
# Check Prometheus targets
curl -s http://localhost:9090/api/v1/targets

# Query service metrics
curl -s "http://localhost:9090/api/v1/query?query=up" | jq .

# Check analytics service metrics
curl -s http://localhost:8005/metrics
```

### **Grafana Dashboard**
```bash
# Access Grafana
open http://localhost:3000
# Login: admin/admin (or check .env for GRAFANA_PASSWORD)

# Import dashboards for:
# - Service health metrics
# - RAG quality metrics  
# - Response times
# - Error rates
```

## 🧪 Load Testing

### **Basic Load Test with curl**
```bash
#!/bin/bash
# Save as: load_test.sh

echo "🏋️ Starting Load Test..."

# Test concurrent requests
for i in {1..10}; do
  curl -s -X POST "http://localhost:8001/api/v1/chat" \
    -H "Content-Type: application/json" \
    -d "{
      \"message\": \"Test message $i\",
      \"conversation_id\": \"load-test-$i\"
    }" &
done

wait
echo "✅ Load test completed"
```

### **Advanced Load Testing with Apache Bench**
```bash
# Install Apache Bench (if not installed)
brew install httpie

# Test Knowledge Base Service
ab -n 100 -c 10 -H "Content-Type: application/json" \
   -p search_payload.json \
   http://localhost:8002/api/v1/search

# Create search_payload.json:
echo '{"query": "test", "limit": 5}' > search_payload.json
```

## 🔍 Error Testing & Edge Cases

### **Test Error Handling**
```bash
# Test invalid requests
curl -X POST "http://localhost:8002/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{"invalid": "data"}'

# Test non-existent endpoints
curl http://localhost:8001/api/v1/nonexistent

# Test malformed JSON
curl -X POST "http://localhost:8001/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"malformed": json}'
```

### **Test Service Dependencies**
```bash
# Stop one service and test others
docker-compose stop analytics-service

# Test conversation service (should still work)
curl -X POST "http://localhost:8001/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "test without analytics"}'

# Restart service
docker-compose start analytics-service
```

## 📱 API Documentation Testing

### **OpenAPI/Swagger UI**
```bash
# Access interactive API docs
open http://localhost:8001/docs  # Conversation Service
open http://localhost:8002/docs  # Knowledge Base Service  
open http://localhost:8005/docs  # Analytics Service
```

## 🚦 Continuous Integration Testing

### **GitHub Actions Test Workflow** (save as `.github/workflows/test.yml`)
```yaml
name: System Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up environment
      run: |
        cp .env.example .env
        echo "OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}" >> .env
        
    - name: Start services
      run: docker-compose up -d
      
    - name: Wait for services
      run: sleep 30
      
    - name: Run health checks
      run: |
        curl -f http://localhost:8001/health
        curl -f http://localhost:8002/health
        curl -f http://localhost:8005/health
        
    - name: Run integration tests
      run: ./test_complete_workflow.sh
      
    - name: Cleanup
      run: docker-compose down
```

## 🎯 Success Criteria

### ✅ **System is Working Correctly If:**

1. **All health endpoints return 200 OK**
2. **Document creation and search work without errors**
3. **Conversations generate responses**
4. **Analytics evaluate responses and track metrics**
5. **Prometheus scrapes metrics successfully**
6. **No critical errors in service logs**
7. **Response times are reasonable (< 5s for most operations)**

### 📋 **Troubleshooting Checklist**

- [ ] All environment variables set correctly
- [ ] Docker containers are running (`docker-compose ps`)
- [ ] Ports are not conflicting with other services
- [ ] External APIs (OpenAI, Pinecone) are accessible
- [ ] Database connections are working
- [ ] No firewall blocking internal service communication

## 🎉 Ready for Production

Once all tests pass, your LLM-Powered Retrieval System is ready for production deployment!

```bash
# For production deployment
kubectl apply -f customer-support-platform/infrastructure/kubernetes/
```