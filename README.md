# LLM-Powered Retrieval System

A production-ready, microservices-based retrieval system powered by advanced RAG capabilities, built with FastAPI, LangChain, and modern containerization.

## 🚀 Quick Start (2 minutes)

```bash
# 1. Clone and setup
git clone <your-repo>
cd LLM-Powered-Retrieval-System

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys (OPENAI_API_KEY, PINECONE_API_KEY)

# 3. Start everything
./quick-start.sh

# 4. Test the system
./test_complete_workflow.sh
```

## 🏗️ Architecture Overview

### Microservices Structure
```
┌─────────────────────────────────────────────────────────┐
│                 Docker Compose                          │
├─────────────────┬─────────────────┬─────────────────────┤
│ Knowledge Base  │ Conversation    │ Analytics           │
│ Service :8002   │ Service :8001   │ Service :8005       │
│                 │                 │                     │
│ • Document CRUD │ • Chat API      │ • Quality Metrics  │
│ • Vector Search │ • RAG Pipeline  │ • User Feedback    │
│ • Hybrid        │ • Streaming     │ • Prometheus       │
│   Retrieval     │ • Context Mgmt  │   Metrics          │
└─────────────────┴─────────────────┴─────────────────────┘
┌─────────────────────────────────────────────────────────┐
│          Infrastructure Services                        │
│ PostgreSQL | Redis | Prometheus | Grafana              │
└─────────────────────────────────────────────────────────┘
```

### Independent Services
- **✅ Complete isolation** - Each service has its own dependencies, Docker container, and lifecycle
- **✅ Independent deployment** - Services can be built, tested, and deployed separately  
- **✅ Fault tolerance** - Failure of one service doesn't affect others
- **✅ Horizontal scaling** - Scale services independently based on load

## 📁 Project Structure

```
LLM-Powered-Retrieval-System/
├── 🔧 Setup & Configuration
│   ├── .env.example                    # Environment template
│   ├── docker-compose.yml              # Multi-service orchestration
│   ├── setup.sh                        # Full setup script
│   └── quick-start.sh                  # 2-minute quick start
│
├── 🧪 Testing & Quality
│   ├── test_complete_workflow.sh       # End-to-end testing
│   ├── TESTING_GUIDE.md               # Comprehensive test guide
│   ├── postman_collection.json        # API test collection
│   └── load_tests/                     # Performance testing
│
├── 🚀 Services (Independent Microservices)
│   ├── knowledge-base-service/         # Document & RAG operations
│   │   ├── src/
│   │   │   ├── main.py                # FastAPI app
│   │   │   ├── core/                  # Business logic
│   │   │   │   ├── retrieval.py       # Advanced RAG retriever
│   │   │   │   ├── chunking.py        # Document processing
│   │   │   │   └── cache.py           # Vector caching
│   │   │   └── api/routes.py          # REST endpoints
│   │   ├── Dockerfile                 # Container definition
│   │   ├── requirements.txt           # Service dependencies
│   │   └── README.md                  # Service documentation
│   │
│   ├── conversation-service/           # Chat & conversation management
│   │   ├── src/core/
│   │   │   ├── rag_pipeline.py        # Multi-stage RAG
│   │   │   ├── context_manager.py     # Conversation context
│   │   │   ├── streaming.py           # Real-time responses
│   │   │   └── prompts.py             # LLM prompt templates
│   │   └── [same structure as above]
│   │
│   └── analytics-service/              # Metrics & evaluation
│       ├── src/core/rag_metrics.py    # Quality evaluation
│       └── [same structure as above]
│
└── 🏗️ Infrastructure
    └── customer-support-platform/infrastructure/
        ├── kubernetes/                 # K8s deployment manifests
        └── monitoring/                 # Prometheus configuration
```

## 🔥 Key Features

### 🤖 Advanced RAG Capabilities
- **Hybrid Retrieval**: Vector similarity + BM25 + Contextual compression
- **Multi-stage Pipeline**: Query rewriting → Multi-query retrieval → Reranking
- **Context-aware Responses**: Conversation history + User intent + Sentiment analysis
- **Streaming Responses**: Real-time response generation

### 🏛️ Production-Ready Architecture
- **Independent Services**: True microservices with separate containers
- **Health Monitoring**: Comprehensive health checks and metrics
- **Horizontal Scaling**: Kubernetes-ready with HPA support
- **Fault Tolerance**: Circuit breakers and graceful degradation

### 📊 Quality & Observability
- **RAG Quality Metrics**: Retrieval precision, response relevance, context utilization
- **User Feedback Loop**: Satisfaction scoring and continuous improvement
- **Comprehensive Monitoring**: Prometheus + Grafana dashboards
- **Distributed Tracing**: Request tracing across services

## 🛠️ Development

### Local Development
```bash
# Start individual service for development
cd knowledge-base-service
pip install -r requirements.txt  
python -m src.main

# Or use Docker for consistency
docker-compose up knowledge-base-service
```

### Testing
```bash
# Quick health check
curl http://localhost:8001/health

# Run full test suite
./test_complete_workflow.sh

# Load testing
./load_tests/run_load_tests.sh

# Import Postman collection
# File: postman_collection.json
```

### API Documentation
- **Knowledge Base**: http://localhost:8002/docs
- **Conversation**: http://localhost:8001/docs  
- **Analytics**: http://localhost:8005/docs

## 🚢 Deployment

### Docker Compose (Development/Staging)
```bash
docker-compose up -d
```

### Kubernetes (Production)
```bash
kubectl apply -f customer-support-platform/infrastructure/kubernetes/
```

### Environment Variables
Key configurations in `.env`:
- `OPENAI_API_KEY` - OpenAI API access
- `PINECONE_API_KEY` - Vector database access
- `VECTOR_STORE_TYPE` - Vector database type (pinecone/weaviate/chroma)
- Service URLs for inter-service communication

## 🔍 Monitoring & Observability

### Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Service Metrics**: Each service exposes `/metrics` endpoint

### Key Metrics
- **Response Quality**: Retrieval precision, response relevance
- **Performance**: Response times, throughput, error rates
- **System Health**: Service uptime, resource utilization
- **User Satisfaction**: Feedback scores, conversation success rates

## 🤝 API Usage Examples

### Create and Search Documents
```bash
# Create document
curl -X POST "http://localhost:8002/api/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{"title":"API Guide","content":"How to use our API...","category":"docs"}'

# Search documents  
curl "http://localhost:8002/api/v1/search?q=API%20guide&limit=5"
```

### Chat Conversation
```bash
# Send chat message
curl -X POST "http://localhost:8001/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{"message":"How do I use the API?","conversation_id":"chat-1"}'
```

### Quality Evaluation
```bash
# Evaluate response quality
curl -X POST "http://localhost:8005/api/v1/evaluate" \
  -H "Content-Type: application/json" \
  -d '{"query":"...","context":"...","response":"..."}'
```

## 📋 System Requirements

- **Docker** & **Docker Compose**
- **API Keys**: OpenAI, Vector Database (Pinecone/Weaviate)
- **Ports**: 8001, 8002, 8005, 9090, 3000
- **Memory**: 4GB+ recommended
- **Storage**: Vector database + PostgreSQL

## 🆘 Troubleshooting

### Common Issues
- **Services not starting**: Check API keys in `.env`
- **Port conflicts**: Ensure ports 8001, 8002, 8005 are available
- **Connection errors**: Verify Docker network and service communication
- **API errors**: Check service logs with `docker-compose logs [service-name]`

### Get Help
```bash
# View service logs
docker-compose logs knowledge-base-service

# Check service health
curl http://localhost:8002/health

# Run diagnostics
./test_complete_workflow.sh
```

## 🎯 Success Criteria

Your system is working correctly when:
- ✅ All health endpoints return `200 OK`
- ✅ Documents can be created and searched
- ✅ Conversations generate coherent responses
- ✅ Analytics track quality metrics
- ✅ Test script passes all checks

## 📈 What's Next?

- **Scale**: Deploy to Kubernetes for production
- **Extend**: Add more vector stores or LLM providers
- **Optimize**: Fine-tune retrieval and ranking algorithms
- **Integrate**: Connect with external APIs and data sources
- **Monitor**: Set up alerts and performance optimization

---

**🎉 Ready to build amazing RAG applications!**

For detailed testing instructions, see [TESTING_GUIDE.md](TESTING_GUIDE.md)