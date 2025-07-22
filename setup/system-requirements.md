# System Requirements & Prerequisites

## Required System Tools

### 1. Docker & Docker Compose
```bash
# macOS
brew install docker docker-compose

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Verify installation
docker --version
docker-compose --version
```

### 2. Command Line Tools (for testing)
```bash
# macOS
brew install curl jq

# Linux (Ubuntu/Debian)  
sudo apt-get install curl jq

# Verify installation
curl --version
jq --version
```

### 3. Python Environment (Optional - for local development)
```bash
# Python 3.11+ recommended
python3 --version

# Install root requirements (optional)
pip install -r ../requirements.txt
```

## API Keys Required

### OpenAI API Key
1. Go to https://platform.openai.com/api-keys
2. Create a new API key
3. Copy the key (starts with `sk-...`)

### Vector Database API Key (Choose one)

#### Option A: Pinecone (Recommended)
1. Go to https://app.pinecone.io/
2. Create account and get API key
3. Create an index with:
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: cosine
   - Name: llm-retrieval-kb

#### Option B: Weaviate
1. Go to https://weaviate.io/
2. Set up Weaviate instance or use cloud
3. Get connection URL and API key

#### Option C: ChromaDB (Local)
- No API key needed
- Runs locally in Docker

## Environment Setup

1. **Copy environment template:**
   ```bash
   cp .env.example ../.env
   ```

2. **Edit the .env file with your keys:**
   ```bash
   OPENAI_API_KEY=sk-your-actual-openai-key
   PINECONE_API_KEY=your-actual-pinecone-key
   PINECONE_INDEX_NAME=llm-retrieval-kb
   ```

## Port Requirements

Ensure these ports are available:
- `8001` - Conversation Service
- `8002` - Knowledge Base Service  
- `8005` - Analytics Service
- `8080` - API Gateway (optional)
- `5432` - PostgreSQL
- `6379` - Redis
- `9090` - Prometheus
- `3000` - Grafana

## System Resources

**Minimum:**
- 4GB RAM
- 2 CPU cores
- 10GB free disk space

**Recommended:**
- 8GB RAM
- 4 CPU cores
- 20GB free disk space

## Verification Script

Run this to check if your system is ready:

```bash
#!/bin/bash
echo "🔍 System Requirements Check"
echo "=========================="

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker: $(docker --version)"
else
    echo "❌ Docker: Not installed"
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose: $(docker-compose --version)"
else
    echo "❌ Docker Compose: Not installed"
fi

# Check curl
if command -v curl &> /dev/null; then
    echo "✅ curl: Available"
else
    echo "❌ curl: Not installed"
fi

# Check jq
if command -v jq &> /dev/null; then
    echo "✅ jq: Available"
else
    echo "❌ jq: Not installed"
fi

# Check environment file
if [ -f ../.env ]; then
    echo "✅ Environment: .env file exists"
    
    if grep -q "your_openai_api_key_here" ../.env; then
        echo "⚠️  Environment: OpenAI API key not configured"
    else
        echo "✅ Environment: OpenAI API key configured"
    fi
else
    echo "❌ Environment: .env file missing"
fi

echo ""
echo "Once all items show ✅, run: ./quick-start.sh"
```