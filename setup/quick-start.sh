#!/bin/bash
# Quick Start Script for LLM-Powered Retrieval System

echo "⚡ Quick Start - LLM-Powered Retrieval System"
echo "============================================="

# Step 1: Environment setup
echo "1️⃣ Setting up environment..."
if [ ! -f ../.env ]; then
    cp .env.example ../.env
    echo "✅ Created .env file"
    echo ""
    echo "🔑 IMPORTANT: Edit .env file with your API keys:"
    echo "   - OPENAI_API_KEY=your_actual_openai_key"
    echo "   - PINECONE_API_KEY=your_actual_pinecone_key"
    echo "   - PINECONE_INDEX_NAME=your_index_name"
    echo ""
    echo "📝 Open .env file in your editor and add your keys, then run this script again."
    exit 0
else
    echo "✅ Environment file exists"
fi

# Step 2: Check if keys are configured
source ../.env
if [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
    echo "❌ Please configure your OPENAI_API_KEY in .env file"
    exit 1
fi

# Step 3: Start services
echo "2️⃣ Starting services with Docker..."
cd .. && docker-compose -f setup/docker-compose.yml up -d

echo "3️⃣ Waiting for services to start..."
sleep 30

# Step 4: Quick health check
echo "4️⃣ Quick health check..."
if curl -s http://localhost:8002/health > /dev/null; then
    echo "✅ Knowledge Base Service: Running"
else
    echo "❌ Knowledge Base Service: Not responding"
fi

if curl -s http://localhost:8001/health > /dev/null; then
    echo "✅ Conversation Service: Running"
else
    echo "❌ Conversation Service: Not responding"
fi

if curl -s http://localhost:8005/health > /dev/null; then
    echo "✅ Analytics Service: Running"  
else
    echo "❌ Analytics Service: Not responding"
fi

echo ""
echo "🎉 Quick start complete!"
echo ""
echo "🌐 Access your services:"
echo "  - Knowledge Base API: http://localhost:8002/docs"
echo "  - Conversation API:   http://localhost:8001/docs"
echo "  - Analytics API:      http://localhost:8005/docs"
echo ""
echo "🧪 Run tests:"
echo "  ./test_complete_workflow.sh"
echo ""