#!/bin/bash
# System Requirements Check

echo "🔍 System Requirements Check"
echo "=========================="

# Check Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker: $(docker --version)"
else
    echo "❌ Docker: Not installed"
    echo "   Install: https://docs.docker.com/get-docker/"
fi

# Check Docker Compose  
if command -v docker-compose &> /dev/null; then
    echo "✅ Docker Compose: $(docker-compose --version)"
else
    echo "❌ Docker Compose: Not installed"
    echo "   Install: https://docs.docker.com/compose/install/"
fi

# Check curl
if command -v curl &> /dev/null; then
    echo "✅ curl: Available"
else
    echo "❌ curl: Not installed"
    echo "   Install: brew install curl (macOS) or apt-get install curl (Linux)"
fi

# Check jq
if command -v jq &> /dev/null; then
    echo "✅ jq: Available"  
else
    echo "❌ jq: Not installed (needed for testing)"
    echo "   Install: brew install jq (macOS) or apt-get install jq (Linux)"
fi

# Check environment file
if [ -f .env ]; then
    echo "✅ Environment: .env file exists"
    
    if grep -q "your_openai_api_key_here" .env; then
        echo "⚠️  Environment: OpenAI API key not configured"
    else
        echo "✅ Environment: OpenAI API key configured"
    fi
else
    echo "❌ Environment: .env file missing"
    echo "   Run: cp setup/.env.example .env"
fi

# Check ports
echo ""
echo "🔌 Port Availability Check:"
ports=(8001 8002 8005 5432 6379 9090 3000)
for port in "${ports[@]}"; do
    if lsof -i :$port > /dev/null 2>&1; then
        echo "⚠️  Port $port: In use"
    else
        echo "✅ Port $port: Available"
    fi
done

echo ""
echo "📋 Next Steps:"
echo "1. If any ❌ items, install the missing tools"
echo "2. If .env missing or not configured, set up API keys"
echo "3. Once all ✅, run: ./quick-start.sh"