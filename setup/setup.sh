#!/bin/bash
# LLM-Powered Retrieval System - Setup Script

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

echo "🚀 LLM-Powered Retrieval System Setup"
echo "====================================="

# Check prerequisites
log_step "1. Checking Prerequisites..."

# Check Docker
if command -v docker &> /dev/null; then
    log_success "Docker is installed"
else
    log_error "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    log_success "Docker Compose is installed"
else
    log_error "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check for environment file
log_step "2. Setting up Environment Configuration..."

if [ ! -f ../.env ]; then
    if [ -f .env.example ]; then
        cp .env.example ../.env
        log_success "Created ../.env file from .env.example"
        log_warning "Please edit .env file with your API keys before continuing"
        echo ""
        echo "Required configuration:"
        echo "- OPENAI_API_KEY: Your OpenAI API key"
        echo "- PINECONE_API_KEY: Your Pinecone API key (or other vector DB)"
        echo "- PINECONE_INDEX_NAME: Your Pinecone index name"
        echo ""
        read -p "Press Enter after you've configured the .env file..."
    else
        log_error ".env.example file not found"
        exit 1
    fi
else
    log_success "Environment file (.env) already exists"
fi

# Validate environment variables
log_step "3. Validating Environment Configuration..."

if [ -f .env ]; then
    source .env
    
    if [ -z "$OPENAI_API_KEY" ] || [ "$OPENAI_API_KEY" = "your_openai_api_key_here" ]; then
        log_error "OPENAI_API_KEY is not set in .env file"
        exit 1
    else
        log_success "OpenAI API key configured"
    fi
    
    if [ -z "$VECTOR_STORE_TYPE" ]; then
        log_warning "VECTOR_STORE_TYPE not set, defaulting to 'pinecone'"
    else
        log_success "Vector store type: $VECTOR_STORE_TYPE"
    fi
else
    log_error ".env file not found"
    exit 1
fi

# Build and start services
log_step "4. Building Docker Images..."

if docker-compose build; then
    log_success "Docker images built successfully"
else
    log_error "Failed to build Docker images"
    exit 1
fi

log_step "5. Starting Services..."

if docker-compose up -d; then
    log_success "Services started successfully"
else
    log_error "Failed to start services"
    exit 1
fi

# Wait for services to be ready
log_step "6. Waiting for Services to Start..."

echo "Waiting for services to be ready..."
sleep 30

# Health check
log_step "7. Running Health Checks..."

services=("knowledge-base-service:8002" "conversation-service:8001" "analytics-service:8005")
all_healthy=true

for service in "${services[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -s -f "http://localhost:$port/health" > /dev/null; then
        log_success "$name is healthy"
    else
        log_error "$name is not responding on port $port"
        all_healthy=false
    fi
done

if [ "$all_healthy" = true ]; then
    log_success "All services are healthy!"
else
    log_warning "Some services are not responding. Check logs with: docker-compose logs"
fi

# Create test script executable
if [ -f test_complete_workflow.sh ]; then
    chmod +x test_complete_workflow.sh
    log_success "Test script made executable"
fi

# Summary
echo ""
echo "====================================="
log_success "🎉 Setup Complete!"
echo "====================================="
echo ""
echo "🌐 Service URLs:"
echo "  - Knowledge Base Service: http://localhost:8002/docs"
echo "  - Conversation Service:   http://localhost:8001/docs" 
echo "  - Analytics Service:      http://localhost:8005/docs"
echo "  - Prometheus:             http://localhost:9090"
echo "  - Grafana:                http://localhost:3000"
echo ""
echo "🧪 Testing:"
echo "  - Run complete test: ./test_complete_workflow.sh"
echo "  - Import Postman collection: postman_collection.json"
echo ""
echo "🛠️  Management:"
echo "  - View logs:    docker-compose logs [service-name]"
echo "  - Stop services: docker-compose down"
echo "  - Restart:      docker-compose restart"
echo ""
echo "📚 Documentation:"
echo "  - Testing Guide: TESTING_GUIDE.md"
echo "  - Architecture:  README.md"
echo ""

if [ "$all_healthy" = true ]; then
    echo "🚀 Your LLM-Powered Retrieval System is ready to use!"
else
    echo "⚠️  Some services need attention. Check the logs and try again."
fi