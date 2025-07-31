import os
import sys
from datetime import datetime, timezone

# Add shared modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
import httpx
import uvicorn
import asyncio
from contextlib import asynccontextmanager
from prometheus_client import make_asgi_app
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

# Basic configuration class
class APIGatewayConfig:
    def __init__(self):
        self.service_name = "api-gateway"
        self.version = "1.0.0"
        self.host = "0.0.0.0"
        self.port = 8000
        self.rate_limit_requests = 100
        self.conversation_service_url = os.getenv("CONVERSATION_SERVICE_URL", "http://localhost:8001")
        self.knowledge_base_service_url = os.getenv("KNOWLEDGE_BASE_SERVICE_URL", "http://localhost:8002")
        self.analytics_service_url = os.getenv("ANALYTICS_SERVICE_URL", "http://localhost:8005")
        self.service_timeout = 30.0
        self.connection_timeout = 10.0
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"

# Configuration
config = APIGatewayConfig()
logger = structlog.get_logger()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting API Gateway", config=config.service_name)
    yield
    logger.info("Shutting down API Gateway")

app = FastAPI(
    title="API Gateway",
    description="Secure API Gateway for LLM-Powered Retrieval System",
    version="1.0.0",
    lifespan=lifespan
)

# Basic CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
# Rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response(
        content=f"Rate limit exceeded: {exc.detail}",
        status_code=429,
        headers={"X-RateLimit-Limit": str(exc.retry_after)}
    )

# Prometheus metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Service health status
service_health = {
    "conversation": False,
    "knowledge_base": False,
    "analytics": False
}

# Health check background task
async def check_services_health():
    """Periodically check backend services health"""
    services = {
        "conversation": config.conversation_service_url,
        "knowledge_base": config.knowledge_base_service_url,
        "analytics": config.analytics_service_url
    }
    
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, service_url in services.items():
            try:
                response = await client.get(f"{service_url}/health")
                service_health[service_name] = response.status_code == 200
            except Exception:
                service_health[service_name] = False

# Start health check task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_health_check())

async def periodic_health_check():
    while True:
        await check_services_health()
        await asyncio.sleep(30)  # Check every 30 seconds

async def proxy_request(url: str, method: str, request: Request) -> Response:
    """Enhanced proxy with circuit breaker pattern and retry logic"""
    max_retries = 3
    retry_delay = 1.0
    
    # Prepare headers (exclude hop-by-hop headers)
    headers = dict(request.headers)
    hop_by_hop_headers = {
        'connection', 'keep-alive', 'proxy-authenticate',
        'proxy-authorization', 'te', 'trailers', 'transfer-encoding', 'upgrade',
        'host'  # Let httpx set the correct host header
    }
    headers = {k: v for k, v in headers.items() if k.lower() not in hop_by_hop_headers}
    
    # Add request tracing headers
    if hasattr(request.state, 'request_id'):
        headers['X-Request-ID'] = request.state.request_id
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(config.service_timeout, connect=config.connection_timeout),
                follow_redirects=False
            ) as client:
                # Get request body
                body = await request.body()
                
                # Forward request
                response = await client.request(
                    method=method,
                    url=url,
                    content=body,
                    headers=headers,
                    params=dict(request.query_params)
                )
                
                # Return response with proper status code
                return Response(
                    content=response.content,
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    media_type=response.headers.get("content-type")
                )
                
        except httpx.TimeoutException:
            logger.warning(f"Timeout on attempt {attempt + 1} for {url}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "error": "Service timeout",
                        "service_url": url,
                        "timeout": config.service_timeout
                    }
                )
                
        except httpx.ConnectError as e:
            logger.error(f"Connection error on attempt {attempt + 1} for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "error": "Service unavailable",
                        "service_url": url,
                        "details": str(e)
                    }
                )
                
        except httpx.RequestError as e:
            logger.error(f"Request error on attempt {attempt + 1} for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "error": "Request failed",
                        "service_url": url,
                        "details": str(e)
                    }
                )
                
        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt + 1} for {url}: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": "Internal proxy error",
                        "details": str(e)
                    }
                )
        
        # Wait before retry
        if attempt < max_retries - 1:
            await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff

@app.get("/health")
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    await check_services_health()
    
    overall_health = "healthy" if all(service_health.values()) else "degraded"
    
    return {
        "status": overall_health,
        "service": config.service_name,
        "version": "1.0.0",
        "timestamp": str(datetime.now(timezone.utc)),
        "services": service_health
    }

@app.get("/ready")
@limiter.limit("30/minute")
async def readiness_check(request: Request):
    """Readiness check for Kubernetes"""
    await check_services_health()
    
    # Consider ready if at least conversation and knowledge-base services are healthy
    critical_services = ["conversation", "knowledge_base"]
    ready = all(service_health.get(service, False) for service in critical_services)
    
    if not ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Critical services not available"
        )
    
    return {
        "status": "ready",
        "service": config.service_name,
        "critical_services": {service: service_health[service] for service in critical_services}
    }

# Conversation service routes
@app.api_route("/conversation/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@limiter.limit(f"{config.rate_limit_requests}/minute")
async def conversation_proxy(path: str, request: Request):
    """Proxy requests to conversation service"""
    if not service_health.get("conversation", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Conversation service is currently unavailable"
        )
    
    url = f"{config.conversation_service_url}/api/v1/{path}"
    return await proxy_request(url, request.method, request)

# Knowledge base service routes  
@app.api_route("/knowledge/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@limiter.limit(f"{config.rate_limit_requests}/minute")
async def knowledge_proxy(path: str, request: Request):
    """Proxy requests to knowledge base service"""
    if not service_health.get("knowledge_base", False):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base service is currently unavailable"
        )
    
    url = f"{config.knowledge_base_service_url}/api/v1/{path}"
    return await proxy_request(url, request.method, request)

# Analytics service routes
@app.api_route("/analytics/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
@limiter.limit(f"{config.rate_limit_requests}/minute")
async def analytics_proxy(path: str, request: Request):
    """Proxy requests to analytics service"""
    if not service_health.get("analytics", False):
        logger.warning("Analytics service unavailable, request will be forwarded anyway")
    
    url = f"{config.analytics_service_url}/api/v1/{path}"
    return await proxy_request(url, request.method, request)

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """Root endpoint with service information"""
    return {
        "message": "LLM-Powered Retrieval System API Gateway",
        "version": "1.0.0",
        "environment": config.environment,
        "services": {
            "conversation": {
                "url": config.conversation_service_url,
                "healthy": service_health.get("conversation", False)
            },
            "knowledge_base": {
                "url": config.knowledge_base_service_url,
                "healthy": service_health.get("knowledge_base", False)
            },
            "analytics": {
                "url": config.analytics_service_url,
                "healthy": service_health.get("analytics", False)
            }
        },
        "documentation": "/docs",
        "health_check": "/health",
        "metrics": "/metrics"
    }

if __name__ == "__main__":
    import logging
    from datetime import datetime
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if config.debug else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.info(f"Starting {config.service_name} on {config.host}:{config.port}")
    logger.info(f"Environment: {config.environment}")
    logger.info(f"Debug mode: {config.debug}")
    
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if config.debug else "warning"
    )