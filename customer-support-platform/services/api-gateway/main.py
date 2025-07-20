"""
API Gateway for Customer Support Platform

This is the main entry point for all API requests to the customer support platform.
It provides authentication, rate limiting, request routing, and comprehensive API documentation.
"""

import time
import logging
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

import httpx
from prometheus_client import Counter, Histogram, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from shared.config.settings import get_api_gateway_settings
from shared.auth.jwt_handler import JWTHandler
from shared.monitoring.metrics import MetricsCollector
from shared.database.connection import get_database_session

# Configuration
settings = get_api_gateway_settings()

# Observability setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Metrics
REQUEST_COUNT = Counter(
    'api_gateway_requests_total',
    'Total API Gateway requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'api_gateway_request_duration_seconds',
    'API Gateway request duration',
    ['method', 'endpoint']
)

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    # Startup
    logging.info("Starting API Gateway")
    yield
    # Shutdown
    logging.info("Shutting down API Gateway")


# FastAPI application
app = FastAPI(
    title="Customer Support Platform API",
    description="Production-ready customer support platform with LangChain and LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure based on your needs
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Authentication
security = HTTPBearer()
jwt_handler = JWTHandler(settings.jwt.secret_key, settings.jwt.algorithm)


class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        # Check if under limit
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(now)
        return True


rate_limiter = RateLimiter(
    max_requests=settings.security.rate_limit_requests,
    window_seconds=settings.security.rate_limit_window
)


# Dependencies
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get current authenticated user."""
    try:
        payload = jwt_handler.decode_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def check_rate_limit(request: Request):
    """Check rate limiting."""
    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded"
        )


# Health check endpoints
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    """Readiness check endpoint."""
    # Check database connection
    try:
        # Add actual database connectivity check
        return {"status": "ready", "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready"
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()


# API route handlers
class ServiceClient:
    """HTTP client for communicating with microservices."""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def make_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        json: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to a service."""
        try:
            response = await self.client.request(
                method=method,
                url=url,
                headers=headers,
                json=json,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=e.response.text
            )
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unavailable: {str(e)}"
            )


service_client = ServiceClient()


# Conversation endpoints
@app.post("/api/v1/conversations", tags=["Conversations"])
async def create_conversation(
    request: Request,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Create a new conversation.
    
    **Request Body:**
    ```json
    {
        "message": "Hello, I need help with my account",
        "metadata": {
            "channel": "web",
            "user_agent": "Mozilla/5.0..."
        }
    }
    ```
    
    **Response:**
    ```json
    {
        "conversation_id": "uuid",
        "session_id": "session_uuid",
        "response": "Hello! I'm here to help you with your account. What specific issue are you experiencing?",
        "metadata": {
            "intent": "account_support",
            "confidence": 0.95,
            "sentiment": "neutral"
        }
    }
    ```
    """
    body = await request.json()
    
    with tracer.start_as_current_span("create_conversation"):
        response = await service_client.make_request(
            method="POST",
            url=f"{settings.services.conversation_service_url}/conversations",
            json={**body, "user_id": user["sub"]},
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/conversations",
        status="200"
    ).inc()
    
    return response


@app.get("/api/v1/conversations/{conversation_id}", tags=["Conversations"])
async def get_conversation(
    conversation_id: str,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Get conversation details and history.
    
    **Path Parameters:**
    - `conversation_id`: UUID of the conversation
    
    **Response:**
    ```json
    {
        "conversation_id": "uuid",
        "user_id": "user_uuid",
        "status": "active",
        "created_at": "2023-01-01T00:00:00Z",
        "messages": [
            {
                "id": "msg_uuid",
                "content": "Hello, I need help",
                "type": "user",
                "timestamp": "2023-01-01T00:00:00Z"
            }
        ],
        "metadata": {
            "sentiment_score": 0.1,
            "resolution_time": null
        }
    }
    ```
    """
    with tracer.start_as_current_span("get_conversation"):
        response = await service_client.make_request(
            method="GET",
            url=f"{settings.services.conversation_service_url}/conversations/{conversation_id}",
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    return response


@app.post("/api/v1/conversations/{conversation_id}/messages", tags=["Conversations"])
async def send_message(
    conversation_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Send a message in an existing conversation.
    
    **Path Parameters:**
    - `conversation_id`: UUID of the conversation
    
    **Request Body:**
    ```json
    {
        "message": "I can't log into my account",
        "metadata": {
            "timestamp": "2023-01-01T00:00:00Z",
            "channel": "web"
        }
    }
    ```
    
    **Response:**
    ```json
    {
        "message_id": "msg_uuid",
        "response": "I can help you with login issues. Let me check your account status.",
        "metadata": {
            "intent": "login_issue",
            "confidence": 0.92,
            "sentiment": "frustrated",
            "escalation_suggested": false
        }
    }
    ```
    """
    body = await request.json()
    
    with tracer.start_as_current_span("send_message"):
        response = await service_client.make_request(
            method="POST",
            url=f"{settings.services.conversation_service_url}/conversations/{conversation_id}/messages",
            json={**body, "user_id": user["sub"]},
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    return response


# Knowledge base endpoints
@app.get("/api/v1/knowledge-base/search", tags=["Knowledge Base"])
async def search_knowledge_base(
    q: str,
    limit: int = 10,
    category: Optional[str] = None,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Search the knowledge base.
    
    **Query Parameters:**
    - `q`: Search query
    - `limit`: Maximum number of results (default: 10)
    - `category`: Filter by category (optional)
    
    **Response:**
    ```json
    {
        "results": [
            {
                "id": "kb_uuid",
                "title": "How to reset your password",
                "content": "To reset your password...",
                "category": "account",
                "score": 0.95,
                "metadata": {
                    "helpful_count": 25,
                    "last_updated": "2023-01-01T00:00:00Z"
                }
            }
        ],
        "total": 1,
        "query": "reset password"
    }
    ```
    """
    params = {"q": q, "limit": limit}
    if category:
        params["category"] = category
    
    with tracer.start_as_current_span("search_knowledge_base"):
        response = await service_client.make_request(
            method="GET",
            url=f"{settings.services.knowledge_base_service_url}/search",
            params=params,
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    return response


@app.post("/api/v1/knowledge-base/documents", tags=["Knowledge Base"])
async def create_knowledge_base_document(
    request: Request,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Create a new knowledge base document.
    
    **Request Body:**
    ```json
    {
        "title": "How to reset your password",
        "content": "Step-by-step guide to reset your password...",
        "category": "account",
        "tags": ["password", "reset", "account"],
        "metadata": {
            "author": "support_team",
            "difficulty": "easy"
        }
    }
    ```
    
    **Response:**
    ```json
    {
        "id": "kb_uuid",
        "title": "How to reset your password",
        "status": "active",
        "vector_id": "vector_uuid",
        "created_at": "2023-01-01T00:00:00Z"
    }
    ```
    """
    body = await request.json()
    
    with tracer.start_as_current_span("create_knowledge_base_document"):
        response = await service_client.make_request(
            method="POST",
            url=f"{settings.services.knowledge_base_service_url}/documents",
            json=body,
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    return response


# NLP endpoints
@app.post("/api/v1/nlp/analyze", tags=["NLP"])
async def analyze_text(
    request: Request,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Analyze text for sentiment, intent, and other NLP features.
    
    **Request Body:**
    ```json
    {
        "text": "I'm really frustrated with this service",
        "features": ["sentiment", "intent", "spam_detection"],
        "metadata": {
            "language": "en",
            "context": "customer_support"
        }
    }
    ```
    
    **Response:**
    ```json
    {
        "sentiment": {
            "label": "negative",
            "score": 0.85,
            "confidence": 0.92
        },
        "intent": {
            "label": "complaint",
            "score": 0.78,
            "confidence": 0.89
        },
        "spam_detection": {
            "is_spam": false,
            "confidence": 0.95
        },
        "metadata": {
            "processing_time": 0.123,
            "model_version": "v1.2.3"
        }
    }
    ```
    """
    body = await request.json()
    
    with tracer.start_as_current_span("analyze_text"):
        response = await service_client.make_request(
            method="POST",
            url=f"{settings.services.nlp_service_url}/analyze",
            json=body,
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    return response


# Analytics endpoints
@app.get("/api/v1/analytics/metrics", tags=["Analytics"])
async def get_analytics_metrics(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    user: dict = Depends(get_current_user),
    _: None = Depends(check_rate_limit)
):
    """
    Get analytics metrics for the specified time range.
    
    **Query Parameters:**
    - `start_date`: Start date in ISO format (optional)
    - `end_date`: End date in ISO format (optional)
    
    **Response:**
    ```json
    {
        "metrics": {
            "total_conversations": 1250,
            "avg_response_time": 2.3,
            "resolution_rate": 0.85,
            "customer_satisfaction": 4.2,
            "escalation_rate": 0.15
        },
        "time_series": [
            {
                "date": "2023-01-01",
                "conversations": 45,
                "avg_response_time": 2.1
            }
        ],
        "period": {
            "start": "2023-01-01T00:00:00Z",
            "end": "2023-01-31T23:59:59Z"
        }
    }
    ```
    """
    params = {}
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date
    
    with tracer.start_as_current_span("get_analytics_metrics"):
        response = await service_client.make_request(
            method="GET",
            url=f"{settings.services.analytics_service_url}/metrics",
            params=params,
            headers={"Authorization": f"Bearer {user['token']}"}
        )
    
    return response


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Customer Support Platform API",
        version="1.0.0",
        description="""
        # Customer Support Platform API
        
        A production-ready customer support platform powered by LangChain and LangGraph.
        
        ## Features
        - 🤖 AI-powered conversations with context awareness
        - 📚 Intelligent knowledge base search
        - 🎯 Advanced NLP analysis (sentiment, intent, spam detection)
        - 📊 Real-time analytics and metrics
        - 🔒 JWT-based authentication
        - 🚀 Rate limiting and monitoring
        - 📈 Comprehensive observability
        
        ## Authentication
        All endpoints require a valid JWT token in the Authorization header:
        ```
        Authorization: Bearer <your_jwt_token>
        ```
        
        ## Rate Limiting
        API requests are rate-limited to prevent abuse:
        - 100 requests per minute per IP address
        - 429 status code returned when limit is exceeded
        
        ## Error Handling
        The API returns standardized error responses:
        ```json
        {
            "detail": "Error message",
            "error_code": "SPECIFIC_ERROR_CODE",
            "timestamp": "2023-01-01T00:00:00Z"
        }
        ```
        """,
        routes=app.routes,
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "bearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
        }
    }
    
    # Add security requirement to all endpoints
    for path_item in openapi_schema["paths"].values():
        for operation in path_item.values():
            if isinstance(operation, dict) and "tags" in operation:
                operation["security"] = [{"bearerAuth": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "error_code": f"HTTP_{exc.status_code}",
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "timestamp": time.time(),
            "path": request.url.path
        }
    )


# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests and collect metrics."""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=str(response.status_code)
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    # Log request
    logging.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Duration: {duration:.3f}s"
    )
    
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level="info",
        access_log=True
    )