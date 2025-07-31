"""
Shared middleware for security, authentication, and monitoring
"""
import time
import uuid
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi import HTTPException, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import jwt
from datetime import datetime, timedelta, timezone
import logging
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')

# Structured logging
logger = structlog.get_logger()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect Prometheus metrics for each request"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        ACTIVE_CONNECTIONS.inc()
        
        try:
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            return response
        
        except Exception as e:
            # Record failed requests
            duration = time.time() - start_time
            REQUEST_DURATION.observe(duration)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=500
            ).inc()
            raise e
        
        finally:
            ACTIVE_CONNECTIONS.dec()


class LoggingMiddleware(BaseHTTPMiddleware):
    """Structured logging for all requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        request_id = getattr(request.state, 'request_id', 'unknown')
        
        # Log request
        logger.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            user_agent=request.headers.get("user-agent"),
            client_ip=request.client.host if request.client else "unknown"
        )
        
        try:
            response = await call_next(request)
            
            # Log response
            duration = time.time() - start_time
            logger.info(
                "Request completed",
                request_id=request_id,
                status_code=response.status_code,
                duration=round(duration, 3)
            )
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                duration=round(duration, 3),
                exc_info=True
            )
            raise e


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """JWT Authentication middleware (optional)"""
    
    def __init__(self, app, secret_key: str, excluded_paths: Optional[list] = None):
        super().__init__(app)
        self.secret_key = secret_key
        self.excluded_paths = excluded_paths or [
            "/health", "/ready", "/docs", "/openapi.json", "/metrics", "/"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip auth for excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid authorization header"}
            )
        
        # Extract and validate token
        token = auth_header.split(" ")[1]
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            request.state.user = payload
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid token"}
            )
        
        return await call_next(request)


def create_rate_limiter() -> Limiter:
    """Create rate limiter instance"""
    limiter = Limiter(key_func=get_remote_address)
    return limiter


def create_jwt_token(payload: dict, secret_key: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT token"""
    to_encode = payload.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(hours=24)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    
    return encoded_jwt


async def rate_limit_error_handler(request: Request, exc: RateLimitExceeded):
    """Custom rate limit error handler"""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after": exc.retry_after
        }
    )