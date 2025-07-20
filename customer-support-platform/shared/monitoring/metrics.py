"""Metrics collection and monitoring utilities."""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
import logging


# Prometheus metrics
CONVERSATION_REQUESTS = Counter(
    'conversation_requests_total',
    'Total conversation requests',
    ['intent', 'sentiment', 'resolution']
)

CONVERSATION_RESPONSE_TIME = Histogram(
    'conversation_response_time_seconds',
    'Conversation response time in seconds',
    ['intent', 'sentiment']
)

DOCUMENT_OPERATIONS = Counter(
    'document_operations_total',
    'Total document operations',
    ['operation', 'category', 'status']
)

SEARCH_REQUESTS = Counter(
    'search_requests_total',
    'Total search requests',
    ['category', 'results_found']
)

SEARCH_RESPONSE_TIME = Histogram(
    'search_response_time_seconds',
    'Search response time in seconds'
)

ACTIVE_CONVERSATIONS = Gauge(
    'active_conversations',
    'Number of active conversations'
)

ESCALATION_RATE = Gauge(
    'escalation_rate',
    'Human escalation rate'
)

CUSTOMER_SATISFACTION = Gauge(
    'customer_satisfaction_score',
    'Average customer satisfaction score'
)

NLP_PROCESSING_TIME = Histogram(
    'nlp_processing_time_seconds',
    'NLP processing time in seconds',
    ['operation']
)

VECTOR_STORE_OPERATIONS = Counter(
    'vector_store_operations_total',
    'Total vector store operations',
    ['operation', 'status']
)

SYSTEM_INFO = Info(
    'system_info',
    'System information'
)


class MetricsCollector:
    """Collector for application metrics."""
    
    def __init__(self):
        self.tracer = trace.get_tracer(__name__)
        self._active_conversations = 0
        
        # Set system info
        SYSTEM_INFO.info({
            'version': '1.0.0',
            'service': 'customer-support-platform'
        })
    
    def record_conversation_created(
        self,
        intent: str,
        sentiment: str,
        response_time: float
    ):
        """Record conversation creation metrics."""
        
        CONVERSATION_REQUESTS.labels(
            intent=intent,
            sentiment=sentiment,
            resolution='created'
        ).inc()
        
        CONVERSATION_RESPONSE_TIME.labels(
            intent=intent,
            sentiment=sentiment
        ).observe(response_time)
        
        self._active_conversations += 1
        ACTIVE_CONVERSATIONS.set(self._active_conversations)
        
        logging.info(f"Conversation created: intent={intent}, sentiment={sentiment}, time={response_time:.2f}s")
    
    def record_conversation_completed(
        self,
        intent: str,
        sentiment: str,
        resolution: str,
        satisfaction_score: Optional[float] = None
    ):
        """Record conversation completion metrics."""
        
        CONVERSATION_REQUESTS.labels(
            intent=intent,
            sentiment=sentiment,
            resolution=resolution
        ).inc()
        
        self._active_conversations = max(0, self._active_conversations - 1)
        ACTIVE_CONVERSATIONS.set(self._active_conversations)
        
        if satisfaction_score is not None:
            CUSTOMER_SATISFACTION.set(satisfaction_score)
        
        logging.info(f"Conversation completed: intent={intent}, resolution={resolution}")
    
    def record_message_sent(
        self,
        intent: str,
        sentiment: str,
        response_time: float
    ):
        """Record message sending metrics."""
        
        CONVERSATION_RESPONSE_TIME.labels(
            intent=intent,
            sentiment=sentiment
        ).observe(response_time)
        
        logging.info(f"Message sent: intent={intent}, sentiment={sentiment}, time={response_time:.2f}s")
    
    def record_escalation(
        self,
        reason: str,
        escalation_rate: float
    ):
        """Record escalation metrics."""
        
        ESCALATION_RATE.set(escalation_rate)
        
        logging.info(f"Escalation recorded: reason={reason}, rate={escalation_rate:.2f}")
    
    def record_document_created(
        self,
        category: str,
        processing_time: float
    ):
        """Record document creation metrics."""
        
        DOCUMENT_OPERATIONS.labels(
            operation='create',
            category=category,
            status='success'
        ).inc()
        
        logging.info(f"Document created: category={category}, time={processing_time:.2f}s")
    
    def record_document_deleted(
        self,
        document_id: str
    ):
        """Record document deletion metrics."""
        
        DOCUMENT_OPERATIONS.labels(
            operation='delete',
            category='unknown',
            status='success'
        ).inc()
        
        logging.info(f"Document deleted: id={document_id}")
    
    def record_search_performed(
        self,
        query: str,
        results_count: int,
        processing_time: float,
        category: Optional[str] = None
    ):
        """Record search operation metrics."""
        
        results_found = 'found' if results_count > 0 else 'not_found'
        
        SEARCH_REQUESTS.labels(
            category=category or 'all',
            results_found=results_found
        ).inc()
        
        SEARCH_RESPONSE_TIME.observe(processing_time)
        
        logging.info(f"Search performed: query='{query[:50]}...', results={results_count}, time={processing_time:.2f}s")
    
    def record_nlp_processing(
        self,
        operation: str,
        processing_time: float
    ):
        """Record NLP processing metrics."""
        
        NLP_PROCESSING_TIME.labels(operation=operation).observe(processing_time)
        
        logging.info(f"NLP processing: operation={operation}, time={processing_time:.2f}s")
    
    def record_vector_store_operation(
        self,
        operation: str,
        status: str = 'success'
    ):
        """Record vector store operation metrics."""
        
        VECTOR_STORE_OPERATIONS.labels(
            operation=operation,
            status=status
        ).inc()
        
        logging.info(f"Vector store operation: operation={operation}, status={status}")
    
    def track_with_span(self, operation_name: str):
        """Decorator to track operation with distributed tracing."""
        
        def decorator(func):
            def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    start_time = time.time()
                    
                    try:
                        result = func(*args, **kwargs)
                        
                        # Record success
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("operation.success", True)
                        
                        return result
                    
                    except Exception as e:
                        # Record error
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("operation.success", False)
                        span.set_attribute("error.message", str(e))
                        
                        raise
                    
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        span.set_attribute("operation.duration", duration)
            
            return wrapper
        return decorator
    
    def track_async_with_span(self, operation_name: str):
        """Decorator to track async operation with distributed tracing."""
        
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(operation_name) as span:
                    start_time = time.time()
                    
                    try:
                        result = await func(*args, **kwargs)
                        
                        # Record success
                        span.set_status(Status(StatusCode.OK))
                        span.set_attribute("operation.success", True)
                        
                        return result
                    
                    except Exception as e:
                        # Record error
                        span.set_status(Status(StatusCode.ERROR, str(e)))
                        span.set_attribute("operation.success", False)
                        span.set_attribute("error.message", str(e))
                        
                        raise
                    
                    finally:
                        # Record duration
                        duration = time.time() - start_time
                        span.set_attribute("operation.duration", duration)
            
            return wrapper
        return decorator


# Global metrics collector instance
metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


# Decorator functions for easy use
def track_operation(operation_name: str):
    """Decorator to track operation metrics."""
    return metrics_collector.track_with_span(operation_name)


def track_async_operation(operation_name: str):
    """Decorator to track async operation metrics."""
    return metrics_collector.track_async_with_span(operation_name)


# Health check utilities
def get_health_metrics() -> Dict[str, Any]:
    """Get health check metrics."""
    return {
        "active_conversations": metrics_collector._active_conversations,
        "system_status": "healthy",
        "timestamp": time.time()
    }