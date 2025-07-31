"""
LLM Client Manager with Fallback Support
Handles multiple LLM providers with automatic fallback functionality.
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
from datetime import datetime, timedelta

import openai
from anthropic import AsyncAnthropic
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker for LLM provider resilience."""
    
    def __init__(self, 
                 provider: LLMProvider,
                 failure_threshold: int = 5,
                 recovery_timeout: int = 60,
                 success_threshold: int = 2):
        self.provider = provider
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
        logger.info(f"Circuit breaker initialized for {provider.value}: threshold={failure_threshold}, recovery={recovery_timeout}s")
        
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker for {self.provider.value} moved to HALF_OPEN")
                return True
            return False
        else:  # HALF_OPEN
            return True
            
    def record_success(self):
        """Record successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info(f"Circuit breaker for {self.provider.value} CLOSED after {self.success_count} successes")
        elif self.state == CircuitBreakerState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradually reduce failure count
        
    def record_failure(self, error: Exception = None):
        """Record failed execution."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {self.provider.value} OPENED after {self.failure_count} failures. Error: {error}")
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker for {self.provider.value} returned to OPEN state after failure in HALF_OPEN")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "provider": self.provider.value,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "can_execute": self.can_execute()
        }


@dataclass
class LLMResponse:
    """Standardized response from LLM providers."""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    response_time: Optional[float] = None
    circuit_breaker_state: Optional[str] = None
    fallback_used: bool = False


@dataclass
class StreamingLLMResponse:
    """Standardized streaming response from LLM providers."""
    content_delta: str
    provider: LLMProvider
    model: str
    is_complete: bool = False
    finish_reason: Optional[str] = None


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider = self._get_provider()
    
    @abstractmethod
    def _get_provider(self) -> LLMProvider:
        """Return the provider enum."""
        pass
    
    @abstractmethod
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """Generate a streaming response from the LLM."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI LLM client."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        super().__init__(api_key, model)
        self.client = openai.AsyncOpenAI(api_key=api_key)
    
    def _get_provider(self) -> LLMProvider:
        return LLMProvider.OPENAI
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using OpenAI API."""
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider=self.provider,
                model=self.model,
                usage=response.usage.model_dump() if response.usage else None,
                finish_reason=response.choices[0].finish_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """Generate streaming response using OpenAI API."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield StreamingLLMResponse(
                        content_delta=chunk.choices[0].delta.content,
                        provider=self.provider,
                        model=self.model
                    )
                
                if chunk.choices[0].finish_reason:
                    yield StreamingLLMResponse(
                        content_delta="",
                        provider=self.provider,
                        model=self.model,
                        is_complete=True,
                        finish_reason=chunk.choices[0].finish_reason
                    )
                    
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            raise


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude LLM client."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        super().__init__(api_key, model)
        self.client = AsyncAnthropic(api_key=api_key)
    
    def _get_provider(self) -> LLMProvider:
        return LLMProvider.ANTHROPIC
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Convert messages to Anthropic format."""
        system_msg = ""
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        return system_msg, claude_messages
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Anthropic API."""
        start_time = time.time()
        
        try:
            system_msg, claude_messages = self._convert_messages(messages)
            
            response = await self.client.messages.create(
                model=self.model,
                system=system_msg if system_msg else "",
                messages=claude_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            response_time = time.time() - start_time
            
            return LLMResponse(
                content=response.content[0].text,
                provider=self.provider,
                model=self.model,
                usage=response.usage.model_dump() if hasattr(response, 'usage') else None,
                finish_reason=response.stop_reason,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise
    
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """Generate streaming response using Anthropic API."""
        try:
            system_msg, claude_messages = self._convert_messages(messages)
            
            async with self.client.messages.stream(
                model=self.model,
                system=system_msg if system_msg else "",
                messages=claude_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamingLLMResponse(
                        content_delta=text,
                        provider=self.provider,
                        model=self.model
                    )
                
                # Final completion message
                yield StreamingLLMResponse(
                    content_delta="",
                    provider=self.provider,
                    model=self.model,
                    is_complete=True,
                    finish_reason="stop"
                )
                    
        except Exception as e:
            logger.error(f"Anthropic streaming error: {e}")
            raise


class GeminiClient(BaseLLMClient):
    """Google Gemini LLM client."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        super().__init__(api_key, model)
        genai.configure(api_key=api_key)
        self.model_instance = genai.GenerativeModel(model)
    
    def _get_provider(self) -> LLMProvider:
        return LLMProvider.GEMINI
    
    def _convert_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages to Gemini format."""
        conversation = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                conversation += f"System: {content}\n\n"
            elif role == "user":
                conversation += f"Human: {content}\n\n"
            elif role == "assistant":
                conversation += f"Assistant: {content}\n\n"
        
        conversation += "Assistant: "
        return conversation
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response using Gemini API with retry logic."""
        max_retries = 3
        base_timeout = 60  # Increased timeout for Gemini
        
        for attempt in range(max_retries):
            start_time = time.time()
            current_timeout = base_timeout * (attempt + 1)  # Progressive timeout increase
            
            try:
                prompt = self._convert_messages(messages)
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    **kwargs
                )
                
                # Use asyncio.wait_for to control timeout
                response = await asyncio.wait_for(
                    self.model_instance.generate_content_async(
                        prompt,
                        generation_config=generation_config
                    ),
                    timeout=current_timeout
                )
                
                response_time = time.time() - start_time
                
                # Check if response has content
                if not response.text or not response.text.strip():
                    raise Exception("Gemini returned empty response")
                
                logger.info(f"Gemini API successful on attempt {attempt + 1} after {response_time:.2f}s")
                
                return LLMResponse(
                    content=response.text,
                    provider=self.provider,
                    model=self.model,
                    usage=None,  # Gemini doesn't provide detailed usage in free tier
                    finish_reason=response.candidates[0].finish_reason.name if response.candidates else None,
                    response_time=response_time
                )
                
            except asyncio.TimeoutError:
                elapsed = time.time() - start_time
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Gemini API timeout after {elapsed:.2f}s (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Gemini API timeout after {elapsed:.2f}s (final attempt {attempt + 1}/{max_retries})"
                    logger.error(error_msg)
                    raise Exception(error_msg)
            except Exception as e:
                elapsed = time.time() - start_time
                if attempt < max_retries - 1 and "rate" not in str(e).lower():
                    # Retry for non-rate-limit errors
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini API error after {elapsed:.2f}s (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {str(e)}, retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    error_msg = f"Gemini API error after {elapsed:.2f}s: {type(e).__name__}: {str(e)}"
                    if not str(e):  # Handle empty error messages
                        error_msg = f"Gemini API error after {elapsed:.2f}s: {type(e).__name__} with empty message (repr: {repr(e)})"
                    logger.error(error_msg)
                    raise Exception(error_msg)
        
        # Should never reach here, but just in case
        raise Exception("Gemini API failed after all retry attempts")
    
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """Generate streaming response using Gemini API."""
        try:
            prompt = self._convert_messages(messages)
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            response = await self.model_instance.generate_content_async(
                prompt,
                generation_config=generation_config,
                stream=True
            )
            
            async for chunk in response:
                if chunk.text:
                    yield StreamingLLMResponse(
                        content_delta=chunk.text,
                        provider=self.provider,
                        model=self.model
                    )
            
            # Final completion message
            yield StreamingLLMResponse(
                content_delta="",
                provider=self.provider,
                model=self.model,
                is_complete=True,
                finish_reason="stop"
            )
                    
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise


class LLMClientManager:
    """Manages multiple LLM clients with fallback support."""
    
    def __init__(self, config):
        self.config = config
        self.clients: Dict[LLMProvider, BaseLLMClient] = {}
        self.circuit_breakers: Dict[LLMProvider, CircuitBreaker] = {}
        self._initialize_clients()
        self._initialize_circuit_breakers()
    
    def _initialize_clients(self):
        """Initialize all available LLM clients."""
        # Initialize OpenAI client
        if hasattr(self.config, 'openai_api_key') and self.config.openai_api_key:
            self.clients[LLMProvider.OPENAI] = OpenAIClient(
                self.config.openai_api_key, 
                self.config.openai_model
            )
        
        # Initialize Anthropic client
        if hasattr(self.config, 'anthropic_api_key') and self.config.anthropic_api_key:
            self.clients[LLMProvider.ANTHROPIC] = AnthropicClient(
                self.config.anthropic_api_key, 
                self.config.anthropic_model
            )
        
        # Initialize Gemini client
        if hasattr(self.config, 'gemini_api_key') and self.config.gemini_api_key:
            self.clients[LLMProvider.GEMINI] = GeminiClient(
                self.config.gemini_api_key, 
                self.config.gemini_model
            )
        
        logger.info(f"Initialized {len(self.clients)} LLM clients: {list(self.clients.keys())}")
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all clients."""
        for provider in self.clients.keys():
            self.circuit_breakers[provider] = CircuitBreaker(
                provider=provider,
                failure_threshold=getattr(self.config, 'circuit_breaker_failure_threshold', 5),
                recovery_timeout=getattr(self.config, 'circuit_breaker_recovery_timeout', 60),
                success_threshold=getattr(self.config, 'circuit_breaker_success_threshold', 2)
            )
        logger.info(f"Initialized circuit breakers for {len(self.circuit_breakers)} providers")
    
    def _get_provider_order(self) -> List[LLMProvider]:
        """Get the order of providers to try (primary first, then fallbacks)."""
        provider_order = []
        
        # Add primary provider first
        if hasattr(self.config, 'primary_llm_provider'):
            try:
                primary = LLMProvider(self.config.primary_llm_provider)
                if primary in self.clients:
                    provider_order.append(primary)
            except ValueError:
                logger.warning(f"Invalid primary provider: {self.config.primary_llm_provider}")
        
        # Add fallback providers
        if hasattr(self.config, 'enable_fallback') and self.config.enable_fallback and hasattr(self.config, 'fallback_providers'):
            for fallback_name in self.config.fallback_providers:
                try:
                    fallback = LLMProvider(fallback_name)
                    if fallback in self.clients and fallback not in provider_order:
                        provider_order.append(fallback)
                except ValueError:
                    logger.warning(f"Invalid fallback provider: {fallback_name}")
        
        return provider_order
    
    async def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> LLMResponse:
        """Generate response with circuit breaker fallback support."""
        provider_order = self._get_provider_order_with_circuit_breakers()
        
        if not provider_order:
            raise ValueError("No LLM providers available")
        
        last_error = None
        fallback_used = False
        
        for i, provider in enumerate(provider_order):
            if provider not in self.circuit_breakers:
                continue
                
            circuit_breaker = self.circuit_breakers[provider]
            
            # Check circuit breaker
            if not circuit_breaker.can_execute():
                logger.info(f"Circuit breaker for {provider.value} is OPEN, skipping")
                fallback_used = True
                continue
            
            client = self.clients[provider]
            
            try:
                logger.info(f"Attempting to generate response using {provider.value} (attempt {i+1}/{len(provider_order)})")
                start_time = time.time()
                
                # Use timeout for each provider attempt
                response = await asyncio.wait_for(
                    client.generate_response(messages, temperature, max_tokens, **kwargs),
                    timeout=self.config.fallback_timeout
                )
                
                response_time = time.time() - start_time
                
                # Record success in circuit breaker
                circuit_breaker.record_success()
                
                logger.info(f"Successfully generated response using {provider.value}")
                
                # Update response with circuit breaker info
                return LLMResponse(
                    content=response.content,
                    provider=response.provider,
                    model=response.model,
                    usage=response.usage,
                    finish_reason=response.finish_reason,
                    response_time=response_time,
                    circuit_breaker_state=circuit_breaker.state.value,
                    fallback_used=fallback_used
                )
                
            except Exception as e:
                last_error = e
                circuit_breaker.record_failure(e)
                error_message = str(e) if str(e) else f"{type(e).__name__} with empty message (repr: {repr(e)})"
                logger.warning(f"Failed to generate response using {provider.value}: {error_message}")
                fallback_used = True
                
                if not self.config.enable_fallback or provider == provider_order[-1]:
                    # If fallback is disabled or this is the last provider, raise the error
                    break
                
                logger.info(f"Falling back to next provider...")
        
        # If we get here, all providers failed
        circuit_breaker_status = {p.value: cb.get_state() for p, cb in self.circuit_breakers.items()}
        logger.error(f"All LLM providers failed. Circuit breaker status: {circuit_breaker_status}")
        raise last_error or Exception(f"All LLM providers failed. Circuit breaker status: {circuit_breaker_status}")
        
    def _get_provider_order_with_circuit_breakers(self) -> List[LLMProvider]:
        """Get ordered list of providers to try, considering circuit breaker states."""
        # Start with primary provider from config
        primary = getattr(self.config, 'primary_llm_provider', 'gemini')
        
        provider_map = {
            'openai': LLMProvider.OPENAI,
            'anthropic': LLMProvider.ANTHROPIC,
            'gemini': LLMProvider.GEMINI
        }
        
        order = []
        
        # Add primary provider first (if circuit breaker allows)
        if primary in provider_map:
            primary_provider = provider_map[primary]
            if (primary_provider in self.clients and 
                primary_provider in self.circuit_breakers):
                order.append(primary_provider)
        
        # Add fallback providers
        fallback_providers = getattr(self.config, 'fallback_providers', ['openai', 'anthropic'])
        for fb_provider in fallback_providers:
            if fb_provider in provider_map:
                fb_provider_enum = provider_map[fb_provider]
                if (fb_provider_enum in self.clients and 
                    fb_provider_enum in self.circuit_breakers and 
                    fb_provider_enum not in order):
                    order.append(fb_provider_enum)
        
        # Sort by circuit breaker health (closed states first)
        def circuit_breaker_priority(provider):
            if provider not in self.circuit_breakers:
                return 999  # Lowest priority
            cb = self.circuit_breakers[provider]
            if cb.state == CircuitBreakerState.CLOSED:
                return 0  # Highest priority
            elif cb.state == CircuitBreakerState.HALF_OPEN:
                return 1  # Medium priority
            else:  # OPEN
                return 2  # Low priority (but still try if others fail)
        
        order.sort(key=circuit_breaker_priority)
        
        logger.info(f"Provider order with circuit breakers: {[p.value for p in order]}")
        return order
    
    async def generate_streaming_response(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncGenerator[StreamingLLMResponse, None]:
        """Generate streaming response with fallback support."""
        provider_order = self._get_provider_order()
        
        if not provider_order:
            raise ValueError("No LLM providers available")
        
        last_error = None
        
        for provider in provider_order:
            client = self.clients[provider]
            
            try:
                logger.info(f"Attempting to generate streaming response using {provider.value}")
                
                async for chunk in client.generate_streaming_response(
                    messages, temperature, max_tokens, **kwargs
                ):
                    yield chunk
                
                logger.info(f"Successfully completed streaming response using {provider.value}")
                return
                
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to generate streaming response using {provider.value}: {e}")
                
                if not self.config.enable_fallback or provider == provider_order[-1]:
                    break
                
                logger.info(f"Falling back to next provider...")
        
        # If we get here, all providers failed
        logger.error("All LLM providers failed for streaming")
        raise last_error or Exception("All LLM providers failed for streaming")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return [provider.value for provider in self.clients.keys()]
    
    def get_primary_provider(self) -> str:
        """Get the primary provider name."""
        return self.config.primary_llm_provider
    
    def get_fallback_providers(self) -> List[str]:
        """Get list of fallback provider names."""
        return self.config.fallback_providers if self.config.enable_fallback else []