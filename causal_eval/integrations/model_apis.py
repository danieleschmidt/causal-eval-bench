"""
Model API clients for external language model services.
"""

import os
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import random
from functools import wraps

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class APIStatus(Enum):
    """API status for circuit breaker."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    CIRCUIT_OPEN = "circuit_open"


@dataclass
class ModelResponse:
    """Response from a model API."""
    
    content: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    retry_count: int = 0
    api_status: APIStatus = APIStatus.HEALTHY


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    success_threshold: int = 3
    

class CircuitBreaker:
    """Circuit breaker for API calls."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.state = APIStatus.HEALTHY
    
    def call_succeeded(self):
        """Record a successful call."""
        if self.state == APIStatus.CIRCUIT_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = APIStatus.HEALTHY
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0
    
    def call_failed(self):
        """Record a failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        self.success_count = 0
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = APIStatus.CIRCUIT_OPEN
    
    def can_attempt_call(self) -> bool:
        """Check if calls are allowed."""
        if self.state != APIStatus.CIRCUIT_OPEN:
            return True
        
        # Check if enough time has passed to attempt recovery
        if time.time() - self.last_failure_time > self.config.recovery_timeout:
            self.state = APIStatus.DEGRADED  # Half-open state
            return True
        
        return False
    
    def get_status(self) -> APIStatus:
        """Get current circuit breaker status."""
        return self.state


def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
    """Decorator for exponential backoff retry logic."""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    result = await func(self, *args, **kwargs)
                    if hasattr(self, 'circuit_breaker'):
                        self.circuit_breaker.call_succeeded()
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    if hasattr(self, 'circuit_breaker'):
                        self.circuit_breaker.call_failed()
                    
                    if attempt == max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}")
                        break
                    
                    # Calculate delay with exponential backoff and jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__} in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


class ModelAPIClient(ABC):
    """Abstract base class for model API clients."""
    
    def __init__(self, api_key: str, model_name: str, enable_circuit_breaker: bool = True):
        """Initialize the client."""
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
        self.circuit_breaker = CircuitBreaker(CircuitBreakerConfig()) if enable_circuit_breaker else None
        self.request_history: List[Dict[str, Any]] = []
        self.total_tokens_used = 0
        self.total_cost = 0.0
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows calls."""
        if self.circuit_breaker and not self.circuit_breaker.can_attempt_call():
            raise Exception(f"Circuit breaker is open for {self.model_name}. API is temporarily unavailable.")
    
    def _record_request(self, prompt: str, response: ModelResponse) -> None:
        """Record request for monitoring and analytics."""
        self.request_history.append({
            "timestamp": time.time(),
            "prompt_length": len(prompt),
            "response_length": len(response.content),
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "error": response.error
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
        
        # Update totals
        if response.tokens_used:
            self.total_tokens_used += response.tokens_used
        if response.cost:
            self.total_cost += response.cost
    
    @abstractmethod
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response from the model."""
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics for this client."""
        if not self.request_history:
            return {}
        
        recent_requests = [r for r in self.request_history if time.time() - r["timestamp"] < 3600]  # Last hour
        
        return {
            "total_requests": len(self.request_history),
            "recent_requests_1h": len(recent_requests),
            "total_tokens_used": self.total_tokens_used,
            "total_cost": self.total_cost,
            "average_latency_ms": sum(r["latency_ms"] for r in self.request_history if r["latency_ms"]) / len(self.request_history),
            "error_rate": sum(1 for r in self.request_history if r["error"]) / len(self.request_history),
            "circuit_breaker_status": self.circuit_breaker.get_status().value if self.circuit_breaker else "disabled"
        }
    
    @abstractmethod
    async def close(self):
        """Close the client connection."""
        pass


class OpenAIClient(ModelAPIClient):
    """OpenAI API client for GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4", enable_circuit_breaker: bool = True):
        """Initialize OpenAI client."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        super().__init__(api_key, model_name, enable_circuit_breaker)
        self.client = AsyncOpenAI(api_key=api_key)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-4-1106-preview": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-1106": {"input": 0.001, "output": 0.002}
        }
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using OpenAI API."""
        self._check_circuit_breaker()
        
        start_time = time.time()
        response_obj = None
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Handle system prompts if provided
            if "system_prompt" in kwargs:
                messages.insert(0, {"role": "system", "content": kwargs.pop("system_prompt")})
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=kwargs.pop("timeout", 120.0),  # 2 minute timeout
                **kwargs
            )
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Extract response content
            content = response.choices[0].message.content or ""
            
            # Calculate costs
            usage = response.usage
            tokens_used = usage.total_tokens if usage else None
            cost = self._calculate_cost(usage) if usage else None
            
            response_obj = ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "model_version": response.model
                },
                api_status=self.circuit_breaker.get_status() if self.circuit_breaker else APIStatus.HEALTHY
            )
            
            self._record_request(prompt, response_obj)
            return response_obj
            
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            error_response = ModelResponse(
                content=f"Error: {str(e)}",
                model=self.model_name,
                tokens_used=0,
                cost=0.0,
                latency_ms=latency_ms,
                error=str(e),
                api_status=APIStatus.FAILING
            )
            
            self._record_request(prompt, error_response)
            logger.error(f"OpenAI API error for {self.model_name}: {e}")
            raise
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage."""
        if self.model_name not in self.pricing:
            return 0.0
        
        pricing = self.pricing[self.model_name]
        input_cost = (usage.prompt_tokens / 1000) * pricing["input"]
        output_cost = (usage.completion_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def close(self):
        """Close the client connection."""
        if self.client:
            await self.client.close()


class AnthropicClient(ModelAPIClient):
    """Anthropic API client for Claude models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-sonnet-20240229", enable_circuit_breaker: bool = True):
        """Initialize Anthropic client."""
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        super().__init__(api_key, model_name, enable_circuit_breaker)
        self.client = AsyncAnthropic(api_key=api_key)
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015}
        }
    
    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Anthropic API."""
        self._check_circuit_breaker()
        
        start_time = time.time()
        response_obj = None
        
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Handle system prompts
            system_prompt = kwargs.pop("system_prompt", None)
            
            # Prepare request parameters
            request_params = {
                "model": self.model_name,
                "max_tokens": max_tokens or 4000,
                "temperature": temperature,
                "messages": messages,
                "timeout": kwargs.pop("timeout", 120.0),
                **kwargs
            }
            
            if system_prompt:
                request_params["system"] = system_prompt
            
            response = await self.client.messages.create(**request_params)
            
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            # Extract response content
            content = ""
            if response.content and len(response.content) > 0:
                content = response.content[0].text
            
            # Calculate costs
            usage = response.usage
            tokens_used = usage.input_tokens + usage.output_tokens if usage else None
            cost = self._calculate_cost(usage) if usage else None
            
            response_obj = ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None,
                    "model_version": response.model
                },
                api_status=self.circuit_breaker.get_status() if self.circuit_breaker else APIStatus.HEALTHY
            )
            
            self._record_request(prompt, response_obj)
            return response_obj
            
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            error_response = ModelResponse(
                content=f"Error: {str(e)}",
                model=self.model_name,
                tokens_used=0,
                cost=0.0,
                latency_ms=latency_ms,
                error=str(e),
                api_status=APIStatus.FAILING
            )
            
            self._record_request(prompt, error_response)
            logger.error(f"Anthropic API error for {self.model_name}: {e}")
            raise
    
    def _calculate_cost(self, usage) -> float:
        """Calculate cost based on token usage."""
        if self.model_name not in self.pricing:
            return 0.0
        
        pricing = self.pricing[self.model_name]
        input_cost = (usage.input_tokens / 1000) * pricing["input"]
        output_cost = (usage.output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    async def close(self):
        """Close the client connection."""
        if self.client:
            await self.client.aclose()


class HuggingFaceClient(ModelAPIClient):
    """Hugging Face API client for open source models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-2-7b-chat-hf", enable_circuit_breaker: bool = True):
        """Initialize Hugging Face client."""
        api_key = api_key or os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_key:
            raise ValueError("Hugging Face API token is required")
        
        super().__init__(api_key, model_name, enable_circuit_breaker)
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    @retry_with_backoff(max_retries=3, base_delay=2.0, max_delay=60.0)
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Hugging Face Inference API."""
        self._check_circuit_breaker()
        
        start_time = time.time()
        response_obj = None
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens or 512,
                        "return_full_text": False,
                        "do_sample": True,
                        "top_p": kwargs.pop("top_p", 0.9),
                        "top_k": kwargs.pop("top_k", 50),
                        **kwargs
                    },
                    "options": {
                        "wait_for_model": True,
                        "use_cache": False
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/{self.model_name}",
                    headers=self.headers,
                    json=payload
                )
                
                response.raise_for_status()
                result = response.json()
                
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                
                # Handle different response formats
                content = ""
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        content = result[0]["generated_text"]
                    elif "text" in result[0]:
                        content = result[0]["text"]
                    else:
                        content = str(result[0])
                elif isinstance(result, dict):
                    content = result.get("generated_text", result.get("text", str(result)))
                else:
                    content = str(result)
                
                # Estimate token count for HuggingFace
                estimated_tokens = len(prompt.split()) + len(content.split()) if content else len(prompt.split())
                
                response_obj = ModelResponse(
                    content=content,
                    model=self.model_name,
                    tokens_used=estimated_tokens,
                    cost=0.0,  # Typically free for inference API
                    latency_ms=latency_ms,
                    metadata={
                        "raw_response": result,
                        "estimated_tokens": True
                    },
                    api_status=self.circuit_breaker.get_status() if self.circuit_breaker else APIStatus.HEALTHY
                )
                
                self._record_request(prompt, response_obj)
                return response_obj
                
        except Exception as e:
            end_time = time.time()
            latency_ms = int((end_time - start_time) * 1000)
            
            error_response = ModelResponse(
                content=f"Error: {str(e)}",
                model=self.model_name,
                tokens_used=0,
                cost=0.0,
                latency_ms=latency_ms,
                error=str(e),
                api_status=APIStatus.FAILING
            )
            
            self._record_request(prompt, error_response)
            logger.error(f"Hugging Face API error for {self.model_name}: {e}")
            raise
    
    async def close(self):
        """Close the client connection."""
        pass  # httpx client is closed automatically


class ModelAPIManager:
    """Manager for multiple model API clients."""
    
    def __init__(self):
        """Initialize the API manager."""
        self.clients: Dict[str, ModelAPIClient] = {}
    
    def add_client(self, name: str, client: ModelAPIClient):
        """Add a model client."""
        self.clients[name] = client
    
    def get_client(self, name: str) -> Optional[ModelAPIClient]:
        """Get a model client by name."""
        return self.clients.get(name)
    
    async def generate_response(
        self, 
        client_name: str, 
        prompt: str, 
        **kwargs
    ) -> ModelResponse:
        """Generate response using specified client."""
        client = self.get_client(client_name)
        if not client:
            raise ValueError(f"Client '{client_name}' not found")
        
        return await client.generate_response(prompt, **kwargs)
    
    async def batch_generate(
        self,
        requests: List[Dict[str, Any]],
        max_concurrent: int = 5
    ) -> List[ModelResponse]:
        """Generate responses for multiple requests concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_request(request):
            async with semaphore:
                client_name = request["client"]
                prompt = request["prompt"]
                kwargs = request.get("kwargs", {})
                
                try:
                    return await self.generate_response(client_name, prompt, **kwargs)
                except Exception as e:
                    logger.error(f"Batch generation error for {client_name}: {e}")
                    return ModelResponse(
                        content=f"Error: {str(e)}",
                        model=client_name,
                        tokens_used=0,
                        cost=0.0,
                        latency_ms=0,
                        error=str(e),
                        api_status=APIStatus.FAILING
                    )
        
        tasks = [process_request(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all clients."""
        health_status = {}
        
        for name, client in self.clients.items():
            stats = client.get_usage_stats()
            health_status[name] = {
                "status": stats.get("circuit_breaker_status", "unknown"),
                "error_rate": stats.get("error_rate", 0.0),
                "average_latency_ms": stats.get("average_latency_ms", 0),
                "total_requests": stats.get("total_requests", 0),
                "total_cost": stats.get("total_cost", 0.0)
            }
        
        return health_status
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform health check on all clients."""
        health_results = {}
        test_prompt = "Hello, this is a health check."
        
        for name, client in self.clients.items():
            try:
                # Skip health check if circuit breaker is open
                if client.circuit_breaker and not client.circuit_breaker.can_attempt_call():
                    health_results[name] = False
                    continue
                
                # Short timeout for health check
                response = await client.generate_response(
                    test_prompt, 
                    max_tokens=10, 
                    timeout=10.0
                )
                health_results[name] = response.error is None
                
            except Exception as e:
                logger.warning(f"Health check failed for {name}: {e}")
                health_results[name] = False
        
        return health_results
    
    async def close_all(self):
        """Close all client connections."""
        await asyncio.gather(*[client.close() for client in self.clients.values()], return_exceptions=True)
    
    def list_clients(self) -> List[str]:
        """List available client names."""
        return list(self.clients.keys())


# Enhanced factory functions for easy client creation
def create_openai_client(
    model_name: str = "gpt-4",
    api_key: Optional[str] = None,
    enable_circuit_breaker: bool = True
) -> OpenAIClient:
    """Create OpenAI client with API key from environment."""
    return OpenAIClient(
        api_key=api_key,
        model_name=model_name, 
        enable_circuit_breaker=enable_circuit_breaker
    )


def create_anthropic_client(
    model_name: str = "claude-3-sonnet-20240229",
    api_key: Optional[str] = None,
    enable_circuit_breaker: bool = True
) -> AnthropicClient:
    """Create Anthropic client with API key from environment."""
    return AnthropicClient(
        api_key=api_key,
        model_name=model_name,
        enable_circuit_breaker=enable_circuit_breaker
    )


def create_huggingface_client(
    model_name: str = "meta-llama/Llama-2-7b-chat-hf",
    api_key: Optional[str] = None,
    enable_circuit_breaker: bool = True
) -> HuggingFaceClient:
    """Create Hugging Face client with API key from environment."""
    return HuggingFaceClient(
        api_key=api_key,
        model_name=model_name,
        enable_circuit_breaker=enable_circuit_breaker
    )


def create_default_manager(enable_circuit_breakers: bool = True) -> ModelAPIManager:
    """Create manager with default clients."""
    manager = ModelAPIManager()
    
    # Add OpenAI clients if API key available
    if os.getenv("OPENAI_API_KEY"):
        manager.add_client("gpt-4", create_openai_client(
            "gpt-4", enable_circuit_breaker=enable_circuit_breakers
        ))
        manager.add_client("gpt-4-turbo", create_openai_client(
            "gpt-4-turbo", enable_circuit_breaker=enable_circuit_breakers
        ))
        manager.add_client("gpt-3.5-turbo", create_openai_client(
            "gpt-3.5-turbo", enable_circuit_breaker=enable_circuit_breakers
        ))
    
    # Add Anthropic clients if API key available
    if os.getenv("ANTHROPIC_API_KEY"):
        manager.add_client("claude-3-sonnet", create_anthropic_client(
            "claude-3-sonnet-20240229", enable_circuit_breaker=enable_circuit_breakers
        ))
        manager.add_client("claude-3-opus", create_anthropic_client(
            "claude-3-opus-20240229", enable_circuit_breaker=enable_circuit_breakers
        ))
        manager.add_client("claude-3-haiku", create_anthropic_client(
            "claude-3-haiku-20240307", enable_circuit_breaker=enable_circuit_breakers
        ))
    
    # Add Hugging Face clients if token available
    if os.getenv("HUGGINGFACE_API_TOKEN"):
        manager.add_client("llama-2-7b", create_huggingface_client(
            "meta-llama/Llama-2-7b-chat-hf", enable_circuit_breaker=enable_circuit_breakers
        ))
        manager.add_client("llama-2-13b", create_huggingface_client(
            "meta-llama/Llama-2-13b-chat-hf", enable_circuit_breaker=enable_circuit_breakers
        ))
        manager.add_client("mistral-7b", create_huggingface_client(
            "mistralai/Mistral-7B-Instruct-v0.1", enable_circuit_breaker=enable_circuit_breakers
        ))
    
    return manager


class ModelEvaluator:
    """High-level evaluator for causal reasoning tasks using multiple models."""
    
    def __init__(self, manager: Optional[ModelAPIManager] = None):
        """Initialize the evaluator."""
        self.manager = manager or create_default_manager()
        self.evaluation_history: List[Dict[str, Any]] = []
    
    async def evaluate_model(
        self,
        model_name: str,
        task_prompt: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None
    ) -> ModelResponse:
        """Evaluate a specific model on a causal reasoning task."""
        response = await self.manager.generate_response(
            model_name, 
            task_prompt, 
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Record evaluation
        self.evaluation_history.append({
            "timestamp": time.time(),
            "model": model_name,
            "prompt_length": len(task_prompt),
            "response_length": len(response.content),
            "tokens_used": response.tokens_used,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "success": response.error is None
        })
        
        return response
    
    async def compare_models(
        self,
        model_names: List[str],
        task_prompt: str,
        temperature: float = 0.7
    ) -> Dict[str, ModelResponse]:
        """Compare multiple models on the same task."""
        tasks = []
        for model_name in model_names:
            task = self.evaluate_model(model_name, task_prompt, temperature)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        results = {}
        for model_name, response in zip(model_names, responses):
            if isinstance(response, Exception):
                results[model_name] = ModelResponse(
                    content=f"Error: {str(response)}",
                    model=model_name,
                    error=str(response),
                    api_status=APIStatus.FAILING
                )
            else:
                results[model_name] = response
        
        return results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation history."""
        if not self.evaluation_history:
            return {}
        
        by_model = {}
        for eval_record in self.evaluation_history:
            model = eval_record["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(eval_record)
        
        summary = {}
        for model, records in by_model.items():
            total_cost = sum(r["cost"] or 0 for r in records)
            total_tokens = sum(r["tokens_used"] or 0 for r in records)
            avg_latency = sum(r["latency_ms"] or 0 for r in records) / len(records)
            success_rate = sum(1 for r in records if r["success"]) / len(records)
            
            summary[model] = {
                "total_evaluations": len(records),
                "total_cost": total_cost,
                "total_tokens": total_tokens,
                "average_latency_ms": avg_latency,
                "success_rate": success_rate
            }
        
        return summary