"""
Model API clients for external language model services.
"""

import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

import httpx
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from a model API."""
    
    content: str
    model: str
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    latency_ms: Optional[int] = None
    metadata: Dict[str, Any] = None


class ModelAPIClient(ABC):
    """Abstract base class for model API clients."""
    
    def __init__(self, api_key: str, model_name: str):
        """Initialize the client."""
        self.api_key = api_key
        self.model_name = model_name
        self.client = None
    
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
    
    @abstractmethod
    async def close(self):
        """Close the client connection."""
        pass


class OpenAIClient(ModelAPIClient):
    """OpenAI API client for GPT models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4"):
        """Initialize OpenAI client."""
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required")
        
        super().__init__(api_key, model_name)
        self.client = AsyncOpenAI(api_key=api_key)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002}
        }
    
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using OpenAI API."""
        import time
        
        start_time = time.time()
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
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
            
            return ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
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
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-sonnet-20240229"):
        """Initialize Anthropic client."""
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key is required")
        
        super().__init__(api_key, model_name)
        self.client = AsyncAnthropic(api_key=api_key)
        self.pricing = {
            "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125}
        }
    
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Anthropic API."""
        import time
        
        start_time = time.time()
        
        try:
            response = await self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tokens or 1000,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                **kwargs
            )
            
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
            
            return ModelResponse(
                content=content,
                model=self.model_name,
                tokens_used=tokens_used,
                cost=cost,
                latency_ms=latency_ms,
                metadata={
                    "stop_reason": response.stop_reason,
                    "input_tokens": usage.input_tokens if usage else None,
                    "output_tokens": usage.output_tokens if usage else None
                }
            )
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
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
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """Initialize Hugging Face client."""
        api_key = api_key or os.getenv("HUGGINGFACE_API_TOKEN")
        if not api_key:
            raise ValueError("Hugging Face API token is required")
        
        super().__init__(api_key, model_name)
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    async def generate_response(
        self, 
        prompt: str, 
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ModelResponse:
        """Generate response using Hugging Face Inference API."""
        import time
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient() as client:
                payload = {
                    "inputs": prompt,
                    "parameters": {
                        "temperature": temperature,
                        "max_new_tokens": max_tokens or 100,
                        "return_full_text": False,
                        **kwargs
                    }
                }
                
                response = await client.post(
                    f"{self.base_url}/{self.model_name}",
                    headers=self.headers,
                    json=payload,
                    timeout=60.0
                )
                
                response.raise_for_status()
                result = response.json()
                
                end_time = time.time()
                latency_ms = int((end_time - start_time) * 1000)
                
                # Extract response content
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "")
                else:
                    content = str(result)
                
                return ModelResponse(
                    content=content,
                    model=self.model_name,
                    tokens_used=None,  # HF API doesn't provide token counts
                    cost=0.0,  # Typically free for inference API
                    latency_ms=latency_ms,
                    metadata={"raw_response": result}
                )
                
        except Exception as e:
            logger.error(f"Hugging Face API error: {e}")
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
                        metadata={"error": True}
                    )
        
        tasks = [process_request(req) for req in requests]
        return await asyncio.gather(*tasks)
    
    async def close_all(self):
        """Close all client connections."""
        for client in self.clients.values():
            await client.close()
    
    def list_clients(self) -> List[str]:
        """List available client names."""
        return list(self.clients.keys())


# Factory functions for easy client creation
def create_openai_client(model_name: str = "gpt-4") -> OpenAIClient:
    """Create OpenAI client with API key from environment."""
    return OpenAIClient(model_name=model_name)


def create_anthropic_client(model_name: str = "claude-3-sonnet-20240229") -> AnthropicClient:
    """Create Anthropic client with API key from environment."""
    return AnthropicClient(model_name=model_name)


def create_huggingface_client(model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> HuggingFaceClient:
    """Create Hugging Face client with API key from environment."""
    return HuggingFaceClient(model_name=model_name)


def create_default_manager() -> ModelAPIManager:
    """Create manager with default clients."""
    manager = ModelAPIManager()
    
    # Add OpenAI clients if API key available
    if os.getenv("OPENAI_API_KEY"):
        manager.add_client("gpt-4", create_openai_client("gpt-4"))
        manager.add_client("gpt-3.5-turbo", create_openai_client("gpt-3.5-turbo"))
    
    # Add Anthropic clients if API key available
    if os.getenv("ANTHROPIC_API_KEY"):
        manager.add_client("claude-3-sonnet", create_anthropic_client("claude-3-sonnet-20240229"))
        manager.add_client("claude-3-opus", create_anthropic_client("claude-3-opus-20240229"))
    
    # Add Hugging Face clients if token available
    if os.getenv("HUGGINGFACE_API_TOKEN"):
        manager.add_client("llama-2-7b", create_huggingface_client("meta-llama/Llama-2-7b-chat-hf"))
    
    return manager