"""
LLM Client - Standalone wrapper for Language Model APIs.

Supports OpenAI API and compatible endpoints (Azure, local models, etc.)
"""

import os
import time
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM call."""
    content: str
    usage: Dict[str, int]
    model: str
    finish_reason: str


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from the LLM."""
        pass

    @abstractmethod
    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from messages format."""
        pass


class LLMClient(BaseLLMClient):
    """
    OpenAI-compatible LLM client.

    Can be used with:
    - OpenAI API
    - Azure OpenAI
    - Local models with OpenAI-compatible API (vLLM, Ollama, etc.)

    Example:
        ```python
        # OpenAI
        client = LLMClient(
            api_key="sk-...",
            model="gpt-4o-mini"
        )

        # Local model
        client = LLMClient(
            base_url="http://localhost:8000/v1",
            api_key="dummy",
            model="llama-3-8b"
        )

        response = client.generate("What is 2+2?")
        print(response.content)
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL for compatible services
            organization: OpenAI organization ID
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://api.openai.com/v1"
        self.organization = organization
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError(
                "API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        # Track usage
        self.total_tokens = 0
        self.total_calls = 0
        self.history: List[Dict] = []

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict:
        """Make HTTP request to the API."""
        import urllib.request
        import urllib.error

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization

        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        request_data = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=request_data,
            headers=headers,
            method="POST"
        )

        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                error_body = e.read().decode("utf-8") if e.fp else ""
                logger.warning(f"HTTP Error {e.code}: {error_body}")

                if e.code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.info(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif e.code >= 500:  # Server error
                    wait_time = self.retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    raise
            except Exception as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError(f"Failed after {self.max_retries} attempts")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from a simple prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with generated content
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        return self.generate_with_messages(
            messages, temperature, max_tokens, **kwargs
        )

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from messages format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with generated content
        """
        response = self._make_request(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        choice = response["choices"][0]
        usage = response.get("usage", {})

        result = LLMResponse(
            content=choice["message"]["content"],
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            model=response.get("model", self.model),
            finish_reason=choice.get("finish_reason", "unknown")
        )

        # Track usage
        self.total_tokens += result.usage.get("total_tokens", 0)
        self.total_calls += 1
        self.history.append({
            "messages": messages,
            "response": result.content,
            "usage": result.usage,
        })

        return result

    def copy(self, **kwargs) -> "LLMClient":
        """Create a copy with optionally modified parameters."""
        return LLMClient(
            model=kwargs.get("model", self.model),
            api_key=kwargs.get("api_key", self.api_key),
            base_url=kwargs.get("base_url", self.base_url),
            organization=kwargs.get("organization", self.organization),
            timeout=kwargs.get("timeout", self.timeout),
            max_retries=kwargs.get("max_retries", self.max_retries),
            retry_delay=kwargs.get("retry_delay", self.retry_delay),
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "model": self.model,
        }
