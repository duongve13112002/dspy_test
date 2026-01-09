"""
LLM Client - Standalone wrapper for Language Model APIs.

Supports:
- OpenAI API (GPT-4, GPT-4o, GPT-3.5, etc.)
- Google Gemini API (Gemini Pro, Gemini Flash, etc.)
- Azure OpenAI
- Local models with OpenAI-compatible API (vLLM, Ollama, etc.)
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


# ============================================================================
# GOOGLE GEMINI CLIENT
# ============================================================================

class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client.

    Supports Gemini models:
    - gemini-1.5-pro
    - gemini-1.5-flash
    - gemini-1.0-pro
    - gemini-2.0-flash-exp

    Example:
        ```python
        # Using API key
        client = GeminiClient(
            api_key="AIza...",
            model="gemini-1.5-flash"
        )

        response = client.generate("What is 2+2?")
        print(response.content)

        # With system instruction
        response = client.generate(
            "Solve this math problem",
            system_prompt="You are a math tutor."
        )
        ```
    """

    # Gemini API base URL
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize Gemini client.

        Args:
            model: Model name (e.g., "gemini-1.5-flash", "gemini-1.5-pro")
            api_key: Google AI API key (defaults to GOOGLE_API_KEY env var)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise ValueError(
                "API key required. Set GOOGLE_API_KEY env var or pass api_key."
            )

        # Track usage
        self.total_tokens = 0
        self.total_calls = 0
        self.history: List[Dict] = []

    def _make_request(
        self,
        contents: List[Dict],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> Dict:
        """Make HTTP request to Gemini API."""
        import urllib.request
        import urllib.error

        url = f"{self.BASE_URL}/models/{self.model}:generateContent?key={self.api_key}"

        # Build request body
        data = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            }
        }

        # Add system instruction if provided
        if system_instruction:
            data["systemInstruction"] = {
                "parts": [{"text": system_instruction}]
            }

        headers = {
            "Content-Type": "application/json",
        }

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
            system_prompt: Optional system instruction
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with generated content
        """
        contents = [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ]

        response = self._make_request(
            contents=contents,
            system_instruction=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return self._parse_response(response, [{"role": "user", "content": prompt}])

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text from messages format (OpenAI-style).

        Converts OpenAI message format to Gemini format.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional API parameters

        Returns:
            LLMResponse with generated content
        """
        # Extract system prompt if present
        system_prompt = None
        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "assistant":
                contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })
            else:  # user
                contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })

        response = self._make_request(
            contents=contents,
            system_instruction=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        return self._parse_response(response, messages)

    def _parse_response(self, response: Dict, messages: List) -> LLMResponse:
        """Parse Gemini API response."""
        # Extract content
        candidates = response.get("candidates", [])
        if not candidates:
            raise ValueError("No response candidates from Gemini")

        content = ""
        finish_reason = "unknown"

        candidate = candidates[0]
        if "content" in candidate:
            parts = candidate["content"].get("parts", [])
            content = "".join(part.get("text", "") for part in parts)

        finish_reason = candidate.get("finishReason", "unknown")

        # Extract usage
        usage_metadata = response.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0),
        }

        result = LLMResponse(
            content=content,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason
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

    def copy(self, **kwargs) -> "GeminiClient":
        """Create a copy with optionally modified parameters."""
        return GeminiClient(
            model=kwargs.get("model", self.model),
            api_key=kwargs.get("api_key", self.api_key),
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


# ============================================================================
# OPENAI CLIENT
# ============================================================================

class OpenAIClient(BaseLLMClient):
    """
    OpenAI-compatible LLM client.

    Can be used with:
    - OpenAI API (GPT-4, GPT-4o, GPT-3.5, etc.)
    - Azure OpenAI
    - Local models with OpenAI-compatible API (vLLM, Ollama, etc.)

    Example:
        ```python
        # OpenAI
        client = OpenAIClient(
            api_key="sk-...",
            model="gpt-4o-mini"
        )

        # Local model (Ollama)
        client = OpenAIClient(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
            model="llama3"
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

    def copy(self, **kwargs) -> "OpenAIClient":
        """Create a copy with optionally modified parameters."""
        return OpenAIClient(
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


# ============================================================================
# UNIFIED LLM CLIENT (Auto-detect provider)
# ============================================================================

class LLMClient(BaseLLMClient):
    """
    Unified LLM client that auto-detects provider based on model name.

    Supports:
    - OpenAI: gpt-4, gpt-4o, gpt-4o-mini, gpt-3.5-turbo, etc.
    - Google Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash-exp, etc.
    - Local/Custom: Any model with custom base_url

    Example:
        ```python
        # OpenAI (auto-detected)
        client = LLMClient(
            api_key="sk-...",
            model="gpt-4o-mini"
        )

        # Gemini (auto-detected)
        client = LLMClient(
            api_key="AIza...",
            model="gemini-1.5-flash"
        )

        # Or specify provider explicitly
        client = LLMClient(
            api_key="...",
            model="my-model",
            provider="openai"  # or "gemini"
        )

        response = client.generate("What is 2+2?")
        print(response.content)
        ```
    """

    GEMINI_MODELS = [
        "gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro",
        "gemini-2.0-flash-exp", "gemini-pro", "gemini-flash"
    ]

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,  # "openai", "gemini", or None (auto)
        **kwargs
    ):
        """
        Initialize unified LLM client.

        Args:
            model: Model name
            api_key: API key (auto-detects from env if not provided)
            base_url: Custom base URL (forces OpenAI-compatible mode)
            provider: Force provider ("openai" or "gemini")
            **kwargs: Additional provider-specific options
        """
        self.model = model

        # Determine provider
        if provider:
            self._provider = provider.lower()
        elif base_url:
            self._provider = "openai"  # Custom URL = OpenAI-compatible
        elif any(model.startswith(gm) for gm in self.GEMINI_MODELS):
            self._provider = "gemini"
        elif model.startswith("gemini"):
            self._provider = "gemini"
        else:
            self._provider = "openai"

        # Create underlying client
        if self._provider == "gemini":
            api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            self._client = GeminiClient(model=model, api_key=api_key, **kwargs)
        else:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            self._client = OpenAIClient(
                model=model, api_key=api_key, base_url=base_url, **kwargs
            )

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from a simple prompt."""
        return self._client.generate(
            prompt, system_prompt, temperature, max_tokens, **kwargs
        )

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> LLMResponse:
        """Generate text from messages format."""
        return self._client.generate_with_messages(
            messages, temperature, max_tokens, **kwargs
        )

    def copy(self, **kwargs) -> "LLMClient":
        """Create a copy with optionally modified parameters."""
        new_client = LLMClient.__new__(LLMClient)
        new_client.model = kwargs.get("model", self.model)
        new_client._provider = self._provider
        new_client._client = self._client.copy(**kwargs)
        return new_client

    def get_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        stats = self._client.get_stats()
        stats["provider"] = self._provider
        return stats

    @property
    def total_tokens(self) -> int:
        return self._client.total_tokens

    @property
    def total_calls(self) -> int:
        return self._client.total_calls

    @property
    def history(self) -> List[Dict]:
        return self._client.history
