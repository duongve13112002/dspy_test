"""
LLM Client - Unified wrapper for Language Model APIs using official SDKs.

Supports:
- Google Gemini via google-genai SDK (recommended, new unified SDK)
- OpenAI via openai SDK
- LiteLLM for multi-provider support (optional)

This module follows DSPy's approach but without DSPy dependency.
"""

import os
import time
import logging
import uuid
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class LLMResponse:
    """
    Response from an LLM call.

    Attributes:
        text: The generated text content
        usage: Token usage statistics
        model: Model name used
        finish_reason: Why generation stopped
        raw_response: Original response object from SDK
    """
    text: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""
    finish_reason: str = "stop"
    raw_response: Any = None

    @property
    def content(self) -> str:
        """Alias for text (backwards compatibility)."""
        return self.text


@dataclass
class Message:
    """A chat message."""
    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


# ============================================================================
# BASE LLM CLIENT
# ============================================================================

class BaseLLM(ABC):
    """
    Abstract base class for LLM clients.

    Follows DSPy's BaseLM interface pattern.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self.history: List[Dict] = []
        self._call_count = 0
        self._total_tokens = 0

    @abstractmethod
    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text from prompt or messages."""
        pass

    def __call__(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Shorthand for generate()."""
        return self.generate(prompt=prompt, messages=messages, **kwargs)

    def copy(self, **kwargs) -> "BaseLLM":
        """Create a copy with optionally modified parameters."""
        new_instance = deepcopy(self)
        new_instance.history = []

        for key, value in kwargs.items():
            if hasattr(new_instance, key):
                setattr(new_instance, key, value)
            else:
                new_instance.kwargs[key] = value

        return new_instance

    def _update_history(
        self,
        prompt: Optional[str],
        messages: Optional[List],
        response: LLMResponse
    ):
        """Track call in history."""
        self.history.append({
            "prompt": prompt,
            "messages": messages,
            "response": response.text,
            "usage": response.usage,
            "model": response.model,
            "timestamp": time.time(),
        })

        self._call_count += 1
        self._total_tokens += response.usage.get("total_tokens", 0)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model,
            "call_count": self._call_count,
            "total_tokens": self._total_tokens,
        }


# ============================================================================
# GOOGLE GEMINI CLIENT (Using new google-genai SDK)
# ============================================================================

class GeminiLM(BaseLLM):
    """
    Google Gemini client using the new google-genai SDK.

    The new SDK (google-genai) is the recommended way to use Gemini.
    It replaces the deprecated google-generativeai package.

    Installation:
        pip install google-genai

    Example:
        ```python
        # Using API key (Gemini Developer API)
        lm = GeminiLM(
            model="gemini-2.0-flash",
            api_key="your-api-key"  # or set GEMINI_API_KEY env var
        )

        # Using Vertex AI
        lm = GeminiLM(
            model="gemini-2.0-flash",
            vertexai=True,
            project="your-project",
            location="us-central1"
        )

        response = lm.generate("What is 2+2?")
        print(response.text)
        ```
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
        vertexai: bool = False,
        project: Optional[str] = None,
        location: str = "us-central1",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize Gemini client.

        Args:
            model: Model name (e.g., "gemini-2.0-flash", "gemini-1.5-pro")
            api_key: API key (defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var)
            vertexai: Use Vertex AI instead of Gemini Developer API
            project: Google Cloud project (required for Vertex AI)
            location: Google Cloud location (for Vertex AI)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional generation config options
        """
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.vertexai = vertexai
        self.project = project
        self.location = location
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the google-genai client."""
        try:
            from google import genai

            if self.vertexai:
                self._client = genai.Client(
                    vertexai=True,
                    project=self.project,
                    location=self.location,
                )
            else:
                if not self.api_key:
                    raise ValueError(
                        "API key required. Set GEMINI_API_KEY env var or pass api_key."
                    )
                self._client = genai.Client(api_key=self.api_key)

        except ImportError:
            raise ImportError(
                "google-genai package not found. Install with: pip install google-genai"
            )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_instruction: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using Gemini.

        Args:
            prompt: Simple text prompt
            messages: Chat messages (OpenAI format)
            temperature: Override default temperature
            max_tokens: Override default max tokens
            system_instruction: System instruction
            **kwargs: Additional config options

        Returns:
            LLMResponse with generated text
        """
        from google.genai import types

        # Build contents
        if messages:
            contents, system_instruction = self._convert_messages(messages, system_instruction)
        elif prompt:
            contents = prompt
        else:
            raise ValueError("Either prompt or messages required")

        # Build config
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        config = types.GenerateContentConfig(
            temperature=temp,
            max_output_tokens=max_tok,
            **{k: v for k, v in kwargs.items() if v is not None}
        )

        if system_instruction:
            config.system_instruction = system_instruction

        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=config,
                )

                # Parse response
                result = self._parse_response(response)
                self._update_history(prompt, messages, result)
                return result

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Gemini API error (attempt {attempt + 1}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Gemini API failed after {self.max_retries} attempts: {e}")
                    raise

        raise last_error

    def _convert_messages(
        self,
        messages: List[Dict[str, str]],
        system_instruction: Optional[str] = None
    ) -> tuple:
        """Convert OpenAI-style messages to Gemini format."""
        from google.genai import types

        contents = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=content)]
                ))
            else:  # user
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part(text=content)]
                ))

        return contents, system_instruction

    def _parse_response(self, response) -> LLMResponse:
        """Parse Gemini response to LLMResponse."""
        # Extract text
        text = ""
        finish_reason = "stop"

        if hasattr(response, 'text'):
            text = response.text
        elif hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts
                text = "".join(p.text for p in parts if hasattr(p, 'text'))
            if hasattr(candidate, 'finish_reason'):
                finish_reason = str(candidate.finish_reason)

        # Extract usage
        usage = {}
        if hasattr(response, 'usage_metadata'):
            um = response.usage_metadata
            usage = {
                "prompt_tokens": getattr(um, 'prompt_token_count', 0),
                "completion_tokens": getattr(um, 'candidates_token_count', 0),
                "total_tokens": getattr(um, 'total_token_count', 0),
            }

        return LLMResponse(
            text=text,
            usage=usage,
            model=self.model,
            finish_reason=finish_reason,
            raw_response=response,
        )

    def copy(self, **kwargs) -> "GeminiLM":
        """Create a copy with optionally modified parameters."""
        new_instance = GeminiLM(
            model=kwargs.get("model", self.model),
            api_key=kwargs.get("api_key", self.api_key),
            vertexai=kwargs.get("vertexai", self.vertexai),
            project=kwargs.get("project", self.project),
            location=kwargs.get("location", self.location),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=kwargs.get("timeout", self.timeout),
            max_retries=kwargs.get("max_retries", self.max_retries),
            **{**self.kwargs, **{k: v for k, v in kwargs.items()
                                  if k not in ["model", "api_key", "vertexai", "project",
                                              "location", "temperature", "max_tokens",
                                              "timeout", "max_retries"]}}
        )
        return new_instance


# ============================================================================
# OPENAI CLIENT (Using official openai SDK)
# ============================================================================

class OpenAILM(BaseLLM):
    """
    OpenAI client using the official openai SDK.

    Also compatible with:
    - Azure OpenAI
    - Local models with OpenAI-compatible API (vLLM, Ollama, etc.)

    Installation:
        pip install openai

    Example:
        ```python
        # OpenAI
        lm = OpenAILM(
            model="gpt-4o-mini",
            api_key="sk-..."  # or set OPENAI_API_KEY env var
        )

        # Azure OpenAI
        lm = OpenAILM(
            model="gpt-4o-mini",
            api_key="your-azure-key",
            base_url="https://your-resource.openai.azure.com/",
            api_version="2024-02-15-preview"
        )

        # Local model (Ollama)
        lm = OpenAILM(
            model="llama3",
            base_url="http://localhost:11434/v1",
            api_key="ollama"
        )

        response = lm.generate("What is 2+2?")
        print(response.text)
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        api_version: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        timeout: int = 120,
        max_retries: int = 3,
        **kwargs
    ):
        """
        Initialize OpenAI client.

        Args:
            model: Model name (e.g., "gpt-4o-mini", "gpt-4o")
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Custom API base URL
            organization: OpenAI organization ID
            api_version: API version (for Azure)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum output tokens
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            **kwargs: Additional parameters
        """
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        self.organization = organization
        self.api_version = api_version
        self.timeout = timeout
        self.max_retries = max_retries

        # Initialize client
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI, AzureOpenAI

            if self.api_version:
                # Azure OpenAI
                self._client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=self.api_version,
                    azure_endpoint=self.base_url,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                )
            else:
                # Standard OpenAI or compatible
                client_kwargs = {
                    "timeout": self.timeout,
                    "max_retries": self.max_retries,
                }
                if self.api_key:
                    client_kwargs["api_key"] = self.api_key
                if self.base_url:
                    client_kwargs["base_url"] = self.base_url
                if self.organization:
                    client_kwargs["organization"] = self.organization

                self._client = OpenAI(**client_kwargs)

        except ImportError:
            raise ImportError(
                "openai package not found. Install with: pip install openai"
            )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate text using OpenAI.

        Args:
            prompt: Simple text prompt
            messages: Chat messages
            temperature: Override default temperature
            max_tokens: Override default max tokens
            **kwargs: Additional parameters (e.g., top_p, presence_penalty)

        Returns:
            LLMResponse with generated text
        """
        # Build messages
        if messages is None:
            messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("Either prompt or messages required")

        # Build request
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                **kwargs
            )

            result = self._parse_response(response)
            self._update_history(prompt, messages, result)
            return result

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI response to LLMResponse."""
        choice = response.choices[0]

        text = choice.message.content or ""
        finish_reason = choice.finish_reason or "stop"

        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            text=text,
            usage=usage,
            model=response.model,
            finish_reason=finish_reason,
            raw_response=response,
        )

    def copy(self, **kwargs) -> "OpenAILM":
        """Create a copy with optionally modified parameters."""
        new_instance = OpenAILM(
            model=kwargs.get("model", self.model),
            api_key=kwargs.get("api_key", self.api_key),
            base_url=kwargs.get("base_url", self.base_url),
            organization=kwargs.get("organization", self.organization),
            api_version=kwargs.get("api_version", self.api_version),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            timeout=kwargs.get("timeout", self.timeout),
            max_retries=kwargs.get("max_retries", self.max_retries),
            **{**self.kwargs, **{k: v for k, v in kwargs.items()
                                  if k not in ["model", "api_key", "base_url", "organization",
                                              "api_version", "temperature", "max_tokens",
                                              "timeout", "max_retries"]}}
        )
        return new_instance


# ============================================================================
# LITELLM CLIENT (Multi-provider support)
# ============================================================================

class LiteLLM(BaseLLM):
    """
    LiteLLM client for multi-provider support.

    LiteLLM provides a unified interface to 100+ LLM providers.
    This is the same approach used by DSPy's LM class.

    Installation:
        pip install litellm

    Example:
        ```python
        # OpenAI
        lm = LiteLLM(model="openai/gpt-4o-mini")

        # Anthropic
        lm = LiteLLM(model="anthropic/claude-3-sonnet")

        # Google
        lm = LiteLLM(model="gemini/gemini-1.5-flash")

        # Together AI
        lm = LiteLLM(model="together_ai/meta-llama/Llama-3-70b")

        response = lm.generate("What is 2+2?")
        ```
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        num_retries: int = 3,
        **kwargs
    ):
        """
        Initialize LiteLLM client.

        Args:
            model: Model in format "provider/model_name"
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            num_retries: Retry attempts
            **kwargs: Additional LiteLLM parameters
        """
        super().__init__(model, temperature, max_tokens, **kwargs)
        self.num_retries = num_retries

        # Verify litellm is installed
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError(
                "litellm package not found. Install with: pip install litellm"
            )

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using LiteLLM."""
        # Build messages
        if messages is None:
            messages = []
        if prompt:
            messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("Either prompt or messages required")

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        try:
            response = self._litellm.completion(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok,
                num_retries=self.num_retries,
                **{**self.kwargs, **kwargs}
            )

            result = self._parse_response(response)
            self._update_history(prompt, messages, result)
            return result

        except Exception as e:
            logger.error(f"LiteLLM error: {e}")
            raise

    def _parse_response(self, response) -> LLMResponse:
        """Parse LiteLLM response."""
        choice = response.choices[0]
        text = choice.message.content if hasattr(choice, 'message') else choice.get("text", "")

        usage = {}
        if hasattr(response, 'usage') and response.usage:
            usage = dict(response.usage)

        return LLMResponse(
            text=text,
            usage=usage,
            model=response.model if hasattr(response, 'model') else self.model,
            finish_reason=choice.finish_reason if hasattr(choice, 'finish_reason') else "stop",
            raw_response=response,
        )

    def copy(self, **kwargs) -> "LiteLLM":
        """Create a copy with modified parameters."""
        new_kwargs = {**self.kwargs}
        for k, v in kwargs.items():
            if k in ["model", "temperature", "max_tokens", "num_retries"]:
                continue
            new_kwargs[k] = v

        return LiteLLM(
            model=kwargs.get("model", self.model),
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            num_retries=kwargs.get("num_retries", self.num_retries),
            **new_kwargs
        )


# ============================================================================
# UNIFIED LLM CLIENT (Auto-detect provider)
# ============================================================================

class LLM(BaseLLM):
    """
    Unified LLM client that auto-detects and uses the appropriate backend.

    Automatically selects:
    - GeminiLM for gemini-* models
    - OpenAILM for gpt-* models
    - LiteLLM for provider/model format

    Example:
        ```python
        # Auto-detect Gemini
        lm = LLM(model="gemini-2.0-flash")

        # Auto-detect OpenAI
        lm = LLM(model="gpt-4o-mini")

        # Use LiteLLM format
        lm = LLM(model="anthropic/claude-3-sonnet")

        # Force specific backend
        lm = LLM(model="my-model", backend="openai", base_url="http://localhost:8000")
        ```
    """

    def __init__(
        self,
        model: str,
        backend: Optional[str] = None,  # "gemini", "openai", "litellm"
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs
    ):
        """
        Initialize unified LLM client.

        Args:
            model: Model name
            backend: Force specific backend ("gemini", "openai", "litellm")
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            **kwargs: Backend-specific parameters
        """
        super().__init__(model, temperature, max_tokens, **kwargs)

        # Determine backend
        if backend:
            self._backend = backend.lower()
        elif "/" in model:
            # provider/model format -> use litellm
            self._backend = "litellm"
        elif model.startswith("gemini"):
            self._backend = "gemini"
        elif model.startswith(("gpt", "o1", "o3", "text-")):
            self._backend = "openai"
        elif kwargs.get("base_url"):
            # Custom base URL -> assume OpenAI compatible
            self._backend = "openai"
        else:
            # Default to litellm for flexibility
            self._backend = "litellm"

        # Create underlying client
        self._create_client(kwargs)

    def _create_client(self, kwargs: Dict):
        """Create the appropriate backend client."""
        common_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        if self._backend == "gemini":
            gemini_kwargs = {k: kwargs.get(k) for k in
                           ["api_key", "vertexai", "project", "location",
                            "timeout", "max_retries"]
                           if k in kwargs}
            self._client = GeminiLM(**common_kwargs, **gemini_kwargs)

        elif self._backend == "openai":
            openai_kwargs = {k: kwargs.get(k) for k in
                           ["api_key", "base_url", "organization", "api_version",
                            "timeout", "max_retries"]
                           if k in kwargs}
            self._client = OpenAILM(**common_kwargs, **openai_kwargs)

        else:  # litellm
            litellm_kwargs = {k: v for k, v in kwargs.items()
                            if k not in ["api_key", "base_url"]}
            # Pass through api_key and base_url for litellm
            if "api_key" in kwargs:
                litellm_kwargs["api_key"] = kwargs["api_key"]
            if "base_url" in kwargs:
                litellm_kwargs["api_base"] = kwargs["base_url"]
            self._client = LiteLLM(**common_kwargs, **litellm_kwargs)

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate text using the appropriate backend."""
        result = self._client.generate(prompt=prompt, messages=messages, **kwargs)
        self._update_history(prompt, messages, result)
        return result

    def copy(self, **kwargs) -> "LLM":
        """Create a copy with modified parameters."""
        new_kwargs = {**self.kwargs, **kwargs}
        return LLM(
            model=kwargs.get("model", self.model),
            backend=self._backend,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            **new_kwargs
        )

    @property
    def history(self) -> List[Dict]:
        """Get combined history from underlying client."""
        return self._client.history

    @property
    def stats(self) -> Dict[str, Any]:
        """Get stats from underlying client."""
        stats = self._client.stats
        stats["backend"] = self._backend
        return stats


# ============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# ============================================================================

# Keep old names for backwards compatibility
LLMClient = LLM
GeminiClient = GeminiLM
OpenAIClient = OpenAILM
