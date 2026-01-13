"""
MIPROv2 - Standalone Implementation (No DSPy Dependency)

A from-scratch implementation of the MIPROv2 prompt optimizer.
Based on the paper: "Optimizing Instructions and Demonstrations for
Multi-Stage Language Model Programs" (arXiv:2406.11695)

Features:
- Official SDK support: google-genai (Gemini), openai (OpenAI)
- Multi-provider via LiteLLM (100+ providers)
- Complete 3-step MIPROv2 algorithm: Bootstrap → Propose → Optimize
- Following paper and DSPy library design

Usage:
    ```python
    from mipro_v2_scratch import LLM, MIPROv2, Example, Module, Predictor, Signature

    # Create LLM client
    lm = LLM(model="gemini-2.0-flash")  # or "gpt-4o-mini"

    # Define metric
    def accuracy(example, prediction):
        return example.answer.lower() in prediction.answer.lower()

    # Optimize
    optimizer = MIPROv2(metric=accuracy, llm=lm, auto="light")
    best_program, score = optimizer.compile(program=my_module, trainset=trainset)
    ```
"""

# Core LLM clients
from .llm_client import (
    LLM,           # Unified auto-detect client
    GeminiLM,      # Google Gemini (google-genai SDK)
    OpenAILM,      # OpenAI/Azure (openai SDK)
    LiteLLM,       # Multi-provider (litellm)
    BaseLLM,       # Base class for custom clients
    LLMResponse,   # Response dataclass
    Message,       # Message dataclass
    # Backwards compatibility aliases
    LLMClient,
    GeminiClient,
    OpenAIClient,
)

# Module classes
from .module import (
    Module,
    Predictor,
    Signature,
    Field,
    SimplePredictor,
)

# Data containers
from .example import Example, Prediction

# Main optimizer
from .mipro_optimizer import MIPROv2, setup_logging, OptimizationLogger

__all__ = [
    # LLM Clients
    "LLM",
    "GeminiLM",
    "OpenAILM",
    "LiteLLM",
    "BaseLLM",
    "LLMResponse",
    "Message",
    # Backwards compatibility
    "LLMClient",
    "GeminiClient",
    "OpenAIClient",
    # Modules
    "Module",
    "Predictor",
    "Signature",
    "Field",
    "SimplePredictor",
    # Data
    "Example",
    "Prediction",
    # Optimizer
    "MIPROv2",
    # Logging utilities
    "setup_logging",
    "OptimizationLogger",
]

__version__ = "2.0.0"
