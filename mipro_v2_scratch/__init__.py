"""
MIPROv2 - Standalone Implementation (No DSPy Dependency)

A from-scratch implementation of the MIPROv2 prompt optimizer.
Based on the paper: "Optimizing Instructions and Demonstrations for
Multi-Stage Language Model Programs" (arXiv:2406.11695)

This implementation uses only:
- OpenAI API (or compatible) for LLM calls
- Optuna for Bayesian optimization
- NumPy for numerical operations
- Standard Python libraries

Author: Standalone implementation
"""

from .mipro_optimizer import MIPROv2
from .llm_client import LLMClient, GeminiClient, OpenAIClient
from .module import Module, Predictor, Signature, SimplePredictor
from .example import Example, Prediction

__all__ = [
    "MIPROv2",
    "LLMClient",
    "GeminiClient",
    "OpenAIClient",
    "Module",
    "Predictor",
    "Signature",
    "SimplePredictor",
    "Example",
    "Prediction",
]

__version__ = "2.0.0"
