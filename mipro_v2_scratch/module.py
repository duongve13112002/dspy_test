"""
Module - Base classes for LLM-based prediction modules.

Provides Signature (defines input/output schema) and Predictor (executes LLM calls).
"""

import re
import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field

from .example import Example, Prediction
from .llm_client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class Field:
    """
    Definition of an input or output field.

    Attributes:
        name: Field name
        description: Human-readable description
        prefix: Prefix for formatting in prompts
        field_type: "input" or "output"
    """
    name: str
    description: str = ""
    prefix: Optional[str] = None
    field_type: str = "input"  # "input" or "output"

    def __post_init__(self):
        if self.prefix is None:
            self.prefix = f"{self.name.replace('_', ' ').title()}:"


@dataclass
class Signature:
    """
    Defines the input/output schema for a predictor.

    A signature specifies:
    - What input fields the predictor expects
    - What output fields it should produce
    - Instructions for the LLM

    Usage:
        ```python
        # Simple string syntax
        sig = Signature.from_string("question -> answer")

        # With descriptions
        sig = Signature.from_string(
            "question: the question to answer -> answer: the response"
        )

        # Programmatic
        sig = Signature(
            input_fields=[Field("question", "The question to answer")],
            output_fields=[Field("answer", "The response")],
            instructions="Answer the question accurately."
        )
        ```
    """
    input_fields: List[Field] = field(default_factory=list)
    output_fields: List[Field] = field(default_factory=list)
    instructions: str = ""

    @classmethod
    def from_string(cls, sig_string: str, instructions: str = "") -> "Signature":
        """
        Parse signature from string format.

        Formats supported:
        - "input1, input2 -> output1, output2"
        - "input1: desc1, input2: desc2 -> output1: desc1"

        Args:
            sig_string: Signature string
            instructions: Optional instructions

        Returns:
            Parsed Signature
        """
        # Split by arrow
        if "->" not in sig_string:
            raise ValueError(f"Signature must contain '->': {sig_string}")

        input_part, output_part = sig_string.split("->", 1)

        def parse_fields(part: str, field_type: str) -> List[Field]:
            fields = []
            for item in part.split(","):
                item = item.strip()
                if not item:
                    continue

                if ":" in item:
                    name, desc = item.split(":", 1)
                    name = name.strip()
                    desc = desc.strip()
                else:
                    name = item
                    desc = ""

                fields.append(Field(
                    name=name,
                    description=desc,
                    field_type=field_type
                ))
            return fields

        return cls(
            input_fields=parse_fields(input_part, "input"),
            output_fields=parse_fields(output_part, "output"),
            instructions=instructions
        )

    def with_instructions(self, instructions: str) -> "Signature":
        """Create a copy with new instructions."""
        return Signature(
            input_fields=deepcopy(self.input_fields),
            output_fields=deepcopy(self.output_fields),
            instructions=instructions
        )

    @property
    def fields(self) -> Dict[str, Field]:
        """Get all fields as a dictionary."""
        result = {}
        for f in self.input_fields:
            result[f.name] = f
        for f in self.output_fields:
            result[f.name] = f
        return result

    def __repr__(self) -> str:
        inputs = ", ".join(f.name for f in self.input_fields)
        outputs = ", ".join(f.name for f in self.output_fields)
        return f"Signature({inputs} -> {outputs})"


class Predictor:
    """
    A module that uses an LLM to map inputs to outputs.

    The Predictor:
    1. Takes a Signature defining input/output fields
    2. Formats inputs into a prompt
    3. Calls the LLM
    4. Parses outputs from the response

    Usage:
        ```python
        # Create predictor
        predictor = Predictor(
            signature=Signature.from_string("question -> answer"),
            llm=LLMClient(model="gpt-4o-mini")
        )

        # Add few-shot demos
        predictor.demos = [
            Example(question="What is 2+2?", answer="4"),
        ]

        # Run prediction
        result = predictor(question="What is 3+3?")
        print(result.answer)  # "6"
        ```
    """

    def __init__(
        self,
        signature: Signature,
        llm: Optional[LLMClient] = None,
    ):
        """
        Initialize predictor.

        Args:
            signature: Input/output signature
            llm: LLM client (can be set later)
        """
        self.signature = signature
        self.llm = llm
        self.demos: List[Example] = []
        self.traces: List[tuple] = []  # For bootstrapping

    def __call__(self, **kwargs) -> Prediction:
        """
        Run prediction with given inputs.

        Args:
            **kwargs: Input field values

        Returns:
            Prediction with output field values
        """
        if self.llm is None:
            raise ValueError("LLM client not set. Pass llm to Predictor or set predictor.llm")

        # Build prompt
        prompt = self._build_prompt(kwargs)

        # Call LLM
        response = self.llm.generate(
            prompt=prompt,
            temperature=0.7,
            max_tokens=1000
        )

        # Parse response
        outputs = self._parse_response(response.content)

        # Create prediction
        prediction = Prediction(**outputs)

        # Store trace for bootstrapping
        self.traces.append((self, kwargs, outputs))

        return prediction

    def _build_prompt(self, inputs: Dict[str, Any]) -> str:
        """Build the full prompt with instructions, demos, and inputs."""
        parts = []

        # Instructions
        if self.signature.instructions:
            parts.append(self.signature.instructions)
            parts.append("")

        # Few-shot demonstrations
        if self.demos:
            parts.append("---")
            parts.append("")
            for demo in self.demos:
                # Format demo inputs
                for f in self.signature.input_fields:
                    if f.name in demo:
                        parts.append(f"{f.prefix} {demo[f.name]}")

                # Format demo outputs
                for f in self.signature.output_fields:
                    if f.name in demo:
                        parts.append(f"{f.prefix} {demo[f.name]}")

                parts.append("")
                parts.append("---")
                parts.append("")

        # Current inputs
        for f in self.signature.input_fields:
            if f.name in inputs:
                parts.append(f"{f.prefix} {inputs[f.name]}")

        # Output prompts
        for f in self.signature.output_fields:
            parts.append(f"{f.prefix}")

        return "\n".join(parts)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract output fields."""
        outputs = {}

        # Try to parse each output field
        for i, field in enumerate(self.signature.output_fields):
            prefix = field.prefix.rstrip(":")

            # Pattern to find field value
            if i < len(self.signature.output_fields) - 1:
                # Not the last field - look for next prefix
                next_prefix = self.signature.output_fields[i + 1].prefix.rstrip(":")
                pattern = rf"{re.escape(prefix)}[:\s]*(.+?)(?={re.escape(next_prefix)}|$)"
            else:
                # Last field - take everything after prefix
                pattern = rf"{re.escape(prefix)}[:\s]*(.+)"

            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                outputs[field.name] = match.group(1).strip()
            else:
                # Fallback: if only one output field, use entire response
                if len(self.signature.output_fields) == 1:
                    outputs[field.name] = response.strip()
                else:
                    outputs[field.name] = ""

        return outputs

    def reset(self) -> "Predictor":
        """Reset predictor state (clear demos and traces)."""
        self.demos = []
        self.traces = []
        return self

    def copy(self) -> "Predictor":
        """Create a copy of this predictor."""
        new_pred = Predictor(
            signature=deepcopy(self.signature),
            llm=self.llm,
        )
        new_pred.demos = deepcopy(self.demos)
        return new_pred


class Module:
    """
    Base class for composable LLM programs.

    A Module can contain multiple Predictors and orchestrate their execution.

    Usage:
        ```python
        class QAModule(Module):
            def __init__(self):
                super().__init__()
                self.qa = Predictor(
                    Signature.from_string("question -> answer")
                )

            def forward(self, question):
                return self.qa(question=question)

        # Create and use
        module = QAModule()
        module.set_llm(LLMClient(model="gpt-4o-mini"))
        result = module(question="What is AI?")
        ```
    """

    def __init__(self):
        self._predictors: Dict[str, Predictor] = {}
        self._compiled = False

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Predictor):
            if not hasattr(self, "_predictors"):
                object.__setattr__(self, "_predictors", {})
            self._predictors[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, **kwargs) -> Prediction:
        """Execute the module."""
        return self.forward(**kwargs)

    def forward(self, **kwargs) -> Prediction:
        """
        Override this method to define module behavior.

        Args:
            **kwargs: Input arguments

        Returns:
            Prediction result
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def predictors(self) -> List[Predictor]:
        """Get all predictors in this module."""
        return list(self._predictors.values())

    def named_predictors(self) -> List[tuple]:
        """Get (name, predictor) pairs."""
        return list(self._predictors.items())

    def set_llm(self, llm: LLMClient) -> "Module":
        """Set LLM for all predictors."""
        for pred in self.predictors():
            pred.llm = llm
        return self

    def reset(self) -> "Module":
        """Reset all predictors."""
        for pred in self.predictors():
            pred.reset()
        self._compiled = False
        return self

    def copy(self) -> "Module":
        """Create a shallow copy."""
        new_module = self.__class__.__new__(self.__class__)
        new_module._predictors = {}
        new_module._compiled = self._compiled

        for name, pred in self._predictors.items():
            new_pred = pred.copy()
            new_module._predictors[name] = new_pred
            object.__setattr__(new_module, name, new_pred)

        return new_module

    def deepcopy(self) -> "Module":
        """Create a deep copy."""
        return deepcopy(self)


class SimplePredictor(Module):
    """
    A simple single-predictor module.

    Convenience class when you just need one predictor.

    Usage:
        ```python
        qa = SimplePredictor("question -> answer")
        qa.set_llm(llm)
        result = qa(question="What is 2+2?")
        ```
    """

    def __init__(self, signature: str | Signature, instructions: str = ""):
        super().__init__()

        if isinstance(signature, str):
            signature = Signature.from_string(signature, instructions)

        self.predict = Predictor(signature)

    def forward(self, **kwargs) -> Prediction:
        return self.predict(**kwargs)
