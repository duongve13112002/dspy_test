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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "prefix": self.prefix,
            "field_type": self.field_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Field":
        """Create Field from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            prefix=data.get("prefix"),
            field_type=data.get("field_type", "input"),
        )


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_fields": [f.to_dict() for f in self.input_fields],
            "output_fields": [f.to_dict() for f in self.output_fields],
            "instructions": self.instructions,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Signature":
        """Create Signature from dictionary."""
        return cls(
            input_fields=[Field.from_dict(f) for f in data.get("input_fields", [])],
            output_fields=[Field.from_dict(f) for f in data.get("output_fields", [])],
            instructions=data.get("instructions", ""),
        )


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

    def save(self, path: str, include_metadata: bool = True) -> None:
        """
        Save optimized prompts and demos to a JSON file.

        Args:
            path: Path to save the JSON file
            include_metadata: Include optimization metadata (score, trial logs)

        Example:
            ```python
            optimized_module.save("optimized_prompts.json")
            ```
        """
        data = {
            "version": "2.0.0",
            "module_class": self.__class__.__name__,
            "predictors": {},
        }

        # Save each predictor's state
        for name, pred in self.named_predictors():
            pred_data = {
                "signature": pred.signature.to_dict(),
                "demos": [
                    demo.to_dict() if hasattr(demo, 'to_dict') else dict(demo._data)
                    for demo in pred.demos
                ],
                "demo_input_keys": [
                    list(demo._input_keys) if hasattr(demo, '_input_keys') else []
                    for demo in pred.demos
                ],
            }
            data["predictors"][name] = pred_data

        # Include metadata if available
        if include_metadata:
            data["metadata"] = {
                "compiled": getattr(self, '_compiled', False),
                "score": getattr(self, '_score', None),
                "trial_logs": getattr(self, '_trial_logs', None),
            }

        # Write to file
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved optimized module to {path}")

    def load(self, path: str) -> "Module":
        """
        Load optimized prompts and demos from a JSON file.

        Args:
            path: Path to the JSON file

        Returns:
            Self for chaining

        Example:
            ```python
            module.load("optimized_prompts.json")
            # or
            module = MyModule()
            module.set_llm(lm)
            module.load("optimized_prompts.json")
            ```
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Load each predictor's state
        for name, pred_data in data.get("predictors", {}).items():
            if name not in self._predictors:
                logger.warning(f"Predictor '{name}' not found in module, skipping")
                continue

            pred = self._predictors[name]

            # Load signature (mainly instructions)
            if "signature" in pred_data:
                sig_data = pred_data["signature"]
                pred.signature = Signature.from_dict(sig_data)

            # Load demos
            if "demos" in pred_data:
                pred.demos = []
                demo_input_keys = pred_data.get("demo_input_keys", [])

                for i, demo_dict in enumerate(pred_data["demos"]):
                    demo = Example(**demo_dict)
                    # Restore input keys if available
                    if i < len(demo_input_keys) and demo_input_keys[i]:
                        demo.with_inputs(*demo_input_keys[i])
                    pred.demos.append(demo)

        # Load metadata if available
        if "metadata" in data:
            self._compiled = data["metadata"].get("compiled", True)
            self._score = data["metadata"].get("score")
            self._trial_logs = data["metadata"].get("trial_logs")

        logger.info(f"Loaded optimized module from {path}")
        return self

    @classmethod
    def load_state(cls, path: str, llm=None) -> "Module":
        """
        Class method to create a new instance and load state.

        Note: This creates a base Module, not a subclass instance.
        For subclasses, use instance.load() instead.

        Args:
            path: Path to the JSON file
            llm: Optional LLM client to set

        Returns:
            New Module instance with loaded state
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create a simple module with predictors from the saved data
        module = cls()

        for name, pred_data in data.get("predictors", {}).items():
            sig = Signature.from_dict(pred_data["signature"])
            pred = Predictor(sig, llm=llm)

            # Load demos
            demo_input_keys = pred_data.get("demo_input_keys", [])
            for i, demo_dict in enumerate(pred_data.get("demos", [])):
                demo = Example(**demo_dict)
                if i < len(demo_input_keys) and demo_input_keys[i]:
                    demo.with_inputs(*demo_input_keys[i])
                pred.demos.append(demo)

            module._predictors[name] = pred
            object.__setattr__(module, name, pred)

        # Load metadata
        if "metadata" in data:
            module._compiled = data["metadata"].get("compiled", True)
            module._score = data["metadata"].get("score")
            module._trial_logs = data["metadata"].get("trial_logs")

        return module

    def get_optimized_state(self) -> Dict[str, Any]:
        """
        Get a summary of the optimized state.

        Returns:
            Dictionary with instructions and demo counts for each predictor

        Example:
            ```python
            state = optimized_module.get_optimized_state()
            for name, info in state["predictors"].items():
                print(f"{name}: {info['instruction'][:50]}...")
                print(f"  Demos: {info['num_demos']}")
            ```
        """
        return {
            "compiled": getattr(self, '_compiled', False),
            "score": getattr(self, '_score', None),
            "predictors": {
                name: {
                    "instruction": pred.signature.instructions,
                    "num_demos": len(pred.demos),
                    "demo_preview": [
                        {k: str(v)[:100] for k, v in demo._data.items()}
                        for demo in pred.demos[:2]  # First 2 demos as preview
                    ],
                }
                for name, pred in self.named_predictors()
            },
        }


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
