"""
Example - Data container for training/evaluation examples.

Similar to a flexible dictionary that tracks which fields are inputs vs outputs.
"""

from typing import Any, Dict, List, Optional, Set
from copy import deepcopy


class Example:
    """
    A flexible data container for examples.

    Examples store input/output fields and track which fields are inputs.
    This allows automatic extraction of inputs for prediction and
    outputs for evaluation.

    Usage:
        ```python
        # Create an example
        ex = Example(
            question="What is 2+2?",
            answer="4"
        ).with_inputs("question")

        # Access fields
        print(ex.question)  # "What is 2+2?"
        print(ex.answer)    # "4"

        # Get only inputs
        print(ex.inputs())  # {"question": "What is 2+2?"}

        # Get only labels (outputs)
        print(ex.labels())  # {"answer": "4"}
        ```
    """

    def __init__(self, **kwargs):
        """
        Initialize example with field values.

        Args:
            **kwargs: Field name-value pairs
        """
        self._data: Dict[str, Any] = kwargs
        self._input_keys: Set[str] = set()

    def __getattr__(self, name: str) -> Any:
        """Get field value by attribute access."""
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        data = object.__getattribute__(self, "_data")
        if name in data:
            return data[name]
        raise AttributeError(f"Example has no field '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        """Set field value by attribute access."""
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            self._data[name] = value

    def __getitem__(self, key: str) -> Any:
        """Get field value by key access."""
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Set field value by key access."""
        self._data[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if field exists."""
        return key in self._data

    def __repr__(self) -> str:
        """String representation."""
        fields = ", ".join(f"{k}={repr(v)[:50]}" for k, v in self._data.items())
        return f"Example({fields})"

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Example):
            return False
        return self._data == other._data

    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(tuple(sorted(self._data.items(), key=lambda x: x[0])))

    def keys(self) -> List[str]:
        """Get all field names."""
        return list(self._data.keys())

    def values(self) -> List[Any]:
        """Get all field values."""
        return list(self._data.values())

    def items(self):
        """Get all field name-value pairs."""
        return self._data.items()

    def get(self, key: str, default: Any = None) -> Any:
        """Get field value with default."""
        return self._data.get(key, default)

    def with_inputs(self, *keys: str) -> "Example":
        """
        Mark fields as inputs.

        Args:
            *keys: Field names to mark as inputs

        Returns:
            Self for chaining
        """
        self._input_keys = set(keys)
        return self

    def inputs(self) -> Dict[str, Any]:
        """
        Get only input fields.

        Returns:
            Dictionary of input field name-value pairs
        """
        if not self._input_keys:
            # If no inputs specified, return all except 'augmented' marker
            return {k: v for k, v in self._data.items() if k != "augmented"}
        return {k: v for k, v in self._data.items() if k in self._input_keys}

    def labels(self) -> Dict[str, Any]:
        """
        Get only output/label fields.

        Returns:
            Dictionary of output field name-value pairs
        """
        if not self._input_keys:
            return {}
        return {
            k: v for k, v in self._data.items()
            if k not in self._input_keys and k != "augmented"
        }

    def copy(self) -> "Example":
        """Create a shallow copy."""
        new_ex = Example(**self._data.copy())
        new_ex._input_keys = self._input_keys.copy()
        return new_ex

    def deepcopy(self) -> "Example":
        """Create a deep copy."""
        new_ex = Example(**deepcopy(self._data))
        new_ex._input_keys = self._input_keys.copy()
        return new_ex

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self._data.copy()

    @classmethod
    def from_dict(cls, data: Dict[str, Any], inputs: Optional[List[str]] = None) -> "Example":
        """
        Create Example from dictionary.

        Args:
            data: Field name-value pairs
            inputs: Optional list of input field names

        Returns:
            New Example instance
        """
        ex = cls(**data)
        if inputs:
            ex.with_inputs(*inputs)
        return ex


class Prediction(Example):
    """
    A prediction output from a module.

    Same as Example but semantically represents a prediction/output.
    """

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={repr(v)[:50]}" for k, v in self._data.items())
        return f"Prediction({fields})"
