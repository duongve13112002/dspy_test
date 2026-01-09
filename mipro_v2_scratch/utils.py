"""
Utility functions for MIPROv2 optimizer.

This module contains helper functions for:
- Minibatch creation
- Program evaluation
- Signature manipulation
- Logging utilities
"""

import logging
import random
import os
from collections import defaultdict
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


def create_minibatch(dataset: list, batch_size: int, rng: Optional[random.Random] = None) -> list:
    """
    Create a random minibatch from the dataset.

    Args:
        dataset: The full dataset to sample from
        batch_size: Number of samples to include in minibatch
        rng: Random number generator for reproducibility

    Returns:
        A list containing the sampled minibatch
    """
    batch_size = min(batch_size, len(dataset))
    rng = rng or random.Random()
    sampled_indices = rng.sample(range(len(dataset)), batch_size)
    return [dataset[i] for i in sampled_indices]


def get_signature(predictor) -> Any:
    """
    Extract the signature from a predictor.

    Args:
        predictor: DSPy predictor module

    Returns:
        The predictor's signature
    """
    assert hasattr(predictor, "signature"), "Predictor must have a signature attribute"
    return predictor.signature


def set_signature(predictor, new_signature) -> None:
    """
    Update the signature of a predictor.

    Args:
        predictor: DSPy predictor module
        new_signature: The new signature to set
    """
    assert hasattr(predictor, "signature"), "Predictor must have a signature attribute"
    predictor.signature = new_signature


def get_program_with_highest_avg_score(
    param_score_dict: dict,
    fully_evaled_param_combos: dict
) -> tuple:
    """
    Find the program with highest average score from minibatch evaluations.

    Used in Bayesian optimization with minibatching to identify the most
    promising candidate for full evaluation.

    Args:
        param_score_dict: Dictionary mapping parameter combinations to (score, program, params) tuples
        fully_evaled_param_combos: Set of parameter combinations already fully evaluated

    Returns:
        Tuple of (program, mean_score, combo_key, params)
    """
    results = []

    for key, values in param_score_dict.items():
        scores = np.array([v[0] for v in values])
        mean = np.average(scores)
        program = values[0][1]
        params = values[0][2]
        results.append((key, mean, program, params))

    # Sort by mean score descending
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

    # Find highest scoring combo not yet fully evaluated
    for combination in sorted_results:
        key, mean, program, params = combination
        if key not in fully_evaled_param_combos:
            return program, mean, key, params

    # Fallback to last valid one
    return program, mean, key, params


def print_full_program(program) -> None:
    """
    Print the program's instructions and prefixes for each predictor.

    Args:
        program: DSPy program to print
    """
    for i, predictor in enumerate(program.predictors()):
        print(f"Predictor {i}")
        print(f"Instructions: {get_signature(predictor).instructions}")
        *_, last_field = get_signature(predictor).fields.values()
        print(f"Prefix: {last_field.json_schema_extra.get('prefix', 'N/A')}")
    print("\n")


def save_candidate_program(
    program,
    log_dir: Optional[str],
    trial_num: int,
    note: Optional[str] = None
) -> Optional[str]:
    """
    Save a candidate program to the log directory.

    Args:
        program: DSPy program to save
        log_dir: Directory to save to (if None, returns None)
        trial_num: Trial number for filename
        note: Optional note to append to filename

    Returns:
        Path to saved file, or None if log_dir is None
    """
    if log_dir is None:
        return None

    eval_programs_dir = os.path.join(log_dir, "evaluated_programs")
    os.makedirs(eval_programs_dir, exist_ok=True)

    if note:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}_{note}.json")
    else:
        save_path = os.path.join(eval_programs_dir, f"program_{trial_num}.json")

    program.save(save_path)
    return save_path


def create_example_string(fields: dict, example) -> str:
    """
    Create a formatted string representation of an example.

    Args:
        fields: Dictionary of field definitions
        example: Example to format

    Returns:
        Formatted string representation
    """
    parts = []
    for field_name, field in fields.items():
        if field_name in example:
            value = example[field_name]
            prefix = field.json_schema_extra.get("prefix", field_name + ":")
            parts.append(f"{prefix} {value}")
    return "\n".join(parts)


def strip_prefix(text: str) -> str:
    """
    Remove common prefixes from generated text.

    Args:
        text: Text to clean

    Returns:
        Cleaned text
    """
    # Common prefixes to remove
    prefixes = [
        "PROPOSED INSTRUCTION:",
        "Proposed Instruction:",
        "proposed instruction:",
        "INSTRUCTION:",
        "Instruction:",
    ]

    text = text.strip()
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    return text


# ANSI color codes for terminal output
class Colors:
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    BLUE = "\033[94m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    ENDC = "\033[0m"


def colored_print(message: str, color: str = Colors.ENDC) -> None:
    """Print a message with ANSI color codes."""
    print(f"{color}{message}{Colors.ENDC}")
