"""
MIPROv2 - Model-based Instruction Prompt Optimization v2

A from-scratch implementation of the MIPROv2 optimizer for DSPy.
This implementation follows the original DSPy library architecture.

Author: Implementation from scratch based on DSPy MIPROv2
"""

from .mipro_v2 import MIPROv2
from .proposer import GroundedProposer, InstructionGenerator
from .dataset_summary import DatasetSummaryGenerator
from .bootstrap import BootstrapFewShot, LabeledFewShot
from .evaluator import Evaluator

__all__ = [
    "MIPROv2",
    "GroundedProposer",
    "InstructionGenerator",
    "DatasetSummaryGenerator",
    "BootstrapFewShot",
    "LabeledFewShot",
    "Evaluator",
]

__version__ = "1.0.0"
