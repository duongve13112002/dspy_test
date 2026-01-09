"""
Evaluator Module for MIPROv2.

This module handles evaluation of candidate programs during optimization.
It supports:
- Full evaluation on validation set
- Minibatch evaluation for efficiency
- Parallel execution
"""

import logging
from typing import Any, Callable, Optional

import dspy
from dspy.utils.parallelizer import ParallelExecutor

from .utils import create_minibatch

logger = logging.getLogger(__name__)


class EvaluationResult:
    """
    Container for evaluation results.

    Attributes:
        score: Numerical score (percentage)
        results: List of (example, prediction, score) tuples
    """

    def __init__(self, score: float, results: list):
        self.score = score
        self.results = results

    def __repr__(self):
        return f"EvaluationResult(score={self.score}, results=<{len(self.results)} items>)"


class Evaluator:
    """
    Evaluates DSPy programs on a dataset.

    This class provides flexible evaluation capabilities:
    - Single-threaded or parallel execution
    - Progress display
    - Error handling with configurable limits
    """

    def __init__(
        self,
        devset: list,
        metric: Callable,
        num_threads: Optional[int] = None,
        display_progress: bool = False,
        max_errors: Optional[int] = None,
        provide_traceback: Optional[bool] = None,
        failure_score: float = 0.0,
    ):
        """
        Args:
            devset: Evaluation dataset
            metric: Scoring function (example, prediction) -> score
            num_threads: Parallel threads (None = sequential)
            display_progress: Show progress bar
            max_errors: Maximum errors before stopping
            provide_traceback: Include error tracebacks
            failure_score: Default score on failure
        """
        self.devset = devset
        self.metric = metric
        self.num_threads = num_threads
        self.display_progress = display_progress
        self.max_errors = max_errors
        self.provide_traceback = provide_traceback
        self.failure_score = failure_score

    def __call__(
        self,
        program,
        metric: Optional[Callable] = None,
        devset: Optional[list] = None,
        num_threads: Optional[int] = None,
        display_progress: Optional[bool] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate a program on the dataset.

        Args:
            program: DSPy program to evaluate
            metric: Override metric function
            devset: Override evaluation dataset
            num_threads: Override thread count
            display_progress: Override progress display

        Returns:
            EvaluationResult with score and detailed results
        """
        metric = metric or self.metric
        devset = devset or self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress

        # Create parallel executor
        executor = ParallelExecutor(
            num_threads=num_threads,
            disable_progress_bar=not display_progress,
            max_errors=self.max_errors or dspy.settings.max_errors,
            provide_traceback=self.provide_traceback,
            compare_results=True,
        )

        def process_item(example):
            prediction = program(**example.inputs())
            score = metric(example, prediction)
            return prediction, score

        # Execute evaluation
        results = executor.execute(process_item, devset)
        assert len(devset) == len(results)

        # Handle failures
        results = [
            ((dspy.Prediction(), self.failure_score) if r is None else r)
            for r in results
        ]
        results = [
            (example, prediction, score)
            for example, (prediction, score) in zip(devset, results)
        ]

        # Calculate score
        total_score = sum(score for *_, score in results)
        n_total = len(devset)
        percentage = round(100 * total_score / n_total, 2) if n_total > 0 else 0.0

        logger.info(f"Average Metric: {total_score} / {n_total} ({percentage}%)")

        return EvaluationResult(score=percentage, results=results)


def eval_candidate_program(
    batch_size: int,
    trainset: list,
    candidate_program,
    evaluator: Evaluator,
    rng=None
) -> EvaluationResult:
    """
    Evaluate a candidate program with optional minibatching.

    Args:
        batch_size: Size of evaluation batch
        trainset: Full dataset
        candidate_program: Program to evaluate
        evaluator: Evaluator instance
        rng: Random number generator for minibatch sampling

    Returns:
        EvaluationResult with score
    """
    try:
        if batch_size >= len(trainset):
            # Full evaluation
            return evaluator(candidate_program, devset=trainset)
        else:
            # Minibatch evaluation
            minibatch = create_minibatch(trainset, batch_size, rng)
            return evaluator(candidate_program, devset=minibatch)

    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        return EvaluationResult(score=0.0, results=[])
