"""
Dataset Summary Generator for MIPROv2.

This module generates a summary of the training dataset using LLM calls.
The summary is used to inform instruction generation by providing context
about the data characteristics.

The process:
1. Analyze initial batch of examples
2. Iteratively refine observations with more batches
3. Summarize all observations into concise description
"""

import re
from typing import Optional

import dspy


# Signature for initial dataset observation
class DatasetDescriptor(dspy.Signature):
    """
    Given several examples from a dataset, write observations about trends
    that hold for most or all of the samples.

    Consider: topics, content, syntax, conciseness, etc.
    Make an educated guess about the task this dataset enables.
    """
    examples = dspy.InputField(
        desc="Sample data points from the dataset"
    )
    observations = dspy.OutputField(
        desc="Things that hold true for most or all of the data observed"
    )


# Signature for iterative refinement with prior observations
class DatasetDescriptorWithPriorObservations(dspy.Signature):
    """
    Given examples from a dataset and prior observations, add new observations
    or say 'COMPLETE' if observations are comprehensive.

    Consider: topics, content, syntax, conciseness, etc.
    """
    examples = dspy.InputField(
        desc="Sample data points from the dataset"
    )
    prior_observations = dspy.InputField(
        desc="Prior observations made about the data"
    )
    observations = dspy.OutputField(
        desc="New observations or 'COMPLETE' if nothing to add"
    )


# Signature for final summary
class ObservationSummarizer(dspy.Signature):
    """
    Given observations about a dataset, summarize them into a brief
    2-3 sentence summary highlighting only the most important details.
    """
    observations = dspy.InputField(
        desc="Observations made about the dataset"
    )
    summary = dspy.OutputField(
        desc="Two to three sentence summary of the most significant observations"
    )


def _order_input_keys_in_string(unordered_repr: str) -> str:
    """
    Reorder input_keys in string representation for consistency.

    Args:
        unordered_repr: String representation with potentially unordered keys

    Returns:
        String with ordered input_keys
    """
    pattern = r"input_keys=\{([^\}]+)\}"

    def reorder_keys(match):
        keys_str = match.group(1)
        keys = sorted(key.strip() for key in keys_str.split(","))
        return f"input_keys={{{', '.join(keys)}}}"

    return re.sub(pattern, reorder_keys, unordered_repr)


class DatasetSummaryGenerator:
    """
    Generates a summary of the training dataset.

    The summary provides context for instruction generation, helping
    the proposer understand what kind of data and task we're working with.
    """

    def __init__(
        self,
        prompt_model=None,
        view_data_batch_size: int = 10,
        max_iterations: int = 10,
        verbose: bool = False
    ):
        """
        Args:
            prompt_model: Language model for generating summaries
            view_data_batch_size: Number of examples per batch
            max_iterations: Maximum refinement iterations
            verbose: Whether to print progress
        """
        self.prompt_model = prompt_model
        self.view_data_batch_size = view_data_batch_size
        self.max_iterations = max_iterations
        self.verbose = verbose

    def generate(self, trainset: list) -> str:
        """
        Generate a summary of the training dataset.

        Args:
            trainset: Training dataset to summarize

        Returns:
            String summary of dataset characteristics
        """
        if self.verbose:
            print("\nGenerating dataset summary...")

        prompt_model = self.prompt_model or dspy.settings.lm

        # Initial observation from first batch
        upper_lim = min(len(trainset), self.view_data_batch_size)
        examples_str = _order_input_keys_in_string(repr(trainset[0:upper_lim]))

        with dspy.context(lm=prompt_model):
            initial = dspy.Predict(DatasetDescriptor, n=1, temperature=1.0)(
                examples=examples_str
            )

        observations = initial.observations

        if self.verbose:
            print(f"Initial observations: {observations[:100]}...")

        # Iterative refinement
        skips = 0
        calls = 0

        try:
            for batch_start in range(
                self.view_data_batch_size,
                len(trainset),
                self.view_data_batch_size
            ):
                calls += 1
                if calls >= self.max_iterations:
                    break

                batch_end = min(len(trainset), batch_start + self.view_data_batch_size)
                batch_str = _order_input_keys_in_string(
                    repr(trainset[batch_start:batch_end])
                )

                with dspy.context(lm=prompt_model):
                    output = dspy.Predict(
                        DatasetDescriptorWithPriorObservations,
                        n=1,
                        temperature=1.0
                    )(
                        prior_observations=observations,
                        examples=batch_str
                    )

                # Check if observations are complete
                if (len(output.observations) >= 8 and
                    output.observations[:8].upper() == "COMPLETE"):
                    skips += 1
                    if skips >= 5:
                        break
                    continue

                observations += " " + output.observations

        except Exception as e:
            if self.verbose:
                print(f"Error during refinement: {e}. Using current observations.")

        # Generate final summary
        with dspy.context(lm=prompt_model):
            summary_result = dspy.Predict(
                ObservationSummarizer,
                n=1,
                temperature=1.0
            )(observations=observations)

        summary = self._strip_prefix(summary_result.summary)

        if self.verbose:
            print(f"\nGenerated summary: {summary}")

        return summary

    @staticmethod
    def _strip_prefix(text: str) -> str:
        """Remove common prefixes from text."""
        text = text.strip()
        prefixes = ["SUMMARY:", "Summary:", "summary:"]
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return text


def create_dataset_summary(
    trainset: list,
    view_data_batch_size: int = 10,
    prompt_model=None,
    verbose: bool = False
) -> str:
    """
    Convenience function to generate dataset summary.

    Args:
        trainset: Training dataset
        view_data_batch_size: Examples per batch
        prompt_model: Language model to use
        verbose: Print progress

    Returns:
        Dataset summary string
    """
    generator = DatasetSummaryGenerator(
        prompt_model=prompt_model,
        view_data_batch_size=view_data_batch_size,
        verbose=verbose
    )
    return generator.generate(trainset)
