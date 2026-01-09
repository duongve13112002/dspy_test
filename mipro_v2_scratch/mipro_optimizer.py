"""
MIPROv2 - Standalone Multi-Prompt Instruction Optimization v2

A complete implementation of MIPRO without any DSPy dependencies.
Based on: "Optimizing Instructions and Demonstrations for Multi-Stage
Language Model Programs" (arXiv:2406.11695)

The algorithm has 3 main steps:
1. Bootstrap Demonstrations - Generate diverse few-shot example sets
2. Propose Instructions - Generate instruction candidates using LLM
3. Bayesian Optimization - Find optimal combination using TPE

This implementation follows the paper and DSPy library closely.
"""

import random
import logging
import inspect
from copy import deepcopy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from .llm_client import LLMClient
from .example import Example, Prediction
from .module import Module, Predictor, Signature

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS (Aligned with DSPy library)
# ============================================================================

# Constants from DSPy
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50
MAX_INSTRUCT_IN_HISTORY = 5
DEFAULT_VIEW_DATA_BATCH_SIZE = 10

# Auto settings (from DSPy library)
AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}

# Prompting tips for instruction generation (Exact from paper - Appendix C.2)
TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona that is relevant to the task in the instruction (ie. "You are a ...")',
}


# ============================================================================
# DATASET SUMMARY GENERATOR (Following paper/DSPy exactly)
# ============================================================================

class DatasetSummaryGenerator:
    """
    Generates dataset summary using iterative observation gathering.

    Process (from paper):
    1. Loop through training data in batches
    2. Ask LLM to write observations about trends
    3. Stop when LLM outputs "COMPLETE" 5 consecutive times
    4. Summarize observations into 2-3 sentences
    """

    OBSERVATION_PROMPT = """Given several examples from a dataset please write observations about trends that hold for most or all of the samples.
Some areas you may consider in your observations: topics, content, syntax, conciseness, etc.
It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative.

Examples:
{examples}

Observations:"""

    OBSERVATION_WITH_PRIOR_PROMPT = """Given several examples from a dataset please write observations about trends that hold for most or all of the samples.
I will also provide you with a few observations I have already made. Please add your own observations or if you feel the observations are comprehensive say 'COMPLETE'.
Some areas you may consider in your observations: topics, content, syntax, conciseness, etc.
It will be useful to make an educated guess as to the nature of the task this dataset will enable. Don't be afraid to be creative.

Examples:
{examples}

Prior observations:
{prior_observations}

Additional observations (or 'COMPLETE' if nothing to add):"""

    SUMMARIZE_PROMPT = """Given a series of observations I have made about my dataset, please summarize them into a brief 2-3 sentence summary which highlights only the most important details.

Observations:
{observations}

Summary (2-3 sentences):"""

    def __init__(self, llm: LLMClient, batch_size: int = 10, max_calls: int = 10):
        self.llm = llm
        self.batch_size = batch_size
        self.max_calls = max_calls

    def generate(self, trainset: List[Example]) -> str:
        """Generate dataset summary using iterative observation."""
        if not trainset:
            return "No dataset provided."

        # Initial observation on first batch
        upper_lim = min(len(trainset), self.batch_size)
        examples_str = self._format_examples(trainset[:upper_lim])

        try:
            response = self.llm.generate(
                self.OBSERVATION_PROMPT.format(examples=examples_str),
                temperature=1.0,
                max_tokens=500
            )
            observations = response.content.strip()
        except Exception as e:
            logger.warning(f"Error in initial observation: {e}")
            return "Dataset summary not available."

        # Iterate through remaining batches
        skips = 0
        calls = 0

        for b in range(self.batch_size, len(trainset), self.batch_size):
            calls += 1
            if calls >= self.max_calls:
                break

            upper_lim = min(len(trainset), b + self.batch_size)
            examples_str = self._format_examples(trainset[b:upper_lim])

            try:
                response = self.llm.generate(
                    self.OBSERVATION_WITH_PRIOR_PROMPT.format(
                        examples=examples_str,
                        prior_observations=observations
                    ),
                    temperature=1.0,
                    max_tokens=500
                )
                new_obs = response.content.strip()

                # Check for COMPLETE
                if len(new_obs) >= 8 and new_obs[:8].upper() == "COMPLETE":
                    skips += 1
                    if skips >= 5:
                        break
                    continue

                observations += "\n" + new_obs

            except Exception as e:
                logger.warning(f"Error in observation iteration: {e}")
                break

        # Summarize observations
        try:
            response = self.llm.generate(
                self.SUMMARIZE_PROMPT.format(observations=observations),
                temperature=1.0,
                max_tokens=200
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error summarizing: {e}")
            return observations[:500]  # Return truncated observations as fallback

    def _format_examples(self, examples: List[Example]) -> str:
        """Format examples for prompt."""
        parts = []
        for i, ex in enumerate(examples):
            ex_parts = [f"Example {i + 1}:"]
            for k, v in ex.items():
                if k not in ("augmented", "_input_keys"):
                    ex_parts.append(f"  {k}: {v}")
            parts.append("\n".join(ex_parts))
        return "\n\n".join(parts)


# ============================================================================
# PROGRAM DESCRIPTION GENERATOR (Following paper/DSPy)
# ============================================================================

class ProgramDescriptionGenerator:
    """
    Generates program description by analyzing code structure.

    From paper: LLM analyzes DSPy pseudo-code and describes:
    1. What task the program is designed to solve
    2. How it appears to work
    """

    DESCRIBE_PROGRAM_PROMPT = """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe what type of task this program appears to be designed to solve, and how it appears to work.

PROGRAM CODE:
{program_code}

EXAMPLE OF PROGRAM IN USE:
{program_example}

SUMMARY OF PROGRAM ABOVE:"""

    DESCRIBE_MODULE_PROMPT = """Below is some pseudo-code for a pipeline that solves tasks with calls to language models. Please describe the purpose of one of the specified modules in this pipeline.

PROGRAM CODE:
{program_code}

SUMMARY OF PROGRAM ABOVE:
{program_description}

EXAMPLE OF PROGRAM IN USE:
{program_example}

MODULE:
{module}

MODULE DESCRIPTION:"""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def describe_program(self, program: Module, task_demos: str) -> str:
        """Generate program description."""
        program_code = self._get_program_code(program)

        try:
            response = self.llm.generate(
                self.DESCRIBE_PROGRAM_PROMPT.format(
                    program_code=program_code,
                    program_example=task_demos
                ),
                temperature=0.7,
                max_tokens=300
            )
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error describing program: {e}")
            return "Program description not available."

    def describe_module(
        self,
        program: Module,
        predictor: Predictor,
        program_description: str,
        task_demos: str
    ) -> Tuple[str, str]:
        """Generate module code representation and description."""
        program_code = self._get_program_code(program)

        # Create module code string
        inputs = [f.name for f in predictor.signature.input_fields]
        outputs = [f.name for f in predictor.signature.output_fields]
        module_code = f"Predictor({', '.join(inputs)}) -> {', '.join(outputs)}"

        try:
            response = self.llm.generate(
                self.DESCRIBE_MODULE_PROMPT.format(
                    program_code=program_code,
                    program_description=program_description,
                    program_example=task_demos,
                    module=module_code
                ),
                temperature=0.7,
                max_tokens=200
            )
            return module_code, response.content.strip()
        except Exception as e:
            logger.warning(f"Error describing module: {e}")
            return module_code, "Module description not available."

    def _get_program_code(self, program: Module) -> str:
        """Get program source code or representation."""
        try:
            # Try to get actual source code
            source = inspect.getsource(program.__class__)
            return source
        except Exception:
            # Fallback to string representation
            return self._create_pseudo_code(program)

    def _create_pseudo_code(self, program: Module) -> str:
        """Create pseudo-code representation of program."""
        lines = [f"class {program.__class__.__name__}(Module):"]
        lines.append("    def __init__(self):")

        for name, pred in program.named_predictors():
            inputs = [f.name for f in pred.signature.input_fields]
            outputs = [f.name for f in pred.signature.output_fields]
            sig_str = f"{', '.join(inputs)} -> {', '.join(outputs)}"
            lines.append(f"        self.{name} = Predictor(\"{sig_str}\")")

        lines.append("")
        lines.append("    def forward(self, **inputs):")
        lines.append("        # Process through each predictor")
        for name, _ in program.named_predictors():
            lines.append(f"        result = self.{name}(**inputs)")
        lines.append("        return result")

        return "\n".join(lines)


# ============================================================================
# STEP 1: BOOTSTRAP DEMONSTRATIONS
# ============================================================================

class DemonstrationBootstrapper:
    """
    Generates diverse sets of few-shot demonstrations through bootstrapping.

    The bootstrapping process:
    1. Run training examples through the program
    2. Evaluate outputs with the metric
    3. Keep successful traces as demonstrations
    4. Create multiple diverse demo sets
    """

    def __init__(
        self,
        metric: Callable,
        llm: LLMClient,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        metric_threshold: Optional[float] = None,
        max_errors: int = 10,
    ):
        self.metric = metric
        self.llm = llm
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.metric_threshold = metric_threshold
        self.max_errors = max_errors

    def bootstrap(
        self,
        program: Module,
        trainset: List[Example],
        num_candidate_sets: int,
        rng: random.Random,
    ) -> Dict[int, List[List[Example]]]:
        """
        Create N diverse demonstration sets.

        Args:
            program: The LLM program
            trainset: Training examples
            num_candidate_sets: Number of demo sets to create
            rng: Random number generator

        Returns:
            Dict mapping predictor index to list of demo sets
        """
        logger.info(f"Bootstrapping {num_candidate_sets} demonstration sets...")

        demo_candidates = {i: [] for i in range(len(program.predictors()))}

        for set_idx in range(num_candidate_sets):
            logger.info(f"  Creating set {set_idx + 1}/{num_candidate_sets}")

            if set_idx == 0:
                # Zero-shot set (empty demos)
                demos_per_pred = self._create_empty_demos(program)

            elif set_idx == 1 and self.max_labeled_demos > 0:
                # Labeled-only set (random samples)
                demos_per_pred = self._create_labeled_demos(
                    program, trainset, rng
                )

            else:
                # Bootstrapped set (metric-validated)
                trainset_shuffled = list(trainset)
                rng.shuffle(trainset_shuffled)

                demos_per_pred = self._bootstrap_demos(
                    program, trainset_shuffled, rng
                )

            # Add to candidates
            for pred_idx, demos in demos_per_pred.items():
                demo_candidates[pred_idx].append(demos)

        return demo_candidates

    def _create_empty_demos(self, program: Module) -> Dict[int, List[Example]]:
        """Create empty demo sets (zero-shot)."""
        return {i: [] for i in range(len(program.predictors()))}

    def _create_labeled_demos(
        self,
        program: Module,
        trainset: List[Example],
        rng: random.Random,
    ) -> Dict[int, List[Example]]:
        """Create demo sets from random labeled samples."""
        k = min(self.max_labeled_demos, len(trainset))
        sampled = rng.sample(trainset, k)

        # Same demos for all predictors
        return {i: list(sampled) for i in range(len(program.predictors()))}

    def _bootstrap_demos(
        self,
        program: Module,
        trainset: List[Example],
        rng: random.Random,
    ) -> Dict[int, List[Example]]:
        """Bootstrap demos by running and validating with metric."""
        demos_per_pred = {i: [] for i in range(len(program.predictors()))}
        errors = 0

        for example in trainset:
            # Check if we have enough demos
            min_demos = min(len(d) for d in demos_per_pred.values())
            if min_demos >= self.max_bootstrapped_demos:
                break

            try:
                # Run program on example
                program_copy = program.copy()
                prediction = program_copy(**example.inputs())

                # Evaluate with metric
                score = self.metric(example, prediction)

                # Check threshold
                if self.metric_threshold is not None:
                    success = score >= self.metric_threshold
                else:
                    success = bool(score)

                if success:
                    # Collect traces from all predictors
                    for pred_idx, pred in enumerate(program_copy.predictors()):
                        if pred.traces:
                            _, inputs, outputs = pred.traces[-1]
                            demo = Example(augmented=True, **inputs, **outputs)
                            demos_per_pred[pred_idx].append(demo)

            except Exception as e:
                errors += 1
                logger.warning(f"Bootstrap error: {e}")
                if errors >= self.max_errors:
                    logger.error("Max errors reached in bootstrapping")
                    break

        # Pad with labeled demos if needed
        for pred_idx in demos_per_pred:
            current = len(demos_per_pred[pred_idx])
            if current < self.max_bootstrapped_demos:
                needed = min(
                    self.max_labeled_demos,
                    len(trainset),
                    self.max_bootstrapped_demos - current
                )
                sampled = rng.sample(trainset, needed)
                demos_per_pred[pred_idx].extend(sampled)

        return demos_per_pred


# ============================================================================
# STEP 2: PROPOSE INSTRUCTIONS (Following paper/DSPy exactly)
# ============================================================================

class InstructionProposer:
    """
    Generates instruction candidates using LLM-based grounded proposal.

    The "grounded" approach uses (from paper):
    - Dataset summary (iterative observation)
    - Program code & description
    - Module description
    - Task demonstrations
    - Previous instructions with scores
    - Prompting tips
    """

    INSTRUCTION_PROPOSAL_PROMPT = """Use the information below to learn about a task that we are trying to solve using calls to an LM, then generate a new instruction that will be used to prompt a Language Model to better solve the task.

{dataset_section}

{program_section}

TASK DEMO(S):
{task_demos}

{history_section}

BASIC INSTRUCTION:
{basic_instruction}

{tip_section}

PROPOSED INSTRUCTION:"""

    def __init__(
        self,
        llm: LLMClient,
        program_aware: bool = True,
        data_aware: bool = True,
        tip_aware: bool = True,
        use_instruct_history: bool = True,
        view_data_batch_size: int = 10,
        num_demos_in_context: int = 3,
        verbose: bool = False,
    ):
        self.llm = llm
        self.program_aware = program_aware
        self.data_aware = data_aware
        self.tip_aware = tip_aware
        self.use_instruct_history = use_instruct_history
        self.view_data_batch_size = view_data_batch_size
        self.num_demos_in_context = num_demos_in_context
        self.verbose = verbose

        # Generators
        self.dataset_summarizer = DatasetSummaryGenerator(llm, view_data_batch_size)
        self.program_describer = ProgramDescriptionGenerator(llm)

    def propose(
        self,
        program: Module,
        trainset: List[Example],
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
        num_candidates: int,
        trial_logs: Dict,
        rng: random.Random,
    ) -> Dict[int, List[str]]:
        """
        Generate instruction candidates for all predictors.
        """
        logger.info(f"Proposing {num_candidates} instructions per predictor...")

        # Generate dataset summary (iterative)
        data_summary = None
        if self.data_aware:
            logger.info("  Generating dataset summary...")
            data_summary = self.dataset_summarizer.generate(trainset)
            if self.verbose:
                logger.info(f"  Dataset summary: {data_summary}")

        # Get program description
        program_description = None
        if self.program_aware:
            logger.info("  Generating program description...")
            # Use first demo set for example
            task_demos_str = self._format_demos_for_context(
                program, demo_candidates, 0, 0
            )
            program_description = self.program_describer.describe_program(
                program, task_demos_str
            )
            if self.verbose:
                logger.info(f"  Program description: {program_description}")

        instruction_candidates = {}

        # Determine how many demos we have
        if demo_candidates:
            num_demos = max(len(demo_candidates.get(0, [])), 1)
        else:
            num_demos = num_candidates

        for pred_idx, (name, predictor) in enumerate(program.named_predictors()):
            logger.info(f"  Predictor {pred_idx} ({name})")

            # Get module description
            module_code = None
            module_description = None
            if self.program_aware and program_description:
                task_demos_str = self._format_demos_for_context(
                    program, demo_candidates, pred_idx, 0
                )
                module_code, module_description = self.program_describer.describe_module(
                    program, predictor, program_description, task_demos_str
                )

            # Start with original instruction
            original = predictor.signature.instructions or "Complete the task."
            candidates = [original]

            # Generate new candidates
            for demo_set_i in range(min(num_candidates - 1, num_demos)):
                # Randomly decide whether to use history (50% chance, like DSPy)
                use_history = rng.random() < 0.5 if self.use_instruct_history else False

                # Select random tip
                if self.tip_aware:
                    tip_key = rng.choice(list(TIPS.keys()))
                    tip = TIPS[tip_key]
                else:
                    tip = ""

                # Get demos for context
                task_demos_str = self._format_demos_for_context(
                    program, demo_candidates, pred_idx, demo_set_i
                )

                # Get instruction history
                history_str = ""
                if use_history:
                    history_str = self._get_instruction_history(
                        pred_idx, trial_logs, MAX_INSTRUCT_IN_HISTORY
                    )

                # Generate instruction
                instruction = self._propose_single(
                    predictor=predictor,
                    data_summary=data_summary,
                    program_description=program_description,
                    module_code=module_code,
                    module_description=module_description,
                    task_demos=task_demos_str,
                    history=history_str,
                    basic_instruction=original,
                    tip=tip,
                )

                candidates.append(instruction)

                if self.verbose:
                    logger.info(f"    Candidate {demo_set_i + 1}: {instruction[:60]}...")

            instruction_candidates[pred_idx] = candidates

        return instruction_candidates

    def _format_demos_for_context(
        self,
        program: Module,
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
        pred_idx: int,
        demo_set_i: int,
    ) -> str:
        """Format demonstrations for proposal context."""
        if not demo_candidates or pred_idx not in demo_candidates:
            return "No task demos provided."

        demo_sets = demo_candidates[pred_idx]
        if not demo_sets or demo_set_i >= len(demo_sets):
            return "No task demos provided."

        demos = demo_sets[demo_set_i]
        if not demos:
            return "No task demos provided."

        predictor = program.predictors()[pred_idx]

        # Format up to num_demos_in_context demos
        demo_strs = []
        for demo in demos[:self.num_demos_in_context]:
            if not hasattr(demo, 'get') or 'augmented' not in demo:
                continue
            parts = []
            for field in predictor.signature.input_fields:
                if field.name in demo:
                    parts.append(f"{field.prefix} {demo[field.name]}")
            for field in predictor.signature.output_fields:
                if field.name in demo:
                    parts.append(f"{field.prefix} {demo[field.name]}")
            if parts:
                demo_strs.append("\n".join(parts))

        if not demo_strs:
            return "No task demos provided."

        return "\n\n---\n\n".join(demo_strs)

    def _get_instruction_history(
        self,
        pred_idx: int,
        trial_logs: Dict,
        max_history: int,
    ) -> str:
        """Get previous instructions and their scores for a predictor."""
        if not trial_logs:
            return ""

        history = []
        for trial_num, log in sorted(trial_logs.items(), reverse=True):
            if len(history) >= max_history:
                break

            instr_key = f"pred_{pred_idx}_instruction"
            if instr_key in log:
                score = log.get("score", log.get("full_eval_score", "N/A"))
                instruction = log.get(f"pred_{pred_idx}_instruction_text", f"Instruction {log[instr_key]}")
                history.append(f"- Score {score}: {instruction}")

        if not history:
            return ""

        return "Previous instructions and scores:\n" + "\n".join(history)

    def _propose_single(
        self,
        predictor: Predictor,
        data_summary: Optional[str],
        program_description: Optional[str],
        module_code: Optional[str],
        module_description: Optional[str],
        task_demos: str,
        history: str,
        basic_instruction: str,
        tip: str,
    ) -> str:
        """Generate a single instruction candidate."""

        # Build dataset section
        if data_summary:
            dataset_section = f"DATASET SUMMARY:\n{data_summary}"
        else:
            dataset_section = ""

        # Build program section
        program_section = ""
        if program_description and module_code:
            program_section = f"""PROGRAM DESCRIPTION:
{program_description}

MODULE:
{module_code}

MODULE DESCRIPTION:
{module_description or 'Not available'}"""

        # Build history section
        if history:
            history_section = f"PREVIOUS INSTRUCTIONS:\n{history}"
        else:
            history_section = ""

        # Build tip section
        if tip:
            tip_section = f"TIP:\n{tip}"
        else:
            tip_section = ""

        # Build full prompt
        prompt = self.INSTRUCTION_PROPOSAL_PROMPT.format(
            dataset_section=dataset_section,
            program_section=program_section,
            task_demos=task_demos,
            history_section=history_section,
            basic_instruction=basic_instruction,
            tip_section=tip_section,
        )

        # Clean up empty sections
        prompt = "\n".join(line for line in prompt.split("\n") if line.strip())

        try:
            response = self.llm.generate(
                prompt,
                temperature=1.0,
                max_tokens=300
            )
            instruction = response.content.strip()

            # Clean up common prefixes
            prefixes = ["PROPOSED INSTRUCTION:", "Proposed instruction:",
                       "New instruction:", "Instruction:"]
            for prefix in prefixes:
                if instruction.lower().startswith(prefix.lower()):
                    instruction = instruction[len(prefix):].strip()

            return instruction

        except Exception as e:
            logger.warning(f"Failed to generate instruction: {e}")
            return basic_instruction


# ============================================================================
# STEP 3: BAYESIAN OPTIMIZATION
# ============================================================================

class BayesianOptimizer:
    """
    Finds optimal instruction/demo combination using Bayesian Optimization.

    Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler to
    efficiently search the combinatorial space of:
    - Which instruction to use for each predictor
    - Which demo set to use for each predictor
    """

    def __init__(
        self,
        metric: Callable,
        llm: LLMClient,
        num_threads: int = 1,
        verbose: bool = False,
    ):
        self.metric = metric
        self.llm = llm
        self.num_threads = num_threads
        self.verbose = verbose

    def optimize(
        self,
        program: Module,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
        valset: List[Example],
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
        rng: random.Random,
    ) -> Tuple[Module, float, Dict]:
        """
        Run Bayesian optimization to find best parameter combination.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info("Starting Bayesian optimization...")

        # Initialize tracking
        best_score = 0.0
        best_program = program.copy()
        trial_logs = {}
        param_score_dict = defaultdict(list)
        fully_evaled = {}

        # Evaluate default program
        logger.info("Evaluating default program...")
        default_score = self._evaluate(program, valset)
        logger.info(f"Default score: {default_score:.2f}%")

        best_score = default_score
        trial_logs[0] = {"score": default_score, "type": "default"}

        # Create Optuna study
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Add default as baseline
        default_params = self._create_default_params(
            program, instruction_candidates, demo_candidates
        )
        baseline_trial = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_distributions(
                program, instruction_candidates, demo_candidates
            ),
            value=default_score,
        )
        study.add_trial(baseline_trial)

        # Optimization loop
        use_minibatch = minibatch and minibatch_size < len(valset)
        full_eval_counter = 0

        def objective(trial):
            nonlocal best_score, best_program, full_eval_counter

            trial_num = trial.number + 1
            logger.info(f"Trial {trial_num}/{num_trials}")

            # Create candidate program
            candidate = program.copy()
            params = {}

            for pred_idx, predictor in enumerate(candidate.predictors()):
                # Select instruction
                instr_idx = trial.suggest_categorical(
                    f"pred_{pred_idx}_instruction",
                    list(range(len(instruction_candidates[pred_idx])))
                )
                instruction = instruction_candidates[pred_idx][instr_idx]
                predictor.signature = predictor.signature.with_instructions(instruction)
                params[f"pred_{pred_idx}_instruction"] = instr_idx

                # Select demos
                if demo_candidates and pred_idx in demo_candidates:
                    demo_idx = trial.suggest_categorical(
                        f"pred_{pred_idx}_demos",
                        list(range(len(demo_candidates[pred_idx])))
                    )
                    predictor.demos = demo_candidates[pred_idx][demo_idx]
                    params[f"pred_{pred_idx}_demos"] = demo_idx

            # Evaluate
            if use_minibatch:
                batch = rng.sample(valset, minibatch_size)
                score = self._evaluate(candidate, batch)
            else:
                score = self._evaluate(candidate, valset)

            # Track results
            param_key = str(sorted(params.items()))
            param_score_dict[param_key].append((score, candidate.copy(), params))

            trial_logs[trial_num] = {
                "score": score,
                "params": params,
                "type": "minibatch" if use_minibatch else "full"
            }

            logger.info(f"  Score: {score:.2f}%")

            # Periodic full evaluation
            if use_minibatch:
                full_eval_counter += 1
                if full_eval_counter >= minibatch_full_eval_steps:
                    full_eval_counter = 0

                    # Find best averaging candidate
                    top_candidate, avg_score, top_params = self._get_top_candidate(
                        param_score_dict, fully_evaled
                    )

                    if top_candidate is not None:
                        logger.info(f"  Full evaluation (avg {avg_score:.2f}%)...")
                        full_score = self._evaluate(top_candidate, valset)
                        logger.info(f"  Full score: {full_score:.2f}%")

                        fully_evaled[str(sorted(top_params.items()))] = full_score

                        if full_score > best_score:
                            best_score = full_score
                            best_program = top_candidate.copy()
                            logger.info(f"  New best: {best_score:.2f}%")

            elif score > best_score:
                best_score = score
                best_program = candidate.copy()
                logger.info(f"  New best: {best_score:.2f}%")

            return score

        # Run optimization
        study.optimize(objective, n_trials=num_trials, show_progress_bar=False)

        # Final full evaluation of top candidates
        if use_minibatch:
            for _ in range(3):  # Evaluate top 3 not yet fully evaluated
                top_candidate, avg_score, top_params = self._get_top_candidate(
                    param_score_dict, fully_evaled
                )
                if top_candidate is None:
                    break

                full_score = self._evaluate(top_candidate, valset)
                fully_evaled[str(sorted(top_params.items()))] = full_score

                if full_score > best_score:
                    best_score = full_score
                    best_program = top_candidate.copy()

        logger.info(f"Optimization complete. Best score: {best_score:.2f}%")

        return best_program, best_score, trial_logs

    def _evaluate(self, program: Module, dataset: List[Example]) -> float:
        """Evaluate program on dataset, return percentage score."""
        if not dataset:
            return 0.0

        correct = 0
        for example in dataset:
            try:
                prediction = program(**example.inputs())
                score = self.metric(example, prediction)
                if isinstance(score, bool):
                    correct += 1 if score else 0
                else:
                    correct += float(score)
            except Exception as e:
                logger.warning(f"Evaluation error: {e}")

        return 100.0 * correct / len(dataset)

    def _create_default_params(
        self,
        program: Module,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
    ) -> Dict[str, int]:
        """Create default parameter dict (all zeros)."""
        params = {}
        for pred_idx in range(len(program.predictors())):
            params[f"pred_{pred_idx}_instruction"] = 0
            if demo_candidates and pred_idx in demo_candidates:
                params[f"pred_{pred_idx}_demos"] = 0
        return params

    def _get_distributions(
        self,
        program: Module,
        instruction_candidates: Dict[int, List[str]],
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
    ):
        """Create Optuna parameter distributions."""
        from optuna.distributions import CategoricalDistribution

        distributions = {}
        for pred_idx in range(len(program.predictors())):
            distributions[f"pred_{pred_idx}_instruction"] = CategoricalDistribution(
                list(range(len(instruction_candidates[pred_idx])))
            )
            if demo_candidates and pred_idx in demo_candidates:
                distributions[f"pred_{pred_idx}_demos"] = CategoricalDistribution(
                    list(range(len(demo_candidates[pred_idx])))
                )
        return distributions

    def _get_top_candidate(
        self,
        param_score_dict: Dict,
        fully_evaled: Dict,
    ) -> Tuple[Optional[Module], float, Dict]:
        """Get candidate with highest average score not yet fully evaluated."""
        results = []
        for key, values in param_score_dict.items():
            if key in fully_evaled:
                continue
            scores = [v[0] for v in values]
            avg = np.mean(scores)
            program = values[0][1]
            params = values[0][2]
            results.append((avg, program, params))

        if not results:
            return None, 0.0, {}

        results.sort(key=lambda x: x[0], reverse=True)
        return results[0][1], results[0][0], results[0][2]


# ============================================================================
# MAIN MIPROV2 CLASS
# ============================================================================

class MIPROv2:
    """
    MIPROv2 - Multi-Prompt Instruction Optimization v2.

    A complete standalone implementation that optimizes LLM programs by:
    1. Bootstrapping diverse few-shot demonstration sets
    2. Proposing instruction candidates using grounded LLM proposal
    3. Finding optimal combination via Bayesian optimization (TPE)

    Based on: "Optimizing Instructions and Demonstrations for Multi-Stage
    Language Model Programs" (arXiv:2406.11695)

    This implementation follows the paper and DSPy library closely.
    """

    def __init__(
        self,
        metric: Callable,
        llm: LLMClient,
        prompt_llm: Optional[LLMClient] = None,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: Optional[int] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        metric_threshold: Optional[float] = None,
        num_threads: int = 1,
        seed: int = 9,
        init_temperature: float = 1.0,
        verbose: bool = False,
        # Proposer settings
        program_aware: bool = True,
        data_aware: bool = True,
        tip_aware: bool = True,
        view_data_batch_size: int = 10,
    ):
        """
        Initialize MIPROv2 optimizer.

        Args:
            metric: Evaluation function (example, prediction) -> score
            llm: LLM client for running the program
            prompt_llm: LLM for generating instructions (defaults to llm)
            auto: Auto mode ("light", "medium", "heavy") or None for manual
            num_candidates: Number of candidates (required if auto=None)
            max_bootstrapped_demos: Max bootstrapped demos per set
            max_labeled_demos: Max labeled demos per set
            metric_threshold: Threshold for bootstrap acceptance
            num_threads: Parallel evaluation threads
            seed: Random seed
            init_temperature: Temperature for proposal generation
            verbose: Print detailed progress
            program_aware: Use program code in proposal
            data_aware: Use dataset summary in proposal
            tip_aware: Use tips in proposal
            view_data_batch_size: Batch size for dataset summarization
        """
        self.metric = metric
        self.llm = llm
        self.prompt_llm = prompt_llm or llm
        self.auto = auto
        self.num_candidates = num_candidates
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.metric_threshold = metric_threshold
        self.num_threads = num_threads
        self.seed = seed
        self.init_temperature = init_temperature
        self.verbose = verbose
        self.program_aware = program_aware
        self.data_aware = data_aware
        self.tip_aware = tip_aware
        self.view_data_batch_size = view_data_batch_size

        self.rng = None

        # Validate
        if auto is None and num_candidates is None:
            raise ValueError(
                "If auto=None, must provide num_candidates"
            )

    def compile(
        self,
        program: Module,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: int = 35,  # DSPy default
        minibatch_full_eval_steps: int = 5,
    ) -> Tuple[Module, float]:
        """
        Optimize the program.

        Args:
            program: Program to optimize
            trainset: Training examples
            valset: Validation examples (auto-split if None)
            num_trials: Number of optimization trials
            max_bootstrapped_demos: Override bootstrapped demos
            max_labeled_demos: Override labeled demos
            minibatch: Use minibatch evaluation
            minibatch_size: Size of evaluation minibatches
            minibatch_full_eval_steps: Full evaluation interval

        Returns:
            Tuple of (optimized_program, best_score)
        """
        # Set random seeds
        self.rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Effective demo settings
        eff_max_bootstrapped = max_bootstrapped_demos or self.max_bootstrapped_demos
        eff_max_labeled = max_labeled_demos or self.max_labeled_demos
        zeroshot_opt = (eff_max_bootstrapped == 0) and (eff_max_labeled == 0)

        # Validate and prepare datasets
        trainset, valset = self._prepare_datasets(trainset, valset)

        # Set hyperparameters based on auto mode
        (
            num_trials,
            valset,
            minibatch,
            num_instruct_candidates,
            num_fewshot_candidates,
        ) = self._set_hyperparams(
            program, num_trials, minibatch, zeroshot_opt, valset
        )

        logger.info(f"\nMIPROv2 Configuration:")
        logger.info(f"  Training set: {len(trainset)} examples")
        logger.info(f"  Validation set: {len(valset)} examples")
        logger.info(f"  Instruction candidates: {num_instruct_candidates}")
        logger.info(f"  Few-shot candidates: {num_fewshot_candidates}")
        logger.info(f"  Trials: {num_trials}")
        logger.info(f"  Minibatch: {minibatch} (size: {minibatch_size})")

        # Ensure LLM is set
        program.set_llm(self.llm)

        trial_logs = {}

        # ========== STEP 1: Bootstrap Demonstrations ==========
        logger.info("\n" + "="*50)
        logger.info("STEP 1: BOOTSTRAP DEMONSTRATIONS")
        logger.info("="*50)

        bootstrapper = DemonstrationBootstrapper(
            metric=self.metric,
            llm=self.llm,
            max_bootstrapped_demos=eff_max_bootstrapped,
            max_labeled_demos=eff_max_labeled,
            metric_threshold=self.metric_threshold,
        )

        demo_candidates = bootstrapper.bootstrap(
            program=program,
            trainset=trainset,
            num_candidate_sets=num_fewshot_candidates,
            rng=self.rng,
        )

        # ========== STEP 2: Propose Instructions ==========
        logger.info("\n" + "="*50)
        logger.info("STEP 2: PROPOSE INSTRUCTIONS")
        logger.info("="*50)

        proposer = InstructionProposer(
            llm=self.prompt_llm,
            program_aware=self.program_aware,
            data_aware=self.data_aware,
            tip_aware=self.tip_aware,
            use_instruct_history=True,
            view_data_batch_size=self.view_data_batch_size,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            verbose=self.verbose,
        )

        instruction_candidates = proposer.propose(
            program=program,
            trainset=trainset,
            demo_candidates=demo_candidates,
            num_candidates=num_instruct_candidates,
            trial_logs=trial_logs,
            rng=self.rng,
        )

        # If zero-shot, discard demos
        if zeroshot_opt:
            demo_candidates = None

        # ========== STEP 3: Bayesian Optimization ==========
        logger.info("\n" + "="*50)
        logger.info("STEP 3: BAYESIAN OPTIMIZATION")
        logger.info("="*50)

        optimizer = BayesianOptimizer(
            metric=self.metric,
            llm=self.llm,
            num_threads=self.num_threads,
            verbose=self.verbose,
        )

        best_program, best_score, trial_logs = optimizer.optimize(
            program=program,
            instruction_candidates=instruction_candidates,
            demo_candidates=demo_candidates,
            valset=valset,
            num_trials=num_trials,
            minibatch=minibatch,
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=minibatch_full_eval_steps,
            seed=self.seed,
            rng=self.rng,
        )

        # Attach metadata
        best_program._optimized = True
        best_program._score = best_score
        best_program._trial_logs = trial_logs

        logger.info("\n" + "="*50)
        logger.info(f"OPTIMIZATION COMPLETE - Best Score: {best_score:.2f}%")
        logger.info("="*50)

        return best_program, best_score

    def _prepare_datasets(
        self,
        trainset: List[Example],
        valset: Optional[List[Example]],
    ) -> Tuple[List[Example], List[Example]]:
        """Prepare and split datasets."""
        if not trainset:
            raise ValueError("trainset cannot be empty")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError("Need at least 2 examples for auto-split")

            # DSPy style split: last 80% for val, first 20% for train
            valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]

        return trainset, valset

    def _set_hyperparams(
        self,
        program: Module,
        num_trials: Optional[int],
        minibatch: bool,
        zeroshot_opt: bool,
        valset: List[Example],
    ) -> Tuple[int, List[Example], bool, int, int]:
        """Set hyperparameters based on auto mode."""

        if self.auto is None:
            if self.num_candidates is None:
                raise ValueError("num_candidates must be provided when auto is None")

            num_instruct_candidates = self.num_candidates
            num_fewshot_candidates = self.num_candidates

            if num_trials is None:
                # Calculate recommended trials (from DSPy)
                num_trials = self._calc_num_trials(program, zeroshot_opt, self.num_candidates)

            return num_trials, valset, minibatch, num_instruct_candidates, num_fewshot_candidates

        # Auto mode settings
        auto_settings = AUTO_RUN_SETTINGS[self.auto]
        n = auto_settings["n"]
        val_size = auto_settings["val_size"]

        # Limit valset size
        if len(valset) > val_size:
            valset = self.rng.sample(valset, val_size)

        # Determine minibatch
        minibatch = len(valset) > MIN_MINIBATCH_SIZE

        # Set candidates (from DSPy: instruction = N/2 if using demos, else N)
        num_instruct_candidates = n if zeroshot_opt else int(n * 0.5)
        num_fewshot_candidates = n

        # Calculate trials
        num_trials = self._calc_num_trials(program, zeroshot_opt, n)

        return num_trials, valset, minibatch, num_instruct_candidates, num_fewshot_candidates

    def _calc_num_trials(self, program: Module, zeroshot_opt: bool, n: int) -> int:
        """Calculate recommended number of trials (from DSPy)."""
        num_vars = len(program.predictors())
        if not zeroshot_opt:
            num_vars *= 2  # Account for demos + instructions

        # Formula from DSPy: max(2*M*log2(N), 1.5*N)
        num_trials = int(max(2 * num_vars * np.log2(n), 1.5 * n))
        return num_trials
