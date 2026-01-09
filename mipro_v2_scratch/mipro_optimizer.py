"""
MIPROv2 - Standalone Multi-Prompt Instruction Optimization v2

A complete implementation of MIPRO without any DSPy dependencies.
Based on: "Optimizing Instructions and Demonstrations for Multi-Stage
Language Model Programs" (arXiv:2406.11695)

The algorithm has 3 main steps:
1. Bootstrap Demonstrations - Generate diverse few-shot example sets
2. Propose Instructions - Generate instruction candidates using LLM
3. Bayesian Optimization - Find optimal combination using TPE
"""

import random
import logging
from copy import deepcopy
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from .llm_client import LLMClient
from .example import Example, Prediction
from .module import Module, Predictor, Signature

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

# Auto settings based on paper recommendations
# Paper uses N=10-70 candidates and 20-50 trials depending on task complexity
AUTO_RUN_SETTINGS = {
    "light": {"num_candidates": 10, "val_size": 100, "num_trials": 20},
    "medium": {"num_candidates": 30, "val_size": 300, "num_trials": 35},  # Most common in paper
    "heavy": {"num_candidates": 50, "val_size": 1000, "num_trials": 50},
}

# Prompting tips for instruction generation (Exact from paper - Appendix C.2)
# These tips are sampled randomly during instruction proposal to increase diversity
INSTRUCTION_TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": "Provide the LM with a persona that is relevant to the task (ie. 'You are a...')",
}

# Default N values from paper Table 4 (task-dependent)
# For 0-Shot MIPRO: ranges 15-70 depending on task
# For MIPRO with demos: ranges 10-70 depending on task
# Common defaults: 30 for most tasks


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
# STEP 2: PROPOSE INSTRUCTIONS
# ============================================================================

class InstructionProposer:
    """
    Generates instruction candidates using LLM-based proposal.

    The "grounded" approach uses:
    - Dataset summary (data characteristics)
    - Program structure (code awareness)
    - Task demonstrations (concrete examples)
    - Prompting tips (creative suggestions)
    """

    # Prompt templates
    DATASET_SUMMARY_PROMPT = """Analyze these examples from a dataset and describe:
1. What type of task this data is for
2. Key patterns you observe
3. Input/output characteristics

Examples:
{examples}

Provide a 2-3 sentence summary:"""

    INSTRUCTION_PROPOSAL_PROMPT = """You are creating an instruction for a Language Model module.

{dataset_info}

{demo_info}

The module has these fields:
- Inputs: {input_fields}
- Outputs: {output_fields}

Current basic instruction: {basic_instruction}

{tip}

Generate a new, improved instruction that will help the LM perform this task better.
Focus on clarity, specificity, and task guidance.

New instruction:"""

    def __init__(
        self,
        llm: LLMClient,
        temperature: float = 1.0,
        verbose: bool = False,
    ):
        self.llm = llm
        self.temperature = temperature
        self.verbose = verbose

    def propose(
        self,
        program: Module,
        trainset: List[Example],
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
        num_candidates: int,
        rng: random.Random,
    ) -> Dict[int, List[str]]:
        """
        Generate instruction candidates for all predictors.

        Args:
            program: The LLM program
            trainset: Training examples
            demo_candidates: Demo sets from bootstrapping
            num_candidates: Instructions to generate per predictor
            rng: Random number generator

        Returns:
            Dict mapping predictor index to list of instructions
        """
        logger.info(f"Proposing {num_candidates} instructions per predictor...")

        # Generate dataset summary
        dataset_summary = self._generate_dataset_summary(trainset)

        instruction_candidates = {}

        for pred_idx, (name, predictor) in enumerate(program.named_predictors()):
            logger.info(f"  Predictor {pred_idx} ({name})")

            # Start with original instruction
            original = predictor.signature.instructions or "Complete the task."
            candidates = [original]

            # Generate new candidates
            for i in range(num_candidates - 1):
                # Select random tip
                tip_key = rng.choice(list(INSTRUCTION_TIPS.keys()))
                tip = INSTRUCTION_TIPS[tip_key]

                # Get demos for context
                demo_str = self._format_demos(
                    predictor, demo_candidates, pred_idx, i, rng
                )

                # Generate instruction
                instruction = self._propose_single(
                    predictor=predictor,
                    dataset_summary=dataset_summary,
                    demo_str=demo_str,
                    basic_instruction=original,
                    tip=tip,
                )

                candidates.append(instruction)

                if self.verbose:
                    logger.info(f"    Candidate {i + 1}: {instruction[:60]}...")

            instruction_candidates[pred_idx] = candidates

        return instruction_candidates

    def _generate_dataset_summary(self, trainset: List[Example]) -> str:
        """Generate a summary of the dataset."""
        # Sample a few examples
        sample_size = min(5, len(trainset))
        samples = trainset[:sample_size]

        # Format examples
        examples_str = "\n\n".join(
            "\n".join(f"{k}: {v}" for k, v in ex.items() if k != "augmented")
            for ex in samples
        )

        prompt = self.DATASET_SUMMARY_PROMPT.format(examples=examples_str)

        try:
            response = self.llm.generate(prompt, temperature=0.7, max_tokens=200)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Failed to generate dataset summary: {e}")
            return "Dataset characteristics not available."

    def _format_demos(
        self,
        predictor: Predictor,
        demo_candidates: Optional[Dict[int, List[List[Example]]]],
        pred_idx: int,
        candidate_idx: int,
        rng: random.Random,
    ) -> str:
        """Format demonstrations for context."""
        if not demo_candidates or pred_idx not in demo_candidates:
            return "No demonstrations available."

        demo_sets = demo_candidates[pred_idx]
        if not demo_sets:
            return "No demonstrations available."

        # Pick a demo set
        set_idx = candidate_idx % len(demo_sets)
        demos = demo_sets[set_idx]

        if not demos:
            return "No demonstrations available."

        # Format up to 3 demos
        demo_strs = []
        for demo in demos[:3]:
            parts = []
            for field in predictor.signature.input_fields:
                if field.name in demo:
                    parts.append(f"{field.prefix} {demo[field.name]}")
            for field in predictor.signature.output_fields:
                if field.name in demo:
                    parts.append(f"{field.prefix} {demo[field.name]}")
            demo_strs.append("\n".join(parts))

        return "Example demonstrations:\n\n" + "\n\n---\n\n".join(demo_strs)

    def _propose_single(
        self,
        predictor: Predictor,
        dataset_summary: str,
        demo_str: str,
        basic_instruction: str,
        tip: str,
    ) -> str:
        """Generate a single instruction candidate."""
        # Format field info
        input_fields = ", ".join(f.name for f in predictor.signature.input_fields)
        output_fields = ", ".join(f.name for f in predictor.signature.output_fields)

        # Build prompt
        prompt = self.INSTRUCTION_PROPOSAL_PROMPT.format(
            dataset_info=f"Dataset summary: {dataset_summary}",
            demo_info=demo_str,
            input_fields=input_fields,
            output_fields=output_fields,
            basic_instruction=basic_instruction,
            tip=f"Tip: {tip}" if tip else "",
        )

        try:
            response = self.llm.generate(
                prompt,
                temperature=self.temperature,
                max_tokens=300
            )
            instruction = response.content.strip()

            # Clean up common prefixes
            for prefix in ["New instruction:", "Instruction:", "New Instruction:"]:
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
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
        rng: random.Random,
    ) -> Tuple[Module, float, Dict]:
        """
        Run Bayesian optimization to find best parameter combination.

        Args:
            program: Base program to optimize
            instruction_candidates: Instructions per predictor
            demo_candidates: Demo sets per predictor
            valset: Validation set
            num_trials: Number of optimization trials
            minibatch_size: Size of evaluation minibatches
            minibatch_full_eval_steps: Full eval interval
            seed: Random seed
            rng: Random number generator

        Returns:
            Tuple of (best_program, best_score, trial_logs)
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
        use_minibatch = minibatch_size < len(valset)
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
    2. Proposing instruction candidates using LLM
    3. Finding optimal combination via Bayesian optimization

    Based on: "Optimizing Instructions and Demonstrations for Multi-Stage
    Language Model Programs" (arXiv:2406.11695)

    Usage:
        ```python
        from mipro_v2_scratch import MIPROv2, LLMClient, SimplePredictor

        # Create LLM client
        llm = LLMClient(api_key="sk-...", model="gpt-4o-mini")

        # Create program
        program = SimplePredictor("question -> answer")
        program.set_llm(llm)

        # Define metric
        def exact_match(example, prediction):
            return prediction.answer.lower() == example.answer.lower()

        # Create optimizer
        optimizer = MIPROv2(
            metric=exact_match,
            llm=llm,
            auto="light"
        )

        # Optimize
        best_program, score = optimizer.compile(
            program=program,
            trainset=train_examples,
        )

        print(f"Best score: {score}%")
        ```
    """

    def __init__(
        self,
        metric: Callable,
        llm: LLMClient,
        prompt_llm: Optional[LLMClient] = None,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: Optional[int] = None,
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        metric_threshold: Optional[float] = None,
        num_threads: int = 1,
        seed: int = 42,
        verbose: bool = False,
    ):
        """
        Initialize MIPROv2 optimizer.

        Args:
            metric: Evaluation function (example, prediction) -> score
            llm: LLM client for running the program
            prompt_llm: LLM for generating instructions (defaults to llm)
            auto: Auto mode ("light", "medium", "heavy") or None for manual
            num_candidates: Number of candidates (required if auto=None)
            num_trials: Number of trials (required if auto=None)
            max_bootstrapped_demos: Max bootstrapped demos per set
            max_labeled_demos: Max labeled demos per set
            metric_threshold: Threshold for bootstrap acceptance
            num_threads: Parallel evaluation threads
            seed: Random seed
            verbose: Print detailed progress
        """
        self.metric = metric
        self.llm = llm
        self.prompt_llm = prompt_llm or llm
        self.auto = auto
        self.num_candidates = num_candidates
        self.num_trials = num_trials
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.metric_threshold = metric_threshold
        self.num_threads = num_threads
        self.seed = seed
        self.verbose = verbose

        # Validate
        if auto is None and (num_candidates is None or num_trials is None):
            raise ValueError(
                "If auto=None, must provide num_candidates and num_trials"
            )

    def compile(
        self,
        program: Module,
        trainset: List[Example],
        valset: Optional[List[Example]] = None,
        minibatch_size: int = 25,
        minibatch_full_eval_steps: int = 5,
    ) -> Tuple[Module, float]:
        """
        Optimize the program.

        Args:
            program: Program to optimize
            trainset: Training examples
            valset: Validation examples (auto-split if None)
            minibatch_size: Size of evaluation minibatches
            minibatch_full_eval_steps: Full evaluation interval

        Returns:
            Tuple of (optimized_program, best_score)
        """
        # Initialize RNG
        rng = random.Random(self.seed)
        np.random.seed(self.seed)

        # Apply auto settings
        if self.auto:
            settings = AUTO_RUN_SETTINGS[self.auto]
            num_candidates = settings["num_candidates"]
            num_trials = settings["num_trials"]
            val_size = settings["val_size"]
        else:
            num_candidates = self.num_candidates
            num_trials = self.num_trials
            val_size = 1000

        # Split datasets
        trainset, valset = self._prepare_datasets(trainset, valset, val_size, rng)

        logger.info(f"Training set: {len(trainset)} examples")
        logger.info(f"Validation set: {len(valset)} examples")
        logger.info(f"Candidates: {num_candidates}, Trials: {num_trials}")

        # Ensure LLM is set
        program.set_llm(self.llm)

        # ========== STEP 1: Bootstrap Demonstrations ==========
        logger.info("\n" + "="*50)
        logger.info("STEP 1: BOOTSTRAP DEMONSTRATIONS")
        logger.info("="*50)

        bootstrapper = DemonstrationBootstrapper(
            metric=self.metric,
            llm=self.llm,
            max_bootstrapped_demos=self.max_bootstrapped_demos,
            max_labeled_demos=self.max_labeled_demos,
            metric_threshold=self.metric_threshold,
        )

        demo_candidates = bootstrapper.bootstrap(
            program=program,
            trainset=trainset,
            num_candidate_sets=num_candidates,
            rng=rng,
        )

        # ========== STEP 2: Propose Instructions ==========
        logger.info("\n" + "="*50)
        logger.info("STEP 2: PROPOSE INSTRUCTIONS")
        logger.info("="*50)

        proposer = InstructionProposer(
            llm=self.prompt_llm,
            temperature=1.0,
            verbose=self.verbose,
        )

        instruction_candidates = proposer.propose(
            program=program,
            trainset=trainset,
            demo_candidates=demo_candidates,
            num_candidates=num_candidates,
            rng=rng,
        )

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
            minibatch_size=minibatch_size,
            minibatch_full_eval_steps=minibatch_full_eval_steps,
            seed=self.seed,
            rng=rng,
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
        val_size: int,
        rng: random.Random,
    ) -> Tuple[List[Example], List[Example]]:
        """Prepare and split datasets."""
        if not trainset:
            raise ValueError("trainset cannot be empty")

        if valset is None:
            # Auto-split
            if len(trainset) < 2:
                raise ValueError("Need at least 2 examples for auto-split")

            all_data = list(trainset)
            rng.shuffle(all_data)

            split_size = min(val_size, int(len(all_data) * 0.8))
            split_size = max(1, split_size)

            valset = all_data[:split_size]
            trainset = all_data[split_size:]

            if not trainset:
                trainset = all_data[:max(1, len(all_data) // 5)]

        # Limit validation size
        if len(valset) > val_size:
            valset = rng.sample(valset, val_size)

        return trainset, valset
