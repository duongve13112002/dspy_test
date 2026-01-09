"""
MIPROv2 - Model-based Instruction Prompt Optimization v2

This is the main optimizer class that orchestrates the three-step optimization:
1. Bootstrap Few-Shot Examples - Generate diverse demonstration candidates
2. Propose Instructions - Generate instruction candidates using LLM
3. Bayesian Optimization - Find optimal combination of instructions and demos

Based on the paper: "MIPRO: Multi-Prompt Instruction Optimization"
"""

import logging
import random
from collections import defaultdict
from typing import Any, Callable, Literal, Optional

import numpy as np

import dspy

from .bootstrap import create_n_fewshot_demo_sets
from .proposer import GroundedProposer
from .evaluator import Evaluator, eval_candidate_program
from .utils import (
    create_minibatch,
    get_signature,
    set_signature,
    get_program_with_highest_avg_score,
    print_full_program,
    save_candidate_program,
    Colors,
)

logger = logging.getLogger(__name__)

# Constants
BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT = 3
LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT = 0
MIN_MINIBATCH_SIZE = 50

# Auto-run mode settings
AUTO_RUN_SETTINGS = {
    "light": {"n": 6, "val_size": 100},
    "medium": {"n": 12, "val_size": 300},
    "heavy": {"n": 18, "val_size": 1000},
}


class MIPROv2:
    """
    MIPROv2 Optimizer - Multi-Prompt Instruction Optimization v2.

    This optimizer improves DSPy programs by:
    1. Generating diverse few-shot demonstration sets
    2. Proposing multiple instruction candidates per predictor
    3. Using Bayesian optimization to find the best combination

    The optimization is "grounded" because it uses actual data characteristics,
    program structure, and concrete examples to inform instruction generation.

    Example usage:
        ```python
        import dspy
        from mipro_v2_scratch import MIPROv2

        # Define your metric
        def my_metric(example, prediction):
            return prediction.answer == example.answer

        # Create optimizer
        optimizer = MIPROv2(
            metric=my_metric,
            auto="light",  # or "medium", "heavy"
        )

        # Compile your program
        optimized_program = optimizer.compile(
            student=my_program,
            trainset=train_data,
        )
        ```
    """

    def __init__(
        self,
        metric: Callable,
        prompt_model: Optional[Any] = None,
        task_model: Optional[Any] = None,
        teacher_settings: Optional[dict] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 4,
        auto: Literal["light", "medium", "heavy"] | None = "light",
        num_candidates: Optional[int] = None,
        num_threads: Optional[int] = None,
        max_errors: Optional[int] = None,
        seed: int = 9,
        init_temperature: float = 1.0,
        verbose: bool = False,
        track_stats: bool = True,
        log_dir: Optional[str] = None,
        metric_threshold: Optional[float] = None,
    ):
        """
        Initialize MIPROv2 optimizer.

        Args:
            metric: Function to evaluate (example, prediction) -> score
            prompt_model: LLM for generating instructions (default: dspy.settings.lm)
            task_model: LLM for running the program (default: dspy.settings.lm)
            teacher_settings: Settings for teacher model during bootstrap
            max_bootstrapped_demos: Maximum bootstrapped demos per predictor
            max_labeled_demos: Maximum labeled demos per predictor
            auto: Automatic configuration mode ("light", "medium", "heavy", or None)
            num_candidates: Number of candidates (required if auto=None)
            num_threads: Parallel evaluation threads
            max_errors: Maximum errors before stopping
            seed: Random seed for reproducibility
            init_temperature: Temperature for instruction generation
            verbose: Print detailed progress
            track_stats: Track optimization statistics
            log_dir: Directory for saving logs and programs
            metric_threshold: Minimum metric for accepting bootstrap examples
        """
        # Validate auto mode
        allowed_modes = {None, "light", "medium", "heavy"}
        if auto not in allowed_modes:
            raise ValueError(f"Invalid auto mode: {auto}. Must be one of {allowed_modes}")

        self.auto = auto
        self.num_candidates = num_candidates
        self.num_fewshot_candidates = num_candidates
        self.num_instruct_candidates = num_candidates
        self.metric = metric
        self.init_temperature = init_temperature
        self.task_model = task_model or dspy.settings.lm
        self.prompt_model = prompt_model or dspy.settings.lm
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.verbose = verbose
        self.track_stats = track_stats
        self.log_dir = log_dir
        self.teacher_settings = teacher_settings or {}
        self.num_threads = num_threads
        self.max_errors = max_errors
        self.metric_threshold = metric_threshold
        self.seed = seed
        self.rng = None

        # Track LM calls
        self.prompt_model_total_calls = 0
        self.total_calls = 0

        if not self.prompt_model or not self.task_model:
            raise ValueError(
                "Either provide prompt_model and task_model, or set default LM via dspy.configure(lm=...)"
            )

    def compile(
        self,
        student: Any,
        *,
        trainset: list,
        teacher: Any = None,
        valset: Optional[list] = None,
        num_trials: Optional[int] = None,
        max_bootstrapped_demos: Optional[int] = None,
        max_labeled_demos: Optional[int] = None,
        seed: Optional[int] = None,
        minibatch: bool = True,
        minibatch_size: int = 35,
        minibatch_full_eval_steps: int = 5,
        program_aware_proposer: bool = True,
        data_aware_proposer: bool = True,
        view_data_batch_size: int = 10,
        tip_aware_proposer: bool = True,
        fewshot_aware_proposer: bool = True,
        provide_traceback: Optional[bool] = None,
    ) -> Any:
        """
        Compile (optimize) a DSPy program.

        This is the main entry point. It runs the three-step MIPRO optimization:
        1. Bootstrap diverse few-shot demonstration sets
        2. Propose instruction candidates for each predictor
        3. Use Bayesian optimization to find best combination

        Args:
            student: DSPy program to optimize
            trainset: Training dataset
            teacher: Optional teacher program for bootstrapping
            valset: Validation dataset (auto-split from trainset if None)
            num_trials: Number of optimization trials (required if auto=None)
            max_bootstrapped_demos: Override max bootstrapped demos
            max_labeled_demos: Override max labeled demos
            seed: Override random seed
            minibatch: Use minibatch evaluation
            minibatch_size: Size of minibatches
            minibatch_full_eval_steps: Full eval every N steps
            program_aware_proposer: Include program structure in proposals
            data_aware_proposer: Include dataset summary in proposals
            view_data_batch_size: Batch size for dataset summary
            tip_aware_proposer: Include prompting tips
            fewshot_aware_proposer: Include task demos in proposals
            provide_traceback: Include error tracebacks

        Returns:
            Optimized DSPy program with best instructions and demos
        """
        # Parameter validation
        effective_max_errors = self.max_errors or dspy.settings.max_errors
        effective_max_bootstrapped = max_bootstrapped_demos or self.max_bootstrapped_demos
        effective_max_labeled = max_labeled_demos or self.max_labeled_demos

        zeroshot_opt = (effective_max_bootstrapped == 0) and (effective_max_labeled == 0)

        # Validate auto mode vs manual settings
        if self.auto is None:
            if self.num_candidates is not None and num_trials is None:
                suggested = self._set_num_trials_from_num_candidates(
                    student, zeroshot_opt, self.num_candidates
                )
                raise ValueError(
                    f"If auto=None, num_trials is required. "
                    f"Suggested: num_trials={suggested} for num_candidates={self.num_candidates}"
                )
            if self.num_candidates is None or num_trials is None:
                raise ValueError("If auto=None, both num_candidates and num_trials are required")

        if self.auto is not None and (self.num_candidates is not None or num_trials is not None):
            raise ValueError(
                "If auto is set, num_candidates and num_trials cannot be specified "
                "(they are set automatically)"
            )

        # Set random seeds
        seed = seed or self.seed
        self._set_random_seeds(seed)

        # Prepare datasets
        trainset, valset = self._set_and_validate_datasets(trainset, valset)

        # Determine candidate counts
        num_instruct = self.num_instruct_candidates or self.num_candidates
        num_fewshot = self.num_fewshot_candidates or self.num_candidates

        # Apply auto settings
        num_trials, valset, minibatch, num_instruct, num_fewshot = self._set_hyperparams_from_run_mode(
            student, num_trials, minibatch, zeroshot_opt, valset, num_instruct, num_fewshot
        )

        if self.auto:
            self._print_auto_run_settings(num_trials, minibatch, valset, num_fewshot, num_instruct)

        if minibatch and minibatch_size > len(valset):
            raise ValueError(f"minibatch_size ({minibatch_size}) exceeds valset size ({len(valset)})")

        # Initialize program and evaluator
        program = student.deepcopy()
        evaluator = Evaluator(
            devset=valset,
            metric=self.metric,
            num_threads=self.num_threads,
            max_errors=effective_max_errors,
            display_progress=True,
            provide_traceback=provide_traceback,
        )

        # ========== STEP 1: Bootstrap Few-Shot Examples ==========
        with dspy.context(lm=self.task_model):
            demo_candidates = self._bootstrap_fewshot_examples(
                program=program,
                trainset=trainset,
                seed=seed,
                teacher=teacher,
                num_fewshot_candidates=num_fewshot,
                max_bootstrapped_demos=effective_max_bootstrapped,
                max_labeled_demos=effective_max_labeled,
                max_errors=effective_max_errors,
                metric_threshold=self.metric_threshold,
            )

        # ========== STEP 2: Propose Instruction Candidates ==========
        instruction_candidates = self._propose_instructions(
            program=program,
            trainset=trainset,
            demo_candidates=demo_candidates,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware_proposer,
            data_aware=data_aware_proposer,
            tip_aware=tip_aware_proposer,
            fewshot_aware=fewshot_aware_proposer,
            num_instruct_candidates=num_instruct,
        )

        # Zero-shot mode: discard demos
        if zeroshot_opt:
            demo_candidates = None

        # ========== STEP 3: Bayesian Optimization ==========
        with dspy.context(lm=self.task_model):
            best_program = self._optimize_prompt_parameters(
                program=program,
                instruction_candidates=instruction_candidates,
                demo_candidates=demo_candidates,
                evaluator=evaluator,
                valset=valset,
                num_trials=num_trials,
                minibatch=minibatch,
                minibatch_size=minibatch_size,
                minibatch_full_eval_steps=minibatch_full_eval_steps,
                seed=seed,
            )

        return best_program

    def _set_random_seeds(self, seed: int):
        """Initialize random number generators."""
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _set_num_trials_from_num_candidates(
        self, program, zeroshot_opt: bool, num_candidates: int
    ) -> int:
        """Calculate recommended number of trials based on candidates."""
        num_vars = len(program.predictors())
        if not zeroshot_opt:
            num_vars *= 2  # Instructions + demos
        # Formula: max(c*M*log(N), 1.5*N) where c=2
        return int(max(2 * num_vars * np.log2(num_candidates), 1.5 * num_candidates))

    def _set_hyperparams_from_run_mode(
        self,
        program,
        num_trials: Optional[int],
        minibatch: bool,
        zeroshot_opt: bool,
        valset: list,
        num_instruct: Optional[int],
        num_fewshot: Optional[int],
    ):
        """Apply auto mode settings."""
        if self.auto is None:
            if num_instruct is None or num_fewshot is None:
                raise ValueError("num_candidates required when auto=None")
            return num_trials, valset, minibatch, num_instruct, num_fewshot

        settings = AUTO_RUN_SETTINGS[self.auto]

        # Sample validation set
        valset = create_minibatch(valset, settings["val_size"], self.rng)
        minibatch = len(valset) > MIN_MINIBATCH_SIZE

        # Set candidate counts
        # Fewer instruction candidates when using few-shot (spend budget on demos)
        num_instruct = settings["n"] if zeroshot_opt else int(settings["n"] * 0.5)
        num_fewshot = settings["n"]

        # Calculate trials
        num_trials = self._set_num_trials_from_num_candidates(program, zeroshot_opt, settings["n"])

        return num_trials, valset, minibatch, num_instruct, num_fewshot

    def _set_and_validate_datasets(self, trainset: list, valset: Optional[list]):
        """Validate and optionally split datasets."""
        if not trainset:
            raise ValueError("trainset cannot be empty")

        if valset is None:
            if len(trainset) < 2:
                raise ValueError("trainset must have >= 2 examples if valset not provided")
            # 80% for validation, min 1, max 1000
            valset_size = min(1000, max(1, int(len(trainset) * 0.80)))
            cutoff = len(trainset) - valset_size
            valset = trainset[cutoff:]
            trainset = trainset[:cutoff]
        elif len(valset) < 1:
            raise ValueError("valset must have >= 1 example")

        return trainset, valset

    def _print_auto_run_settings(
        self, num_trials, minibatch, valset, num_fewshot, num_instruct
    ):
        """Log auto-run configuration."""
        logger.info(
            f"\n{Colors.BOLD}RUNNING WITH {self.auto.upper()} AUTO SETTINGS:{Colors.ENDC}\n"
            f"  num_trials: {num_trials}\n"
            f"  minibatch: {minibatch}\n"
            f"  num_fewshot_candidates: {num_fewshot}\n"
            f"  num_instruct_candidates: {num_instruct}\n"
            f"  valset size: {len(valset)}\n"
        )

    def _bootstrap_fewshot_examples(
        self,
        program,
        trainset: list,
        seed: int,
        teacher,
        num_fewshot_candidates: int,
        max_bootstrapped_demos: int,
        max_labeled_demos: int,
        max_errors: Optional[int],
        metric_threshold: Optional[float],
    ) -> Optional[dict]:
        """
        Step 1: Generate diverse few-shot demonstration sets.

        This creates N different sets of demonstrations by:
        - Zero-shot (no demos)
        - Labeled only (random samples)
        - Bootstrapped (metric-validated traces)
        - Multiple shuffled bootstraps
        """
        logger.info(f"\n{Colors.BOLD}==> STEP 1: BOOTSTRAP FEWSHOT EXAMPLES <=={Colors.ENDC}")

        if max_bootstrapped_demos > 0:
            logger.info("Creating demonstration candidates for optimization and instruction proposal.\n")
        else:
            logger.info("Creating demonstrations for informing instruction proposal.\n")

        logger.info(f"Bootstrapping N={num_fewshot_candidates} demonstration sets...")

        zeroshot = max_bootstrapped_demos == 0 and max_labeled_demos == 0
        max_errors = max_errors or dspy.settings.max_errors

        demo_candidates = create_n_fewshot_demo_sets(
            student=program,
            num_candidate_sets=num_fewshot_candidates,
            trainset=trainset,
            max_labeled_demos=LABELED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_labeled_demos,
            max_bootstrapped_demos=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT if zeroshot else max_bootstrapped_demos,
            metric=self.metric,
            max_errors=max_errors,
            teacher=teacher,
            teacher_settings=self.teacher_settings,
            seed=seed,
            metric_threshold=metric_threshold,
            rng=self.rng,
        )

        return demo_candidates

    def _propose_instructions(
        self,
        program,
        trainset: list,
        demo_candidates: Optional[dict],
        view_data_batch_size: int,
        program_aware: bool,
        data_aware: bool,
        tip_aware: bool,
        fewshot_aware: bool,
        num_instruct_candidates: int,
    ) -> dict:
        """
        Step 2: Generate instruction candidates for each predictor.

        Uses the GroundedProposer to create diverse instructions informed by:
        - Dataset characteristics
        - Program structure
        - Task demonstrations
        - Prompting tips
        """
        logger.info(f"\n{Colors.BOLD}==> STEP 2: PROPOSE INSTRUCTION CANDIDATES <=={Colors.ENDC}")
        logger.info(
            "Generating instructions using dataset summary, program structure, "
            "demonstrations, and prompting tips.\n"
        )

        proposer = GroundedProposer(
            program=program,
            trainset=trainset,
            prompt_model=self.prompt_model,
            view_data_batch_size=view_data_batch_size,
            program_aware=program_aware,
            use_dataset_summary=data_aware,
            use_task_demos=fewshot_aware,
            num_demos_in_context=BOOTSTRAPPED_FEWSHOT_EXAMPLES_IN_CONTEXT,
            use_tip=tip_aware,
            set_tip_randomly=tip_aware,
            use_instruct_history=False,
            set_history_randomly=False,
            verbose=self.verbose,
            rng=self.rng,
            init_temperature=self.init_temperature,
        )

        logger.info(f"Proposing N={num_instruct_candidates} instructions per predictor...\n")

        instruction_candidates = proposer.propose_instructions_for_program(
            trainset=trainset,
            program=program,
            demo_candidates=demo_candidates,
            N=num_instruct_candidates,
            trial_logs={},
        )

        # Include original instructions as first candidate
        for i, pred in enumerate(program.predictors()):
            original = get_signature(pred).instructions
            instruction_candidates[i].insert(0, original)
            logger.info(f"Predictor {i} - {len(instruction_candidates[i])} instruction candidates")

            if self.verbose:
                for j, instr in enumerate(instruction_candidates[i]):
                    logger.info(f"  {j}: {instr[:80]}...")

        return instruction_candidates

    def _optimize_prompt_parameters(
        self,
        program,
        instruction_candidates: dict,
        demo_candidates: Optional[dict],
        evaluator: Evaluator,
        valset: list,
        num_trials: int,
        minibatch: bool,
        minibatch_size: int,
        minibatch_full_eval_steps: int,
        seed: int,
    ):
        """
        Step 3: Find optimal combination using Bayesian optimization.

        Uses Optuna's TPE sampler to efficiently search the space of:
        - Instruction index per predictor
        - Demonstration set index per predictor (if not zero-shot)

        With minibatching, we evaluate on small batches and periodically
        do full evaluations on the most promising candidates.
        """
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        logger.info(f"\n{Colors.BOLD}==> STEP 3: BAYESIAN OPTIMIZATION <=={Colors.ENDC}")
        logger.info("Finding optimal combination of instructions and demonstrations.\n")

        # Calculate total trials including full evals
        run_final_eval = 1 if num_trials % minibatch_full_eval_steps != 0 else 0
        adjusted_trials = int(
            (num_trials + num_trials // minibatch_full_eval_steps + 1 + run_final_eval)
            if minibatch else num_trials
        )

        # Evaluate default program
        logger.info(f"== Trial 1 / {adjusted_trials} - Evaluating Default Program ==")
        default_score = eval_candidate_program(
            len(valset), valset, program, evaluator, self.rng
        ).score
        logger.info(f"Default program score: {default_score}\n")

        # Initialize tracking
        trial_logs = {1: {
            "full_eval_program_path": save_candidate_program(program, self.log_dir, -1),
            "full_eval_score": default_score,
            "total_eval_calls_so_far": len(valset),
            "full_eval_program": program.deepcopy(),
        }}

        best_score = default_score
        best_program = program.deepcopy()
        total_eval_calls = len(valset)
        score_data = [{"score": best_score, "program": program.deepcopy(), "full_eval": True}]
        param_score_dict = defaultdict(list)
        fully_evaled = {}

        def objective(trial):
            nonlocal best_program, best_score, total_eval_calls

            trial_num = trial.number + 1
            if minibatch:
                logger.info(f"== Trial {trial_num} / {adjusted_trials} - Minibatch ==")
            else:
                logger.info(f"== Trial {trial_num} / {num_trials} ==")

            trial_logs[trial_num] = {}

            # Create candidate program
            candidate = program.deepcopy()

            # Select and insert parameters
            chosen, raw_params = self._select_and_insert_params(
                candidate, instruction_candidates, demo_candidates, trial, trial_logs, trial_num
            )

            if self.verbose:
                logger.info("Evaluating candidate program...")
                print_full_program(candidate)

            # Evaluate
            batch = minibatch_size if minibatch else len(valset)
            score = eval_candidate_program(batch, valset, candidate, evaluator, self.rng).score
            total_eval_calls += batch

            # Update best (only for full eval)
            if not minibatch and score > best_score:
                best_score = score
                best_program = candidate.deepcopy()
                logger.info(f"{Colors.GREEN}New best score: {score}{Colors.ENDC}")

            # Log results
            score_data.append({
                "score": score,
                "program": candidate,
                "full_eval": batch >= len(valset)
            })

            self._log_trial(
                score, best_score, batch, chosen, score_data, trial,
                adjusted_trials if minibatch else num_trials,
                trial_logs, trial_num, candidate, total_eval_calls, minibatch
            )

            # Track for averaging
            key = ",".join(map(str, chosen))
            param_score_dict[key].append((score, candidate, raw_params))

            # Periodic full evaluation (minibatch mode)
            if minibatch:
                is_eval_step = trial_num % (minibatch_full_eval_steps + 1) == 0
                is_final = trial_num == (adjusted_trials - 1)

                if is_eval_step or is_final:
                    best_score, best_program, total_eval_calls = self._perform_full_evaluation(
                        trial_num, adjusted_trials, param_score_dict, fully_evaled,
                        evaluator, valset, trial_logs, total_eval_calls, score_data,
                        best_score, best_program, study, instruction_candidates, demo_candidates
                    )

            return score

        # Create Optuna study
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # Add default as baseline
        default_params = {f"{i}_predictor_instruction": 0 for i in range(len(program.predictors()))}
        if demo_candidates:
            default_params.update({f"{i}_predictor_demos": 0 for i in range(len(program.predictors()))})

        baseline = optuna.trial.create_trial(
            params=default_params,
            distributions=self._get_param_distributions(program, instruction_candidates, demo_candidates),
            value=default_score,
        )
        study.add_trial(baseline)

        # Run optimization
        study.optimize(objective, n_trials=num_trials)

        # Attach stats to best program
        if best_program is not None and self.track_stats:
            best_program.trial_logs = trial_logs
            best_program.score = best_score
            best_program.prompt_model_total_calls = self.prompt_model_total_calls
            best_program.total_calls = self.total_calls

            sorted_candidates = sorted(score_data, key=lambda x: x["score"], reverse=True)
            best_program.mb_candidate_programs = [s for s in sorted_candidates if not s["full_eval"]]
            best_program.candidate_programs = [s for s in sorted_candidates if s["full_eval"]]

        logger.info(f"\n{Colors.GREEN}Returning best program with score {best_score}!{Colors.ENDC}")
        return best_program

    def _select_and_insert_params(
        self,
        candidate,
        instruction_candidates: dict,
        demo_candidates: Optional[dict],
        trial,
        trial_logs: dict,
        trial_num: int,
    ):
        """Select parameter values and insert into candidate program."""
        chosen = []
        raw_params = {}

        for i, predictor in enumerate(candidate.predictors()):
            # Select instruction
            instr_idx = trial.suggest_categorical(
                f"{i}_predictor_instruction",
                range(len(instruction_candidates[i]))
            )
            instruction = instruction_candidates[i][instr_idx]
            new_sig = get_signature(predictor).with_instructions(instruction)
            set_signature(predictor, new_sig)

            trial_logs[trial_num][f"{i}_predictor_instruction"] = instr_idx
            chosen.append(f"P{i}:I{instr_idx}")
            raw_params[f"{i}_predictor_instruction"] = instr_idx

            # Select demos
            if demo_candidates:
                demo_idx = trial.suggest_categorical(
                    f"{i}_predictor_demos",
                    range(len(demo_candidates[i]))
                )
                predictor.demos = demo_candidates[i][demo_idx]

                trial_logs[trial_num][f"{i}_predictor_demos"] = demo_idx
                chosen.append(f"P{i}:D{demo_idx}")
                raw_params[f"{i}_predictor_demos"] = demo_idx

        return chosen, raw_params

    def _get_param_distributions(self, program, instruction_candidates, demo_candidates):
        """Create Optuna parameter distributions."""
        from optuna.distributions import CategoricalDistribution

        distributions = {}
        for i in range(len(instruction_candidates)):
            distributions[f"{i}_predictor_instruction"] = CategoricalDistribution(
                range(len(instruction_candidates[i]))
            )
            if demo_candidates:
                distributions[f"{i}_predictor_demos"] = CategoricalDistribution(
                    range(len(demo_candidates[i]))
                )
        return distributions

    def _log_trial(
        self, score, best_score, batch_size, chosen, score_data, trial,
        total_trials, trial_logs, trial_num, candidate, total_eval_calls, minibatch
    ):
        """Log trial results."""
        if minibatch:
            trial_logs[trial_num]["mb_program_path"] = save_candidate_program(
                candidate, self.log_dir, trial_num
            )
            trial_logs[trial_num]["mb_score"] = score
            trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
            trial_logs[trial_num]["mb_program"] = candidate.deepcopy()

            mb_scores = ", ".join([f"{s['score']}" for s in score_data if not s["full_eval"]])
            full_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])

            logger.info(f"Score: {score} (minibatch={batch_size}) params={chosen}")
            logger.info(f"Minibatch scores: [{mb_scores}]")
            logger.info(f"Full eval scores: [{full_scores}]")
            logger.info(f"Best full score: {best_score}\n")
        else:
            trial_logs[trial_num]["full_eval_program_path"] = save_candidate_program(
                candidate, self.log_dir, trial_num
            )
            trial_logs[trial_num]["full_eval_score"] = score
            trial_logs[trial_num]["total_eval_calls_so_far"] = total_eval_calls
            trial_logs[trial_num]["full_eval_program"] = candidate.deepcopy()

            full_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
            logger.info(f"Score: {score} params={chosen}")
            logger.info(f"All scores: [{full_scores}]")
            logger.info(f"Best score: {best_score}\n")

    def _perform_full_evaluation(
        self,
        trial_num: int,
        adjusted_trials: int,
        param_score_dict: dict,
        fully_evaled: dict,
        evaluator: Evaluator,
        valset: list,
        trial_logs: dict,
        total_eval_calls: int,
        score_data: list,
        best_score: float,
        best_program,
        study,
        instruction_candidates: dict,
        demo_candidates: Optional[dict],
    ):
        """Perform full evaluation on most promising candidate."""
        import optuna

        logger.info(f"== Trial {trial_num + 1} / {adjusted_trials} - Full Evaluation ==")

        # Find best averaging program
        top_program, mean_score, key, params = get_program_with_highest_avg_score(
            param_score_dict, fully_evaled
        )

        logger.info(f"Evaluating top averaging program (avg={mean_score:.2f})...")

        # Full evaluation
        full_score = eval_candidate_program(
            len(valset), valset, top_program, evaluator, self.rng
        ).score

        score_data.append({"score": full_score, "program": top_program, "full_eval": True})

        # Add to Optuna study
        trial = optuna.trial.create_trial(
            params=params,
            distributions=self._get_param_distributions(
                best_program, instruction_candidates, demo_candidates
            ),
            value=full_score,
        )
        study.add_trial(trial)

        # Track
        fully_evaled[key] = {"program": top_program, "score": full_score}
        total_eval_calls += len(valset)

        trial_logs[trial_num + 1] = {
            "total_eval_calls_so_far": total_eval_calls,
            "full_eval_program_path": save_candidate_program(
                top_program, self.log_dir, trial_num + 1, note="full_eval"
            ),
            "full_eval_program": top_program,
            "full_eval_score": full_score,
        }

        # Update best
        if full_score > best_score:
            logger.info(f"{Colors.GREEN}New best full eval score: {full_score}{Colors.ENDC}")
            best_score = full_score
            best_program = top_program.deepcopy()

        full_scores = ", ".join([f"{s['score']}" for s in score_data if s["full_eval"]])
        logger.info(f"Full eval scores: [{full_scores}]")
        logger.info(f"Best full score: {best_score}\n")

        return best_score, best_program, total_eval_calls
