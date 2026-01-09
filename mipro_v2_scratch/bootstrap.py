"""
Bootstrap Few-Shot Module for MIPROv2.

This module handles the creation of few-shot demonstrations through:
1. LabeledFewShot - Simple random sampling from labeled examples
2. BootstrapFewShot - Intelligent bootstrapping with metric validation

The bootstrapping process is essential for MIPRO as it provides:
- Diverse demonstration sets for instruction proposal
- High-quality examples that pass the metric threshold
"""

import logging
import random
import threading
from typing import Any, Callable, Optional

import dspy

logger = logging.getLogger(__name__)


class LabeledFewShot:
    """
    Simple few-shot sampler that randomly selects examples from the training set.

    This is the simplest approach - just randomly sample k examples
    to use as demonstrations.
    """

    def __init__(self, k: int = 4):
        """
        Args:
            k: Number of examples to sample for few-shot demonstrations
        """
        self.k = k

    def compile(
        self,
        student,
        trainset: list,
        sample: bool = True
    ):
        """
        Compile the student program with few-shot examples.

        Args:
            student: DSPy program to compile
            trainset: Training dataset to sample from
            sample: If True, randomly sample; if False, take first k

        Returns:
            Compiled student program with demos
        """
        student = student.reset_copy()

        if sample:
            demos = random.sample(trainset, min(self.k, len(trainset)))
        else:
            demos = trainset[:self.k]

        # Assign demos to all predictors
        for predictor in student.predictors():
            predictor.demos = demos

        return student


class BootstrapFewShot:
    """
    Bootstrap Few-Shot Teleprompter.

    This class generates high-quality few-shot demonstrations by:
    1. Running examples through a teacher model
    2. Evaluating the outputs with a metric function
    3. Keeping only examples that meet the metric threshold

    The result is a set of diverse, validated demonstrations.
    """

    def __init__(
        self,
        metric: Optional[Callable] = None,
        metric_threshold: Optional[float] = None,
        teacher_settings: Optional[dict] = None,
        max_bootstrapped_demos: int = 4,
        max_labeled_demos: int = 16,
        max_rounds: int = 1,
        max_errors: Optional[int] = None,
    ):
        """
        Args:
            metric: Function to evaluate (example, prediction) -> score
            metric_threshold: Minimum score to accept a bootstrap example
            teacher_settings: Settings for the teacher model
            max_bootstrapped_demos: Maximum number of bootstrapped examples
            max_labeled_demos: Maximum number of raw labeled examples
            max_rounds: Number of bootstrap attempts per example
            max_errors: Maximum errors before stopping
        """
        self.metric = metric
        self.metric_threshold = metric_threshold
        self.teacher_settings = teacher_settings or {}
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos
        self.max_rounds = max_rounds
        self.max_errors = max_errors
        self.error_count = 0
        self.error_lock = threading.Lock()

    def compile(
        self,
        student,
        *,
        teacher=None,
        trainset: list
    ):
        """
        Compile the student program with bootstrapped demonstrations.

        Args:
            student: DSPy program to compile
            teacher: Teacher program (defaults to student copy)
            trainset: Training dataset

        Returns:
            Compiled student with bootstrapped demos
        """
        self.trainset = trainset
        self._prepare_student_and_teacher(student, teacher)
        self._prepare_predictor_mappings()
        self._bootstrap()
        self.student = self._train()
        self.student._compiled = True
        return self.student

    def _prepare_student_and_teacher(self, student, teacher):
        """Initialize student and teacher programs."""
        self.student = student.reset_copy()
        self.teacher = teacher.deepcopy() if teacher is not None else student.deepcopy()

        assert not getattr(self.student, "_compiled", False), "Student must be uncompiled"

        # If teacher is uncompiled, add labeled demos
        if self.max_labeled_demos and not getattr(self.teacher, "_compiled", False):
            teleprompter = LabeledFewShot(k=self.max_labeled_demos)
            self.teacher = teleprompter.compile(
                self.teacher.reset_copy(),
                trainset=self.trainset
            )

    def _prepare_predictor_mappings(self):
        """Create mappings between predictor names and IDs."""
        name2predictor = {}
        predictor2name = {}

        student, teacher = self.student, self.teacher

        assert len(student.predictors()) == len(teacher.predictors()), \
            "Student and teacher must have same number of predictors"

        for (name1, pred1), (name2, pred2) in zip(
            student.named_predictors(),
            teacher.named_predictors()
        ):
            assert name1 == name2, "Student and teacher must have same structure"
            name2predictor[name1] = None
            predictor2name[id(pred1)] = name1
            predictor2name[id(pred2)] = name2

        self.name2predictor = name2predictor
        self.predictor2name = predictor2name

    def _bootstrap(self, max_bootstraps: Optional[int] = None):
        """
        Run the bootstrap process to collect demonstrations.

        This is the core algorithm:
        1. Iterate through training examples
        2. Run each through teacher model
        3. Evaluate with metric
        4. Keep successful traces as demos
        """
        max_bootstraps = max_bootstraps or self.max_bootstrapped_demos
        bootstrap_attempts = 0
        bootstrapped = {}
        self.name2traces = {name: [] for name in self.name2predictor}

        for example_idx, example in enumerate(self.trainset):
            if len(bootstrapped) >= max_bootstraps:
                break

            for round_idx in range(self.max_rounds):
                bootstrap_attempts += 1
                if self._bootstrap_one_example(example, round_idx):
                    bootstrapped[example_idx] = True
                    break

        logger.info(
            f"Bootstrapped {len(bootstrapped)} traces after {example_idx + 1} examples "
            f"for up to {self.max_rounds} rounds ({bootstrap_attempts} attempts)"
        )

        # Unbootstrapped examples for validation
        self.validation = [
            x for idx, x in enumerate(self.trainset)
            if idx not in bootstrapped
        ]
        random.Random(0).shuffle(self.validation)

    def _bootstrap_one_example(self, example, round_idx: int = 0) -> bool:
        """
        Attempt to bootstrap a single example.

        Args:
            example: Training example to bootstrap
            round_idx: Current round number

        Returns:
            True if bootstrap was successful
        """
        name2traces = {}
        teacher = self.teacher
        predictor_cache = {}

        try:
            with dspy.context(trace=[], **self.teacher_settings):
                lm = dspy.settings.lm

                # Use fresh rollout with temperature=1.0 to bypass cache
                if round_idx > 0:
                    lm = lm.copy(rollout_id=round_idx, temperature=1.0)
                    new_settings = {"lm": lm}
                else:
                    new_settings = {}

                with dspy.context(**new_settings):
                    # Temporarily remove current example from demos
                    for name, predictor in teacher.named_predictors():
                        predictor_cache[name] = predictor.demos
                        predictor.demos = [x for x in predictor.demos if x != example]

                    # Run teacher
                    prediction = teacher(**example.inputs())
                    trace = dspy.settings.trace

                    # Restore demos
                    for name, predictor in teacher.named_predictors():
                        predictor.demos = predictor_cache[name]

                # Evaluate with metric
                if self.metric:
                    metric_val = self.metric(example, prediction, trace)
                    if self.metric_threshold:
                        success = metric_val >= self.metric_threshold
                    else:
                        success = bool(metric_val)
                else:
                    success = True

        except Exception as e:
            success = False
            with self.error_lock:
                self.error_count += 1
                current_errors = self.error_count

            effective_max = self.max_errors or dspy.settings.max_errors
            if current_errors >= effective_max:
                raise e
            logger.error(f"Failed to bootstrap example: {e}")

        # Collect successful traces
        if success:
            for step in trace:
                predictor, inputs, outputs = step
                demo = dspy.Example(augmented=True, **inputs, **outputs)

                try:
                    predictor_name = self.predictor2name[id(predictor)]
                except KeyError:
                    continue

                if predictor_name not in name2traces:
                    name2traces[predictor_name] = []
                name2traces[predictor_name].append(demo)

            # Update global traces
            for name, demos in name2traces.items():
                # Sample from multiple traces for same predictor
                if len(demos) > 1:
                    rng = random.Random(hash(tuple(str(d) for d in demos)))
                    demos = [rng.choice(demos[:-1]) if rng.random() < 0.5 else demos[-1]]
                self.name2traces[name].extend(demos)

        return success

    def _train(self):
        """Assign collected demos to student predictors."""
        rng = random.Random(0)
        raw_demos = self.validation

        for name, predictor in self.student.named_predictors():
            augmented_demos = self.name2traces[name][:self.max_bootstrapped_demos]

            sample_size = min(
                self.max_labeled_demos - len(augmented_demos),
                len(raw_demos)
            )
            sample_size = max(0, sample_size)

            sampled_raw = rng.sample(raw_demos, sample_size) if sample_size > 0 else []
            predictor.demos = augmented_demos + sampled_raw

        return self.student


def create_n_fewshot_demo_sets(
    student,
    num_candidate_sets: int,
    trainset: list,
    max_labeled_demos: int,
    max_bootstrapped_demos: int,
    metric: Optional[Callable],
    teacher_settings: dict,
    max_errors: Optional[int] = None,
    max_rounds: int = 1,
    metric_threshold: Optional[float] = None,
    teacher=None,
    seed: int = 0,
    rng: Optional[random.Random] = None,
) -> dict:
    """
    Create N diverse sets of few-shot demonstrations.

    This is the Step 1 of MIPRO - generating multiple candidate
    demonstration sets to explore during optimization.

    Args:
        student: DSPy program
        num_candidate_sets: Number of demo sets to create
        trainset: Training dataset
        max_labeled_demos: Max labeled demos per set
        max_bootstrapped_demos: Max bootstrapped demos per set
        metric: Evaluation metric
        teacher_settings: Teacher model settings
        max_errors: Max errors allowed
        max_rounds: Bootstrap rounds per example
        metric_threshold: Metric threshold for acceptance
        teacher: Optional teacher program
        seed: Random seed
        rng: Random number generator

    Returns:
        Dictionary mapping predictor index to list of demo sets
    """
    max_errors = max_errors or dspy.settings.max_errors
    demo_candidates = {i: [] for i in range(len(student.predictors()))}

    # Account for 3 special sets (zero-shot, labels-only, unshuffled)
    num_candidate_sets -= 3
    rng = rng or random.Random(seed)

    for set_idx in range(-3, num_candidate_sets):
        print(f"Bootstrapping set {set_idx + 4}/{num_candidate_sets + 3}")

        trainset_copy = list(trainset)

        if set_idx == -3:
            # Zero-shot set (no demos)
            program2 = student.reset_copy()

        elif set_idx == -2 and max_labeled_demos > 0:
            # Labels only (no bootstrapping)
            teleprompter = LabeledFewShot(k=max_labeled_demos)
            program2 = teleprompter.compile(student, trainset=trainset_copy)

        elif set_idx == -1:
            # Unshuffled bootstrap
            program = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                max_bootstrapped_demos=max_bootstrapped_demos,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = program.compile(student, teacher=teacher, trainset=trainset_copy)

        else:
            # Shuffled bootstrap with random size
            rng.shuffle(trainset_copy)
            size = rng.randint(1, max_bootstrapped_demos)

            teleprompter = BootstrapFewShot(
                metric=metric,
                max_errors=max_errors,
                metric_threshold=metric_threshold,
                max_bootstrapped_demos=size,
                max_labeled_demos=max_labeled_demos,
                teacher_settings=teacher_settings,
                max_rounds=max_rounds,
            )
            program2 = teleprompter.compile(
                student,
                teacher=teacher,
                trainset=trainset_copy
            )

        # Collect demos for each predictor
        for i, predictor in enumerate(student.predictors()):
            demo_candidates[i].append(program2.predictors()[i].demos)

    return demo_candidates
