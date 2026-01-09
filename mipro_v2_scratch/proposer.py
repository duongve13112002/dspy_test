"""
Instruction Proposer for MIPROv2.

This module handles Step 2 of MIPRO - generating instruction candidates.
It uses a "grounded" approach that considers:
- Dataset characteristics (data-aware)
- Program structure (program-aware)
- Task demonstrations (fewshot-aware)
- Prompting tips (tip-aware)

The proposer generates multiple instruction candidates for each predictor
in the program, which are then optimized in Step 3.
"""

import inspect
import random
from typing import Any, Callable, Optional

import dspy

from .utils import get_signature, strip_prefix, create_example_string
from .dataset_summary import create_dataset_summary


# Prompting tips for instruction generation
TIPS = {
    "none": "",
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": 'Include a persona relevant to the task (e.g., "You are a...").',
}

MAX_INSTRUCT_IN_HISTORY = 5


class DescribeProgram(dspy.Signature):
    """
    Describe what task this program solves and how it works.
    """
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program",
        prefix="PROGRAM CODE:"
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use",
        prefix="EXAMPLE OF PROGRAM IN USE:"
    )
    program_description = dspy.OutputField(
        desc="Description of what task the program solves and how",
        prefix="SUMMARY OF PROGRAM ABOVE:"
    )


class DescribeModule(dspy.Signature):
    """
    Describe the purpose of a specific module in the pipeline.
    """
    program_code = dspy.InputField(
        format=str,
        desc="Pseudocode for a language model program",
        prefix="PROGRAM CODE:"
    )
    program_example = dspy.InputField(
        format=str,
        desc="An example of the program in use",
        prefix="EXAMPLE OF PROGRAM IN USE:"
    )
    program_description = dspy.InputField(
        desc="Summary of the program's task and approach",
        prefix="SUMMARY OF PROGRAM ABOVE:"
    )
    module = dspy.InputField(
        desc="The module to describe",
        prefix="MODULE:"
    )
    module_description = dspy.OutputField(
        desc="Description of the module's role in the program",
        prefix="MODULE DESCRIPTION:"
    )


def generate_instruction_class(
    use_dataset_summary: bool = True,
    program_aware: bool = True,
    use_task_demos: bool = True,
    use_instruct_history: bool = True,
    use_tip: bool = True,
):
    """
    Dynamically create an instruction generation signature.

    This function creates a custom DSPy Signature based on which
    features are enabled. The signature is used to prompt the LLM
    to generate new instructions.

    Args:
        use_dataset_summary: Include dataset description
        program_aware: Include program code and description
        use_task_demos: Include task demonstrations
        use_instruct_history: Include previous instructions
        use_tip: Include prompting tip

    Returns:
        DSPy Predict module with the custom signature
    """

    class GenerateSingleModuleInstruction(dspy.Signature):
        """
        Use the information below to generate a new instruction that will
        prompt a Language Model to better solve the task.
        """
        pass

    # Dynamically add input fields based on enabled features
    if use_dataset_summary:
        GenerateSingleModuleInstruction.dataset_description = dspy.InputField(
            desc="Description of the dataset",
            prefix="DATASET SUMMARY:"
        )

    if program_aware:
        GenerateSingleModuleInstruction.program_code = dspy.InputField(
            format=str,
            desc="Language model program code",
            prefix="PROGRAM CODE:"
        )
        GenerateSingleModuleInstruction.program_description = dspy.InputField(
            desc="Summary of the program's task",
            prefix="PROGRAM DESCRIPTION:"
        )
        GenerateSingleModuleInstruction.module = dspy.InputField(
            desc="The module to create instruction for",
            prefix="MODULE:"
        )
        GenerateSingleModuleInstruction.module_description = dspy.InputField(
            desc="Description of the module",
            prefix="MODULE DESCRIPTION:"
        )

    # Always include task demos field (content varies)
    GenerateSingleModuleInstruction.task_demos = dspy.InputField(
        format=str,
        desc="Example inputs/outputs of the module",
        prefix="TASK DEMO(S):"
    )

    if use_instruct_history:
        GenerateSingleModuleInstruction.previous_instructions = dspy.InputField(
            format=str,
            desc="Previous instructions with their scores",
            prefix="PREVIOUS INSTRUCTIONS:"
        )

    GenerateSingleModuleInstruction.basic_instruction = dspy.InputField(
        format=str,
        desc="Basic instruction",
        prefix="BASIC INSTRUCTION:"
    )

    if use_tip:
        GenerateSingleModuleInstruction.tip = dspy.InputField(
            format=str,
            desc="Suggestion for generating the instruction",
            prefix="TIP:"
        )

    GenerateSingleModuleInstruction.proposed_instruction = dspy.OutputField(
        desc="Proposed instruction for the Language Model",
        prefix="PROPOSED INSTRUCTION:"
    )

    return dspy.Predict(GenerateSingleModuleInstruction)


def get_dspy_source_code(program) -> str:
    """
    Get the source code of a DSPy program.

    Args:
        program: DSPy program

    Returns:
        Source code string
    """
    try:
        return inspect.getsource(program.__class__)
    except (TypeError, OSError):
        # Fallback: describe the program structure
        lines = [f"class {program.__class__.__name__}:"]
        for name, pred in program.named_predictors():
            sig = get_signature(pred)
            inputs = [f for f in sig.input_fields]
            outputs = [f for f in sig.output_fields]
            lines.append(f"    {name}: {', '.join(inputs)} -> {', '.join(outputs)}")
        return "\n".join(lines)


class InstructionGenerator(dspy.Module):
    """
    Module that generates a single instruction for a predictor.

    This module orchestrates the instruction generation process by:
    1. Optionally describing the program (if program-aware)
    2. Optionally describing the specific module
    3. Generating a new instruction based on all available context
    """

    def __init__(
        self,
        program_code_string: Optional[str] = None,
        use_dataset_summary: bool = True,
        program_aware: bool = False,
        use_task_demos: bool = True,
        use_instruct_history: bool = True,
        use_tip: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.use_dataset_summary = use_dataset_summary
        self.program_aware = program_aware
        self.use_task_demos = use_task_demos
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.verbose = verbose
        self.program_code_string = program_code_string

        # Initialize sub-modules
        self.describe_program = dspy.Predict(DescribeProgram)
        self.describe_module = dspy.Predict(DescribeModule)
        self.generate_module_instruction = generate_instruction_class(
            use_dataset_summary=use_dataset_summary,
            program_aware=program_aware,
            use_task_demos=use_task_demos,
            use_instruct_history=use_instruct_history,
            use_tip=use_tip,
        )

    def forward(
        self,
        demo_candidates: Optional[dict],
        pred_i: int,
        demo_set_i: int,
        program,
        previous_instructions: str,
        data_summary: str,
        num_demos_in_context: int = 3,
        tip: Optional[str] = None,
    ):
        """
        Generate a new instruction for a specific predictor.

        Args:
            demo_candidates: Dictionary of demo sets per predictor
            pred_i: Predictor index
            demo_set_i: Demo set index to use
            program: DSPy program
            previous_instructions: History of previous instructions
            data_summary: Dataset summary
            num_demos_in_context: Number of demos to include
            tip: Optional prompting tip

        Returns:
            Prediction with proposed_instruction
        """

        def gather_examples_from_sets(candidate_sets, max_examples):
            """Gather examples from demo sets."""
            count = 0
            for candidate_set in candidate_sets:
                for example in candidate_set:
                    if "augmented" in example.keys():
                        fields = get_signature(program.predictors()[pred_i]).fields
                        yield create_example_string(fields, example)
                        count += 1
                        if count >= max_examples:
                            return

        # Get basic instruction
        basic_instruction = get_signature(program.predictors()[pred_i]).instructions
        task_demos = ""

        # Gather task demos if enabled
        if self.use_task_demos and demo_candidates:
            adjacent_sets = (
                [demo_candidates[pred_i][demo_set_i]] +
                demo_candidates[pred_i][demo_set_i + 1:] +
                demo_candidates[pred_i][:demo_set_i]
            )
            example_strings = list(gather_examples_from_sets(
                adjacent_sets, num_demos_in_context
            ))
            task_demos = "\n\n".join(example_strings) + "\n\n"

        if not task_demos.strip() or demo_set_i == 0:
            task_demos = "No task demos provided."

        # Program-aware context
        program_description = "Not available"
        module_code = "Not provided"
        module_description = "Not provided"

        if self.program_aware:
            try:
                program_description = strip_prefix(
                    self.describe_program(
                        program_code=self.program_code_string,
                        program_example=task_demos,
                    ).program_description
                )

                if self.verbose:
                    print(f"PROGRAM DESCRIPTION: {program_description}")

                # Build module signature string
                sig = get_signature(program.predictors()[pred_i])
                inputs = list(sig.input_fields.keys())
                outputs = list(sig.output_fields.keys())
                pred_class = program.predictors()[pred_i].__class__.__name__
                module_code = f"{pred_class}({', '.join(inputs)}) -> {', '.join(outputs)}"

                module_description = self.describe_module(
                    program_code=self.program_code_string,
                    program_description=program_description,
                    program_example=task_demos,
                    module=module_code,
                ).module_description

            except Exception as e:
                if self.verbose:
                    print(f"Error in program-aware: {e}. Running without.")
                self.program_aware = False

        # Generate instruction
        instruct = self.generate_module_instruction(
            dataset_description=data_summary,
            program_code=self.program_code_string,
            module=module_code,
            program_description=program_description,
            module_description=module_description,
            task_demos=task_demos,
            tip=tip,
            basic_instruction=basic_instruction,
            previous_instructions=previous_instructions,
        )

        proposed = strip_prefix(instruct.proposed_instruction)
        return dspy.Prediction(proposed_instruction=proposed)


def create_predictor_level_history_string(
    program,
    pred_i: int,
    trial_logs: dict,
    max_history: int = 5
) -> str:
    """
    Create a history string of previous instructions for a predictor.

    Args:
        program: DSPy program
        pred_i: Predictor index
        trial_logs: Dictionary of trial logs
        max_history: Maximum history entries

    Returns:
        Formatted history string
    """
    if not trial_logs:
        return ""

    history_entries = []
    for trial_num, log in list(trial_logs.items())[-max_history:]:
        key = f"{pred_i}_predictor_instruction"
        if key in log:
            score = log.get("full_eval_score", log.get("mb_score", "N/A"))
            history_entries.append(f"Trial {trial_num}: Score={score}")

    return "\n".join(history_entries) if history_entries else ""


class GroundedProposer:
    """
    Grounded Instruction Proposer.

    This class generates instruction candidates for each predictor in
    a DSPy program using a "grounded" approach that leverages:

    1. Dataset Summary - Understanding of data characteristics
    2. Program Code - Awareness of overall program structure
    3. Task Demos - Concrete examples of inputs/outputs
    4. Prompting Tips - Creative suggestions for instruction writing

    The "grounded" aspect means that instruction generation is informed
    by actual data and program structure, not just abstract prompts.
    """

    def __init__(
        self,
        prompt_model,
        program,
        trainset: list,
        view_data_batch_size: int = 10,
        use_dataset_summary: bool = True,
        program_aware: bool = True,
        use_task_demos: bool = True,
        num_demos_in_context: int = 3,
        use_instruct_history: bool = True,
        use_tip: bool = True,
        set_tip_randomly: bool = True,
        set_history_randomly: bool = True,
        verbose: bool = False,
        rng: Optional[random.Random] = None,
        init_temperature: float = 1.0,
    ):
        """
        Args:
            prompt_model: Language model for instruction generation
            program: DSPy program to optimize
            trainset: Training dataset
            view_data_batch_size: Batch size for dataset summary
            use_dataset_summary: Include dataset analysis
            program_aware: Include program structure
            use_task_demos: Include task examples
            num_demos_in_context: Number of demos per instruction
            use_instruct_history: Include previous attempts
            use_tip: Include prompting tips
            set_tip_randomly: Randomize tip selection
            set_history_randomly: Randomize history usage
            verbose: Print debug information
            rng: Random number generator
            init_temperature: Temperature for generation
        """
        self.program_aware = program_aware
        self.use_dataset_summary = use_dataset_summary
        self.use_task_demos = use_task_demos
        self.num_demos_in_context = num_demos_in_context
        self.use_instruct_history = use_instruct_history
        self.use_tip = use_tip
        self.set_tip_randomly = set_tip_randomly
        self.set_history_randomly = set_history_randomly
        self.verbose = verbose
        self.rng = rng or random.Random()
        self.prompt_model = prompt_model
        self.init_temperature = init_temperature

        # Get program source code
        self.program_code_string = None
        if self.program_aware:
            try:
                self.program_code_string = get_dspy_source_code(program)
                if self.verbose:
                    print(f"SOURCE CODE:\n{self.program_code_string}")
            except Exception as e:
                print(f"Error getting source code: {e}. Running without program awareness.")
                self.program_aware = False

        # Generate dataset summary
        self.data_summary = None
        if self.use_dataset_summary:
            try:
                self.data_summary = create_dataset_summary(
                    trainset=trainset,
                    view_data_batch_size=view_data_batch_size,
                    prompt_model=prompt_model,
                    verbose=verbose
                )
                if self.verbose:
                    print(f"DATA SUMMARY: {self.data_summary}")
            except Exception as e:
                print(f"Error generating data summary: {e}. Running without.")
                self.use_dataset_summary = False

    def propose_instructions_for_program(
        self,
        trainset: list,
        program,
        demo_candidates: Optional[dict],
        trial_logs: dict,
        N: int,
    ) -> dict:
        """
        Generate instruction candidates for all predictors.

        Args:
            trainset: Training dataset
            program: DSPy program
            demo_candidates: Demo sets per predictor
            trial_logs: History of trials
            N: Number of instructions to generate per predictor

        Returns:
            Dictionary mapping predictor index to list of instructions
        """
        proposed_instructions = {}

        # Randomly decide whether to use history
        if self.set_history_randomly:
            use_history = self.rng.random() < 0.5
            self.use_instruct_history = use_history
            if self.verbose:
                print(f"Using instruction history: {use_history}")

        # Handle missing demo candidates
        if not demo_candidates:
            if self.verbose:
                print("No demo candidates. Running without task demos.")
            self.use_task_demos = False
            num_demos = N
        else:
            num_demos = max(len(demo_candidates[0]), 1)

        # Generate instructions for each predictor
        for pred_i, predictor in enumerate(program.predictors()):
            proposed_instructions[pred_i] = []

            for demo_set_i in range(min(N, num_demos)):
                # Select tip
                selected_tip = None
                if self.set_tip_randomly:
                    tip_key = self.rng.choice(list(TIPS.keys()))
                    selected_tip = TIPS[tip_key]
                    self.use_tip = bool(selected_tip)
                    if self.verbose:
                        print(f"Selected tip: {tip_key}")

                # Generate instruction
                instruction = self.propose_instruction_for_predictor(
                    program=program,
                    predictor=predictor,
                    pred_i=pred_i,
                    demo_candidates=demo_candidates,
                    demo_set_i=demo_set_i,
                    trial_logs=trial_logs,
                    tip=selected_tip,
                )
                proposed_instructions[pred_i].append(instruction)

        return proposed_instructions

    def propose_instruction_for_predictor(
        self,
        program,
        predictor,
        pred_i: int,
        demo_candidates: Optional[dict],
        demo_set_i: int,
        trial_logs: dict,
        tip: Optional[str] = None,
    ) -> str:
        """
        Generate a single instruction for a specific predictor.

        Args:
            program: DSPy program
            predictor: Target predictor
            pred_i: Predictor index
            demo_candidates: Demo sets
            demo_set_i: Demo set index
            trial_logs: Trial history
            tip: Prompting tip

        Returns:
            Generated instruction string
        """
        # Build instruction history
        instruction_history = create_predictor_level_history_string(
            program, pred_i, trial_logs, MAX_INSTRUCT_IN_HISTORY
        )

        # Create instruction generator
        generator = InstructionGenerator(
            program_code_string=self.program_code_string,
            use_dataset_summary=self.use_dataset_summary,
            program_aware=self.program_aware,
            use_task_demos=self.use_task_demos and demo_candidates,
            use_instruct_history=self.use_instruct_history and instruction_history,
            use_tip=self.use_tip,
            verbose=self.verbose,
        )

        # Generate with unique rollout to bypass cache
        rollout_lm = self.prompt_model.copy(
            rollout_id=self.rng.randint(0, 10**9),
            temperature=self.init_temperature,
        )

        with dspy.context(lm=rollout_lm):
            result = generator(
                demo_candidates=demo_candidates,
                pred_i=pred_i,
                demo_set_i=demo_set_i,
                program=program,
                data_summary=self.data_summary,
                previous_instructions=instruction_history,
                num_demos_in_context=self.num_demos_in_context,
                tip=tip,
            )

        proposed = strip_prefix(result.proposed_instruction)

        if self.verbose:
            print(f"PROPOSED INSTRUCTION: {proposed}")

        return proposed
