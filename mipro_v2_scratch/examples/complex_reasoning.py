#!/usr/bin/env python3
"""
Complex Example: Multi-Step Reasoning for Word Problems

This example demonstrates:
1. Multi-predictor pipeline (Chain-of-Thought)
2. Complex metric with partial scoring
3. Realistic word problems requiring reasoning
4. Save/load optimized prompts

Task: Solve word problems that require:
- Information extraction
- Multi-step reasoning
- Mathematical computation
"""

import os
import re
import logging
from typing import Optional

# Setup path for local import
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mipro_v2_scratch import (
    LLM, GeminiLM, OpenAILM,
    MIPROv2,
    Example, Prediction,
    Module, Predictor, Signature,
    setup_logging
)

# ============================================
# 1. Setup Logging
# ============================================

setup_logging(level="INFO", format_style="simple")
logging.basicConfig(level=logging.INFO)

# ============================================
# 2. Define Multi-Step Reasoning Module
# ============================================

class ChainOfThoughtSolver(Module):
    """
    A multi-step reasoning module for word problems.

    Pipeline:
    1. Extract: Identify key information (numbers, entities, relationships)
    2. Plan: Create a step-by-step plan to solve the problem
    3. Solve: Execute the plan and compute the answer
    """

    def __init__(self):
        super().__init__()

        # Step 1: Extract key information
        self.extract = Predictor(
            Signature.from_string(
                "problem: word problem text -> "
                "entities: key entities and their values, "
                "relationships: how entities relate to each other",
                instructions="Extract all relevant numbers, entities, and their relationships from the problem."
            )
        )

        # Step 2: Create solution plan
        self.plan = Predictor(
            Signature.from_string(
                "problem: word problem text, "
                "entities: extracted entities, "
                "relationships: entity relationships -> "
                "steps: numbered step-by-step solution plan",
                instructions="Create a clear step-by-step plan to solve this problem."
            )
        )

        # Step 3: Execute and solve
        self.solve = Predictor(
            Signature.from_string(
                "problem: word problem text, "
                "steps: solution plan -> "
                "reasoning: show your work, "
                "answer: final numerical answer only",
                instructions="Follow the plan, show your reasoning, and provide the final answer."
            )
        )

    def forward(self, problem: str) -> Prediction:
        # Step 1: Extract information
        extraction = self.extract(problem=problem)

        # Step 2: Plan solution
        plan = self.plan(
            problem=problem,
            entities=extraction.entities,
            relationships=extraction.relationships
        )

        # Step 3: Solve
        solution = self.solve(
            problem=problem,
            steps=plan.steps
        )

        # Combine all outputs
        return Prediction(
            entities=extraction.entities,
            relationships=extraction.relationships,
            steps=plan.steps,
            reasoning=solution.reasoning,
            answer=solution.answer
        )


# ============================================
# 3. Define Complex Metric
# ============================================

def extract_number(text: str) -> Optional[float]:
    """Extract the first number from text."""
    if not text:
        return None
    # Handle fractions
    fraction_match = re.search(r'(\d+)/(\d+)', str(text))
    if fraction_match:
        return float(fraction_match.group(1)) / float(fraction_match.group(2))
    # Handle decimals and integers
    num_match = re.search(r'-?\d+\.?\d*', str(text))
    if num_match:
        return float(num_match.group())
    return None


def reasoning_metric(example: Example, prediction: Prediction) -> float:
    """
    Complex metric that scores:
    - Answer correctness (60%)
    - Reasoning quality (40%)

    Returns score between 0 and 1.
    """
    score = 0.0

    # 1. Check answer correctness (60%)
    expected = extract_number(str(example.answer))
    predicted = extract_number(str(prediction.answer))

    if expected is not None and predicted is not None:
        # Allow small tolerance for floating point
        if abs(expected - predicted) < 0.01:
            score += 0.6
        elif abs(expected - predicted) / max(abs(expected), 1) < 0.05:
            # Within 5% - partial credit
            score += 0.3

    # 2. Check reasoning quality (40%)
    reasoning = str(getattr(prediction, 'reasoning', ''))
    steps = str(getattr(prediction, 'steps', ''))

    # Has steps/reasoning
    if len(reasoning) > 20 or len(steps) > 20:
        score += 0.1

    # Contains numbers (shows calculation)
    if re.search(r'\d+', reasoning):
        score += 0.1

    # Contains mathematical operations
    if re.search(r'[+\-*/=]', reasoning):
        score += 0.1

    # Multi-step reasoning (has multiple sentences/lines)
    if len(reasoning.split('.')) > 2 or len(steps.split('\n')) > 1:
        score += 0.1

    return score


def simple_accuracy(example: Example, prediction: Prediction) -> bool:
    """Simple binary accuracy for comparison."""
    expected = extract_number(str(example.answer))
    predicted = extract_number(str(prediction.answer))

    if expected is None or predicted is None:
        return False

    return abs(expected - predicted) < 0.01


# ============================================
# 4. Create Training Data
# ============================================

# Word problems requiring multi-step reasoning
trainset = [
    # Basic arithmetic in context
    Example(
        problem="Sarah has 24 apples. She gives 1/3 of them to her friend and then buys 8 more. How many apples does Sarah have now?",
        answer="24"
    ).with_inputs("problem"),

    Example(
        problem="A train travels at 60 mph for 2.5 hours, then at 80 mph for 1.5 hours. What is the total distance traveled?",
        answer="270"
    ).with_inputs("problem"),

    Example(
        problem="If a rectangle has a perimeter of 36 cm and its length is twice its width, what is its area?",
        answer="72"
    ).with_inputs("problem"),

    Example(
        problem="John had $150. He spent 40% on books and 25% of the remainder on food. How much money does he have left?",
        answer="67.5"
    ).with_inputs("problem"),

    Example(
        problem="A store offers a 20% discount on a $80 item, then adds 10% tax on the discounted price. What is the final price?",
        answer="70.4"
    ).with_inputs("problem"),

    # Rate and ratio problems
    Example(
        problem="Three workers can complete a job in 12 days. How many days would it take for 4 workers to complete the same job?",
        answer="9"
    ).with_inputs("problem"),

    Example(
        problem="A mixture contains water and alcohol in the ratio 3:2. If there are 15 liters of water, how many liters of alcohol are there?",
        answer="10"
    ).with_inputs("problem"),

    Example(
        problem="If 5 machines produce 100 items in 4 hours, how many items can 8 machines produce in 5 hours?",
        answer="200"
    ).with_inputs("problem"),

    # Age problems
    Example(
        problem="Tom is 3 times as old as his son. In 12 years, Tom will be twice as old as his son. How old is Tom now?",
        answer="36"
    ).with_inputs("problem"),

    Example(
        problem="The sum of ages of a father and son is 56 years. After 4 years, the father will be 3 times as old as the son. How old is the son now?",
        answer="12"
    ).with_inputs("problem"),

    # Geometry
    Example(
        problem="A circular pool has a radius of 7 meters. A path of width 2 meters surrounds it. What is the area of the path? (Use pi=22/7)",
        answer="100.57"
    ).with_inputs("problem"),

    Example(
        problem="A cone has a base radius of 3 cm and height of 4 cm. What is its slant height?",
        answer="5"
    ).with_inputs("problem"),

    # Percentage and profit
    Example(
        problem="A shopkeeper buys an item for $200 and wants to make a 25% profit after giving a 10% discount. What should be the marked price?",
        answer="277.78"
    ).with_inputs("problem"),

    Example(
        problem="The price of a laptop increased by 20% and then decreased by 20%. If the original price was $1000, what is the final price?",
        answer="960"
    ).with_inputs("problem"),

    # Time and work
    Example(
        problem="Pipe A fills a tank in 6 hours, Pipe B fills it in 4 hours. If both pipes are opened together, how long to fill the tank?",
        answer="2.4"
    ).with_inputs("problem"),

    # Speed/distance
    Example(
        problem="A car travels from A to B at 40 km/h and returns at 60 km/h. If the distance is 120 km, what is the average speed for the round trip?",
        answer="48"
    ).with_inputs("problem"),

    # Complex multi-step
    Example(
        problem="A company has 120 employees. 60% are men. 25% of men and 40% of women have advanced degrees. How many employees have advanced degrees?",
        answer="37"
    ).with_inputs("problem"),

    Example(
        problem="In a class, the ratio of boys to girls is 3:2. If 5 girls join the class, the ratio becomes 6:5. How many boys are in the class?",
        answer="30"
    ).with_inputs("problem"),

    # Interest problems
    Example(
        problem="$5000 is invested at 8% compound interest annually. What is the amount after 2 years?",
        answer="5832"
    ).with_inputs("problem"),

    Example(
        problem="A sum doubles in 5 years at simple interest. What is the rate of interest per annum?",
        answer="20"
    ).with_inputs("problem"),
]

# Test set (unseen problems)
testset = [
    Example(
        problem="A farmer has chickens and cows. If there are 50 heads and 140 legs, how many chickens are there?",
        answer="30"
    ).with_inputs("problem"),

    Example(
        problem="Two trains start from the same station in opposite directions. One travels at 50 km/h and the other at 70 km/h. After how many hours will they be 360 km apart?",
        answer="3"
    ).with_inputs("problem"),

    Example(
        problem="A tank is filled by pipe A in 10 hours and emptied by pipe B in 15 hours. If both are opened, how long to fill the tank?",
        answer="30"
    ).with_inputs("problem"),
]


# ============================================
# 5. Main Execution
# ============================================

def main():
    # Setup LLM - Choose one:

    # Option A: Google Gemini
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        print("Example: export GEMINI_API_KEY='your-api-key'")
        return

    lm = GeminiLM(model="gemini-2.0-flash", temperature=0.7)

    # Option B: OpenAI
    # lm = OpenAILM(model="gpt-4o-mini", temperature=0.7)

    # ----------------------------------------
    # Test BEFORE optimization
    # ----------------------------------------
    print("\n" + "="*60)
    print("TESTING BEFORE OPTIMIZATION")
    print("="*60)

    module = ChainOfThoughtSolver()
    module.set_llm(lm)

    correct = 0
    for ex in testset[:3]:
        print(f"\nProblem: {ex.problem[:80]}...")
        try:
            result = module(problem=ex.problem)
            is_correct = simple_accuracy(ex, result)
            correct += is_correct

            print(f"Expected: {ex.answer}")
            print(f"Got: {result.answer}")
            print(f"Correct: {'Yes' if is_correct else 'No'}")
        except Exception as e:
            print(f"Error: {e}")

    print(f"\nBaseline accuracy: {correct}/{len(testset[:3])}")

    # ----------------------------------------
    # Optimize with MIPROv2
    # ----------------------------------------
    print("\n" + "="*60)
    print("STARTING MIPRO v2 OPTIMIZATION")
    print("="*60)

    optimizer = MIPROv2(
        metric=reasoning_metric,  # Complex metric with partial scoring
        llm=lm,
        prompt_llm=lm,
        auto="light",  # Use "medium" or "heavy" for better results
        max_bootstrapped_demos=3,
        max_labeled_demos=2,
        verbose=True,
        seed=42,
    )

    # Provide explicit valset since trainset is not huge
    valset = trainset[15:]  # Last 5 for validation
    train_subset = trainset[:15]  # First 15 for training

    optimized_module, best_score = optimizer.compile(
        program=module,
        trainset=train_subset,
        valset=valset,
        minibatch=False,  # Full evaluation for small dataset
    )

    print(f"\nOptimization complete! Best score: {best_score:.2f}%")

    # ----------------------------------------
    # Test AFTER optimization
    # ----------------------------------------
    print("\n" + "="*60)
    print("TESTING AFTER OPTIMIZATION")
    print("="*60)

    correct = 0
    total_score = 0.0

    for ex in testset:
        print(f"\nProblem: {ex.problem[:80]}...")
        try:
            result = optimized_module(problem=ex.problem)
            score = reasoning_metric(ex, result)
            is_correct = simple_accuracy(ex, result)

            correct += is_correct
            total_score += score

            print(f"Expected: {ex.answer}")
            print(f"Got: {result.answer}")
            print(f"Score: {score:.2f}")

            # Show reasoning for interesting cases
            if hasattr(result, 'reasoning') and result.reasoning:
                print(f"Reasoning: {result.reasoning[:150]}...")

        except Exception as e:
            print(f"Error: {e}")

    print(f"\n{'='*40}")
    print(f"Final Results:")
    print(f"  Accuracy: {correct}/{len(testset)}")
    print(f"  Avg Score: {total_score/len(testset):.2f}")

    # ----------------------------------------
    # Inspect optimized prompts
    # ----------------------------------------
    print("\n" + "="*60)
    print("OPTIMIZED PROMPTS")
    print("="*60)

    for name, pred in optimized_module.named_predictors():
        print(f"\n[{name}]")
        print(f"Instruction: {pred.signature.instructions[:200]}...")
        print(f"Demos: {len(pred.demos)} examples")

        if pred.demos:
            print("Demo preview:")
            demo = pred.demos[0]
            for key in list(demo._data.keys())[:2]:
                val = str(demo._data[key])[:80]
                print(f"  {key}: {val}...")

    # ----------------------------------------
    # Save optimized module
    # ----------------------------------------
    save_path = "optimized_cot_solver.json"
    optimized_module.save(save_path)
    print(f"\nSaved optimized module to: {save_path}")

    # ----------------------------------------
    # Demo: Load and use saved module
    # ----------------------------------------
    print("\n" + "="*60)
    print("DEMO: LOADING SAVED MODULE")
    print("="*60)

    # Create fresh module and load
    fresh_module = ChainOfThoughtSolver()
    fresh_module.set_llm(lm)
    fresh_module.load(save_path)

    test_problem = "If 8 workers can build a wall in 10 days, how many days will 5 workers take to build the same wall?"
    result = fresh_module(problem=test_problem)

    print(f"Problem: {test_problem}")
    print(f"Answer: {result.answer}")
    print(f"Reasoning: {result.reasoning[:200] if hasattr(result, 'reasoning') else 'N/A'}...")


if __name__ == "__main__":
    main()
