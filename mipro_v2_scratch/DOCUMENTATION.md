# MIPROv2 Standalone Implementation

## Tổng Quan

Đây là implementation hoàn chỉnh của **MIPROv2** (Multi-Prompt Instruction Proposal Optimizer v2) **không phụ thuộc vào thư viện DSPy**.

Dựa trên paper: ["Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs"](https://arxiv.org/abs/2406.11695)

### Tính Năng Chính

- ✅ **Không phụ thuộc DSPy** - Hoàn toàn standalone
- ✅ **Official SDKs** - Sử dụng `google-genai` cho Gemini, `openai` cho OpenAI
- ✅ **Multi-provider** - Hỗ trợ OpenAI, Google Gemini, Azure, Anthropic (qua LiteLLM)
- ✅ **Đầy đủ 3 bước** - Bootstrap → Propose → Optimize
- ✅ **Theo paper gốc** - Tuân thủ chính xác các thuật toán và parameters

---

## Mục Lục

1. [Cài Đặt](#cài-đặt)
2. [Quick Start](#quick-start)
3. [LLM Client Chi Tiết](#llm-client-chi-tiết)
4. [Thuật Toán MIPROv2](#thuật-toán-miprov2)
5. [Cấu Hình và Parameters](#cấu-hình-và-parameters)
6. [Ví Dụ Thực Tế](#ví-dụ-thực-tế)
7. [Troubleshooting](#troubleshooting)

---

## Cài Đặt

### Requirements

```bash
# Core (chọn 1 hoặc nhiều providers)
pip install google-genai       # Google Gemini (SDK mới - recommended)
pip install openai             # OpenAI
pip install litellm            # Multi-provider support (optional)

# Required
pip install numpy optuna
```

### Cấu Trúc Files

```
mipro_v2_scratch/
├── __init__.py           # Package exports
├── llm_client.py         # LLM clients (GeminiLM, OpenAILM, LiteLLM)
├── example.py            # Data containers (Example, Prediction)
├── module.py             # Module classes (Predictor, Signature)
├── mipro_optimizer.py    # Main MIPROv2 implementation
└── DOCUMENTATION.md      # This file
```

---

## Quick Start

### 1. Sử Dụng LLM Client

```python
from mipro_v2_scratch import LLM, GeminiLM, OpenAILM
import os

# ==========================================
# GOOGLE GEMINI (Recommended - SDK mới)
# ==========================================

# Set API key
os.environ["GEMINI_API_KEY"] = "your-api-key"

# Cách 1: Auto-detect từ model name
lm = LLM(model="gemini-2.0-flash")

# Cách 2: Explicit GeminiLM
lm = GeminiLM(
    model="gemini-2.0-flash",
    temperature=0.7,
    max_tokens=1000
)

# Cách 3: Vertex AI
lm = GeminiLM(
    model="gemini-1.5-pro",
    vertexai=True,
    project="your-gcp-project",
    location="us-central1"
)

# Generate
response = lm.generate("What is machine learning?")
print(response.text)
print(response.usage)  # Token usage

# ==========================================
# OPENAI
# ==========================================

os.environ["OPENAI_API_KEY"] = "sk-..."

lm = OpenAILM(model="gpt-4o-mini")
response = lm.generate("Explain AI")
print(response.text)

# Azure OpenAI
lm = OpenAILM(
    model="gpt-4o-mini",
    api_key="azure-key",
    base_url="https://your-resource.openai.azure.com/",
    api_version="2024-02-15-preview"
)

# Local model (Ollama)
lm = OpenAILM(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

# ==========================================
# LITELLM (Multi-provider - 100+ providers)
# ==========================================

from mipro_v2_scratch import LiteLLM

lm = LiteLLM(model="anthropic/claude-3-sonnet")
lm = LiteLLM(model="together_ai/meta-llama/Llama-3-70b")
lm = LiteLLM(model="groq/llama3-70b")
```

### 2. Tạo Example và Module

```python
from mipro_v2_scratch import Example, Module, Predictor, Signature

# Tạo training examples
trainset = [
    Example(question="What is 2+2?", answer="4").with_inputs("question"),
    Example(question="What is 3*3?", answer="9").with_inputs("question"),
    Example(question="What is 10/2?", answer="5").with_inputs("question"),
]

# Tạo Module
class MathSolver(Module):
    def __init__(self):
        super().__init__()
        self.solve = Predictor(
            Signature.from_string(
                "question: math question -> answer: numerical answer",
                instructions="Solve the math problem step by step."
            )
        )

    def forward(self, question):
        return self.solve(question=question)

# Sử dụng
solver = MathSolver()
solver.set_llm(lm)
result = solver(question="What is 15+7?")
print(result.answer)
```

### 3. Tối Ưu với MIPROv2

```python
from mipro_v2_scratch import MIPROv2, Example, LLM

# Định nghĩa metric
def accuracy_metric(example, prediction):
    expected = str(example.answer).strip().lower()
    predicted = str(prediction.answer).strip().lower()
    return expected == predicted

# Tạo optimizer
optimizer = MIPROv2(
    metric=accuracy_metric,
    llm=LLM(model="gemini-2.0-flash"),      # Task LLM
    prompt_llm=LLM(model="gemini-1.5-pro"),  # Proposal LLM (optional)
    auto="light",                            # light/medium/heavy
    verbose=True
)

# Optimize
optimized_solver, score = optimizer.compile(
    program=solver,
    trainset=trainset,
)

print(f"Best score: {score:.2f}%")

# Use optimized program
result = optimized_solver(question="What is 100/4?")
print(result.answer)
```

---

## LLM Client Chi Tiết

### Classes Hierarchy

```
BaseLLM (Abstract)
├── GeminiLM      # Google Gemini via google-genai SDK
├── OpenAILM      # OpenAI/Azure via openai SDK
├── LiteLLM       # Multi-provider via litellm
└── LLM           # Unified auto-detect wrapper
```

### GeminiLM - Google Gemini

Sử dụng **SDK mới** `google-genai` (thay thế `google-generativeai` đã deprecated).

```python
from mipro_v2_scratch import GeminiLM

# Basic usage
lm = GeminiLM(
    model="gemini-2.0-flash",     # Model name
    api_key=None,                  # From GEMINI_API_KEY env if None
    temperature=0.0,               # Sampling temperature
    max_tokens=1000,               # Max output tokens
    timeout=120,                   # Request timeout
    max_retries=3,                 # Retry attempts
)

# Vertex AI
lm = GeminiLM(
    model="gemini-1.5-pro",
    vertexai=True,
    project="your-project",
    location="us-central1",
)

# Generate
response = lm.generate(
    prompt="Explain quantum computing",
    temperature=0.7,
    max_tokens=500,
)
print(response.text)
print(response.usage)  # {"prompt_tokens": X, "completion_tokens": Y, "total_tokens": Z}

# With messages (OpenAI format)
response = lm.generate(messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
])

# Copy with modified params
lm2 = lm.copy(temperature=1.0, max_tokens=2000)
```

### OpenAILM - OpenAI & Compatible

```python
from mipro_v2_scratch import OpenAILM

# OpenAI
lm = OpenAILM(
    model="gpt-4o-mini",
    api_key=None,                  # From OPENAI_API_KEY env if None
    temperature=0.0,
    max_tokens=1000,
)

# Azure OpenAI
lm = OpenAILM(
    model="gpt-4o-mini",
    api_key="azure-api-key",
    base_url="https://your-resource.openai.azure.com/",
    api_version="2024-02-15-preview",
)

# Local model (Ollama, vLLM, etc.)
lm = OpenAILM(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="ollama",  # Some local servers require dummy key
)
```

### LiteLLM - Multi-Provider

Hỗ trợ 100+ providers với format `provider/model`.

```python
from mipro_v2_scratch import LiteLLM

# OpenAI
lm = LiteLLM(model="openai/gpt-4o-mini")

# Anthropic
lm = LiteLLM(model="anthropic/claude-3-5-sonnet")

# Google Gemini
lm = LiteLLM(model="gemini/gemini-1.5-flash")

# Together AI
lm = LiteLLM(model="together_ai/meta-llama/Llama-3-70b")

# Groq
lm = LiteLLM(model="groq/llama3-70b-8192")

# Bedrock
lm = LiteLLM(model="bedrock/anthropic.claude-3-sonnet")
```

### LLM - Unified Client

Auto-detects backend từ model name.

```python
from mipro_v2_scratch import LLM

# Auto-detect Gemini
lm = LLM(model="gemini-2.0-flash")

# Auto-detect OpenAI
lm = LLM(model="gpt-4o-mini")

# Auto-detect LiteLLM format
lm = LLM(model="anthropic/claude-3-sonnet")

# Force specific backend
lm = LLM(model="my-model", backend="openai", base_url="http://localhost:8000")
```

---

## Thuật Toán MIPROv2

### Tổng Quan 3 Bước

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MIPROv2 Algorithm                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Program Φ, Training Data D, Metric μ                           │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Bootstrap Demonstrations                                │   │
│  │                                                                 │   │
│  │  For i = 1 to N:                                                │   │
│  │    1. Run training examples through program                     │   │
│  │    2. Filter by metric (keep successful traces)                 │   │
│  │    3. Create diverse demo sets                                  │   │
│  │                                                                 │   │
│  │  Output: demo_candidates[module] = [DemoSet₁, ..., DemoSetₙ]   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Propose Instructions                                    │   │
│  │                                                                 │   │
│  │  1. Summarize dataset (iterative + COMPLETE mechanism)          │   │
│  │  2. Analyze program code                                        │   │
│  │  3. For each module, generate N instructions using:             │   │
│  │     - Dataset summary                                           │   │
│  │     - Program/module descriptions                               │   │
│  │     - Task demonstrations                                       │   │
│  │     - Random prompting tips                                     │   │
│  │     - Previous instruction history                              │   │
│  │                                                                 │   │
│  │  Output: instr_candidates[module] = [Instr₁, ..., Instrₙ]      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Bayesian Optimization (TPE)                             │   │
│  │                                                                 │   │
│  │  Search space: (instruction_idx, demo_idx) for each module      │   │
│  │                                                                 │   │
│  │  For trial t = 1 to T:                                          │   │
│  │    1. TPE sampler suggests parameter combination                │   │
│  │    2. Evaluate on validation set (or minibatch)                 │   │
│  │    3. Update surrogate model with score                         │   │
│  │    4. Periodically: full evaluation on top candidates           │   │
│  │                                                                 │   │
│  │  Output: Best (instruction, demos) combination                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  OUTPUT: Optimized Program Φ*                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cơ Chế "COMPLETE" (Dataset Summary)

LLM tự động dừng khi đã hiểu đủ về dataset:

```
Batch 1 (10 examples):
  → LLM: "Dataset contains math QA pairs..."

Batch 2 (10 examples):
  → LLM: "Also has word problems..."

Batch 3 (10 examples):
  → LLM: "COMPLETE"  ← Đã đủ thông tin

Batch 4-7: "COMPLETE" (count=2,3,4,5)
  → Khi count >= 5 → DỪNG iterating

→ Tổng hợp thành 2-3 câu summary
```

### Prompting TIPS

| Tip | Description |
|-----|-------------|
| `none` | Không có tip |
| `creative` | "Don't be afraid to be creative..." |
| `simple` | "Keep the instruction clear and concise." |
| `description` | "Make sure your instruction is very informative..." |
| `high_stakes` | "Include a high stakes scenario..." |
| `persona` | "Include a persona that is relevant..." |

---

## Cấu Hình và Parameters

### Auto Modes

| Mode | N (candidates) | Val Size | Approx Trials | Use Case |
|------|---------------|----------|---------------|----------|
| `light` | 6 | 100 | 12-15 | Development, small datasets |
| `medium` | 12 | 300 | 25-30 | Balanced quality/cost |
| `heavy` | 18 | 1000 | 40-50 | Production, large datasets |

### MIPROv2 Parameters

```python
MIPROv2(
    # Required
    metric=accuracy_metric,       # (example, prediction) -> score
    llm=task_lm,                  # LLM for running the program

    # Optional
    prompt_llm=None,              # LLM for generating instructions
    auto="light",                 # "light", "medium", "heavy", or None
    num_candidates=None,          # Required if auto=None
    max_bootstrapped_demos=4,     # Max demos from bootstrapping
    max_labeled_demos=4,          # Max demos from labels
    metric_threshold=None,        # Threshold for bootstrap acceptance
    num_threads=1,                # Parallel evaluation
    seed=9,                       # Random seed
    verbose=False,                # Print details

    # Proposer settings
    program_aware=True,           # Use program code analysis
    data_aware=True,              # Use dataset summary
    tip_aware=True,               # Use prompting tips
    view_data_batch_size=10,      # Batch size for summary
)
```

### Compile Parameters

```python
optimizer.compile(
    program=my_program,
    trainset=train_examples,
    valset=None,                     # Auto-split if None
    num_trials=None,                 # Auto-calculated if None
    max_bootstrapped_demos=None,     # Override init setting
    max_labeled_demos=None,
    minibatch=True,                  # Use minibatch evaluation
    minibatch_size=35,               # Minibatch size
    minibatch_full_eval_steps=5,     # Full eval every N steps
)
```

### Công Thức Từ Paper

```python
# Number of instruction candidates
num_instruct = N * 0.5  # If using demos
num_instruct = N        # If zero-shot

# Number of trials
num_trials = max(2 * M * log2(N), 1.5 * N)
# Where M = number of predictors * 2 (if using demos)
```

---

## Ví Dụ Thực Tế

### Question Answering

```python
from mipro_v2_scratch import *
import os

os.environ["GEMINI_API_KEY"] = "your-key"
llm = GeminiLM(model="gemini-2.0-flash")

class QAModule(Module):
    def __init__(self):
        super().__init__()
        self.qa = Predictor(
            Signature.from_string(
                "question: the question -> answer: concise answer"
            )
        )

    def forward(self, question):
        return self.qa(question=question)

trainset = [
    Example(question="What is 2+2?", answer="4").with_inputs("question"),
    Example(question="Capital of Japan?", answer="Tokyo").with_inputs("question"),
    Example(question="Who painted Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
]

def match(example, prediction):
    return example.answer.lower() in prediction.answer.lower()

program = QAModule()
program.set_llm(llm)

optimizer = MIPROv2(metric=match, llm=llm, auto="light", verbose=True)
best, score = optimizer.compile(program=program, trainset=trainset)

print(f"Score: {score:.2f}%")
result = best(question="What is the speed of light?")
print(result.answer)
```

### Chain-of-Thought

```python
class ChainOfThought(Module):
    def __init__(self):
        super().__init__()
        self.reason = Predictor(
            Signature.from_string("question -> reasoning: step-by-step thinking")
        )
        self.answer = Predictor(
            Signature.from_string("question, reasoning -> answer: final answer")
        )

    def forward(self, question):
        r = self.reason(question=question)
        return self.answer(question=question, reasoning=r.reasoning)

# MIPROv2 optimizes BOTH predictors
```

---

## Troubleshooting

### API Key Issues

```python
import os

# Gemini
os.environ["GEMINI_API_KEY"] = "AIza..."

# OpenAI
os.environ["OPENAI_API_KEY"] = "sk-..."

# Or pass directly
lm = GeminiLM(api_key="your-key")
```

### Import Errors

```bash
pip install google-genai   # For GeminiLM
pip install openai         # For OpenAILM
pip install litellm        # For LiteLLM
pip install optuna numpy   # Required
```

### Low Optimization Scores

1. **Check metric**: Ensure it returns correct True/False or 0-1
2. **More data**: Add more training examples (50-100+ recommended)
3. **Higher mode**: Try `auto="medium"` or `auto="heavy"`
4. **Better LLM**: Use stronger model for proposals

---

## Complete Working Example

Dưới đây là một ví dụ hoàn chỉnh có thể chạy được:

```python
#!/usr/bin/env python3
"""
Complete example: Optimizing a Math QA system with MIPROv2
"""

import os
import logging
from mipro_v2_scratch import (
    LLM, GeminiLM, OpenAILM,
    MIPROv2,
    Example, Prediction,
    Module, Predictor, Signature
)

# Enable logging to see progress
logging.basicConfig(level=logging.INFO)

# ============================================
# 1. Setup LLM Client
# ============================================

# Option A: Google Gemini (recommended)
os.environ["GEMINI_API_KEY"] = "YOUR_API_KEY_HERE"
lm = GeminiLM(model="gemini-2.0-flash", temperature=0.7)

# Option B: OpenAI
# os.environ["OPENAI_API_KEY"] = "sk-..."
# lm = OpenAILM(model="gpt-4o-mini")

# Option C: Auto-detect
# lm = LLM(model="gemini-2.0-flash")

# ============================================
# 2. Create Training Data
# ============================================

trainset = [
    Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    Example(question="What is 10 - 3?", answer="7").with_inputs("question"),
    Example(question="What is 5 * 6?", answer="30").with_inputs("question"),
    Example(question="What is 20 / 4?", answer="5").with_inputs("question"),
    Example(question="What is 15 + 8?", answer="23").with_inputs("question"),
    Example(question="What is 100 - 37?", answer="63").with_inputs("question"),
    Example(question="What is 7 * 8?", answer="56").with_inputs("question"),
    Example(question="What is 81 / 9?", answer="9").with_inputs("question"),
    Example(question="What is 25 + 17?", answer="42").with_inputs("question"),
    Example(question="What is 50 - 23?", answer="27").with_inputs("question"),
]

# ============================================
# 3. Define Module
# ============================================

class MathSolver(Module):
    """A simple math question answering module."""

    def __init__(self):
        super().__init__()
        self.solve = Predictor(
            Signature.from_string(
                "question: math question -> answer: numerical answer",
                instructions="Solve the math problem and return only the numerical answer."
            )
        )

    def forward(self, question):
        return self.solve(question=question)

# ============================================
# 4. Define Metric
# ============================================

def exact_match(example: Example, prediction: Prediction) -> bool:
    """Check if prediction exactly matches the expected answer."""
    expected = str(example.answer).strip().lower()
    predicted = str(prediction.answer).strip().lower()

    # Extract just the number if there's extra text
    import re
    expected_nums = re.findall(r'\d+', expected)
    predicted_nums = re.findall(r'\d+', predicted)

    if expected_nums and predicted_nums:
        return expected_nums[0] == predicted_nums[0]

    return expected == predicted

# ============================================
# 5. Create and Test Module
# ============================================

print("\n" + "="*50)
print("Testing before optimization...")
print("="*50)

module = MathSolver()
module.set_llm(lm)

# Test a few examples
for ex in trainset[:3]:
    result = module(question=ex.question)
    match = exact_match(ex, result)
    print(f"Q: {ex.question}")
    print(f"A: {result.answer} (expected: {ex.answer}) {'✓' if match else '✗'}")
    print()

# ============================================
# 6. Optimize with MIPROv2
# ============================================

print("\n" + "="*50)
print("Starting MIPROv2 optimization...")
print("="*50)

optimizer = MIPROv2(
    metric=exact_match,
    llm=lm,
    prompt_llm=lm,  # Use same LLM for proposals (or use a stronger one)
    auto="light",   # "light", "medium", or "heavy"
    verbose=True,
    seed=42,
)

optimized_module, best_score = optimizer.compile(
    program=module,
    trainset=trainset,
    minibatch=False,  # Disable minibatch for small datasets
)

print(f"\nOptimization complete! Best score: {best_score:.1f}%")

# ============================================
# 7. Test Optimized Module
# ============================================

print("\n" + "="*50)
print("Testing after optimization...")
print("="*50)

test_questions = [
    "What is 12 + 15?",
    "What is 99 - 44?",
    "What is 11 * 11?",
    "What is 144 / 12?",
]

for q in test_questions:
    result = optimized_module(question=q)
    print(f"Q: {q}")
    print(f"A: {result.answer}")
    print()

# ============================================
# 8. Inspect Optimized Instructions
# ============================================

print("\n" + "="*50)
print("Optimized Instructions:")
print("="*50)

for name, pred in optimized_module.named_predictors():
    print(f"\nPredictor: {name}")
    print(f"Instruction: {pred.signature.instructions}")
    print(f"Demos: {len(pred.demos)} examples")
```

---

## API Reference

### LLM Clients

#### `LLM(model, backend=None, temperature=0.0, max_tokens=1000, **kwargs)`
Unified client with auto-detection.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | Model name |
| `backend` | str | Force "gemini", "openai", or "litellm" |
| `temperature` | float | Sampling temperature (0-2) |
| `max_tokens` | int | Max output tokens |

#### `GeminiLM(model, api_key=None, vertexai=False, ...)`
Google Gemini client.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | e.g., "gemini-2.0-flash" |
| `api_key` | str | API key (or GEMINI_API_KEY env) |
| `vertexai` | bool | Use Vertex AI |
| `project` | str | GCP project (for Vertex) |
| `location` | str | GCP location (default: "us-central1") |

#### `OpenAILM(model, api_key=None, base_url=None, ...)`
OpenAI-compatible client.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | str | e.g., "gpt-4o-mini" |
| `api_key` | str | API key (or OPENAI_API_KEY env) |
| `base_url` | str | Custom endpoint |
| `api_version` | str | For Azure OpenAI |

#### `LiteLLM(model, temperature=0.0, max_tokens=1000, ...)`
Multi-provider client via litellm.

### Data Classes

#### `Example(**fields)`
Training example container.

```python
ex = Example(question="What is AI?", answer="Artificial Intelligence")
ex = ex.with_inputs("question")  # Mark input fields
ex.inputs()   # {"question": "What is AI?"}
ex.labels()   # {"answer": "Artificial Intelligence"}
```

#### `Prediction(**fields)`
Prediction output container (same interface as Example).

### Module Classes

#### `Signature.from_string(sig_string, instructions="")`
Parse signature from string.

```python
sig = Signature.from_string("question -> answer")
sig = Signature.from_string("question: the question -> answer: the response")
```

#### `Predictor(signature, llm=None)`
Single LLM call module.

```python
pred = Predictor(Signature.from_string("question -> answer"))
pred.llm = lm
pred.demos = [Example(...)]  # Few-shot examples
result = pred(question="What is AI?")
```

#### `Module`
Base class for multi-step programs.

```python
class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.step1 = Predictor(...)
        self.step2 = Predictor(...)

    def forward(self, **inputs):
        r1 = self.step1(**inputs)
        return self.step2(**inputs, **r1._data)
```

### MIPROv2 Optimizer

#### `MIPROv2(metric, llm, prompt_llm=None, auto="light", ...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `metric` | Callable | (example, prediction) -> score |
| `llm` | BaseLLM | LLM for running program |
| `prompt_llm` | BaseLLM | LLM for proposals (optional) |
| `auto` | str | "light", "medium", "heavy", or None |
| `num_candidates` | int | Required if auto=None |
| `max_bootstrapped_demos` | int | Max bootstrapped demos (default: 4) |
| `max_labeled_demos` | int | Max labeled demos (default: 4) |
| `verbose` | bool | Print progress |

#### `optimizer.compile(program, trainset, valset=None, ...)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `program` | Module | Program to optimize |
| `trainset` | List[Example] | Training examples |
| `valset` | List[Example] | Validation (auto-split if None) |
| `num_trials` | int | Override trial count |
| `minibatch` | bool | Use minibatch eval (default: True) |
| `minibatch_size` | int | Minibatch size (default: 35) |

Returns: `(optimized_program, best_score)`

---

## Save/Load Optimized Prompts

Lưu và load prompt đã tối ưu vào/từ file JSON.

### Save Optimized Module

```python
# Sau khi tối ưu
optimized_module, score = optimizer.compile(program=module, trainset=trainset)

# Lưu vào file JSON
optimized_module.save("optimized_prompts.json")

# Lưu không kèm metadata (chỉ prompts và demos)
optimized_module.save("prompts_only.json", include_metadata=False)
```

### Load Optimized Module

```python
# Cách 1: Load vào module đã có
module = MathSolver()
module.set_llm(lm)
module.load("optimized_prompts.json")

# Cách 2: Load và sử dụng ngay
module = MathSolver()
module.set_llm(lm).load("optimized_prompts.json")
result = module(question="What is 5+5?")
```

### JSON File Structure

```json
{
  "version": "2.0.0",
  "module_class": "MathSolver",
  "predictors": {
    "solve": {
      "signature": {
        "input_fields": [
          {"name": "question", "description": "math question", "prefix": "Question:", "field_type": "input"}
        ],
        "output_fields": [
          {"name": "answer", "description": "numerical answer", "prefix": "Answer:", "field_type": "output"}
        ],
        "instructions": "You are an expert math tutor. Solve the problem step by step..."
      },
      "demos": [
        {"question": "What is 2+2?", "answer": "4"},
        {"question": "What is 10-3?", "answer": "7"}
      ],
      "demo_input_keys": [["question"], ["question"]]
    }
  },
  "metadata": {
    "compiled": true,
    "score": 95.0,
    "trial_logs": {...}
  }
}
```

### Get Optimized State Summary

```python
# Xem tóm tắt state đã tối ưu
state = optimized_module.get_optimized_state()
print(f"Score: {state['score']}%")

for name, info in state["predictors"].items():
    print(f"\nPredictor: {name}")
    print(f"  Instruction: {info['instruction'][:80]}...")
    print(f"  Demos: {info['num_demos']}")
```

---

## Logging Configuration

Cấu hình logging để theo dõi quá trình tối ưu.

### Basic Setup

```python
from mipro_v2_scratch import setup_logging, MIPROv2

# Hiển thị log chi tiết trên console
setup_logging(level="INFO", format_style="detailed")

# Chỉ hiển thị warnings
setup_logging(level="WARNING")

# Log ra file
setup_logging(level="DEBUG", log_file="optimization.log")
```

### Log Levels

| Level | Description |
|-------|-------------|
| `DEBUG` | Tất cả thông tin chi tiết (prompts, responses, etc.) |
| `INFO` | Tiến trình chính (steps, trials, scores) |
| `WARNING` | Chỉ cảnh báo và errors |
| `ERROR` | Chỉ errors |

### Format Styles

| Style | Output |
|-------|--------|
| `minimal` | `Message only` |
| `simple` | `[INFO] Message` |
| `detailed` | `[10:30:45] [INFO] mipro_v2_scratch: Message` |

### Verbose Parameter

```python
# MIPROv2 có parameter verbose
optimizer = MIPROv2(
    metric=accuracy,
    llm=lm,
    verbose=True,  # Hiển thị progress
)

# Hoặc tắt hoàn toàn
optimizer = MIPROv2(
    metric=accuracy,
    llm=lm,
    verbose=False,  # Chỉ hiển thị warnings/errors
)
```

### Complete Example với Logging

```python
from mipro_v2_scratch import (
    LLM, MIPROv2, Example, Module, Predictor, Signature,
    setup_logging
)

# Setup logging chi tiết
setup_logging(level="INFO", format_style="simple", log_file="my_optimization.log")

# ... định nghĩa module và trainset ...

# Optimizer với verbose
optimizer = MIPROv2(
    metric=accuracy,
    llm=lm,
    verbose=True,  # Hiển thị progress
)

# Optimize - logs sẽ được ghi ra console và file
optimized, score = optimizer.compile(program=module, trainset=trainset)

# Lưu kết quả
optimized.save("best_prompts.json")
print(f"Saved! Score: {score}%")
```

---

## So Sánh với DSPy

| Feature | DSPy | Standalone MIPROv2 |
|---------|------|-------------------|
| Dependencies | DSPy + LiteLLM | google-genai, openai (optional: litellm) |
| Installation | `pip install dspy` | Copy folder + install SDKs |
| LLM Support | Via LiteLLM | Native Gemini/OpenAI + optional LiteLLM |
| Algorithm | MIPROv2 | Exact same algorithm |
| Code Style | DSPy abstractions | Minimal, readable |
| Customization | Limited | Easy to modify |

### Khi nào dùng DSPy?
- Project lớn cần nhiều optimizers (BootstrapFewShot, COPRO, etc.)
- Cần integration với DSPy ecosystem
- Không muốn maintain code riêng

### Khi nào dùng Standalone?
- Chỉ cần MIPROv2
- Muốn kiểm soát hoàn toàn code
- Dự án nhẹ, không muốn dependencies phức tạp
- Học tập/nghiên cứu thuật toán

---

## References

- [MIPROv2 Paper (arXiv:2406.11695)](https://arxiv.org/abs/2406.11695)
- [DSPy Library](https://github.com/stanfordnlp/dspy)
- [Google GenAI SDK](https://googleapis.github.io/python-genai/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
