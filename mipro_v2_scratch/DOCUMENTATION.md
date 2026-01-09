# MIPROv2 - Tài liệu chi tiết (Comprehensive Documentation)

## Mục lục
1. [Giới thiệu](#1-giới-thiệu)
2. [Cài đặt và Sử dụng](#2-cài-đặt-và-sử-dụng)
3. [Thuật toán MIPRO chi tiết](#3-thuật-toán-mipro-chi-tiết)
4. [Giải thích từng bước](#4-giải-thích-từng-bước)
5. [Bayesian Optimization & TPE](#5-bayesian-optimization--tpe)
6. [Tham số và Cấu hình](#6-tham-số-và-cấu-hình)
7. [Ví dụ thực tế](#7-ví-dụ-thực-tế)

---

## 1. Giới thiệu

### 1.1 MIPRO là gì?

**MIPRO** (Multi-prompt Instruction PRoposal Optimizer) là thuật toán tối ưu hóa prompt tự động cho các chương trình sử dụng Language Model (LLM). Thay vì phải viết prompt thủ công (prompt engineering), MIPRO tự động tìm ra:

1. **Instructions tốt nhất** - Hướng dẫn cho LLM về cách thực hiện task
2. **Few-shot demonstrations tốt nhất** - Các ví dụ mẫu để LLM học theo

### 1.2 Tại sao cần MIPRO?

```
❌ Prompt Engineering truyền thống:
   ┌─────────────────────────────────────────────┐
   │  Con người viết prompt                      │
   │       ↓                                     │
   │  Test thủ công                              │
   │       ↓                                     │
   │  Sửa prompt dựa trên cảm tính               │
   │       ↓                                     │
   │  Lặp lại nhiều lần...                       │
   │                                             │
   │  Vấn đề:                                    │
   │  • Tốn thời gian                            │
   │  • Phụ thuộc kinh nghiệm                    │
   │  • Không có cơ sở khoa học                  │
   │  • Khó tìm được tối ưu toàn cục             │
   └─────────────────────────────────────────────┘

✅ MIPRO:
   ┌─────────────────────────────────────────────┐
   │  Dữ liệu training + Metric đánh giá         │
   │       ↓                                     │
   │  MIPRO tự động sinh candidates              │
   │       ↓                                     │
   │  Bayesian Optimization tìm tối ưu           │
   │       ↓                                     │
   │  Output: Prompt tối ưu                      │
   │                                             │
   │  Ưu điểm:                                   │
   │  • Hoàn toàn tự động                        │
   │  • Data-driven (dựa trên dữ liệu)           │
   │  • Có cơ sở toán học (Bayesian)             │
   │  • Tìm được tối ưu hiệu quả                 │
   └─────────────────────────────────────────────┘
```

### 1.3 Paper gốc

- **Tiêu đề**: "Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs"
- **arXiv**: [2406.11695](https://arxiv.org/abs/2406.11695)
- **Tác giả**: Nhóm nghiên cứu từ Stanford (Omar Khattab et al.)
- **Năm**: 2024

---

## 2. Cài đặt và Sử dụng

### 2.1 Dependencies

```bash
pip install openai optuna numpy
```

### 2.2 Cấu trúc thư mục

```
mipro_v2_scratch/
├── __init__.py           # Package exports
├── llm_client.py         # LLM API wrapper (OpenAI compatible)
├── example.py            # Data containers (Example, Prediction)
├── module.py             # Module classes (Predictor, Signature)
├── mipro_optimizer.py    # Main MIPROv2 implementation
└── DOCUMENTATION.md      # This file
```

### 2.3 Quick Start

```python
import logging
logging.basicConfig(level=logging.INFO)

from mipro_v2_scratch import (
    MIPROv2,
    LLMClient,
    Example,
    SimplePredictor
)

# 1. Tạo LLM client
llm = LLMClient(
    api_key="sk-your-api-key",
    model="gpt-4o-mini"
)

# 2. Tạo chương trình cần tối ưu
program = SimplePredictor("question -> answer")
program.set_llm(llm)

# 3. Chuẩn bị dữ liệu training
trainset = [
    Example(question="What is 2+2?", answer="4").with_inputs("question"),
    Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    Example(question="Who wrote Hamlet?", answer="Shakespeare").with_inputs("question"),
    # ... thêm nhiều ví dụ
]

# 4. Định nghĩa metric đánh giá
def exact_match(example, prediction):
    pred_answer = prediction.answer.strip().lower()
    true_answer = example.answer.strip().lower()
    return pred_answer == true_answer

# 5. Tạo optimizer và chạy
optimizer = MIPROv2(
    metric=exact_match,
    llm=llm,
    auto="light",  # "light", "medium", "heavy"
    verbose=True
)

best_program, best_score = optimizer.compile(
    program=program,
    trainset=trainset,
)

print(f"Best score: {best_score:.2f}%")

# 6. Sử dụng chương trình đã tối ưu
result = best_program(question="What is 3+3?")
print(f"Answer: {result.answer}")
```

---

## 3. Thuật toán MIPRO chi tiết

### 3.1 Tổng quan 3 bước

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MIPRO Algorithm Overview                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Program Φ, Training Data D, Metric μ                            │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Bootstrap Demonstrations                                │   │
│  │                                                                 │   │
│  │  For each module m in Φ:                                        │   │
│  │    - Run program on training examples                           │   │
│  │    - Filter by metric μ (keep successful traces)                │   │
│  │    - Create N diverse demo sets                                 │   │
│  │                                                                 │   │
│  │  Output: demo_candidates[m] = [DemoSet₁, DemoSet₂, ..., DemoSetₙ]│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Propose Instructions                                    │   │
│  │                                                                 │   │
│  │  For each module m in Φ:                                        │   │
│  │    - Summarize dataset characteristics                          │   │
│  │    - Use demos as context                                       │   │
│  │    - Apply random prompting tips                                │   │
│  │    - Generate N instruction candidates via LLM                  │   │
│  │                                                                 │   │
│  │  Output: instr_candidates[m] = [Instr₁, Instr₂, ..., Instrₙ]    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 3: Bayesian Optimization                                   │   │
│  │                                                                 │   │
│  │  Search space: (instruction_idx, demo_idx) for each module      │   │
│  │                                                                 │   │
│  │  For trial t = 1 to T:                                          │   │
│  │    - TPE sampler suggests (instr_idx, demo_idx) combination     │   │
│  │    - Evaluate on minibatch from validation set                  │   │
│  │    - Update surrogate model with score                          │   │
│  │    - Periodically: full evaluation on top candidates            │   │
│  │                                                                 │   │
│  │  Output: Best (instruction, demos) combination                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                               ↓                                         │
│  OUTPUT: Optimized Program Φ* with best instructions and demos          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Ký hiệu toán học

| Ký hiệu | Ý nghĩa |
|---------|---------|
| Φ | Chương trình LLM (program) |
| m | Module/Predictor trong chương trình |
| D | Tập dữ liệu training |
| V | Tập dữ liệu validation |
| μ | Metric đánh giá |
| I_m | Tập instructions cho module m |
| E_m | Tập demo sets cho module m |
| θ = (i, e) | Tham số: i = instruction index, e = demo index |
| f(θ) | Objective function: score của program với tham số θ |

---

## 4. Giải thích từng bước

### 4.1 Step 1: Bootstrap Demonstrations

#### Mục đích
Tạo ra **nhiều bộ demonstrations đa dạng** để:
1. Làm few-shot examples trong prompt
2. Cung cấp context cho instruction generation
3. Tạo diversity cho Bayesian search

#### Thuật toán Bootstrap

```
Algorithm: BootstrapDemonstrations(Φ, D, μ, N, K)
─────────────────────────────────────────────────
Input:
  Φ: Program
  D: Training data
  μ: Metric function
  N: Number of demo sets to create
  K: Max demos per set

Output:
  demo_candidates: Dict[module_idx → List[DemoSet]]

1. Initialize demo_candidates = {}

2. For set_idx = 1 to N:

   2.1. If set_idx == 1:
        # Zero-shot set (no demos)
        demos = []

   2.2. Else if set_idx == 2:
        # Labeled-only set
        demos = RandomSample(D, K)

   2.3. Else:
        # Bootstrapped set
        Shuffle(D)
        demos = []
        errors = 0

        For each example x in D:
            If len(demos) >= K: break

            Try:
                # Run program
                prediction = Φ(x.inputs)

                # Evaluate
                score = μ(x, prediction)

                # Keep if passes threshold
                If score >= threshold:
                    trace = GetExecutionTrace(Φ)
                    demos.append(trace)

            Except:
                errors += 1
                If errors >= max_errors: break

   2.4. For each module m in Φ:
        demo_candidates[m].append(demos)

3. Return demo_candidates
```

#### Ví dụ minh họa

```
Training Data:
┌─────────────────────────────────────────────┐
│ Example 1: Q="2+2?" A="4"                   │
│ Example 2: Q="Capital of France?" A="Paris" │
│ Example 3: Q="3×3?" A="9"                   │
│ ...                                         │
└─────────────────────────────────────────────┘
            │
            ▼ Run through Program
┌─────────────────────────────────────────────┐
│ Example 1 → Prediction: "4" → Metric: ✓     │
│ Example 2 → Prediction: "London" → Metric: ✗│
│ Example 3 → Prediction: "9" → Metric: ✓     │
└─────────────────────────────────────────────┘
            │
            ▼ Keep successful traces
┌─────────────────────────────────────────────┐
│ DemoSet 1: [Empty] (zero-shot)              │
│ DemoSet 2: [Ex1, Ex3] (random labeled)      │
│ DemoSet 3: [Ex1, Ex3] (bootstrapped)        │
│ DemoSet 4: [Ex3, Ex1] (shuffled bootstrap)  │
│ ...                                         │
└─────────────────────────────────────────────┘
```

### 4.2 Step 2: Propose Instructions

#### 4.2.1 Dataset Summarization (Iterative Observation)

Trước khi tạo instructions, MIPRO cần hiểu dataset. Quá trình này sử dụng **iterative observation** với cơ chế **"COMPLETE"**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DATASET SUMMARIZATION PROCESS                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Training Data (500 examples)                                           │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ Ex1, Ex2, Ex3, ... Ex500                                        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ BATCH 1 (Ex1-Ex10):                                             │   │
│  │                                                                 │   │
│  │   Prompt: "Given examples, write observations about trends..."  │   │
│  │                                                                 │   │
│  │   LLM Output: "The dataset contains QA pairs about math.        │   │
│  │               Questions are short (5-10 words).                 │   │
│  │               Answers are typically single numbers."            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ BATCH 2 (Ex11-Ex20):                                            │   │
│  │                                                                 │   │
│  │   Prompt: "Here are more examples + prior observations.         │   │
│  │           Add new observations or say 'COMPLETE'."              │   │
│  │                                                                 │   │
│  │   LLM Output: "Also noticed: some questions involve            │   │
│  │               percentages and fractions. Answer format          │   │
│  │               is consistent (just the number, no units)."       │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ BATCH 3 (Ex21-Ex30):                                            │   │
│  │                                                                 │   │
│  │   LLM Output: "COMPLETE"  ← Không có gì mới để thêm             │   │
│  │   skips = 1                                                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ BATCH 4-7: LLM tiếp tục output "COMPLETE"                       │   │
│  │   skips = 2, 3, 4, 5                                            │   │
│  │                                                                 │   │
│  │   Khi skips >= 5 → DỪNG iterating                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ FINAL STEP: Summarize observations                              │   │
│  │                                                                 │   │
│  │   Prompt: "Summarize these observations into 2-3 sentences"     │   │
│  │                                                                 │   │
│  │   Output: "This dataset contains math QA pairs where users      │   │
│  │           ask calculation questions. Answers are short          │   │
│  │           numeric values, typically 1-3 digits."                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Cơ chế "COMPLETE" chi tiết

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     CƠ CHẾ "COMPLETE" STOPPING                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  MỤC ĐÍCH:                                                              │
│  • Tiết kiệm LLM calls (không cần xem hết 500 examples)                │
│  • Dừng khi đã hiểu đủ về dataset                                       │
│  • Tránh over-summarization (thông tin dư thừa)                        │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  ALGORITHM:                                                             │
│                                                                         │
│  skips = 0                                                              │
│  max_calls = 10  # Giới hạn tối đa                                      │
│                                                                         │
│  For each batch:                                                        │
│      response = LLM(examples + prior_observations)                      │
│                                                                         │
│      If response starts with "COMPLETE":                                │
│          skips += 1                                                     │
│          If skips >= 5:                                                 │
│              BREAK  ← Dừng vòng lặp                                     │
│          Continue  ← Bỏ qua batch này, không thêm vào observations      │
│      Else:                                                              │
│          observations += response                                       │
│          skips = 0  ← Reset counter (có observation mới)                │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  TẠI SAO 5 LẦN LIÊN TIẾP?                                               │
│                                                                         │
│  • 1 lần "COMPLETE" có thể là do batch đó tương tự batch trước          │
│  • 5 lần liên tiếp = LLM thực sự đã thấy đủ patterns                   │
│  • Đảm bảo robustness (tránh dừng sớm do 1 batch "boring")             │
│                                                                         │
│  ─────────────────────────────────────────────────────────────────────  │
│                                                                         │
│  VÍ DỤ THỰC TẾ:                                                         │
│                                                                         │
│  Batch 1: "Questions are about math..." → observations += this         │
│  Batch 2: "Also has geography questions" → observations += this        │
│  Batch 3: "COMPLETE" → skips=1, continue                               │
│  Batch 4: "Found science questions too!" → observations += this        │
│           skips=0 (reset vì có observation mới)                        │
│  Batch 5: "COMPLETE" → skips=1                                         │
│  Batch 6: "COMPLETE" → skips=2                                         │
│  Batch 7: "COMPLETE" → skips=3                                         │
│  Batch 8: "COMPLETE" → skips=4                                         │
│  Batch 9: "COMPLETE" → skips=5 → DỪNG!                                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Prompts chính xác từ DSPy

```python
# Prompt cho batch đầu tiên
OBSERVATION_PROMPT = """
Given several examples from a dataset please write observations
about trends that hold for most or all of the samples.

Some areas you may consider in your observations:
topics, content, syntax, conciseness, etc.

It will be useful to make an educated guess as to the nature of
the task this dataset will enable. Don't be afraid to be creative.

Examples:
{examples}

Observations:
"""

# Prompt cho các batch tiếp theo
OBSERVATION_WITH_PRIOR_PROMPT = """
Given several examples from a dataset please write observations
about trends that hold for most or all of the samples.

I will also provide you with a few observations I have already made.
Please add your own observations or if you feel the observations
are comprehensive say 'COMPLETE'.

Some areas you may consider: topics, content, syntax, conciseness, etc.

Examples:
{examples}

Prior observations:
{prior_observations}

Additional observations (or 'COMPLETE' if nothing to add):
"""

# Prompt để tổng hợp cuối cùng
SUMMARIZE_PROMPT = """
Given a series of observations I have made about my dataset,
please summarize them into a brief 2-3 sentence summary
which highlights only the most important details.

Observations:
{observations}

Summary (2-3 sentences):
"""
```

#### 4.2.2 Program Description (Program-Aware Proposal)

MIPRO phân tích **code của program** để hiểu cấu trúc:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PROGRAM DESCRIPTION PROCESS                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Program Source Code                                             │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  class MathSolver(Module):                                      │   │
│  │      def __init__(self):                                        │   │
│  │          self.reason = Predictor("question -> reasoning")       │   │
│  │          self.answer = Predictor("question, reasoning -> ans")  │   │
│  │                                                                 │   │
│  │      def forward(self, question):                               │   │
│  │          r = self.reason(question=question)                     │   │
│  │          a = self.answer(question=question, reasoning=r)        │   │
│  │          return a                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 1: Describe Program                                        │   │
│  │                                                                 │   │
│  │   Prompt: "Describe what task this program solves and how       │   │
│  │           it appears to work"                                   │   │
│  │                                                                 │   │
│  │   Output: "This program solves math problems using a two-stage  │   │
│  │           approach: first generating step-by-step reasoning,    │   │
│  │           then extracting the final answer from the reasoning." │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                         │                                               │
│                         ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ STEP 2: Describe Each Module                                    │   │
│  │                                                                 │   │
│  │   Module 0: Predictor(question) -> reasoning                    │   │
│  │   Description: "Generates step-by-step reasoning to solve       │   │
│  │                 the math problem"                               │   │
│  │                                                                 │   │
│  │   Module 1: Predictor(question, reasoning) -> answer            │   │
│  │   Description: "Extracts the final numeric answer from the      │   │
│  │                 reasoning provided"                             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 4.2.3 Full Instruction Proposal Prompt

Tất cả components được kết hợp thành **Grounded Proposal Prompt**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              COMPLETE INSTRUCTION PROPOSAL PROMPT                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Use the information below to learn about a task that we are trying     │
│  to solve using calls to an LM, then generate a new instruction that    │
│  will be used to prompt a Language Model to better solve the task.      │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  DATASET SUMMARY:                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "This dataset contains math word problems where users ask      │   │
│  │   calculation questions. Answers are short numeric values."     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PROGRAM CODE:                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  class MathSolver(Module):                                      │   │
│  │      def __init__(self):                                        │   │
│  │          self.reason = Predictor("question -> reasoning")       │   │
│  │          self.answer = Predictor("question, reasoning -> ans")  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PROGRAM DESCRIPTION:                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "This program uses two-stage reasoning to solve math problems" │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  MODULE:                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Predictor(question) -> reasoning                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  MODULE DESCRIPTION:                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "Generates step-by-step reasoning to solve the math problem"   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  TASK DEMO(S):                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Question: What is 15% of 200?                                  │   │
│  │  Reasoning: To find 15% of 200, I multiply: 200 × 0.15 = 30    │   │
│  │  ---                                                            │   │
│  │  Question: What is 25 + 37?                                     │   │
│  │  Reasoning: Adding 25 and 37: 25 + 37 = 62                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  PREVIOUS INSTRUCTIONS:                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  - Score 65%: "Solve the math problem"                          │   │
│  │  - Score 72%: "Think step by step to solve the problem"         │   │
│  │  - Score 68%: "Calculate the answer carefully"                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  BASIC INSTRUCTION:                                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "Given the question, provide step-by-step reasoning"           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  TIP:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  "Make sure your instruction is very informative and            │   │
│  │   descriptive."                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                                                                         │
│  PROPOSED INSTRUCTION:                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  (LLM generates new instruction here based on all context)      │   │
│  │                                                                 │   │
│  │  Example output:                                                │   │
│  │  "You are a math tutor. Given a math problem, break it down    │   │
│  │   into clear steps. Show your work by explaining each          │   │
│  │   calculation. Be precise with numbers and operations."        │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Mục đích của từng component

| Component | Mục đích | Khi nào dùng |
|-----------|----------|--------------|
| **Dataset Summary** | Giúp LLM hiểu đặc điểm data | `data_aware=True` |
| **Program Code** | Hiểu cấu trúc chương trình | `program_aware=True` |
| **Program Description** | Hiểu task tổng thể | `program_aware=True` |
| **Module Description** | Hiểu vai trò module cụ thể | `program_aware=True` |
| **Task Demos** | Cung cấp ví dụ cụ thể | `fewshot_aware=True` |
| **Previous Instructions** | Tránh lặp lại, học từ failures | `use_instruct_history=True` (50% random) |
| **Basic Instruction** | Baseline để improve upon | Luôn có |
| **Tip** | Hướng dẫn style của instruction | `tip_aware=True` (random tip) |

#### "Grounded" Proposal

"Grounded" có nghĩa là instruction được sinh dựa trên **dữ liệu thực tế**, không phải abstract:

```
┌────────────────────────────────────────────────────────────┐
│                    Grounded Proposer                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Input Context:                                            │
│  ┌──────────────────────────────────────────────────────┐ │
│  │ 1. Dataset Summary                                   │ │
│  │    "The dataset contains QA pairs about math and     │ │
│  │     general knowledge. Answers are typically short   │ │
│  │     (1-3 words)..."                                  │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │ 2. Task Demonstrations                               │ │
│  │    Q: "What is 2+2?"                                 │ │
│  │    A: "4"                                            │ │
│  │    ---                                               │ │
│  │    Q: "Capital of Japan?"                            │ │
│  │    A: "Tokyo"                                        │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │ 3. Current Instruction                               │ │
│  │    "Given the question, provide the answer."         │ │
│  ├──────────────────────────────────────────────────────┤ │
│  │ 4. Random Tip                                        │ │
│  │    "Keep the instruction clear and concise."         │ │
│  └──────────────────────────────────────────────────────┘ │
│                          │                                 │
│                          ▼                                 │
│  ┌──────────────────────────────────────────────────────┐ │
│  │               LLM Instruction Generator              │ │
│  └──────────────────────────────────────────────────────┘ │
│                          │                                 │
│                          ▼                                 │
│  Output:                                                   │
│  "You are a precise QA system. Given a question, provide  │
│   a concise, accurate answer. For math questions, compute │
│   and return only the numerical result."                  │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Các loại Tips

| Tip | Mục đích |
|-----|----------|
| `none` | Không có tip |
| `creative` | Khuyến khích sáng tạo |
| `simple` | Giữ đơn giản, súc tích |
| `descriptive` | Chi tiết, mô tả đầy đủ |
| `high_stakes` | Thêm scenario quan trọng |
| `persona` | Thêm persona ("You are an expert...") |

### 4.3 Step 3: Bayesian Optimization

#### Search Space

Với M modules, N instructions, K demo sets:
- Mỗi module có N choices cho instruction
- Mỗi module có K choices cho demo set
- Tổng: (N × K)^M combinations

**Ví dụ**: 2 modules, 10 instructions, 6 demo sets = (10 × 6)² = 3,600 combinations

#### Tại sao Bayesian Optimization?

```
❌ Grid Search:
   - Test tất cả 3,600 combinations
   - Quá tốn kém!

❌ Random Search:
   - Test ngẫu nhiên
   - Không học từ kết quả trước

✅ Bayesian Optimization:
   - Xây dựng surrogate model của f(θ)
   - Chọn điểm tiếp theo thông minh
   - Cân bằng exploration vs exploitation
   - Chỉ cần ~50 trials thay vì 3,600
```

---

## 5. Bayesian Optimization & TPE

### 5.1 Bayesian Optimization là gì?

Bayesian Optimization là phương pháp tối ưu hóa cho các hàm **expensive-to-evaluate** (tốn kém để đánh giá).

**Ý tưởng chính**:
1. Xây dựng **surrogate model** (mô hình xấp xỉ) của objective function
2. Sử dụng **acquisition function** để chọn điểm tiếp theo
3. Cân bằng giữa **exploration** (khám phá vùng mới) và **exploitation** (khai thác vùng tốt)

### 5.2 TPE (Tree-structured Parzen Estimator)

TPE là một biến thể của Bayesian Optimization đặc biệt hiệu quả cho **categorical parameters** (như indices của instructions/demos).

#### Cách hoạt động của TPE

```
┌────────────────────────────────────────────────────────────┐
│                    TPE Algorithm                           │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Trial History:                                            │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Trial 1: params=(2,3) → score=65                   │   │
│  │ Trial 2: params=(1,5) → score=72                   │   │
│  │ Trial 3: params=(4,1) → score=58                   │   │
│  │ Trial 4: params=(1,4) → score=75                   │   │
│  │ ...                                                │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                  │
│                         ▼                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Split by threshold γ (e.g., top 25%)               │   │
│  │                                                    │   │
│  │ GOOD (l): {Trial 2, Trial 4}  → scores ≥ 72       │   │
│  │ BAD  (g): {Trial 1, Trial 3}  → scores < 72       │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                  │
│                         ▼                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Build density estimators:                          │   │
│  │                                                    │   │
│  │ l(x) = P(params | score ∈ GOOD)                   │   │
│  │ g(x) = P(params | score ∈ BAD)                    │   │
│  │                                                    │   │
│  │ For categorical: just count frequencies            │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                  │
│                         ▼                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Expected Improvement:                              │   │
│  │                                                    │   │
│  │ EI(x) ∝ l(x) / g(x)                               │   │
│  │                                                    │   │
│  │ High EI = likely good AND rare in bad trials       │   │
│  │                                                    │   │
│  │ Sample x that maximizes EI                         │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                  │
│                         ▼                                  │
│  Next trial: Try x with highest EI                         │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

#### Công thức TPE

1. **Chia trials theo threshold**:
   - γ = quantile (thường 25%)
   - y* = threshold score tại quantile γ
   - Good: {(x,y) | y ≥ y*}
   - Bad: {(x,y) | y < y*}

2. **Ước lượng density**:
   - l(x) = density của x trong Good trials
   - g(x) = density của x trong Bad trials

3. **Expected Improvement**:
   ```
   EI(x) = ∫_{-∞}^{y*} (y* - y) · p(y|x) dy

   Với TPE approximation:
   EI(x) ∝ γ + (1-γ) · g(x)/l(x)

   Tối đa hóa EI ⟺ Tối đa hóa l(x)/g(x)
   ```

### 5.3 Minibatch Strategy

Để tiết kiệm chi phí, MIPRO sử dụng minibatch evaluation:

```
┌────────────────────────────────────────────────────────────┐
│                    Minibatch Strategy                      │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Trial 1: Full eval on default → Score = 70%              │
│                                                            │
│  Trial 2-6: Minibatch (25 examples each)                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │ Trial 2: params=(2,3) → mb_score=72                │   │
│  │ Trial 3: params=(1,5) → mb_score=68                │   │
│  │ Trial 4: params=(3,2) → mb_score=76                │   │
│  │ Trial 5: params=(1,3) → mb_score=74                │   │
│  │ Trial 6: params=(4,1) → mb_score=70                │   │
│  └────────────────────────────────────────────────────┘   │
│                         │                                  │
│                         ▼ Every 5 trials                   │
│  Trial 7: Full eval on top candidate (params=(3,2))       │
│           Full score = 75%? → New best!                    │
│                                                            │
│  Trial 8-12: Minibatch again...                           │
│                                                            │
│  Benefit:                                                  │
│  • 6 minibatch + 1 full ≈ 6×25 + 100 = 250 examples       │
│  • vs 7 full evals = 700 examples                          │
│  • 64% cost reduction!                                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 6. Tham số và Cấu hình

### 6.1 Thông Số Chính Xác Từ Paper Gốc (arXiv:2406.11695)

#### N - Số Lượng Instruction Candidates (Table 4)

| Task | 0-Shot MIPRO | MIPRO (có demos) |
|------|--------------|------------------|
| HotPotQA | 60 | 30 |
| HotPotQA Conditional | 35 | 30 |
| Iris | 50 | 30 |
| Heart Disease | 30 | 15 |
| ScoNe | 70 | 70 |
| HoVer | 15 | 10 |

**Giá trị phổ biến**: 30 candidates cho hầu hết tasks

#### K - Số Demo Sets
- Paper không định nghĩa K riêng biệt
- **N demo sets** được tạo cho mỗi module (cùng với N instructions)
- Bayesian optimization chọn **1 demo set** từ N candidates trong mỗi trial

#### TIPS Chính Xác (Appendix C.2)

```python
TIPS = {
    "none": "",  # Không có tip
    "creative": "Don't be afraid to be creative when creating the new instruction!",
    "simple": "Keep the instruction clear and concise.",
    "description": "Make sure your instruction is very informative and descriptive.",
    "high_stakes": "The instruction should include a high stakes scenario in which the LM must solve the task!",
    "persona": "Provide the LM with a persona that is relevant to the task (ie. 'You are a...')"
}
```

**Tips được chọn ngẫu nhiên** trong mỗi lần generate instruction để tạo diversity.

#### Các Hyperparameters Khác

| Parameter | Giá Trị | Ghi chú |
|-----------|---------|---------|
| Proposer LM temperature | 0.7 | Default, có thể optimize |
| Task LM temperature | 0.7 | Default |
| top_p sampling | 1.0 | Full sampling |
| Optimization trials | 20-50 | Tùy task complexity |
| Minibatch size B | Variable | Tùy dataset size |

### 6.2 Auto Modes (Implementation)

| Mode | num_candidates | val_size | num_trials | Use case |
|------|---------------|----------|------------|----------|
| `light` | 10 | 100 | 20 | Testing, small datasets |
| `medium` | 30 | 300 | 35 | Balanced quality/cost (recommended) |
| `heavy` | 50 | 1000 | 50 | Production, large datasets |

### 6.3 Manual Configuration

```python
optimizer = MIPROv2(
    metric=my_metric,
    llm=llm,
    auto=None,              # Disable auto mode
    num_candidates=10,      # Instructions & demo sets
    num_trials=30,          # Bayesian optimization trials
    max_bootstrapped_demos=4,  # Demos from successful runs
    max_labeled_demos=4,    # Random labeled demos
    metric_threshold=0.8,   # Bootstrap acceptance threshold
    seed=42,                # Reproducibility
    verbose=True            # Detailed logging
)
```

### 6.4 Compile Options

```python
best_program, score = optimizer.compile(
    program=my_program,
    trainset=train_data,
    valset=val_data,        # Optional, auto-split if None
    minibatch_size=25,      # Examples per minibatch eval
    minibatch_full_eval_steps=5,  # Full eval interval
)
```

---

## 7. Ví dụ thực tế

### 7.1 Question Answering

```python
from mipro_v2_scratch import *

# Setup
llm = LLMClient(api_key="sk-...", model="gpt-4o-mini")

# Custom module for QA
class QAModule(Module):
    def __init__(self):
        super().__init__()
        self.qa = Predictor(
            Signature.from_string(
                "question: the question to answer -> answer: concise answer"
            )
        )

    def forward(self, question):
        return self.qa(question=question)

# Data
trainset = [
    Example(question="What is 2+2?", answer="4").with_inputs("question"),
    Example(question="What is the capital of Japan?", answer="Tokyo").with_inputs("question"),
    Example(question="Who painted the Mona Lisa?", answer="Leonardo da Vinci").with_inputs("question"),
    # ... more examples
]

# Metric
def answer_match(example, prediction):
    pred = prediction.answer.strip().lower()
    true = example.answer.strip().lower()
    return pred == true or true in pred

# Optimize
program = QAModule()
program.set_llm(llm)

optimizer = MIPROv2(
    metric=answer_match,
    llm=llm,
    auto="light"
)

best_program, score = optimizer.compile(
    program=program,
    trainset=trainset
)

print(f"Optimized score: {score:.2f}%")

# Use optimized program
result = best_program(question="What is the speed of light?")
print(f"Answer: {result.answer}")
```

### 7.2 Multi-step Pipeline

```python
# Chain of Thought QA
class ChainOfThoughtQA(Module):
    def __init__(self):
        super().__init__()

        # Step 1: Generate reasoning
        self.reason = Predictor(
            Signature.from_string(
                "question -> reasoning: step-by-step thinking"
            )
        )

        # Step 2: Generate answer from reasoning
        self.answer = Predictor(
            Signature.from_string(
                "question, reasoning -> answer: final answer"
            )
        )

    def forward(self, question):
        reasoning_result = self.reason(question=question)
        answer_result = self.answer(
            question=question,
            reasoning=reasoning_result.reasoning
        )
        return answer_result

# MIPRO will optimize BOTH predictors
program = ChainOfThoughtQA()
program.set_llm(llm)

optimizer = MIPROv2(metric=my_metric, llm=llm, auto="medium")
best_program, score = optimizer.compile(program=program, trainset=trainset)

# Now both reason and answer predictors have optimized instructions and demos
```

---

## Tham khảo (References)

1. **Paper gốc**: [Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs](https://arxiv.org/abs/2406.11695)
2. **DSPy Documentation**: [dspy.ai](https://dspy.ai)
3. **Optuna (TPE)**: [optuna.org](https://optuna.org)
4. **Bayesian Optimization**: [A Tutorial on Bayesian Optimization](https://arxiv.org/abs/1807.02811)

---

## Sources từ tìm kiếm:

- [arXiv:2406.11695 - MIPRO Paper](https://arxiv.org/abs/2406.11695)
- [DSPy Optimizers Documentation](https://dspy.ai/learn/optimization/optimizers/)
- [Grokking MIPROv2 - Langtrace](https://www.langtrace.ai/blog/grokking-miprov2-the-new-optimizer-from-dspy)
- [MIPROv2 - DeepEval](https://deepeval.com/docs/prompt-optimization-miprov2)
- [MIPROv2 - Emergent Mind](https://www.emergentmind.com/topics/miprov2-prompt-optimization)
