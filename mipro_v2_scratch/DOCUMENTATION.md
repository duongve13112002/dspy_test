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

#### Mục đích
Sinh ra **nhiều instruction candidates đa dạng** bằng cách sử dụng LLM với các context khác nhau.

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

### 6.1 Auto Modes

| Mode | num_candidates | val_size | num_trials | Use case |
|------|---------------|----------|------------|----------|
| `light` | 6 | 100 | 15 | Testing, small datasets |
| `medium` | 12 | 300 | 30 | Balanced quality/cost |
| `heavy` | 18 | 1000 | 50 | Production, large datasets |

### 6.2 Manual Configuration

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

### 6.3 Compile Options

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
