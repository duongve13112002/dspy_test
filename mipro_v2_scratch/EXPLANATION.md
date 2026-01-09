# Giải thích chi tiết về MIPROv2 (Detailed Explanation)

## Mục lục
1. [Tổng quan về MIPRO](#1-tổng-quan-về-mipro)
2. [Step 1: Bootstrap Demonstrations](#2-step-1-bootstrap-demonstrations)
3. [Step 2: Propose Instructions](#3-step-2-propose-instructions)
4. [Step 3: Bayesian Optimization](#4-step-3-bayesian-optimization)
5. [Kiến trúc kỹ thuật](#5-kiến-trúc-kỹ-thuật)
6. [Ví dụ minh họa](#6-ví-dụ-minh-họa)

---

## 1. Tổng quan về MIPRO

### 1.1 MIPRO là gì?

**MIPRO** (Multi-Prompt Instruction Optimization) là một phương pháp tối ưu hóa prompt tự động cho các Language Model programs. Thay vì phải viết prompt thủ công, MIPRO tự động:

1. **Tạo ra nhiều instruction candidates** cho mỗi module
2. **Bootstrap các few-shot examples** chất lượng cao
3. **Tìm tổ hợp tối ưu** bằng Bayesian Optimization

### 1.2 Tại sao cần MIPRO?

```
Vấn đề truyền thống:
┌─────────────────────────────────────────────────────────────┐
│  Developer viết prompt → Test → Sửa → Test → Sửa → ...     │
│                                                             │
│  ❌ Tốn thời gian                                           │
│  ❌ Phụ thuộc vào kinh nghiệm                               │
│  ❌ Khó tìm được tối ưu toàn cục                            │
└─────────────────────────────────────────────────────────────┘

Giải pháp MIPRO:
┌─────────────────────────────────────────────────────────────┐
│  Data + Program → MIPRO → Optimal Instructions + Demos      │
│                                                             │
│  ✅ Tự động                                                 │
│  ✅ Data-driven                                             │
│  ✅ Bayesian search tìm tối ưu                              │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 Luồng hoạt động tổng quan

```
                        MIPROv2 Pipeline
                        ================

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   STEP 1    │    │   STEP 2    │    │   STEP 3    │
    │  Bootstrap  │───▶│   Propose   │───▶│  Bayesian   │
    │Demonstrations│    │Instructions │    │Optimization │
    └─────────────┘    └─────────────┘    └─────────────┘
          │                   │                   │
          ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │  N sets of  │    │ M candidate │    │   Search    │
    │  few-shot   │    │instructions │    │   optimal   │
    │   demos     │    │per predictor│    │combination  │
    └─────────────┘    └─────────────┘    └─────────────┘
                                                │
                                                ▼
                                    ┌───────────────────┐
                                    │ Optimized Program │
                                    │   with best       │
                                    │ instructions &    │
                                    │     demos         │
                                    └───────────────────┘
```

---

## 2. Step 1: Bootstrap Demonstrations

### 2.1 Mục đích

Tạo ra **N bộ demonstrations đa dạng** để:
1. Cung cấp context cho instruction generation
2. Làm few-shot examples trong program cuối cùng
3. Tạo diversity cho Bayesian search

### 2.2 Quy trình Bootstrap

```
                    Bootstrap Process
                    =================

    Training Data                    Teacher Model
    ┌───────────┐                    ┌───────────┐
    │ Example 1 │───┐                │  Program  │
    │ Example 2 │───┼───▶ Run ───▶   │  with     │───▶ Prediction
    │ Example 3 │───┤                │  demos    │
    │    ...    │   │                └───────────┘
    └───────────┘   │                      │
                    │                      ▼
                    │                ┌───────────┐
                    │                │  Metric   │
                    │                │ Evaluate  │
                    │                └───────────┘
                    │                      │
                    │            ┌─────────┴─────────┐
                    │            ▼                   ▼
                    │      Pass (✓)            Fail (✗)
                    │            │                   │
                    │            ▼                   │
                    │   ┌──────────────┐             │
                    │   │ Save trace   │             │
                    └──▶│ as demo      │◀────────────┘
                        └──────────────┘   (not saved)
```

### 2.3 Các loại Demo Sets

MIPROv2 tạo ra nhiều loại demo sets khác nhau:

| Set | Mô tả | Mục đích |
|-----|-------|----------|
| Set -3 | Zero-shot (không có demos) | Baseline |
| Set -2 | Labeled only (random samples) | Simple baseline |
| Set -1 | Unshuffled bootstrap | Deterministic baseline |
| Set 0+ | Shuffled bootstrap với random size | Diversity |

### 2.4 Code minh họa

```python
# Pseudocode cho bootstrap process
def bootstrap_one_example(example, teacher, metric):
    # 1. Run teacher model
    with trace_context():
        prediction = teacher(example.inputs())
        trace = get_trace()  # Lấy intermediate steps

    # 2. Evaluate with metric
    score = metric(example, prediction)

    # 3. Keep if passes threshold
    if score >= threshold:
        for step in trace:
            predictor, inputs, outputs = step
            demo = Example(augmented=True, **inputs, **outputs)
            save_demo(predictor, demo)
        return True

    return False
```

---

## 3. Step 2: Propose Instructions

### 3.1 Grounded Proposer

"Grounded" có nghĩa là instruction generation được **căn cứ vào dữ liệu thực tế**, không chỉ là abstract prompting.

```
                    Grounded Proposer Architecture
                    ==============================

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Dataset    │  │   Program    │  │    Task      │      │
│  │   Summary    │  │   Structure  │  │   Demos      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │               │
│         │    ┌────────────┴────────────┐    │               │
│         └───▶│                         │◀───┘               │
│              │  Instruction Generator  │                    │
│         ┌───▶│        (LLM)            │◀───┐               │
│         │    └──────────┬──────────────┘    │               │
│         │               │                   │               │
│  ┌──────┴───────┐       │            ┌──────┴───────┐      │
│  │  Prompting   │       ▼            │  Instruction │      │
│  │    Tips      │  ┌─────────┐       │   History    │      │
│  └──────────────┘  │ New     │       └──────────────┘      │
│                    │Instruc- │                              │
│                    │tion     │                              │
│                    └─────────┘                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Các thành phần của Grounded Proposer

#### 3.2.1 Dataset Summary (Data-aware)

```python
# Ví dụ Dataset Summary
"""
The dataset contains question-answer pairs focused on arithmetic
and basic knowledge. Questions are typically short (5-15 words)
and answers are concise single words or numbers. The task appears
to be a factoid QA system requiring precise recall.
"""
```

#### 3.2.2 Program Structure (Program-aware)

```python
# Program được phân tích để hiểu structure
class QAProgram(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)  # Retrieval step
        self.answer = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question)
        return self.answer(context=context, question=question)

# Proposer sẽ hiểu: "Đây là RAG pipeline với retrieval và reasoning"
```

#### 3.2.3 Task Demonstrations (Fewshot-aware)

```
Input: What is the capital of France?
Output: Paris

Input: Who wrote Romeo and Juliet?
Output: William Shakespeare
```

#### 3.2.4 Prompting Tips

| Tip Type | Ví dụ |
|----------|-------|
| `creative` | "Don't be afraid to be creative!" |
| `simple` | "Keep the instruction clear and concise." |
| `description` | "Make instruction very informative and descriptive." |
| `high_stakes` | "Include a high stakes scenario!" |
| `persona` | 'Include "You are a..."' |

### 3.3 Instruction Generation Flow

```
For each predictor P_i:
    For each demo_set_j:
        1. Select random tip
        2. Build context:
           - dataset_summary (if data_aware)
           - program_description (if program_aware)
           - module_description (if program_aware)
           - task_demos from demo_set_j
           - basic_instruction (current)
           - previous_instructions (if using history)
           - tip

        3. Generate with LLM:
           "Generate a new instruction for this module..."

        4. Save instruction_candidates[P_i].append(new_instruction)
```

---

## 4. Step 3: Bayesian Optimization

### 4.1 Tại sao Bayesian Optimization?

```
Vấn đề: Tìm tổ hợp tối ưu trong không gian lớn

Ví dụ với 2 predictors, 10 instructions mỗi predictor, 6 demo sets:
- Số tổ hợp = 10 × 6 × 10 × 6 = 3,600 combinations
- Không thể test hết!

Giải pháp: Bayesian Optimization
- Xây dựng surrogate model của objective function
- Chọn điểm tiếp theo dựa trên acquisition function
- Cân bằng exploration vs exploitation
```

### 4.2 Search Space

```python
# Parameter space cho mỗi predictor
search_space = {
    "0_predictor_instruction": [0, 1, 2, ..., N-1],  # N instruction candidates
    "0_predictor_demos": [0, 1, 2, ..., M-1],        # M demo sets
    "1_predictor_instruction": [0, 1, 2, ..., N-1],
    "1_predictor_demos": [0, 1, 2, ..., M-1],
    # ... cho mỗi predictor
}
```

### 4.3 TPE Sampler (Tree-structured Parzen Estimator)

```
                    TPE Algorithm
                    =============

    Trial History                    Next Trial
    ┌───────────┐                    ┌───────────┐
    │ Trial 1   │                    │  Sample   │
    │ Score: 65 │                    │  from     │
    ├───────────┤         ┌──────▶  │  l(x)/g(x)│
    │ Trial 2   │         │          └───────────┘
    │ Score: 72 │         │
    ├───────────┤         │
    │ Trial 3   │   Split │
    │ Score: 58 │   ──────┤
    ├───────────┤         │
    │    ...    │         │
    └───────────┘         │
          │               │
          ▼               │
    ┌───────────┐         │
    │ Good/Bad  │         │
    │ Threshold │─────────┘
    │  (γ=0.25) │
    └───────────┘

    l(x) = density của params trong "good" trials
    g(x) = density của params trong "bad" trials

    EI(x) ∝ l(x)/g(x)  →  Sample từ l(x), ưu tiên cao l(x)/g(x)
```

### 4.4 Minibatching Strategy

Để tiết kiệm compute, MIPROv2 dùng minibatching:

```
                    Minibatch Strategy
                    ==================

Trial 1: Full eval on default program
         Score = 70%

Trial 2: Minibatch (35 examples)     ─┐
Trial 3: Minibatch (35 examples)      │
Trial 4: Minibatch (35 examples)      ├─ Collect scores
Trial 5: Minibatch (35 examples)      │
Trial 6: Minibatch (35 examples)     ─┘
         │
         ▼ Find highest averaging program

Trial 7: Full eval on top candidate
         Score = 75%? → New best!

Trial 8: Minibatch ...
...
```

### 4.5 Optimization Loop

```python
# Pseudocode cho optimization loop
def optimize(program, instruction_candidates, demo_candidates):
    # Create Optuna study with TPE sampler
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True)
    )

    # Add default program as baseline
    study.add_trial(create_trial(default_params, default_score))

    for trial_num in range(num_trials):
        # 1. Sample parameters using TPE
        params = trial.suggest_categorical(...)

        # 2. Build candidate program
        candidate = program.deepcopy()
        for i, predictor in enumerate(candidate.predictors()):
            set_instruction(predictor, instruction_candidates[i][params[f"{i}_instruction"]])
            if demo_candidates:
                set_demos(predictor, demo_candidates[i][params[f"{i}_demos"]])

        # 3. Evaluate (minibatch or full)
        score = evaluate(candidate, batch_size)

        # 4. Report to Optuna (updates surrogate model)
        return score

    return best_program
```

---

## 5. Kiến trúc kỹ thuật

### 5.1 Class Diagram

```
                         MIPROv2 Architecture
                         ====================

┌─────────────────────────────────────────────────────────────┐
│                         MIPROv2                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ - metric: Callable                                   │    │
│  │ - prompt_model: LM                                   │    │
│  │ - task_model: LM                                     │    │
│  │ - auto: "light"|"medium"|"heavy"|None                │    │
│  ├─────────────────────────────────────────────────────┤    │
│  │ + compile(student, trainset, ...) -> Module          │    │
│  │ - _bootstrap_fewshot_examples()                      │    │
│  │ - _propose_instructions()                            │    │
│  │ - _optimize_prompt_parameters()                      │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                  │
│           ┌───────────────┼───────────────┐                 │
│           ▼               ▼               ▼                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │BootstrapFew- │ │  Grounded-   │ │  Evaluator   │        │
│  │    Shot      │ │   Proposer   │ │              │        │
│  │              │ │              │ │              │        │
│  │- metric      │ │- prompt_model│ │- devset      │        │
│  │- max_demos   │ │- program_code│ │- metric      │        │
│  │- max_rounds  │ │- data_summary│ │- num_threads │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│         │                │                                  │
│         ▼                ▼                                  │
│  ┌──────────────┐ ┌──────────────┐                         │
│  │ LabeledFew-  │ │ Instruction- │                         │
│  │    Shot      │ │  Generator   │                         │
│  └──────────────┘ └──────────────┘                         │
│                          │                                  │
│                          ▼                                  │
│                  ┌──────────────┐                           │
│                  │ Dataset-     │                           │
│                  │ Summary-     │                           │
│                  │ Generator    │                           │
│                  └──────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Data Flow

```
Input:
├── student (DSPy Module)
├── trainset (list[Example])
├── metric (Callable)
└── config (auto, num_candidates, ...)

                    ┌────────────────┐
trainset ──────────▶│ Step 1         │
student  ──────────▶│ Bootstrap      │──────▶ demo_candidates
metric   ──────────▶│                │        Dict[int, List[List[Demo]]]
                    └────────────────┘
                           │
                           ▼
                    ┌────────────────┐
trainset ──────────▶│ Step 2         │
student  ──────────▶│ Propose        │──────▶ instruction_candidates
demo_candidates ───▶│                │        Dict[int, List[str]]
                    └────────────────┘
                           │
                           ▼
                    ┌────────────────┐
student ───────────▶│ Step 3         │
instruction_cands ─▶│ Optimize       │──────▶ best_program
demo_candidates ───▶│                │        (DSPy Module with
valset ────────────▶│                │         optimized params)
metric ────────────▶└────────────────┘

Output:
├── best_program.score (float)
├── best_program.trial_logs (dict)
├── best_program.candidate_programs (list)
└── best_program (với optimized instructions & demos)
```

---

## 6. Ví dụ minh họa

### 6.1 Ví dụ: Tối ưu Question Answering

```python
import dspy
from mipro_v2_scratch import MIPROv2

# Setup
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Program ban đầu với instruction đơn giản
class SimpleQA(dspy.Module):
    def __init__(self):
        # Instruction mặc định: "question -> answer"
        self.qa = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.qa(question=question)

# Data
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="Capital of Japan?", answer="Tokyo").with_inputs("question"),
    # ... more examples
]

# Metric
def exact_match(example, pred):
    return pred.answer.strip().lower() == example.answer.strip().lower()

# Optimize
optimizer = MIPROv2(metric=exact_match, auto="light", verbose=True)
optimized = optimizer.compile(SimpleQA(), trainset=trainset)

# Kết quả
print(f"Best score: {optimized.score}%")

# Xem instruction đã được tối ưu
for i, pred in enumerate(optimized.predictors()):
    print(f"\nPredictor {i}:")
    print(f"Instruction: {pred.signature.instructions}")
    print(f"Demos: {len(pred.demos)} examples")
```

### 6.2 Kết quả mong đợi

**Trước tối ưu:**
```
Instruction: "Given the fields `question`, produce the fields `answer`."
Demos: []
Score: 45%
```

**Sau tối ưu:**
```
Instruction: "You are a precise question answering system. Given a question,
provide a concise, accurate answer. For factual questions, respond with
the specific fact requested. For math questions, compute and return only
the numerical result."
Demos: [
    Example(question="What is 5+3?", answer="8"),
    Example(question="Capital of France?", answer="Paris"),
    ...
]
Score: 82%
```

---

## Tổng kết

MIPROv2 là một powerful optimizer cho DSPy programs với 3 bước chính:

1. **Bootstrap Demonstrations**: Tạo diverse demo sets bằng metric-validated bootstrapping
2. **Propose Instructions**: Sinh instruction candidates sử dụng grounded approach (data-aware, program-aware)
3. **Bayesian Optimization**: Tìm tổ hợp tối ưu bằng TPE với minibatching

Key advantages:
- **Automatic**: Không cần prompt engineering thủ công
- **Data-driven**: Instructions được sinh dựa trên data characteristics
- **Efficient**: TPE + minibatching giúp tìm optimum nhanh
- **Reproducible**: Với seed cố định, kết quả reproducible

---

*Tài liệu này được viết cho mục đích giáo dục và tham khảo. Xem thêm paper gốc về MIPRO để hiểu sâu hơn về lý thuyết.*
