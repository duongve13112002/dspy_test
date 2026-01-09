# MIPROv2 - Model-based Instruction Prompt Optimization v2

## Tổng quan (Overview)

MIPROv2 là một optimizer mạnh mẽ cho DSPy programs, sử dụng phương pháp tối ưu hóa Bayesian để tìm ra sự kết hợp tốt nhất giữa instructions và few-shot demonstrations cho mỗi predictor trong chương trình.

**English**: MIPROv2 is a powerful optimizer for DSPy programs that uses Bayesian optimization to find the best combination of instructions and few-shot demonstrations for each predictor in your program.

## Cài đặt (Installation)

```bash
# Đảm bảo bạn đã cài đặt DSPy và các dependencies
pip install dspy-ai optuna numpy tqdm
```

## Cách sử dụng cơ bản (Basic Usage)

```python
import dspy
from mipro_v2_scratch import MIPROv2

# 1. Cấu hình Language Model
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 2. Định nghĩa chương trình DSPy của bạn
class MyProgram(dspy.Module):
    def __init__(self):
        self.predict = dspy.Predict("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# 3. Định nghĩa metric đánh giá
def my_metric(example, prediction):
    return prediction.answer.lower() == example.answer.lower()

# 4. Chuẩn bị dữ liệu
trainset = [
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    # ... thêm nhiều ví dụ khác
]

# 5. Tạo optimizer và chạy tối ưu
optimizer = MIPROv2(
    metric=my_metric,
    auto="light",  # Có thể là "light", "medium", "heavy"
)

optimized_program = optimizer.compile(
    student=MyProgram(),
    trainset=trainset,
)

# 6. Sử dụng chương trình đã tối ưu
result = optimized_program(question="What is 3+3?")
print(result.answer)
```

## Các tham số cấu hình (Configuration Parameters)

### MIPROv2 Constructor

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `metric` | Hàm đánh giá `(example, prediction) -> score` | **Bắt buộc** |
| `prompt_model` | Model để sinh instructions | `dspy.settings.lm` |
| `task_model` | Model để chạy program | `dspy.settings.lm` |
| `auto` | Chế độ tự động: "light", "medium", "heavy" | `"light"` |
| `num_candidates` | Số lượng candidates (khi auto=None) | `None` |
| `max_bootstrapped_demos` | Số demos bootstrapped tối đa | `4` |
| `max_labeled_demos` | Số labeled demos tối đa | `4` |
| `num_threads` | Số threads song song | `None` |
| `seed` | Random seed | `9` |
| `verbose` | In chi tiết quá trình | `False` |

### Compile Method

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `student` | DSPy program cần tối ưu | **Bắt buộc** |
| `trainset` | Tập dữ liệu training | **Bắt buộc** |
| `teacher` | Teacher program (optional) | `None` |
| `valset` | Tập validation (tự chia nếu None) | `None` |
| `num_trials` | Số trials tối ưu (khi auto=None) | `None` |
| `minibatch` | Sử dụng minibatch | `True` |
| `minibatch_size` | Kích thước minibatch | `35` |
| `program_aware_proposer` | Xem xét cấu trúc program | `True` |
| `data_aware_proposer` | Xem xét đặc điểm dữ liệu | `True` |
| `tip_aware_proposer` | Sử dụng tips khi sinh instruction | `True` |

## Chế độ Auto (Auto Modes)

| Mode | Số candidates | Val size | Phù hợp cho |
|------|---------------|----------|-------------|
| `"light"` | 6 | 100 | Test nhanh, dataset nhỏ |
| `"medium"` | 12 | 300 | Cân bằng chất lượng/thời gian |
| `"heavy"` | 18 | 1000 | Production, dataset lớn |

## Ví dụ nâng cao (Advanced Examples)

### Sử dụng Manual Mode

```python
optimizer = MIPROv2(
    metric=my_metric,
    auto=None,  # Tắt auto mode
    num_candidates=10,  # Tự định số candidates
)

optimized = optimizer.compile(
    student=my_program,
    trainset=trainset,
    num_trials=50,  # Bắt buộc khi auto=None
)
```

### Tùy chỉnh Models

```python
# Dùng model mạnh hơn để sinh instructions
prompt_lm = dspy.LM("openai/gpt-4o")
task_lm = dspy.LM("openai/gpt-4o-mini")

optimizer = MIPROv2(
    metric=my_metric,
    prompt_model=prompt_lm,  # Model để sinh instructions
    task_model=task_lm,      # Model để chạy program
    auto="medium",
)
```

### Zero-shot Optimization

```python
# Chỉ tối ưu instructions, không dùng few-shot demos
optimizer = MIPROv2(
    metric=my_metric,
    max_bootstrapped_demos=0,
    max_labeled_demos=0,
    auto="medium",
)
```

### Xem kết quả chi tiết

```python
optimized = optimizer.compile(student=my_program, trainset=trainset)

# Xem điểm tốt nhất
print(f"Best score: {optimized.score}")

# Xem logs của các trials
for trial_num, log in optimized.trial_logs.items():
    print(f"Trial {trial_num}: {log.get('full_eval_score', log.get('mb_score'))}")

# Xem các chương trình được đánh giá đầy đủ
for candidate in optimized.candidate_programs:
    print(f"Score: {candidate['score']}")
```

## Cấu trúc thư mục (Directory Structure)

```
mipro_v2_scratch/
├── __init__.py          # Package initialization
├── mipro_v2.py          # Main MIPROv2 optimizer class
├── proposer.py          # Instruction proposal module
├── dataset_summary.py   # Dataset summary generator
├── bootstrap.py         # Few-shot bootstrapping
├── evaluator.py         # Evaluation utilities
├── utils.py             # Helper utilities
└── README.md            # Documentation (this file)
```

## Các module chính (Main Modules)

### 1. MIPROv2 (`mipro_v2.py`)
Class chính điều phối toàn bộ quá trình tối ưu 3 bước.

### 2. GroundedProposer (`proposer.py`)
Sinh instruction candidates dựa trên:
- Dataset summary (data-aware)
- Program structure (program-aware)
- Task demonstrations (fewshot-aware)
- Prompting tips (tip-aware)

### 3. BootstrapFewShot (`bootstrap.py`)
Tạo các bộ demonstrations bằng cách:
- Chạy examples qua teacher model
- Đánh giá với metric
- Giữ lại các traces đạt ngưỡng

### 4. DatasetSummaryGenerator (`dataset_summary.py`)
Phân tích và tóm tắt đặc điểm của training data.

### 5. Evaluator (`evaluator.py`)
Đánh giá programs với hỗ trợ:
- Parallel execution
- Minibatch evaluation
- Error handling

## Lưu ý quan trọng (Important Notes)

1. **Memory**: Với dataset lớn và nhiều candidates, optimizer có thể sử dụng nhiều memory. Sử dụng mode "light" cho testing.

2. **API Calls**: Quá trình tối ưu yêu cầu nhiều LLM calls. Ước tính số calls:
   - Prompt model: ~10 (data summary) + N × M (N candidates × M predictors)
   - Task model: trials × batch_size

3. **Reproducibility**: Luôn set `seed` để có kết quả reproducible.

4. **Metric Function**: Metric nên trả về số (0-1) hoặc boolean. Giá trị cao = tốt hơn.

## Troubleshooting

### "trainset cannot be empty"
Đảm bảo trainset có ít nhất 2 examples.

### "num_candidates required when auto=None"
Khi auto=None, phải cung cấp cả `num_candidates` và `num_trials`.

### "minibatch_size exceeds valset size"
Giảm `minibatch_size` hoặc tăng kích thước validation set.

## License

MIT License - xem file LICENSE để biết thêm chi tiết.
