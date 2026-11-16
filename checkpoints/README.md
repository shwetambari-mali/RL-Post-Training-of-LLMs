# Checkpoints

LoRA adapters only. Load on top of `Qwen/Qwen2.5-1.5B-Instruct`.

| adapter | task | held-out accuracy |
|---|---|---|
| `gsm8k_lora_r64` | GSM8K | 75.67% avg@1, base 55.25% |
| `mult_curriculum_lora_r16` | multiplication, trained on 1-4 digits | 24.65% avg@1 on unseen 5-digit, base 0.46% |

```python
from peft import AutoPeftModelForCausalLM
m = AutoPeftModelForCausalLM.from_pretrained("checkpoints/gsm8k_lora_r64").merge_and_unload()
```

Or evaluate directly:

```bash
python evaluate.py --model checkpoints/gsm8k_lora_r64 --tag check --dataset gsm8k --n 8 --limit 500
```

`mult_curriculum_lora_r16` was trained with a larger group size (16 generations, batch 32,
grad-accum 4) than the T4 profile in `run_all.sh`, which reaches about 17% pass@32 rather than the
66.60% this adapter achieves. Reproducing it needs roughly 40 GB of VRAM.

The adapter directories are git-ignored because they exceed GitHub's comfortable file size. To
publish them, push to the Hugging Face Hub:

```bash
huggingface-cli upload <your-user>/grpo-math-gsm8k-r64 checkpoints/gsm8k_lora_r64
```
