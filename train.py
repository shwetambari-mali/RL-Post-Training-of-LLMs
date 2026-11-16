"""GRPO training with colocated vLLM generation and optional LoRA."""
import argparse
import json
import os
import socket
import time
from pathlib import Path

if "MASTER_PORT" not in os.environ:
    with socket.socket() as _s:
        _s.bind(("", 0))
        os.environ["MASTER_PORT"] = str(_s.getsockname()[1])

import torch
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import tasks
from rewards import REWARDS

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class Recorder(TrainerCallback):
    def __init__(self, path):
        self.path, self.rows, self.t0 = Path(path), [], time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or "loss" not in logs:
            return
        self.rows.append({**logs, "step": state.global_step, "wall_s": time.time() - self.t0})
        self.path.write_text("\n".join(json.dumps(r) for r in self.rows))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--reward", default="correctness_format", choices=list(REWARDS))
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--loss-type", default="dapo", help="grpo | dr_grpo | dapo | bnpo | cispo | sapo")
    p.add_argument("--importance-level", default="token", choices=["token", "sequence"])
    p.add_argument("--num-generations", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--num-iterations", type=int, default=1)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16, help="0 disables LoRA (full fine-tuning)")
    p.add_argument("--vllm-mem", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    bf16 = torch.cuda.is_bf16_supported()

    peft_config = None
    if a.lora_r:
        from peft import LoraConfig

        peft_config = LoraConfig(task_type="CAUSAL_LM", r=a.lora_r, lora_alpha=2 * a.lora_r,
                                 lora_dropout=0.0, target_modules=LORA_TARGETS)

    cfg = GRPOConfig(
        output_dir=str(out), run_name=a.name, seed=a.seed, data_seed=a.seed,
        model_init_kwargs={"dtype": "bfloat16" if bf16 else "float16", "attn_implementation": "sdpa"},
        use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=a.vllm_mem,
        vllm_max_model_length=a.max_tokens + 768, vllm_enable_sleep_mode=True,
        learning_rate=a.lr, lr_scheduler_type="constant_with_warmup", warmup_steps=10,
        max_grad_norm=1.0, adam_beta2=0.98, bf16=bf16, fp16=not bf16,
        gradient_checkpointing=True,
        per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=a.grad_accum,
        num_generations=a.num_generations, num_iterations=a.num_iterations,
        max_completion_length=a.max_tokens, temperature=1.0, top_p=1.0, max_steps=a.steps,
        beta=0.0, loss_type=a.loss_type, scale_rewards="group",
        importance_sampling_level=a.importance_level, epsilon=0.2, epsilon_high=0.28,
        mask_truncated_completions=True, logging_steps=1, save_strategy="no",
        eval_strategy="no", report_to=[], disable_tqdm=False,
    )

    rec = Recorder(out / "metrics.jsonl")
    trainer = GRPOTrainer(model=a.model, reward_funcs=REWARDS[a.reward], args=cfg,
                          train_dataset=tasks.load(a.dataset)[0], peft_config=peft_config,
                          callbacks=[rec])
    trainer.train()
    trainer.save_model(str(out / "final"))
    summary = {"name": a.name, "steps": len(rec.rows),
               "wall_s": rec.rows[-1]["wall_s"] if rec.rows else None,
               "mean_step_s": sum(float(r.get("step_time", 0)) for r in rec.rows) / max(len(rec.rows), 1),
               **vars(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("name", "steps", "wall_s", "mean_step_s")}))


if __name__ == "__main__":
    main()
