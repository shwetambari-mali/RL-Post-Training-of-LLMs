"""Sample n completions per problem and report avg@1, pass@k and maj@k."""
import argparse
import json
import os
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import torch
from transformers import AutoTokenizer

import tasks


def pass_at_k(n, c, k):
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def maj_at_k(votes, k, boot, seed):
    rng, total = random.Random(seed), 0.0
    for per_problem in votes:
        draws = [per_problem] if k >= len(per_problem) else [rng.sample(per_problem, k) for _ in range(boot)]
        hits = 0
        for d in draws:
            tally = Counter(key for key, _ in d if key is not None)
            if tally and any(key == tally.most_common(1)[0][0] and ok for key, ok in d):
                hits += 1
        total += hits / len(draws)
    return total / len(votes)


def resolve(path):
    """Merge a LoRA adapter into its base model so vLLM can load it."""
    cfg = Path(path) / "adapter_config.json"
    if not cfg.exists():
        return path, None
    from peft import AutoPeftModelForCausalLM

    merged = tempfile.mkdtemp(prefix="merged_")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    m = AutoPeftModelForCausalLM.from_pretrained(path, dtype=dtype)
    m.merge_and_unload().save_pretrained(merged)
    AutoTokenizer.from_pretrained(json.loads(cfg.read_text())["base_model_name_or_path"]).save_pretrained(merged)
    del m
    torch.cuda.empty_cache()
    return merged, path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", default="gsm8k")
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--ks", default="1,4,8")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-seqs", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--boot", type=int, default=200)
    p.add_argument("--tag", default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model_path, adapter = resolve(a.model)
    test = tasks.load(a.dataset)[1]
    test = test.select(range(min(a.limit, len(test))))
    tok = AutoTokenizer.from_pretrained(model_path)
    prompts = [tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True) for ex in test]

    from vllm import LLM, SamplingParams

    llm = LLM(model=model_path, dtype="auto", gpu_memory_utilization=a.gpu_mem,
              max_model_len=a.max_tokens + 1024, max_num_seqs=a.max_seqs)
    sp = SamplingParams(n=a.n, temperature=a.temperature, top_p=0.95, max_tokens=a.max_tokens, seed=a.seed)
    outs = llm.generate(prompts, sp)

    corrects, votes, lengths = [], [], []
    for o, ex in zip(outs, test):
        preds = [tasks.extract_answer(x.text) for x in o.outputs]
        c = [tasks.is_correct(q, ex["gold"]) for q in preds]
        corrects.append(sum(c))
        votes.append(list(zip([tasks.vote_key(q) for q in preds], c)))
        lengths.extend(len(x.token_ids) for x in o.outputs)

    res = {"model": a.model, "dataset": a.dataset, "tag": a.tag, "n_problems": len(test),
           "n_samples": a.n, "temperature": a.temperature,
           "avg@1": sum(corrects) / (len(test) * a.n),
           "mean_gen_tokens": sum(lengths) / len(lengths),
           "solved_any": sum(c > 0 for c in corrects) / len(test)}
    for k in (int(x) for x in a.ks.split(",")):
        if k <= a.n:
            res[f"pass@{k}"] = sum(pass_at_k(a.n, c, k) for c in corrects) / len(test)
            res[f"maj@{k}"] = maj_at_k(votes, k, a.boot, a.seed)

    print(json.dumps(res, indent=2))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")
    if adapter:
        shutil.rmtree(model_path, ignore_errors=True)


if __name__ == "__main__":
    main()
