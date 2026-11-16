#!/bin/bash
# Reproduces every number in README.md. Resumable: finished runs and evals are skipped.
set -u
PY=${PY:-python}
R=results
mkdir -p $R runs figures
T4="--num-generations 8 --batch-size 16 --grad-accum 2 --max-tokens 640 --steps 100 --vllm-mem 0.25"
EV="--dataset gsm8k --n 8 --limit 500 --ks 1,4,8 --max-tokens 640 --gpu-mem 0.70"

fit () {
  local n=$1; shift
  [ -d "runs/$n/final" ] && { echo "skip train $n"; return 0; }
  $PY train.py --name "$n" $T4 "$@"
}
score () {
  local f=$1 tag=$2 model=$3; shift 3
  grep -q "\"tag\": \"$tag\"" "$R/$f" 2>/dev/null && { echo "skip eval $tag"; return 0; }
  $PY evaluate.py --model "$model" --tag "$tag" "$@" --out "$R/$f"
}

echo "=== 1. GSM8K base, then GRPO at three seeds ==="
score gsm8k.jsonl base Qwen/Qwen2.5-1.5B-Instruct $EV
for s in 0 1 2; do
  fit "grpo_s$s" --loss-type grpo --lr 1e-5 --lora-r 16 --seed $s
  score seeds.jsonl "grpo_s$s" "runs/grpo_s$s/final" $EV
done

echo "=== 2. seven policy-gradient variants at the tuned LR ==="
for v in "grpo grpo token" "dr_grpo dr_grpo token" "dapo dapo token" "gspo dapo sequence" \
         "bnpo bnpo token" "cispo cispo token" "sapo sapo token"; do
  set -- $v
  fit "loss_$1" --lr 1e-5 --lora-r 16 --loss-type "$2" --importance-level "$3" --num-iterations 2
  score loss.jsonl "loss_$1" "runs/loss_$1/final" $EV
done

echo "=== 3. learning rate and LoRA rank ==="
fit lr_1e-6 --lr 1e-6 --lora-r 16
score sweep.jsonl lr_1e-6 runs/lr_1e-6/final $EV
for r in 1 4 16 64 256; do
  fit "lora_r$r" --lr 1e-5 --lora-r $r
  score sweep.jsonl "lora_r$r" "runs/lora_r$r/final" $EV
done

echo "=== 4. multiplication curriculum, train 1-4 digits, test 5-6 ==="
MEV="--n 32 --limit 512 --ks 1,8,32 --max-tokens 1024 --gpu-mem 0.70"
for d in 5 6; do
  score mult.jsonl "base_d$d" Qwen/Qwen2.5-1.5B-Instruct --dataset "mult$d" $MEV
done
for s in 0 1 2; do
  fit "mult_s$s" --dataset mult_curriculum --lr 1e-4 --lora-r 16 --max-tokens 1024 --seed $s
  for d in 5 6; do
    score mult.jsonl "mult_s${s}_d$d" "runs/mult_s$s/final" --dataset "mult$d" $MEV
  done
done

echo "=== 5. figures and tables ==="
$PY plots.py
$PY report.py | tee $R/tables.md
echo "=== DONE ==="
