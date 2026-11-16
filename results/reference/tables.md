
### Headline

| model | avg@1 % | maj@8 % | pass@8 % | tokens |
|---|---|---|---|---|
| base | 55.25 | 72.00 | 89.00 | 132 |
| GRPO, r=16 | 71.35 | 80.60 | 91.40 | 170 |
| GRPO, r=64 (best) | 75.67 | 83.40 | 93.80 | 180 |

### Seed noise floor

Three identical runs differing only in seed: **70.70, 70.75, 71.35**  
Spread **0.65 points**, standard deviation **0.36**.

### Policy-gradient variants

| variant | avg@1 % | maj@8 % | tokens |
|---|---|---|---|
| cispo | 73.98 | 82.60 | 193 |
| dapo | 72.62 | 82.20 | 185 |
| bnpo | 72.55 | 82.40 | 192 |
| dr_grpo | 72.25 | 81.40 | 173 |
| sapo | 71.38 | 83.40 | 165 |
| gspo | 70.83 | 82.40 | 201 |
| grpo | 69.67 | 80.60 | 158 |

Spread across 7 variants: **4.30 points**. Seed noise is **0.65**, so the spread is **6.6x** noise: these are not tied.

### LoRA rank

| LoRA rank | avg@1 % | maj@8 % | pass@8 % | vs base |
|---|---|---|---|---|
| 1 | 56.85 | 73.40 | 87.80 | +1.60 |
| 4 | 61.88 | 81.20 | 90.00 | +6.63 |
| 16 | 70.97 | 80.40 | 90.80 | +15.72 |
| 64 | 75.67 | 83.40 | 93.80 | +20.43 |
| 256 | 73.30 | 78.60 | 89.00 | +18.05 |

### Learning rate

| learning rate (rank 16) | avg@1 % |
|---|---|
| no training | 55.25 |
| 1e-6 | 55.93 |
| 1e-5 | 70.97 |

### Multiplication curriculum

| held out | metric | base % | GRPO % (3 seeds) | best seed % |
|---|---|---|---|---|
| 5-digit | pass@1 | 0.46 | 2.41 ± 2.6 | 5.18 |
| 5-digit | pass@32 | 4.30 | 10.16 ± 8.8 | 16.99 |
| 6-digit | pass@1 | 0.02 | 0.10 ± 0.1 | 0.24 |
| 6-digit | pass@32 | 0.39 | 0.59 ± 0.7 | 1.37 |
