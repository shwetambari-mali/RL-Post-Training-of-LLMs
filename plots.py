"""Generate the figures used in README.md from results/*.jsonl and runs/*/metrics.jsonl."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results")
if not list(R.glob("*.jsonl")):
    R = R / "reference"
F = Path("figures")
BLUE, GREY, RED, GREEN = "#2563eb", "#94a3b8", "#dc2626", "#059669"
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.facecolor": "white"})


def rows(name):
    p = R / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


def by_tag(name):
    return {r["tag"]: r for r in rows(name)}


def save(fig, path):
    F.mkdir(exist_ok=True)
    fig.savefig(F / path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", F / path)


def bare(ax):
    ax.spines[["top", "right"]].set_visible(False)


def fig_headline():
    g, s = by_tag("gsm8k.jsonl"), by_tag("sweep.jsonl")
    if "base" not in g or "lora_r64" not in s:
        return
    base, best = g["base"], s["lora_r64"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    v = [100 * base["avg@1"], 100 * best["avg@1"]]
    b = ax.bar(["base model", "after GRPO"], v, color=[GREY, BLUE], width=0.5)
    ax.bar_label(b, fmt="%.1f%%", padding=4, fontweight="bold")
    ax.set_ylabel("GSM8K accuracy (avg@1, 500 problems)")
    ax.set_ylim(0, 92)
    ax.set_title(f"Qwen2.5-1.5B-Instruct   {v[1]-v[0]:+.1f} points", color=BLUE)
    bare(ax)
    save(fig, "headline.png")


def fig_rank():
    s = by_tag("sweep.jsonl")
    g = by_tag("gsm8k.jsonl")
    ranks = [r for r in (1, 4, 16, 64, 256) if f"lora_r{r}" in s]
    if len(ranks) < 4:
        return
    acc = [100 * s[f"lora_r{r}"]["avg@1"] for r in ranks]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    if "base" in g:
        ax.axhline(100 * g["base"]["avg@1"], color=GREY, ls="--", lw=1.2)
        ax.text(1.05, 100 * g["base"]["avg@1"] + 0.8, "base model", color=GREY, fontsize=8)
    ax.plot(ranks, acc, "o-", color=BLUE, lw=1.8, ms=6)
    peak = acc.index(max(acc))
    ax.plot(ranks[peak], acc[peak], "o", color=RED, ms=10, mfc="none", mew=2)
    for r, a in zip(ranks, acc):
        ax.annotate(f"{a:.1f}", (r, a), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.set_xlabel("LoRA rank")
    ax.set_ylabel("GSM8K accuracy (avg@1)")
    ax.set_ylim(50, 82)
    ax.set_title(f"capacity threshold, peak at rank {ranks[peak]}")
    bare(ax)
    save(fig, "lora_rank.png")


def fig_loss_variants():
    loss, seeds = by_tag("loss.jsonl"), rows("seeds.jsonl")
    if not loss:
        return
    ks = sorted(loss, key=lambda k: loss[k]["avg@1"])
    v = [100 * loss[k]["avg@1"] for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b = ax.bar([k.replace("loss_", "") for k in ks], v, color=BLUE, width=0.62)
    ax.bar_label(b, fmt="%.2f", padding=3, fontsize=8)
    if len(seeds) >= 2:
        sv = [100 * r["avg@1"] for r in seeds]
        lo, hi = min(sv), max(sv)
        ax.axhspan(lo, hi, color=RED, alpha=0.16, zorder=0)
        ax.axhline(sum(sv) / len(sv), color=RED, lw=1.1, zorder=1)
        ax.text(-0.42, max(v) + 1.3, f"red band = seed noise over {len(sv)} identical runs ({hi-lo:.2f} pts)",
                color=RED, fontsize=8, ha="left")
    ax.set_ylim(min(v) - 2.2, max(v) + 2.2)
    ax.set_ylabel("GSM8K accuracy (avg@1)")
    ax.set_title(f"spread {max(v)-min(v):.2f} points, {(max(v)-min(v))/max(hi-lo, 1e-9):.1f}x the seed noise"
                 if len(seeds) >= 2 else f"spread {max(v)-min(v):.2f} points")
    bare(ax)
    save(fig, "loss_variants.png")


def fig_sharpening():
    g, s = by_tag("gsm8k.jsonl"), by_tag("sweep.jsonl")
    loss = rows("loss.jsonl")
    if "base" not in g or not loss:
        return
    base = g["base"]
    tr = sum(100 * r["avg@1"] for r in loss) / len(loss)
    trp8 = sum(100 * r["pass@8"] for r in loss) / len(loss)
    trm8 = sum(100 * r["maj@8"] for r in loss) / len(loss)
    labels = ["avg@1\n(greedy-ish)", "maj@8\n(self-consistency)", "pass@8\n(coverage)"]
    bv = [100 * base["avg@1"], 100 * base["maj@8"], 100 * base["pass@8"]]
    tv = [tr, trm8, trp8]
    x = range(3)
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar([i - 0.2 for i in x], bv, 0.38, label="base", color=GREY)
    ax.bar([i + 0.2 for i in x], tv, 0.38, label="after GRPO (mean of 7)", color=BLUE)
    for i, (a, c) in enumerate(zip(bv, tv)):
        ax.annotate(f"{a:.1f}", (i - 0.2, a), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
        ax.annotate(f"{c:.1f}", (i + 0.2, c), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax.hlines(bv[1], 0.08, 0.92, color=RED, ls="--", lw=1.2)

    ax.set_xticks(list(x), labels)
    ax.set_ylabel("GSM8K (%)")
    ax.set_ylim(0, 102)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    ax.set_title(f"RL sharpens what the base model already knew\n"
                 f"trained avg@1 {tv[0]:.1f} lands on base maj@8 {bv[1]:.1f}, {abs(bv[1]-tv[0]):.1f} pt apart")
    bare(ax)
    save(fig, "sharpening.png")


def _smooth(y, w):
    out = []
    for i in range(len(y)):
        lo, hi = max(0, i - w // 2), min(len(y), i + w // 2 + 1)
        out.append(sum(y[lo:hi]) / (hi - lo))
    return out


def fig_curves():
    want = [("grpo_s0", "GSM8K, lr 1e-5", BLUE), ("lr_1e-6", "GSM8K, lr 1e-6", GREY),
            ("mult_s0", "multiplication, lr 1e-4", GREEN)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    for name, label, c in want:
        p = Path("runs") / name / "metrics.jsonl"
        if not p.exists():
            continue
        m = [json.loads(l) for l in p.open()]
        st = [r["step"] for r in m]
        for ax, key in ((a1, "rewards/correctness/mean"), (a2, "completions/mean_length")):
            y = [r[key] for r in m]
            ax.plot(st, y, color=c, lw=0.8, alpha=0.22)
            ax.plot(st, _smooth(y, 11), color=c, lw=1.9, label=label if ax is a1 else None)
    a1.set_xlabel("training step")
    a1.set_ylabel("mean correctness reward")
    a1.legend(frameon=False, fontsize=8)
    a1.set_title("reward")
    a2.set_xlabel("training step")
    a2.set_ylabel("mean completion length (tokens)")
    a2.set_title("response length")
    bare(a1)
    bare(a2)
    fig.tight_layout()
    save(fig, "training_curves.png")


def fig_mult():
    d = rows("mult.jsonl")
    if not d:
        return
    ks = [1, 8, 32]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    base = next((r for r in d if r["tag"] == "base_d5"), None)
    tr = [r for r in d if r["tag"].startswith("mult_s") and r["tag"].endswith("d5")]
    if not base or not tr:
        return
    ax.plot(ks, [100 * base[f"pass@{k}"] for k in ks], "o--", color=GREY, label="base")
    for i, r in enumerate(sorted(tr, key=lambda r: r["tag"])):
        ax.plot(ks, [100 * r[f"pass@{k}"] for k in ks], "o-", color=BLUE, alpha=0.85,
                label="after GRPO (3 seeds)" if i == 0 else None)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("k (samples per problem)")
    ax.set_ylabel("pass@k (%)")
    ax.set_title("5-digit multiplication, held out\ntrained on 1-4 digits only")
    ax.legend(frameon=False, fontsize=8)
    bare(ax)
    save(fig, "mult.png")


if __name__ == "__main__":
    for f in (fig_headline, fig_rank, fig_loss_variants, fig_sharpening, fig_curves, fig_mult):
        f()
