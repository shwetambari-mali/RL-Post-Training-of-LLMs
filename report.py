"""Print the markdown result tables used in README.md."""
import json
from pathlib import Path
from statistics import mean, stdev

R = Path("results")
if not list(R.glob("*.jsonl")):
    R = R / "reference"


def rows(name):
    p = R / name
    return [json.loads(l) for l in p.open()] if p.exists() else []


def by_tag(name):
    return {r["tag"]: r for r in rows(name)}


def pct(x):
    return f"{100 * x:.2f}"


def table(header, body):
    if not body:
        return ""
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in body]
    return "\n".join(out)


def headline():
    g, s = by_tag("gsm8k.jsonl"), by_tag("sweep.jsonl")
    if "base" not in g:
        return ""
    body = [["base", pct(g["base"]["avg@1"]), pct(g["base"]["maj@8"]), pct(g["base"]["pass@8"]),
             f"{g['base']['mean_gen_tokens']:.0f}"]]
    for tag, label, src in [("grpo_s0", "GRPO, r=16", by_tag("seeds.jsonl")),
                            ("lora_r64", "GRPO, r=64 (best)", s)]:
        if tag in src:
            r = src[tag]
            body.append([label, pct(r["avg@1"]), pct(r["maj@8"]), pct(r["pass@8"]),
                         f"{r['mean_gen_tokens']:.0f}"])
    return table(["model", "avg@1 %", "maj@8 %", "pass@8 %", "tokens"], body)


def seeds():
    d = rows("seeds.jsonl")
    if len(d) < 2:
        return ""
    v = sorted(100 * r["avg@1"] for r in d)
    return (f"Three identical runs differing only in seed: **{', '.join(f'{x:.2f}' for x in v)}**  \n"
            f"Spread **{v[-1] - v[0]:.2f} points**, standard deviation **{stdev(v):.2f}**.")


def losses():
    d = rows("loss.jsonl")
    sd = rows("seeds.jsonl")
    if not d:
        return ""
    body = [[r["tag"].replace("loss_", ""), pct(r["avg@1"]), pct(r["maj@8"]),
             f"{r['mean_gen_tokens']:.0f}"] for r in sorted(d, key=lambda r: -r["avg@1"])]
    v = [100 * r["avg@1"] for r in d]
    out = table(["variant", "avg@1 %", "maj@8 %", "tokens"], body)
    spread = max(v) - min(v)
    note = f"\n\nSpread across {len(d)} variants: **{spread:.2f} points**."
    if len(sd) > 1:
        sv = [100 * r["avg@1"] for r in sd]
        n = max(sv) - min(sv)
        note += f" Seed noise is **{n:.2f}**, so the spread is **{spread / n:.1f}x** noise: these are not tied."
    return out + note


def ranks():
    s, g = by_tag("sweep.jsonl"), by_tag("gsm8k.jsonl")
    base = g.get("base")
    body = []
    for r in (1, 4, 16, 64, 256):
        t = f"lora_r{r}"
        if t in s:
            d = s[t]
            delta = f"{100 * (d['avg@1'] - base['avg@1']):+.2f}" if base else ""
            body.append([r, pct(d["avg@1"]), pct(d["maj@8"]), pct(d["pass@8"]), delta])
    return table(["LoRA rank", "avg@1 %", "maj@8 %", "pass@8 %", "vs base"], body)


def learning_rate():
    s, g = by_tag("sweep.jsonl"), by_tag("gsm8k.jsonl")
    body = []
    if "base" in g:
        body.append(["no training", pct(g["base"]["avg@1"])])
    if "lr_1e-6" in s:
        body.append(["1e-6", pct(s["lr_1e-6"]["avg@1"])])
    if "lora_r16" in s:
        body.append(["1e-5", pct(s["lora_r16"]["avg@1"])])
    return table(["learning rate (rank 16)", "avg@1 %"], body)


def mult():
    d = rows("mult.jsonl")
    if not d:
        return ""
    body = []
    for dg in (5, 6):
        b = [r for r in d if r["tag"] == f"base_d{dg}"]
        t = [r for r in d if r["tag"].startswith("mult_s") and r["tag"].endswith(f"d{dg}")]
        if not b or not t:
            continue
        for k in (1, 32):
            tv = [100 * r[f"pass@{k}"] for r in t]
            s = f" ± {stdev(tv):.1f}" if len(tv) > 1 else ""
            body.append([f"{dg}-digit", f"pass@{k}", pct(b[0][f"pass@{k}"]),
                         f"{mean(tv):.2f}{s}", f"{max(tv):.2f}"])
    return table(["held out", "metric", "base %", "GRPO % (3 seeds)", "best seed %"], body)


SECTIONS = [("Headline", headline), ("Seed noise floor", seeds),
            ("Policy-gradient variants", losses), ("LoRA rank", ranks),
            ("Learning rate", learning_rate), ("Multiplication curriculum", mult)]

if __name__ == "__main__":
    for title, fn in SECTIONS:
        t = fn()
        if t:
            print(f"\n### {title}\n\n{t}")
