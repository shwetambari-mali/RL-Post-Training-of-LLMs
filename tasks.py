"""GSM8K and a synthetic multiplication curriculum, plus answer grading."""
import random
import re

from datasets import Dataset, concatenate_datasets, load_dataset

SYSTEM_PROMPT = (
    "Reason step by step inside <think> </think>, then give the final answer inside "
    "<answer> </answer>. Put only the answer itself inside the answer tags."
)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
FORMAT_RE = re.compile(r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$", re.DOTALL)
NUMONLY_RE = re.compile(r"^-?\d+(\.\d+)?$")
NUM_RE = re.compile(r"-?\d+\.?\d*")


def _row(question, gold):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        "gold": gold,
    }


def gsm8k():
    ds = load_dataset("openai/gsm8k", "main")
    fn = lambda ex: _row(ex["question"], ex["answer"].split("####")[-1].strip().replace(",", ""))
    return (ds["train"].map(fn, remove_columns=ds["train"].column_names),
            ds["test"].map(fn, remove_columns=ds["test"].column_names))


def mult(digits, n_train=1000, n_test=512, seed=0):
    rng = random.Random(f"mult|{digits}|{seed}")
    lo, hi = 10 ** (digits - 1), 10**digits - 1
    seen, rows, tries = set(), [], 0
    cap = 20 * (n_train + n_test)
    while len(rows) < n_train + n_test:
        a, b = rng.randint(lo, hi), rng.randint(lo, hi)
        tries += 1
        if (a, b) in seen and tries < cap:
            continue
        seen.add((a, b))
        rows.append(_row(f"Compute {a} * {b}.", str(a * b)))
    return Dataset.from_list(rows[:n_train]), Dataset.from_list(rows[n_train:])


def mult_curriculum(digits=(1, 2, 3, 4), n_train=4000, n_test=512, seed=0):
    """Training mix over several digit counts; test split is unused, evaluate per level."""
    per = n_train // len(digits)
    parts = [mult(d, per, 1, seed)[0] for d in digits]
    return concatenate_datasets(parts).shuffle(seed=seed), None


def load(name):
    if name == "gsm8k":
        return gsm8k()
    if name == "mult_curriculum":
        return mult_curriculum()
    if name.startswith("mult"):
        return mult(int(name[4:]))
    raise ValueError(f"unknown dataset {name}")


def extract_answer(text):
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1).strip()
    nums = NUM_RE.findall(text)
    return nums[-1] if nums else None


def _clean(pred):
    return str(pred).strip().replace(",", "").replace("$", "").replace("\\", "").rstrip(".")


def is_correct(pred, gold):
    """Exact numeric match; rejects unevaluated expressions a symbolic grader would accept."""
    if pred is None:
        return False
    p = _clean(pred)
    if not NUMONLY_RE.match(p):
        return False
    try:
        return abs(float(p) - float(_clean(gold))) < 1e-6
    except ValueError:
        return False


def vote_key(pred):
    if pred is None:
        return None
    nums = NUM_RE.findall(_clean(pred))
    return (nums[-1].lstrip("0") or "0") if nums else _clean(pred)
