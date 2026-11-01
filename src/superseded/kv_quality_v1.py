"""SUPERSEDED by src/kv_quality.py. Kept so the measurement history in the
README can be checked. Writes its results to the current directory."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))

"""
kv_quality.py — converts DIVERGENCE into QUALITY COST.

WHY THIS EXISTS
---------------
kv_severity.py measures agreement between the control continuation and the
continuation produced from stale KV. That is divergence, not error: in the
dtype run the stale output was a perfectly good alternative, and in the
checkpoint run one stale answer ("the capital of the United States is
Washington D. C.") was factually correct while disagreeing with the control.

The paper claims a "quantified output-quality cost". To support that we need
tasks with VERIFIABLE ground truth, so a continuation can be scored right or
wrong rather than same or different.

MECHANISM UNDER TEST
--------------------
Only the PREFILL KV comes from model A. Every generated token is computed by
model B. So this is not "the wrong model answered" -- it is "the right model
answered while reading a contaminated representation of the prompt."

PREREGISTERED HYPOTHESIS
------------------------
Accuracy under stale KV is lower than accuracy under the model's own KV, and
the gap widens with the distance between the producing and consuming
checkpoints (dtype skew < fine-tune skew).

FALSIFICATION
-------------
If accuracy under stale KV is statistically indistinguishable from control,
then cross-version reuse costs nothing measurable in quality, the "quality
cost" clause of the paper is unsupported, and the honest finding becomes
"cross-version reuse changes outputs without degrading them" -- a much weaker
and differently-framed result.

    python kv_quality.py --mode dtype      --model Qwen/Qwen2.5-0.5B-Instruct
    python kv_quality.py --mode checkpoint --model-a Qwen/Qwen2.5-0.5B \
                                           --model-b Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# (prompt, accepted answer substrings). Kept short so a 0.5B model can do them
# and so scoring is unambiguous.
TASKS = [
    ("Q: What is the capital of Japan?\nA:", ["tokyo"]),
    ("Q: What is 12 plus 15?\nA:", ["27", "twenty-seven", "twenty seven"]),
    ("Q: How many days are in a week?\nA:", ["7", "seven"]),
    ("Q: What color do you get mixing red and white?\nA:", ["pink"]),
    ("Q: What is the largest planet in our solar system?\nA:", ["jupiter"]),
    ("Q: What is 9 times 6?\nA:", ["54", "fifty-four", "fifty four"]),
    ("Q: In which continent is Egypt located?\nA:", ["africa"]),
    ("Q: What is the chemical symbol for water?\nA:", ["h2o", "h₂o"]),
    ("Q: How many sides does a triangle have?\nA:", ["3", "three"]),
    ("Q: What is the opposite of 'hot'?\nA:", ["cold"]),
    ("Q: What is 100 divided by 4?\nA:", ["25", "twenty-five", "twenty five"]),
    ("Q: Which ocean is the largest?\nA:", ["pacific"]),
    ("Q: What gas do humans breathe in to survive?\nA:", ["oxygen"]),
    ("Q: How many hours are in a day?\nA:", ["24", "twenty-four", "twenty four"]),
    ("Q: What is the freezing point of water in Celsius?\nA:", ["0", "zero"]),
]


def load(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    m.eval()
    return m


def cast_past(past, dtype):
    try:
        return tuple(tuple(t.to(dtype) for t in layer) for layer in past)
    except TypeError:
        for layer in past.key_cache:
            layer.data = layer.data.to(dtype)
        for layer in past.value_cache:
            layer.data = layer.data.to(dtype)
        return past


@torch.no_grad()
def answer(model, tok, ids, past, first_logits, n_new):
    toks = []
    logits = first_logits
    for _ in range(n_new):
        nxt = int(torch.argmax(logits, dim=-1))
        if nxt == tok.eos_token_id:
            break
        toks.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return tok.decode(toks)


def scores(text, accepted):
    t = text.lower()
    t = re.sub(r"[^a-z0-9₂\s\-\.]", " ", t)
    return any(a in t for a in accepted)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--mode", choices=["dtype", "checkpoint"], default="dtype")
    ap.add_argument("--n-new", type=int, default=12)
    args = ap.parse_args()

    if args.mode == "dtype":
        name_a = name_b = args.model
        dtype_a, dtype_b = torch.float32, torch.bfloat16
        label = f"{args.model}  fp32 KV -> bf16 server"
    else:
        name_a, name_b = args.model_a, args.model_b
        dtype_a = dtype_b = torch.float32
        label = f"{name_a} KV -> {name_b} server"

    print(f"=== stale-KV QUALITY probe ===\n{label}\n")
    tok = AutoTokenizer.from_pretrained(name_b)
    model_a = load(name_a, dtype_a)
    model_b = model_a if (name_a == name_b and dtype_a == dtype_b) \
        else load(name_b, dtype_b)

    rows = []
    ctrl_ok = stale_ok = both_ok = flips_bad = flips_good = 0

    for prompt, accepted in TASKS:
        ids = tok(prompt, return_tensors="pt").input_ids
        oa = model_a(ids, use_cache=True)
        ob = model_b(ids, use_cache=True)

        ctrl = answer(model_b, tok, ids, ob.past_key_values,
                      ob.logits[:, -1, :], args.n_new)
        stale = answer(model_b, tok, ids, cast_past(oa.past_key_values, dtype_b),
                       oa.logits[:, -1, :], args.n_new)

        c_ok, s_ok = scores(ctrl, accepted), scores(stale, accepted)
        ctrl_ok += c_ok
        stale_ok += s_ok
        both_ok += (c_ok and s_ok)
        flips_bad += (c_ok and not s_ok)      # stale BROKE a correct answer
        flips_good += (s_ok and not c_ok)     # stale fixed a wrong one

        rows.append({"prompt": prompt.split("\n")[0][3:],
                     "control": ctrl.strip()[:60], "control_ok": c_ok,
                     "stale": stale.strip()[:60], "stale_ok": s_ok})
        mark = "  " if c_ok == s_ok else ("!!" if c_ok else "++")
        print(f"{mark} {rows[-1]['prompt'][:44]:<46} "
              f"ctrl={'Y' if c_ok else 'n'} stale={'Y' if s_ok else 'n'}")
        if c_ok != s_ok:
            print(f"     control: {rows[-1]['control']!r}")
            print(f"     stale  : {rows[-1]['stale']!r}")

    n = len(TASKS)
    print(f"\ncontrol accuracy : {ctrl_ok}/{n} = {ctrl_ok/n:.3f}")
    print(f"stale   accuracy : {stale_ok}/{n} = {stale_ok/n:.3f}")
    print(f"quality cost     : {(ctrl_ok - stale_ok)/n:+.3f}")
    print(f"broke a correct answer : {flips_bad}/{n}")
    print(f"fixed a wrong answer   : {flips_good}/{n}")
    print("\nREAD THIS:")
    print("  stale accuracy clearly below control -> a real quality cost;")
    print("     the paper's 'quantified quality cost' clause is supported.")
    print("  accuracy indistinguishable            -> reuse changes outputs")
    print("     without degrading them. Reframe: this is nondeterminism,")
    print("     not staleness. Report that honestly.")

    with open("kv_quality_results.json", "w") as f:
        json.dump({"label": label, "n": n, "control_correct": ctrl_ok,
                   "stale_correct": stale_ok, "broke": flips_bad,
                   "fixed": flips_good, "rows": rows}, f, indent=2)
    print("\nwrote kv_quality_results.json")


if __name__ == "__main__":
    main()
