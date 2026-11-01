"""SUPERSEDED by src/kv_quality.py. Kept so the measurement history in the
README can be checked. Writes its results to the current directory."""

import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))

"""
kv_quality_v2.py — stratified quality probe with confidence intervals.

WHAT CHANGED FROM v1
--------------------
v1 used 15 easy tasks, control scored 15/15 (ceiling), and the checkpoint run
produced 2 breaks and 0 fixes. Two events cannot support a rate claim: the
Wilson interval on 2/15 spans roughly 2% to 40%.

v2 fixes three things:
  1. 120 tasks instead of 15, so the interval is reportable.
  2. Stratified into recall / arith / context, because the mechanism predicts
     the effect should be largest where the answer depends on information in
     the prompt, and smallest for single-fact recall.
  3. Proper statistics: Wilson intervals on the break rate, and an exact
     McNemar test on the discordant pairs (breaks vs fixes), which is the
     correct test for paired binary outcomes on the same items.

WHY NOT REPEATED TRIALS
-----------------------
Greedy decoding is deterministic: re-running gives identical output. Unlike the
RAG and cache harnesses there is no run-to-run variance to average over, so all
statistical power comes from task count, not repetitions.

PREREGISTERED PREDICTIONS
-------------------------
  H1  stale accuracy < control accuracy (one-sided, McNemar).
  H2  break rate ordering: context > arith > recall.

If H1 fails, cross-version reuse changes outputs without degrading them, and
the paper must be reframed as nondeterminism rather than staleness.
If H1 holds but H2 fails (flat across strata), the effect is generic
perturbation rather than prefix contamination, which weakens the mechanism
story even with a real aggregate rate.

    python kv_quality_v2.py --mode checkpoint \
        --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import math
import re
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from taskbank import TASKS


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(breaks, fixes):
    """Exact one-sided binomial test on discordant pairs."""
    n = breaks + fixes
    if n == 0:
        return 1.0
    # P(X >= breaks) under X ~ Binomial(n, 0.5)
    tail = sum(math.comb(n, i) for i in range(breaks, n + 1))
    return tail / (2 ** n)


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
def generate(model, tok, past, first_logits, n_new):
    toks, logits = [], first_logits
    for _ in range(n_new):
        nxt = int(torch.argmax(logits, dim=-1))
        if nxt == tok.eos_token_id:
            break
        toks.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
    return tok.decode(toks)


def correct(text, accepted):
    t = re.sub(r"[^a-z0-9₂\s\-\.,:]", " ", text.lower())
    t = t.replace(",", "")
    return any(a.lower() in t for a in accepted)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--mode", choices=["dtype", "checkpoint"],
                    default="checkpoint")
    ap.add_argument("--n-new", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    args = ap.parse_args()

    if args.mode == "dtype":
        name_a = name_b = args.model
        dtype_a, dtype_b = torch.float32, torch.bfloat16
        label = f"{args.model}  fp32 KV -> bf16 server"
    else:
        name_a, name_b = args.model_a, args.model_b
        dtype_a = dtype_b = torch.float32
        label = f"{name_a} KV -> {name_b} server"

    tasks = TASKS[:args.limit] if args.limit else TASKS
    print(f"=== stale-KV quality probe v2 ===\n{label}\n{len(tasks)} tasks\n")

    tok = AutoTokenizer.from_pretrained(name_b)
    model_a = load(name_a, dtype_a)
    model_b = model_a if (name_a == name_b and dtype_a == dtype_b) \
        else load(name_b, dtype_b)

    per = defaultdict(lambda: {"n": 0, "ctrl": 0, "stale": 0,
                               "break": 0, "fix": 0})
    rows = []

    for i, (stratum, question, accepted) in enumerate(tasks, 1):
        prompt = f"Q: {question}\nA:"
        ids = tok(prompt, return_tensors="pt").input_ids
        oa = model_a(ids, use_cache=True)
        ob = model_b(ids, use_cache=True)

        ctrl = generate(model_b, tok, ob.past_key_values,
                        ob.logits[:, -1, :], args.n_new)
        stale = generate(model_b, tok, cast_past(oa.past_key_values, dtype_b),
                         oa.logits[:, -1, :], args.n_new)

        c, s = correct(ctrl, accepted), correct(stale, accepted)
        d = per[stratum]
        d["n"] += 1
        d["ctrl"] += c
        d["stale"] += s
        d["break"] += (c and not s)
        d["fix"] += (s and not c)

        rows.append({"stratum": stratum, "q": question, "ctrl_ok": c,
                     "stale_ok": s, "control": ctrl.strip()[:70],
                     "stale": stale.strip()[:70]})
        if c and not s:
            print(f"!! [{stratum}] {question[:52]}")
            print(f"     ctrl : {ctrl.strip()[:70]!r}")
            print(f"     stale: {stale.strip()[:70]!r}")
        if i % 20 == 0:
            print(f"   ...{i}/{len(tasks)}")

    print(f"\n{'stratum':<10} {'n':>4} {'ctrl':>7} {'stale':>7} "
          f"{'break':>6} {'fix':>4}  break rate [95% CI]")
    tot = {"n": 0, "ctrl": 0, "stale": 0, "break": 0, "fix": 0}
    for st in ("recall", "arith", "context"):
        d = per[st]
        if not d["n"]:
            continue
        for k in tot:
            tot[k] += d[k]
        lo, hi = wilson(d["break"], d["ctrl"] or 1)
        print(f"{st:<10} {d['n']:>4} {d['ctrl']/d['n']:>7.3f} "
              f"{d['stale']/d['n']:>7.3f} {d['break']:>6} {d['fix']:>4}  "
              f"{d['break']/max(d['ctrl'],1):.3f} [{lo:.3f},{hi:.3f}]")

    n = tot["n"]
    lo, hi = wilson(tot["break"], tot["ctrl"] or 1)
    p = mcnemar_exact(tot["break"], tot["fix"])
    print(f"{'ALL':<10} {n:>4} {tot['ctrl']/n:>7.3f} {tot['stale']/n:>7.3f} "
          f"{tot['break']:>6} {tot['fix']:>4}  "
          f"{tot['break']/max(tot['ctrl'],1):.3f} [{lo:.3f},{hi:.3f}]")
    print(f"\naccuracy drop : {(tot['ctrl']-tot['stale'])/n:+.4f}")
    print(f"McNemar exact (one-sided), breaks={tot['break']} "
          f"fixes={tot['fix']}: p = {p:.4g}")

    print("\nREAD THIS:")
    print("  p < 0.05 and breaks >> fixes -> real, one-directional quality")
    print("     cost. H1 supported.")
    print("  p not significant            -> reuse changes outputs without")
    print("     degrading them. Reframe as nondeterminism, not staleness.")
    print("  break rate flat across strata -> generic perturbation, not")
    print("     prefix contamination. H2 fails; weaker mechanism story.")

    with open("kv_quality_v2_results.json", "w") as f:
        json.dump({"label": label, "per_stratum": {k: dict(v) for k, v
                                                   in per.items()},
                   "total": tot, "mcnemar_p": p, "rows": rows}, f, indent=2)
    print("\nwrote kv_quality_v2_results.json")


if __name__ == "__main__":
    main()
