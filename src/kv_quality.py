"""
kv_quality.py — correctness cost of stale KV (third and current version).

Produces Table I of the paper. Earlier versions are kept in superseded/
because the bugs they had are part of the record (see README).

WHAT WAS WRONG IN v2
--------------------
v2 reported 10 breaks / 4 fixes, p = 0.0898. Inspection of the traces showed
the numerator was inflated by two measurement bugs, both in the harness:

  1. UNICODE. The scoring regex stripped non-ASCII letters, so control
     "Brasilia" scored correct while stale "Brasilia" with an accented i
     became "bras lia" and scored wrong. Both answers were right.

  2. TRUNCATION x STYLE SHIFT. n_new was 14. Stale (base-model) KV shifts the
     generation toward a discursive style, so several stale answers were
     correct reasoning that simply had not reached the number yet:
        "The total number of shirts sold is 18 + 22"
        "The total number of plants is 6 x 9 = 5"
     Those were scored as breaks. They are not errors; they are longer.

After discounting both, roughly 5 genuine errors remained against 4 fixes,
which is noise. The v2 result does NOT support a quality cost.

WHAT v3 CHANGES
---------------
  * unicodedata NFKD normalisation before matching, so accents do not break
    scoring.
  * n_new raised 14 -> 48 so a verbose-but-correct answer can finish.
  * an explicit "answered" flag: did the arm produce the answer at all inside
    the budget, separately from whether it was right.
  * per-arm generation length recorded. The style shift may be the real
    finding: cross-version reuse may cost TOKENS (latency, cost) rather than
    CORRECTNESS, which is a different and still-publishable claim.

PREREGISTERED (restated, since v2's version was tested against buggy scoring)
----------------------------------------------------------------------------
  H1  stale accuracy < control accuracy            (McNemar, one-sided, 0.05)
  H2  break rate ordering context > arith > recall
  H3  stale generation is longer than control      (paired, one-sided)

H3 is new and is the hypothesis the v2 traces actually suggest. If H1 fails
and H3 holds, the honest paper is about a latency/verbosity cost, not a
correctness cost, and the title and abstract must be rewritten accordingly.

    python kv_quality.py --mode checkpoint \
        --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-0.5B-Instruct
"""

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import results_path
from taskbank import TASKS


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(breaks, fixes):
    n = breaks + fixes
    if n == 0:
        return 1.0
    return sum(math.comb(n, i) for i in range(breaks, n + 1)) / (2 ** n)


def sign_test(pos, neg):
    """Two-sided exact sign test on paired differences."""
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(0, k + 1))
    return min(1.0, 2 * tail / (2 ** n))


def norm(text):
    """NFKD-normalise, drop combining marks, lowercase, strip separators."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = t.lower().replace(",", "")
    return re.sub(r"[^a-z0-9\s\-\.:]", " ", t)


def correct(text, accepted):
    t = norm(text)
    return any(norm(a) in t for a in accepted)


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
    return tok.decode(toks), len(toks)


def tokens_to_answer(text, accepted, tok):
    """How many tokens before the answer appears? None if it never does."""
    t = norm(text)
    best = None
    for a in accepted:
        i = t.find(norm(a))
        if i >= 0:
            best = i if best is None else min(best, i)
    return best


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--mode", choices=["dtype", "checkpoint"],
                    default="checkpoint")
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if args.mode == "dtype":
        name_a = name_b = args.model
        dtype_a, dtype_b = torch.float32, torch.bfloat16
        label = f"{args.model}  fp32 KV -> bf16 server"
    else:
        if not (args.model_a and args.model_b):
            ap.error("--mode checkpoint needs --model-a (wrote the KV) "
                     "and --model-b (the server)")
        name_a, name_b = args.model_a, args.model_b
        dtype_a = dtype_b = torch.float32
        label = f"{name_a} KV -> {name_b} server"

    tasks = TASKS[:args.limit] if args.limit else TASKS
    print(f"=== stale-KV quality probe v3 ===\n{label}")
    print(f"{len(tasks)} tasks, n_new={args.n_new}\n")

    tok = AutoTokenizer.from_pretrained(name_b)
    model_a = load(name_a, dtype_a)
    model_b = model_a if (name_a == name_b and dtype_a == dtype_b) \
        else load(name_b, dtype_b)

    per = defaultdict(lambda: {"n": 0, "ctrl": 0, "stale": 0,
                               "break": 0, "fix": 0})
    rows, len_ctrl, len_stale, longer, shorter = [], [], [], 0, 0

    for i, (stratum, question, accepted) in enumerate(tasks, 1):
        prompt = f"Q: {question}\nA:"
        ids = tok(prompt, return_tensors="pt").input_ids
        oa = model_a(ids, use_cache=True)
        ob = model_b(ids, use_cache=True)

        ctrl, lc = generate(model_b, tok, ob.past_key_values,
                            ob.logits[:, -1, :], args.n_new)
        stale, ls = generate(model_b, tok,
                             cast_past(oa.past_key_values, dtype_b),
                             oa.logits[:, -1, :], args.n_new)

        c, s = correct(ctrl, accepted), correct(stale, accepted)
        pos_c = tokens_to_answer(ctrl, accepted, tok)
        pos_s = tokens_to_answer(stale, accepted, tok)

        d = per[stratum]
        d["n"] += 1
        d["ctrl"] += c
        d["stale"] += s
        d["break"] += (c and not s)
        d["fix"] += (s and not c)

        if c and s and pos_c is not None and pos_s is not None:
            len_ctrl.append(pos_c)
            len_stale.append(pos_s)
            longer += pos_s > pos_c
            shorter += pos_s < pos_c

        rows.append({"stratum": stratum, "q": question,
                     "ctrl_ok": c, "stale_ok": s,
                     "ctrl_chars": lc, "stale_chars": ls,
                     "ctrl_pos": pos_c, "stale_pos": pos_s,
                     "control": ctrl.strip()[:110],
                     "stale": stale.strip()[:110]})
        if c and not s:
            print(f"!! [{stratum}] {question[:50]}")
            print(f"     ctrl : {ctrl.strip()[:100]!r}")
            print(f"     stale: {stale.strip()[:100]!r}")
        if i % 30 == 0:
            print(f"   ...{i}/{len(tasks)}")

    print(f"\n{'stratum':<10} {'n':>4} {'ctrl':>7} {'stale':>7} "
          f"{'brk':>4} {'fix':>4}  break rate [95% CI]")
    tot = {"n": 0, "ctrl": 0, "stale": 0, "break": 0, "fix": 0}
    for st in ("recall", "arith", "context"):
        d = per[st]
        if not d["n"]:
            continue
        for k in tot:
            tot[k] += d[k]
        lo, hi = wilson(d["break"], d["ctrl"] or 1)
        print(f"{st:<10} {d['n']:>4} {d['ctrl']/d['n']:>7.3f} "
              f"{d['stale']/d['n']:>7.3f} {d['break']:>4} {d['fix']:>4}  "
              f"{d['break']/max(d['ctrl'],1):.3f} [{lo:.3f},{hi:.3f}]")

    n = tot["n"]
    lo, hi = wilson(tot["break"], tot["ctrl"] or 1)
    p1 = mcnemar_exact(tot["break"], tot["fix"])
    print(f"{'ALL':<10} {n:>4} {tot['ctrl']/n:>7.3f} {tot['stale']/n:>7.3f} "
          f"{tot['break']:>4} {tot['fix']:>4}  "
          f"{tot['break']/max(tot['ctrl'],1):.3f} [{lo:.3f},{hi:.3f}]")

    print(f"\nH1 correctness: drop={(tot['ctrl']-tot['stale'])/n:+.4f}  "
          f"breaks={tot['break']} fixes={tot['fix']}  McNemar p={p1:.4g}")

    if len_ctrl:
        mc = sum(len_ctrl) / len(len_ctrl)
        ms = sum(len_stale) / len(len_stale)
        p3 = sign_test(longer, shorter)
        print(f"H3 verbosity  : on {len(len_ctrl)} both-correct items, mean "
              f"chars-to-answer ctrl={mc:.1f} stale={ms:.1f} "
              f"({ms - mc:+.1f})")
        print(f"                stale longer on {longer}, shorter on "
              f"{shorter}; sign test p={p3:.4g}")
    else:
        p3 = 1.0
        print("H3 verbosity  : no both-correct items with locatable answers")

    print("\nREAD THIS:")
    print("  H1 significant  -> real correctness cost; original framing holds.")
    print("  H1 null, H3 significant -> cross-version reuse costs TOKENS, not")
    print("     correctness. Reframe the paper as a latency/verbosity cost.")
    print("  both null       -> reuse is behaviourally benign at this version")
    print("     distance; report that and scope the claim to key hygiene.")

    out = results_path("quality", label)
    with open(out, "w") as f:
        json.dump({"label": label, "n_new": args.n_new,
                   "per_stratum": {k: dict(v) for k, v in per.items()},
                   "total": tot, "mcnemar_p": p1,
                   "len_ctrl_mean": (sum(len_ctrl) / len(len_ctrl))
                   if len_ctrl else None,
                   "len_stale_mean": (sum(len_stale) / len(len_stale))
                   if len_stale else None,
                   "longer": longer, "shorter": shorter,
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
