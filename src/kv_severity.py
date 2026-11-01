"""
kv_severity.py — the decisive probe for the stale-KV paper.

QUESTION
--------
LMCache's cache key is  model_name@world_size@worker_id@chunk_hash@dtype.
Precision has been part of it since November 2025 (LMCache PR #1859); model
revision is not. So two servers running the *same model name* at a different
checkpoint revision collide on the key: server B can load KV tensors that
server A computed under different weights. (The script was written when the
documented key had no dtype either, which is why the dtype mode exists; it
is now a control for a case LMCache already separates.)

The engineering answer is "put the version in the key." The interesting
question, and the one that decides whether there is a paper here, is:

    when B reuses A's KV, does output quality degrade GRACEFULLY
    (a tunable staleness/savings trade) or COLLAPSE (no knob, just
    "always invalidate")?

PREREGISTERED HYPOTHESIS
------------------------
Output quality degrades continuously with the "distance" between the producing
and consuming configurations: small for dtype skew, larger for a fine-tune of
the same base, largest across unrelated checkpoints. If instead output is
garbage at every non-identical configuration, there is no bounded-staleness
knob and we report the null.

FALSIFICATION
-------------
If greedy-token agreement collapses to near chance for even the mildest skew
(dtype), the "deliberate trade" framing is dead. We report that and stop.

WHAT IT DOES
------------
1. Prefill a prompt under configuration A, capture past_key_values.
2. Continue generation under configuration B, once with B's OWN KV (control)
   and once with A's KV (stale reuse).
3. Compare the two continuations.

Metrics
    exact_prefix_len : greedy tokens agreeing before first divergence
    token_agreement  : fraction of positions matching the control
    first_div_idx    : where it first diverges
    kl_first_step    : KL(control || stale) on the first next-token distribution
    ppl_ratio        : perplexity of the stale continuation under B / control

Runs on CPU. Start with a 0.5B model.

    pip install "transformers>=4.40" torch --index-url https://download.pytorch.org/whl/cpu
    python kv_severity.py --model Qwen/Qwen2.5-0.5B-Instruct --mode dtype
    python kv_severity.py --model-a <base> --model-b <finetune> --mode checkpoint
"""

import argparse
import json

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import results_path

PROMPTS = [
    "The capital of France is Paris. The capital of Italy is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "In distributed systems, a cache is said to be stale when",
    "Q: What is 17 multiplied by 23?\nA: Let me work through this step by step.",
    "The three primary colors are red, blue, and",
]


def load(name, dtype):
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    m.eval()
    return m


@torch.no_grad()
def prefill(model, ids):
    """Run the prompt and return (past_key_values, last-position logits)."""
    out = model(ids, use_cache=True)
    return out.past_key_values, out.logits[:, -1, :]


@torch.no_grad()
def continue_greedy(model, past, first_logits, n_new, dtype):
    """Greedy-decode n_new tokens starting from `past`."""
    toks, dists = [], []
    logits = first_logits
    for _ in range(n_new):
        dists.append(F.log_softmax(logits.float(), dim=-1))
        nxt = int(torch.argmax(logits, dim=-1))
        toks.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return toks, dists


def cast_past(past, dtype):
    """Cast a KV cache to another dtype (models differing only in precision)."""
    try:
        return tuple(tuple(t.to(dtype) for t in layer) for layer in past)
    except TypeError:
        # transformers Cache object
        for layer in past.key_cache:
            layer.data = layer.data.to(dtype)
        for layer in past.value_cache:
            layer.data = layer.data.to(dtype)
        return past


@torch.no_grad()
def compare(model_b, tok, prompt, past_a, past_b, logits_a, logits_b,
            n_new, dtype_b):
    """Control = B's own KV. Stale = A's KV handed to B."""
    ctrl_toks, ctrl_d = continue_greedy(model_b, past_b, logits_b, n_new, dtype_b)
    stale_toks, stale_d = continue_greedy(
        model_b, cast_past(past_a, dtype_b), logits_a, n_new, dtype_b)

    first_div = next((i for i, (a, b) in enumerate(zip(ctrl_toks, stale_toks))
                      if a != b), n_new)
    agree = sum(a == b for a, b in zip(ctrl_toks, stale_toks)) / max(n_new, 1)
    kl = float(F.kl_div(stale_d[0], ctrl_d[0], log_target=True,
                        reduction="sum"))

    return {
        "prompt": prompt[:48],
        "exact_prefix_len": first_div,
        "first_div_idx": first_div,
        "token_agreement": round(agree, 4),
        "kl_first_step": round(kl, 5),
        "control": tok.decode(ctrl_toks),
        "stale": tok.decode(stale_toks),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--model-a")
    ap.add_argument("--model-b")
    ap.add_argument("--mode", choices=["dtype", "checkpoint"], default="dtype")
    ap.add_argument("--n-new", type=int, default=24)
    args = ap.parse_args()

    if args.mode == "dtype":
        # Same weights, different precision. LMCache keys on dtype since
        # Nov 2025, so this is now a control rather than a live collision.
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

    print(f"=== stale-KV severity probe ===\n{label}\n")
    tok = AutoTokenizer.from_pretrained(name_b)
    model_a = load(name_a, dtype_a)
    model_b = model_a if (name_a == name_b and dtype_a == dtype_b) \
        else load(name_b, dtype_b)

    rows = []
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids
        past_a, log_a = prefill(model_a, ids)
        past_b, log_b = prefill(model_b, ids)
        r = compare(model_b, tok, p, past_a, past_b, log_a, log_b,
                    args.n_new, dtype_b)
        rows.append(r)
        print(f"  {r['prompt']!r}")
        print(f"    agreement={r['token_agreement']}  "
              f"first_div={r['first_div_idx']}/{args.n_new}  "
              f"KL={r['kl_first_step']}")
        print(f"    control: {r['control']!r}")
        print(f"    stale  : {r['stale']!r}\n")

    mean_agree = sum(r["token_agreement"] for r in rows) / len(rows)
    mean_div = sum(r["first_div_idx"] for r in rows) / len(rows)
    print(f"MEAN token agreement : {mean_agree:.4f}")
    print(f"MEAN first divergence: {mean_div:.1f} / {args.n_new}")
    print("\nREAD THIS:")
    print("  agreement near 1.0  -> graceful; a bounded-staleness knob exists.")
    print("  agreement near 0.0  -> collapse; no knob. Report the null, stop.")
    print("  in between          -> the interesting case; map the frontier.")

    out = results_path("severity", label)
    with open(out, "w") as f:
        json.dump({"label": label, "rows": rows,
                   "mean_agreement": mean_agree}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
