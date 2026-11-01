"""
distance_sweep.py — where is the cliff?

THE RESULT SO FAR
-----------------
Three points on the Qwen family:

    fp32 -> bf16          accuracy 0.925 -> 0.942   benign
    base -> instruct      accuracy 0.942 -> 0.925   benign
    Qwen2 -> Qwen2.5      accuracy 0.942 -> 0.033   decoherence

Architecture and tokenizer were verified identical for the third pair, so the
collapse is KV incompatibility, not a shape or token-id mismatch.

Two points benign, one catastrophic, nothing in between. That is consistent
with a CLIFF, but three scattered points cannot distinguish a cliff from a
steep sigmoid, and cannot say where the boundary sits.

WHY PYTHIA
----------
EleutherAI published intermediate TRAINING CHECKPOINTS of each Pythia model
under HuggingFace revisions (step1000, step2000, ... step143000). Same
architecture, same tokenizer, same everything -- only training progress
differs. That gives a genuinely continuous distance axis on one model, which
is exactly what is needed to locate a boundary.

    server  = final checkpoint (step143000)
    cache   = an earlier checkpoint (the "old version" whose KV was written)

As the cache checkpoint moves earlier, weight distance grows. If accuracy
falls off a cliff at some distance, we have located it. If it declines
smoothly, the "binary, not continuous" claim is wrong and the paper must say
so.

DISTANCE METRIC
---------------
Mean per-parameter cosine distance between the two checkpoints' weight
tensors, plus relative L2. Reported alongside accuracy so the x-axis is a
measured quantity rather than a label.

PREREGISTERED
-------------
  H4  accuracy vs weight distance is a step function (a narrow region where
      it falls from near-control to near-zero), not a gradual decline.

  Falsification: a smooth monotonic decline across the sweep. That would make
  the honest claim "quality degrades continuously with weight distance", which
  revives a bounded-staleness knob and changes the paper again.

    python distance_sweep.py --model EleutherAI/pythia-160m
    python distance_sweep.py --model EleutherAI/pythia-410m --limit 60
"""

import argparse
import json
import math

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from taskbank import TASKS
from common import results_path
from kv_quality import correct, generate, mcnemar_exact

# Pythia revisions, newest last. Distance from the final checkpoint grows as
# we move up this list.
DEFAULT_STEPS = ["step1000", "step8000", "step32000", "step64000",
                 "step100000", "step128000", "step143000"]


def weight_distance(model_a, model_b):
    """Mean cosine distance and relative L2 over matching parameter tensors."""
    cos_sum, l2_num, l2_den, n = 0.0, 0.0, 0.0, 0
    pb = dict(model_b.named_parameters())
    for name, pa in model_a.named_parameters():
        if name not in pb:
            continue
        a = pa.detach().flatten().float()
        b = pb[name].detach().flatten().float()
        if a.shape != b.shape:
            continue
        denom = (a.norm() * b.norm()).item()
        if denom > 0:
            cos_sum += 1.0 - float((a @ b).item() / denom)
            n += 1
        l2_num += float(((a - b) ** 2).sum().item())
        l2_den += float((a ** 2).sum().item())
    return (cos_sum / max(n, 1), math.sqrt(l2_num / max(l2_den, 1e-12)))


@torch.no_grad()
def evaluate(model_a, model_b, tok, tasks, n_new):
    ctrl_ok = stale_ok = brk = fix = 0
    for _, question, accepted in tasks:
        ids = tok(f"Q: {question}\nA:", return_tensors="pt").input_ids
        oa = model_a(ids, use_cache=True)
        ob = model_b(ids, use_cache=True)
        ctrl, _ = generate(model_b, tok, ob.past_key_values,
                           ob.logits[:, -1, :], n_new)
        stale, _ = generate(model_b, tok, oa.past_key_values,
                            oa.logits[:, -1, :], n_new)
        c, s = correct(ctrl, accepted), correct(stale, accepted)
        ctrl_ok += c
        stale_ok += s
        brk += (c and not s)
        fix += (s and not c)
    n = len(tasks)
    return {"n": n, "ctrl": ctrl_ok / n, "stale": stale_ok / n,
            "break": brk, "fix": fix,
            "mcnemar_p": mcnemar_exact(brk, fix)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--final", default="step143000")
    ap.add_argument("--steps", nargs="*", default=None)
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    steps = args.steps or DEFAULT_STEPS
    tasks = TASKS[:args.limit] if args.limit else TASKS

    print(f"=== KV distance sweep ===\n{args.model}")
    print(f"server = {args.final};  cache written by each earlier checkpoint")
    print(f"{len(tasks)} tasks, n_new={args.n_new}\n")

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.final)
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.final, torch_dtype=torch.float32)
    model_b.eval()

    rows = []
    print(f"{'cache ckpt':>12} {'cos dist':>9} {'rel L2':>8} "
          f"{'ctrl':>6} {'stale':>6} {'brk':>4} {'fix':>4} {'p':>10}")
    for step in steps:
        model_a = AutoModelForCausalLM.from_pretrained(
            args.model, revision=step, torch_dtype=torch.float32)
        model_a.eval()

        cos_d, l2 = weight_distance(model_a, model_b)
        r = evaluate(model_a, model_b, tok, tasks, args.n_new)
        r.update({"step": step, "cos_dist": cos_d, "rel_l2": l2})
        rows.append(r)
        print(f"{step:>12} {cos_d:>9.5f} {l2:>8.4f} {r['ctrl']:>6.3f} "
              f"{r['stale']:>6.3f} {r['break']:>4} {r['fix']:>4} "
              f"{r['mcnemar_p']:>10.3g}")
        del model_a

    print("\nREAD THIS:")
    print("  a narrow region where stale accuracy drops from ~control to ~0")
    print("     -> CLIFF. H4 supported; validity is binary.")
    print("  a smooth decline across the sweep")
    print("     -> GRADIENT. H4 falsified; a bounded-staleness knob is back")
    print("        and the paper must be reframed again.")

    out = results_path("distance_sweep", args.model)
    with open(out, "w") as f:
        json.dump({"model": args.model, "final": args.final,
                   "n_tasks": len(tasks), "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
