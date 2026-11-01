"""
interp_sweep.py — locate the cliff on a synthesised, continuous distance axis.

WHY THIS REPLACES THE PYTHIA SWEEP
----------------------------------
distance_sweep.py used Pythia training checkpoints to get a continuous
distance axis. The axis worked (cosine distance fell cleanly 0.646 -> 0.0022
-> -0.00006 at the identity check) but the experiment was uninformative:
Pythia-160m scored 0.025 on the task bank as the CONTROL. You cannot measure a
drop from a baseline that is already at the floor. The instrument was right and
the subject was wrong.

This version keeps the continuous axis but restores a capable control by
constructing the intermediate checkpoints instead of downloading them:

    theta(alpha) = (1 - alpha) * theta_A + alpha * theta_B

with A = Qwen2-0.5B (the cache-writing model, known to collapse) and
B = Qwen2.5-0.5B-Instruct (the server, control accuracy 0.942).

alpha = 1 reproduces the server exactly, so it doubles as an identity check.
alpha = 0 reproduces the known-collapsing pair. Everything in between is a
synthetic "version" at a measured weight distance from the server.

WHAT THIS DOES AND DOES NOT SHOW
--------------------------------
Interpolated weights are not a real intermediate release; no vendor ships
theta(0.5). What the sweep establishes is the SHAPE of the relationship
between weight distance and KV compatibility: whether reuse degrades smoothly
with distance or fails past a threshold. That shape is the claim under test,
and it is a property of the representation geometry, not of any particular
release schedule. The paper must state this scoping explicitly.

PREREGISTERED (H4, restated with a capable control)
---------------------------------------------------
  Accuracy vs alpha is a step function: near-control for alpha above some
  threshold, near-zero below it, with a narrow transition.

  Falsification: a smooth monotonic decline across alpha. That revives a
  bounded-staleness knob and the paper is reframed again.

Also reported: interpolated-model SELF accuracy (how good theta(alpha) is on
its own). If a mid-alpha model is itself incoherent, its KV failing tells us
nothing, so that column is needed to interpret the sweep honestly.

    python interp_sweep.py --limit 60
"""

import argparse
import copy
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from taskbank import TASKS
from common import results_path
from kv_quality import correct, generate, mcnemar_exact

ALPHAS = [0.0, 0.25, 0.5, 0.75, 0.875, 0.95, 1.0]


def interpolate(model_a, model_b, alpha):
    """theta = (1-alpha)*A + alpha*B, returned as a fresh model."""
    m = copy.deepcopy(model_b)
    sa = model_a.state_dict()
    sb = model_b.state_dict()
    new = {}
    for k, vb in sb.items():
        va = sa.get(k)
        if va is not None and va.shape == vb.shape and vb.is_floating_point():
            new[k] = (1.0 - alpha) * va.float() + alpha * vb.float()
        else:
            new[k] = vb
    m.load_state_dict(new)
    m.eval()
    return m


def weight_distance(model_a, model_b):
    cos_sum, n = 0.0, 0
    pb = dict(model_b.named_parameters())
    for name, pa in model_a.named_parameters():
        if name not in pb:
            continue
        a = pa.detach().flatten().float()
        b = pb[name].detach().flatten().float()
        if a.shape != b.shape:
            continue
        d = (a.norm() * b.norm()).item()
        if d > 0:
            cos_sum += 1.0 - float((a @ b).item() / d)
            n += 1
    return cos_sum / max(n, 1)


@torch.no_grad()
def evaluate(model_a, model_b, tok, tasks, n_new):
    """model_a wrote the KV; model_b is the server. Also scores A alone."""
    ctrl = stale = self_a = brk = fix = 0
    for _, question, accepted in tasks:
        ids = tok(f"Q: {question}\nA:", return_tensors="pt").input_ids
        oa = model_a(ids, use_cache=True)
        ob = model_b(ids, use_cache=True)

        c_txt, _ = generate(model_b, tok, ob.past_key_values,
                            ob.logits[:, -1, :], n_new)
        s_txt, _ = generate(model_b, tok, oa.past_key_values,
                            oa.logits[:, -1, :], n_new)
        a_txt, _ = generate(model_a, tok, oa.past_key_values,
                            oa.logits[:, -1, :], n_new)

        c, s = correct(c_txt, accepted), correct(s_txt, accepted)
        ctrl += c
        stale += s
        self_a += correct(a_txt, accepted)
        brk += (c and not s)
        fix += (s and not c)
    n = len(tasks)
    return {"ctrl": ctrl / n, "stale": stale / n, "self_a": self_a / n,
            "break": brk, "fix": fix, "mcnemar_p": mcnemar_exact(brk, fix)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", default="Qwen/Qwen2-0.5B")
    ap.add_argument("--model-b", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--alphas", nargs="*", type=float, default=None)
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    alphas = args.alphas or ALPHAS
    tasks = TASKS[:args.limit] if args.limit else TASKS

    print("=== KV interpolation sweep ===")
    print(f"cache = (1-a)*{args.model_a} + a*{args.model_b}")
    print(f"server = {args.model_b}")
    print(f"{len(tasks)} tasks, n_new={args.n_new}\n")

    tok = AutoTokenizer.from_pretrained(args.model_b)
    base_a = AutoModelForCausalLM.from_pretrained(
        args.model_a, torch_dtype=torch.float32).eval()
    model_b = AutoModelForCausalLM.from_pretrained(
        args.model_b, torch_dtype=torch.float32).eval()

    rows = []
    print(f"{'alpha':>7} {'cos dist':>9} {'selfA':>7} {'ctrl':>7} "
          f"{'stale':>7} {'brk':>4} {'fix':>4} {'p':>10}")
    for a in alphas:
        m_a = model_b if a >= 1.0 else interpolate(base_a, model_b, a)
        d = weight_distance(m_a, model_b)
        r = evaluate(m_a, model_b, tok, tasks, args.n_new)
        r.update({"alpha": a, "cos_dist": d})
        rows.append(r)
        print(f"{a:>7.3f} {d:>9.5f} {r['self_a']:>7.3f} {r['ctrl']:>7.3f} "
              f"{r['stale']:>7.3f} {r['break']:>4} {r['fix']:>4} "
              f"{r['mcnemar_p']:>10.3g}")
        if m_a is not model_b:
            del m_a

    print("\nREAD THIS:")
    print("  stale near ctrl above a threshold and near 0 below it, with a")
    print("  narrow transition -> CLIFF. H4 supported.")
    print("  stale declining smoothly across alpha -> GRADIENT. H4 falsified.")
    print("  Ignore any row where selfA is near 0: an incoherent interpolated")
    print("  model tells us nothing about KV compatibility.")

    out = results_path("interp_sweep")
    with open(out, "w") as f:
        json.dump({"model_a": args.model_a, "model_b": args.model_b,
                   "n_tasks": len(tasks), "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
