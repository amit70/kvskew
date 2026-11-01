"""
family_matrix.py — is the cliff Qwen-specific?

The three-point result so far is entirely within the Qwen family. Before
claiming that KV-cache validity is binary in general, the same pattern needs to
appear in at least one other model family, with the same shape:

    within-family fine-tune   -> benign
    across-generation         -> collapse

PAIRS
-----
Each pair is (cache_model, server_model, label). The cache model wrote the KV;
the server model consumes it. Pairs are skipped automatically if the
architectures are incompatible (different layer count, kv head count or head
dim), since that is a crash rather than a silent hazard and is not the effect
under study.

Defaults use SmolLM / SmolLM2 (135M, open, CPU-friendly) which have both a
fine-tune pair and a generation boundary, mirroring the Qwen structure.

    python family_matrix.py                 # SmolLM family
    python family_matrix.py --limit 60      # faster
"""

import argparse
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from taskbank import TASKS
from common import results_path
from distance_sweep import evaluate, weight_distance

DEFAULT_PAIRS = [
    ("HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-135M-Instruct",
     "SmolLM2 base -> instruct (fine-tune)"),
    ("HuggingFaceTB/SmolLM-135M", "HuggingFaceTB/SmolLM2-135M-Instruct",
     "SmolLM -> SmolLM2 (generation)"),
]


def arch_key(name):
    c = AutoConfig.from_pretrained(name)
    heads = getattr(c, "num_attention_heads", None)
    return (
        getattr(c, "num_hidden_layers", None),
        getattr(c, "num_key_value_heads", heads),
        (c.hidden_size // heads) if heads else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-new", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="triples: cache server label")
    args = ap.parse_args()

    pairs = DEFAULT_PAIRS
    if args.pairs:
        it = iter(args.pairs)
        pairs = list(zip(it, it, it))

    tasks = TASKS[:args.limit] if args.limit else TASKS
    print(f"=== cross-family replication ===\n{len(tasks)} tasks, "
          f"n_new={args.n_new}\n")

    rows = []
    for cache_m, server_m, label in pairs:
        ka, kb = arch_key(cache_m), arch_key(server_m)
        if ka != kb:
            print(f"SKIP {label}\n     incompatible architecture "
                  f"{ka} vs {kb} -- KV shapes differ, so this is a crash, "
                  f"not a silent hazard.\n")
            rows.append({"label": label, "skipped": True,
                         "arch_cache": ka, "arch_server": kb})
            continue

        print(f"--- {label}")
        tok = AutoTokenizer.from_pretrained(server_m)
        ma = AutoModelForCausalLM.from_pretrained(
            cache_m, torch_dtype=torch.float32).eval()
        mb = AutoModelForCausalLM.from_pretrained(
            server_m, torch_dtype=torch.float32).eval()

        cos_d, l2 = weight_distance(ma, mb)
        r = evaluate(ma, mb, tok, tasks, args.n_new)
        r.update({"label": label, "cache": cache_m, "server": server_m,
                  "cos_dist": cos_d, "rel_l2": l2, "skipped": False})
        rows.append(r)
        print(f"    cos_dist={cos_d:.5f}  rel_L2={l2:.4f}")
        print(f"    ctrl={r['ctrl']:.3f}  stale={r['stale']:.3f}  "
              f"breaks={r['break']}  fixes={r['fix']}  "
              f"p={r['mcnemar_p']:.3g}\n")
        del ma, mb

    print("READ THIS:")
    print("  fine-tune benign AND generation collapse, in a second family")
    print("     -> the cliff replicates; the claim is not Qwen-specific.")
    print("  a different pattern (e.g. fine-tune also collapses, or the")
    print("  generation pair stays benign)")
    print("     -> the boundary is family-dependent. Report that; it is a")
    print("        weaker but honest claim.")

    out = results_path("family_matrix")
    with open(out, "w") as f:
        json.dump({"n_tasks": len(tasks), "rows": rows}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
