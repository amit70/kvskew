"""
make_figures.py — rebuild the paper's rollout and distance figures from the
files in results/. Needs matplotlib only; no model or GPU.

    python make_figures.py
"""

import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS_DIR  # noqa: E402

FIG_DIR = os.path.normpath(os.path.join(RESULTS_DIR, os.pardir, "figures"))


def rollout():
    rows = list(csv.DictReader(open(os.path.join(RESULTS_DIR,
                                                 "frequency.csv"))))
    agg = defaultdict(list)
    for r in rows:
        agg[(r["arm"], int(r["upgrade_minutes"]))].append(r)
    windows = sorted({int(r["upgrade_minutes"]) for r in rows})

    def mean(arm, field):
        return [100 * sum(float(r[field]) for r in agg[(arm, w)])
                / len(agg[(arm, w)]) for w in windows]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7, 2.6))
    for arm, label, style in (("b1", "name-only key (LMCache today)", "o-"),
                              ("t_version", "version-aware key", "s--")):
        a1.plot(windows, mean(arm, "cross_version_rate"), style, label=label)
        a2.plot(windows, mean(arm, "hit_rate"), style, label=label)
    a1.set(xlabel="upgrade window (min)", ylabel="cross-version hits (% req)")
    a2.set(xlabel="upgrade window (min)", ylabel="overall hit rate (%)")
    a1.legend(fontsize=7, frameon=False)
    for a in (a1, a2):
        a.set_xticks(windows)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "rollout_exposure.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


def distance():
    d = json.load(open(os.path.join(RESULTS_DIR, "interp_sweep.json")))
    # A producing model that scores ~0 on its own says nothing about KV
    # compatibility, so those points are excluded, as in the paper.
    usable = [r for r in d["rows"] if r["self_a"] > 0.05]
    xs = [max(r["cos_dist"], 1e-4) for r in usable]
    ys = [r["break"] for r in usable]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    ax.plot(xs, ys, "o-")
    for r, x, y in zip(usable, xs, ys):
        ax.annotate(f"a={r['alpha']:g}", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=7)
    ax.set_xscale("log")
    ax.set(xlabel="cosine weight distance (log)",
           ylabel=f"breaks / {d['n_tasks']} tasks")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "breaks_vs_distance.png")
    fig.savefig(out, dpi=300)
    print(f"wrote {out}")


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    rollout()
    distance()
