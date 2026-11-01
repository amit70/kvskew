"""
frequency.py — how often does version-skewed KV reuse actually happen?

The severity probe (kv_severity.py) answers "how bad is one stale reuse".
This answers "how often does one occur", which is the other half of the paper.

MODEL
-----
A fleet of N serving instances shares one cross-instance KV store (the LMCache
tier). The store is keyed on the token-chunk hash plus model_name (and, in
LMCache since Nov 2025, dtype) -- but NOT on model revision, so entries
written by an instance running version v are visible to an instance running
version v' of the same model name.

A rolling upgrade walks the fleet from version v0 to v1 over a drain window.
During that window the fleet is mixed. Requests arrive, are routed to an
instance, and look up their prefix chunks in the shared store.

A CROSS-VERSION HIT is a chunk served to an instance whose version differs
from the version that wrote it. That is the event whose rate we want.

The rate is governed by three quantities an operator already knows:
    - upgrade duration relative to cache entry lifetime (TTL / eviction)
    - request rate and prefix reuse rate (how often chunks are shared at all)
    - fleet size and upgrade batch size (how long the fleet stays mixed)

ARMS
----
    b0            : key = (chunk_hash)                 -- version-blind
    b1            : key = (chunk_hash, model_name)     -- LMCache today
    t_version     : key = (chunk_hash, model_name, version)  -- the fix
    t_bounded     : version-aware, but permits reuse across versions whose
                    distance is within a tolerance -- the "knob" arm

Reported per arm: cross-version hit rate, overall hit rate (the savings that
version-strict keying gives up), and recompute count.

Pure simulation. Seeded, CPU-only, no model required.

    python frequency.py --instances 16 --upgrade-minutes 30
"""

import argparse
import csv
import random
from collections import defaultdict

from common import results_path


class Fleet:
    """N instances, upgraded in batches over a drain window."""

    def __init__(self, n, upgrade_start, upgrade_end, batch, rng):
        self.n = n
        self.batch = batch
        self.rng = rng
        self.start = upgrade_start
        self.end = upgrade_end
        # instance i flips to v1 at a time spread across the window
        steps = max(1, n // max(batch, 1))
        span = max(self.end - self.start, 1e-9)
        self.flip_at = {}
        for i in range(n):
            step = i // max(batch, 1)
            self.flip_at[i] = self.start + span * (step / steps)

    def version_of(self, instance, t):
        return "v1" if t >= self.flip_at[instance] else "v0"

    def mixed(self, t):
        vs = {self.version_of(i, t) for i in range(self.n)}
        return len(vs) > 1


class SharedStore:
    """Cross-instance KV chunk store with a configurable key schema."""

    def __init__(self, arm, ttl, tolerance=0):
        self.arm = arm
        self.ttl = ttl
        self.tolerance = tolerance
        self.entries = {}  # key -> (written_at, writer_version)

    def _key(self, chunk, version):
        if self.arm == "b0":
            return (chunk,)
        if self.arm == "b1":                      # LMCache today
            return (chunk, "model_name")
        if self.arm in ("t_version", "t_bounded"):
            return (chunk, "model_name", version)
        raise ValueError(self.arm)

    def lookup(self, chunk, version, t):
        """Return (hit, cross_version) for a reader at `version`."""
        if self.arm == "t_bounded":
            # version-aware key, but accept a neighbouring version within
            # tolerance -- this is the knob.
            for v in self._candidates(version):
                e = self.entries.get((chunk, "model_name", v))
                if e and t - e[0] <= self.ttl:
                    return True, (v != version)
            return False, False

        e = self.entries.get(self._key(chunk, version))
        if not e or t - e[0] > self.ttl:
            return False, False
        return True, (e[1] != version)

    def _candidates(self, version):
        order = ["v0", "v1"]
        if self.tolerance <= 0:
            return [version]
        return [version] + [v for v in order if v != version]

    def write(self, chunk, version, t):
        self.entries[self._key(chunk, version)] = (t, version)


def run(arm, seed, instances, batch, upgrade_start, upgrade_end, duration,
        rate, n_chunks, zipf_s, ttl, tolerance):
    rng = random.Random(seed)
    fleet = Fleet(instances, upgrade_start, upgrade_end, batch, rng)
    store = SharedStore(arm, ttl, tolerance)

    # popularity-skewed chunk selection: a few prefixes dominate, as in
    # shared system prompts / few-shot blocks.
    weights = [1.0 / ((i + 1) ** zipf_s) for i in range(n_chunks)]
    total_w = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total_w
        cum.append(acc)

    def pick_chunk():
        r = rng.random()
        lo, hi = 0, len(cum) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if cum[mid] < r:
                lo = mid + 1
            else:
                hi = mid
        return lo

    hits = misses = cross = 0
    n_req = int(duration * rate)
    for k in range(n_req):
        t = duration * (k / max(n_req, 1))
        inst = rng.randrange(instances)
        ver = fleet.version_of(inst, t)
        chunk = pick_chunk()

        hit, xver = store.lookup(chunk, ver, t)
        if hit:
            hits += 1
            cross += int(xver)
        else:
            misses += 1
            store.write(chunk, ver, t)

    return {
        "arm": arm,
        "seed": seed,
        "requests": n_req,
        "hit_rate": round(hits / max(n_req, 1), 4),
        "cross_version_rate": round(cross / max(n_req, 1), 4),
        "cross_of_hits": round(cross / max(hits, 1), 4),
        "recomputes": misses,
    }


FIELDS = ["arm", "seed", "upgrade_minutes", "requests", "hit_rate",
          "cross_version_rate", "cross_of_hits", "recomputes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--duration", type=float, default=3600.0, help="seconds")
    ap.add_argument("--rate", type=float, default=2.0, help="requests/second")
    ap.add_argument("--chunks", type=int, default=400)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--ttl", type=float, default=1800.0)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()

    rows = []
    print(f"{'arm':>12} {'upg_min':>8} {'hit':>7} {'xver':>7} {'xver/hit':>9}")
    for upgrade_minutes in (5, 15, 30, 60):
        us = args.duration * 0.25
        ue = us + upgrade_minutes * 60
        for arm in ("b0", "b1", "t_version", "t_bounded"):
            agg = defaultdict(list)
            for rep in range(1, args.reps + 1):
                r = run(arm, rep, args.instances, args.batch, us, ue,
                        args.duration, args.rate, args.chunks, args.zipf,
                        args.ttl, tolerance=1 if arm == "t_bounded" else 0)
                r["upgrade_minutes"] = upgrade_minutes
                rows.append(r)
                for k in ("hit_rate", "cross_version_rate", "cross_of_hits"):
                    agg[k].append(r[k])
            m = {k: sum(v) / len(v) for k, v in agg.items()}
            print(f"{arm:>12} {upgrade_minutes:>8} {m['hit_rate']:>7.3f} "
                  f"{m['cross_version_rate']:>7.3f} {m['cross_of_hits']:>9.3f}")

    out = results_path("frequency", ext="csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})
    print(f"\nwrote {out}")
    print("\nREAD THIS:")
    print("  b1 is the LMCache-today schema. Its cross_version_rate is the")
    print("  exposure operators have now. t_version drives it to zero but")
    print("  gives up hit rate -- that gap is the savings the knob trades.")


if __name__ == "__main__":
    main()
