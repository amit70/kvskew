"""
lmcache_key_repro.py — the key collision, in a few seconds, with no GPU.

WHAT IT SHOWS
-------------
Two serving instances run the same model name, the same tensor-parallel
layout and the same precision, but different weights: the instance that has
already been upgraded runs revision B, the one still draining runs revision A.
Both see the same prompt prefix, so both compute the same chunk hash (the hash
is over token ids, which do not depend on the weights).

LMCache's CacheEngineKey has fields for model name, world size, worker id,
chunk hash and dtype. It has no field for the model revision, so the two
instances produce the SAME key, and the shared cache tier treats KV computed
by revision A as valid for revision B.

TWO MODES
---------
default        no dependencies. Reproduces the key string format from
               lmcache/utils.py (CacheEngineKey.to_string), checked against
               LMCache commit a2cd25df (2026-10-25).
--use-lmcache  imports lmcache.utils.CacheEngineKey from an installed LMCache
               and compares real key objects, so the result does not depend
               on this file's copy of the format.

THE OPT-IN WORKAROUND
---------------------
The in-process connector folds request configs named lmcache.tag.* into the
key, so a deployment that sends e.g. lmcache.tag.revision with every request
separates the two revisions. That works only if every client sets it, and
LMCache's multiprocess-mode docs say tags are metadata there, not cache
identity. --show-tag-workaround demonstrates the in-process behaviour.

    python lmcache_key_repro.py
    python lmcache_key_repro.py --use-lmcache --show-tag-workaround
"""

import argparse
import hashlib
import sys

MODEL_NAME = "acme/chat-8b"          # what both instances serve under
REVISION_A = "3f1c9e0"               # old weights, still draining
REVISION_B = "8a72d41"               # new weights, already upgraded
WORLD_SIZE, WORKER_ID = 1, 0
PREFIX_TOKEN_IDS = [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13]


def chunk_hash(token_ids):
    """Stand-in for LMCache's prefix hash: a function of token ids only.
    The real hash differs in algorithm, not in what it depends on."""
    data = ",".join(map(str, token_ids)).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")


def key_string(model_name, world_size, worker_id, chash, dtype_str,
               tags=None):
    """Format of CacheEngineKey.to_string() in lmcache/utils.py."""
    s = f"{model_name}@{world_size}@{worker_id}@{chash:x}@{dtype_str}"
    if tags:
        s += "@" + "@".join(f"{k}%{v}" for k, v in tags)
    return s


def run_stdlib(show_tags):
    h = chunk_hash(PREFIX_TOKEN_IDS)
    # Each instance knows its revision, but the key has nowhere to put it.
    ka = key_string(MODEL_NAME, WORLD_SIZE, WORKER_ID, h, "bfloat16")
    kb = key_string(MODEL_NAME, WORLD_SIZE, WORKER_ID, h, "bfloat16")
    report(f"revision {REVISION_A}", ka, f"revision {REVISION_B}", kb)

    if show_tags:
        print("\nwith lmcache.tag.revision sent on every request "
              "(in-process connector):")
        ta = key_string(MODEL_NAME, WORLD_SIZE, WORKER_ID, h, "bfloat16",
                        [("revision", REVISION_A)])
        tb = key_string(MODEL_NAME, WORLD_SIZE, WORKER_ID, h, "bfloat16",
                        [("revision", REVISION_B)])
        report(f"revision {REVISION_A}", ta, f"revision {REVISION_B}", tb)


def run_lmcache(show_tags):
    try:
        import torch
        from lmcache.utils import CacheEngineKey
    except ImportError as e:
        sys.exit(f"--use-lmcache needs lmcache and torch installed ({e})")

    h = chunk_hash(PREFIX_TOKEN_IDS)

    def make(tags=None):
        cfg = {f"lmcache.tag.{k}": v for k, v in (tags or [])}
        return CacheEngineKey(model_name=MODEL_NAME, world_size=WORLD_SIZE,
                              worker_id=WORKER_ID, chunk_hash=h,
                              dtype=torch.bfloat16, request_configs=cfg)

    ka, kb = make(), make()
    report(f"revision {REVISION_A}", ka.to_string(),
           f"revision {REVISION_B}", kb.to_string(), equal=(ka == kb))

    if show_tags:
        print("\nwith lmcache.tag.revision sent on every request "
              "(in-process connector):")
        ta, tb = make([("revision", REVISION_A)]), make([("revision", REVISION_B)])
        report(f"revision {REVISION_A}", ta.to_string(),
               f"revision {REVISION_B}", tb.to_string(), equal=(ta == tb))


def report(name_a, key_a, name_b, key_b, equal=None):
    equal = (key_a == key_b) if equal is None else equal
    print(f"  {name_a:<18} {key_a}")
    print(f"  {name_b:<18} {key_b}")
    if equal:
        print("  -> SAME KEY: revision B will load KV computed by revision A")
    else:
        print("  -> different keys: the revisions are kept apart")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--use-lmcache", action="store_true",
                    help="compare real CacheEngineKey objects")
    ap.add_argument("--show-tag-workaround", action="store_true")
    args = ap.parse_args()

    print(f"model name {MODEL_NAME!r}, same prefix, same TP layout, "
          f"same dtype, different weights\n")
    if args.use_lmcache:
        run_lmcache(args.show_tag_workaround)
    else:
        run_stdlib(args.show_tag_workaround)


if __name__ == "__main__":
    main()
