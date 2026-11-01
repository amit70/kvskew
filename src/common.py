"""Shared helpers: where results go and how runs are named.

Every probe writes to <repo>/results/, and the file name carries the
configuration, so running the dtype pair and then the checkpoint pair keeps
both results instead of the second overwriting the first.
"""

import os
import re

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           os.pardir, "results")


def slug(text):
    """'Qwen/Qwen2-0.5B KV -> Qwen/Qwen2.5-0.5B-Instruct server' ->
    'qwen2-0.5b_kv_to_qwen2.5-0.5b-instruct_server'."""
    t = text.lower().replace("->", " to ")
    t = re.sub(r"[a-z0-9_.-]+/", "", t)          # drop HF org prefixes
    t = re.sub(r"[^a-z0-9.\-]+", "_", t)
    return t.strip("_")


def results_path(prefix, label=None, ext="json"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    name = prefix if label is None else f"{prefix}_{slug(label)}"
    return os.path.normpath(os.path.join(RESULTS_DIR, f"{name}.{ext}"))
