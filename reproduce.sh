#!/usr/bin/env bash
# Rerun everything behind the paper. Model runs download 0.5B checkpoints
# from Hugging Face and take minutes to tens of minutes each on CPU.
set -euo pipefail
cd "$(dirname "$0")/src"

python lmcache_key_repro.py --show-tag-workaround
python frequency.py

python kv_quality.py --mode dtype      --model Qwen/Qwen2.5-0.5B-Instruct
python kv_quality.py --mode checkpoint --model-a Qwen/Qwen2.5-0.5B --model-b Qwen/Qwen2.5-0.5B-Instruct
python kv_quality.py --mode checkpoint --model-a Qwen/Qwen2-0.5B   --model-b Qwen/Qwen2.5-0.5B-Instruct
python interp_sweep.py --limit 60
python distance_sweep.py --model EleutherAI/pythia-160m
python family_matrix.py

python make_figures.py
