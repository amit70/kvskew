#!/usr/bin/env bash
# gpu_repro.sh — two revisions of one model ID sharing an LMCache MP server.
#
# Run on a single-GPU Linux box (24 GB+) with vLLM installed, e.g. the
# vllm/vllm-openai image on vast.ai. Needs HF_TOKEN in the environment if
# amitrajula/kvskew-demo is private.
#
#   pip install lmcache
#   export HF_TOKEN=hf_...            # read-only token
#   bash gpu_repro.sh
#
# Writes everything to ./repro_out/: versions, server logs, and the three
# outputs below. Takes roughly 10-15 minutes, most of it model loading.
#
#   1_B_alone.txt     revision B, empty cache            -> control
#   2_A_first.txt     revision A writes the cache
#   3_B_after_A.txt   revision B reads A's entries       -> the bug
set -uo pipefail

MODEL=amitrajula/kvskew-demo
REV_A=9f5fb381869af92331669c76bc0fab55d6ce2d63   # Qwen2-0.5B weights
REV_B=6f54247043512e1226da692eec6bde78ab1f3f8a   # Qwen2.5-0.5B-Instruct weights
OUT=repro_out
KV='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":6000}}'
mkdir -p "$OUT"

# A long, fixed prompt (well over one 256-token chunk) ending in a question
# whose answer is stated in the text.
PROMPT=$(python3 - <<'PY'
facts = []
for i in range(1, 61):
    facts.append(f"Warehouse {i} holds {i * 7 % 50 + 10} crates of apples "
                 f"and {i * 3 % 40 + 5} crates of pears.")
facts.append("The manager of the whole network is named Priya Sharma.")
text = " ".join(facts)
print(f"{text}\n\nQuestion: What is the name of the manager of the whole "
      f"network?\nAnswer:")
PY
)

{
  date -u
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python3 -c "import vllm, lmcache; print('vllm', vllm.__version__); print('lmcache', getattr(lmcache, '__version__', 'unknown'))"
  python3 -c "import lmcache.integration.vllm.lmcache_mp_connector as m; print('external MP connector:', m.__file__)"
} > "$OUT/versions.txt" 2>&1
cat "$OUT/versions.txt"

start_lmcache() {
  pkill -f "lmcache server" 2>/dev/null; sleep 2
  lmcache server --host 127.0.0.1 --port 6000 --chunk-size 256 \
    --l1-size-gb 4 --eviction-policy LRU --http-port 8089 > "$OUT/lmcache_$1.log" 2>&1 &
  for _ in $(seq 1 30); do
    if (echo > /dev/tcp/127.0.0.1/6000) 2>/dev/null; then
      sleep 5   # it can open the port and then exit if a later step fails
      pgrep -f "lmcache server" >/dev/null && return 0
      break
    fi
    sleep 2
  done
  echo "LMCache server did not start; see $OUT/lmcache_$1.log"; exit 1
}

start_vllm() {  # $1 revision, $2 port, $3 log name
  vllm serve "$MODEL" --revision "$1" --port "$2" \
    --gpu-memory-utilization 0.35 --max-model-len 4096 \
    --kv-transfer-config "$KV" > "$OUT/vllm_$3.log" 2>&1 &
  for _ in $(seq 1 120); do
    curl -sf "localhost:$2/v1/models" >/dev/null && return 0
    sleep 5
  done
  echo "vLLM on port $2 did not start; see $OUT/vllm_$3.log"; exit 1
}

ask() {  # $1 port, $2 output file
  python3 - "$1" "$2" "$PROMPT" <<'PY'
import json, sys, urllib.request
port, out, prompt = sys.argv[1], sys.argv[2], sys.argv[3]
body = {"model": "amitrajula/kvskew-demo", "prompt": prompt,
        "max_tokens": 48, "temperature": 0}
req = urllib.request.Request(f"http://localhost:{port}/v1/completions",
                             json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req))
text = r["choices"][0]["text"]
open(out, "w").write(text + "\n\nusage: " + json.dumps(r.get("usage")) + "\n")
print(f"--- {out}\n{text!r}\n")
PY
}

stop_all() { pkill -f "vllm serve" 2>/dev/null; sleep 5; }

echo "=== 1. control: revision B alone, empty cache"
start_lmcache 1
start_vllm "$REV_B" 8002 B_alone
ask 8002 "$OUT/1_B_alone.txt"
stop_all

echo "=== 2-3. revision A writes, then revision B reads"
start_lmcache 2
start_vllm "$REV_A" 8001 A
start_vllm "$REV_B" 8002 B_after_A
ask 8001 "$OUT/2_A_first.txt"
sleep 3   # let the store finish
ask 8002 "$OUT/3_B_after_A.txt"
stop_all
pkill -f "lmcache server" 2>/dev/null

echo "=== LMCache retrievals during runs 2-3 (A stores; any retrieval is B reading)"
grep -E "Prefetch request completed|Retrieved [0-9]+ tokens" "$OUT/lmcache_2.log" \
  | tee "$OUT/hits_B_after_A.txt"
[ -s "$OUT/hits_B_after_A.txt" ] || echo "(no retrievals: revision B did not read A's entries)" \
  | tee "$OUT/hits_B_after_A.txt"

tar czf repro_out.tgz "$OUT"
echo "Done. Download repro_out.tgz before destroying the instance."
