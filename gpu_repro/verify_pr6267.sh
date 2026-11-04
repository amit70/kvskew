#!/usr/bin/env bash
# verify_pr6267.sh — end-to-end check of LMCache PR #6267 on a real GPU.
#
# Applies PR #6267's revision logic (functions copied verbatim from head
# df10c9bf) to the installed LMCache connector, then runs:
#   1. pinned A then pinned B  -> B must not read A's KV; answer = control
#   2. unpinned, stock LMCache -> baseline cache name
#   3. unpinned, with #6267    -> cache name must equal the baseline
# Restores the original connector file at the end.
#
#   pip install lmcache            # 0.5.6 at time of writing
#   cd /workspace && bash /verify_pr6267.sh 2>&1 | tee verify.log
set -uo pipefail

MODEL=amitrajula/kvskew-demo
REV_A=9f5fb381869af92331669c76bc0fab55d6ce2d63   # Qwen2-0.5B weights
REV_B=6f54247043512e1226da692eec6bde78ab1f3f8a   # Qwen2.5-0.5B-Instruct weights
OUT=verify_out
KV='{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"127.0.0.1","lmcache.mp.port":6000}}'
rm -rf "$OUT"; mkdir -p "$OUT"

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

CONN=$(python -c "import lmcache.integration.vllm.lmcache_mp_connector as m; print(m.__file__)" 2>/dev/null | tail -n 1)
[ -f "$CONN" ] || { echo "connector not found"; exit 1; }
cp "$CONN" "$CONN.orig"
trap 'cp "$CONN.orig" "$CONN"; echo "restored original $CONN"' EXIT

{
  date -u
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  python -c "import vllm, lmcache; print('vllm', vllm.__version__); print('lmcache', getattr(lmcache, '__version__', '?'))"
} > "$OUT/versions.txt" 2>&1
grep -v INFO "$OUT/versions.txt"

start_lmcache() {
  pkill -f "lmcache server" 2>/dev/null; sleep 2
  lmcache server --host 127.0.0.1 --port 6000 --chunk-size 256 \
    --l1-size-gb 4 --eviction-policy LRU --http-port 8089 > "$OUT/lmcache_$1.log" 2>&1 &
  for _ in $(seq 1 30); do
    if (echo > /dev/tcp/127.0.0.1/6000) 2>/dev/null; then
      sleep 5; pgrep -f "lmcache server" >/dev/null && return 0; break
    fi
    sleep 2
  done
  echo "LMCache server did not start; see $OUT/lmcache_$1.log"; exit 1
}

start_vllm() {  # $1 revision ("" = unpinned), $2 port, $3 log name
  local rev=()
  [ -n "$1" ] && rev=(--revision "$1")
  vllm serve "$MODEL" "${rev[@]}" --port "$2" \
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
open(out, "w").write(text + "\n")
print(f"--- {out}\n{text!r}\n")
PY
}

stop_all() { pkill -f "vllm serve" 2>/dev/null; sleep 5; }

cache_name() {  # $1 log name
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/vllm_$1.log" | grep -o "cache_model_name=[^ ]*" | head -n 1 | sed 's/cache_model_name=//'
}

revision_repr() {  # $1 log name
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/vllm_$1.log" | grep -o "KVSKEW_REVISION=.*" | head -n 1
}

apply_pr6267() {
python - "$CONN" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
funcs = open("/tmp/pr6267_funcs.py").read()
if "import urllib.parse" not in s:
    s = "import urllib.parse\n" + s
anchor = '_DCP_LAYOUT_NAMESPACE = "##lmcache-dcp-layout-v1-"\n'
assert s.count(anchor) == 1, "unexpected connector layout"
s = s.replace(anchor, anchor + '_REVISION_NAMESPACE = "##lmcache-revision-v1-"\n')
anchor = "def get_resolved_attention_block_sizes("
assert s.count(anchor) == 1, "unexpected connector layout"
s = s.replace(anchor, funcs + anchor)
call = "        cache_model_name = get_dcp_decorated_model_name(vllm_config, kv_cache_config)\n"
assert s.count(call) == 1, "unexpected connector layout"
# Instrumentation only: log what vLLM hands the connector as `revision`.
s = s.replace(call,
    "        logger.info('KVSKEW_REVISION=%r initial=%r resolved=%r', "
    "vllm_config.model_config.revision, "
    "getattr(vllm_config.model_config.revision, 'initial', '<no attr>'), "
    "getattr(vllm_config.model_config.revision, 'resolved', '<no attr>'))\n"
    "        cache_model_name = get_cache_model_name(vllm_config, kv_cache_config)\n")
open(p, "w").write(s)
print("applied PR #6267 revision logic to", p)
PY
}

cat > /tmp/pr6267_funcs.py <<'PYFUNCS'
def _revision_cache_token(vllm_config: VllmConfig) -> str | None:
    """Return the model revision that must be part of the cache identity.

    vLLM resolves ``--revision`` to a commit hash for Hub models and exposes
    the hash as ``ResolvedRevision.resolved``. Two engines that pin different
    commits of one model ID produce different KV bytes for the same tokens, so
    the hash is what the identity has to track: the requested ref alone would
    not change when the branch moves. A revision vLLM could not resolve (local
    path, ModelScope, offline lookup) is used verbatim.

    An engine that passes no ``--revision`` still gets the default branch
    resolved, to ``ResolvedRevision(initial=None, ...)``. Tracking that hash
    would rename every existing cache namespace on upgrade without any
    guarantee about the served weights, so only an explicitly pinned revision
    takes part in the identity.

    Args:
        vllm_config: The active vLLM configuration.

    Returns:
        The commit hash, the requested revision, or ``None`` when the engine
        does not pin a revision.
    """
    revision = getattr(vllm_config.model_config, "revision", None)
    if not revision:
        return None
    requested = getattr(revision, "initial", revision)
    if requested is None:
        return None
    return str(getattr(revision, "resolved", None) or requested)


def get_cache_model_name(
    vllm_config: VllmConfig,
    kv_cache_config: "KVCacheConfig | None" = None,
) -> str:
    """Return the cache identity the MP connector stores objects under.

    The identity starts from the model name vLLM serves and appends every
    attribute that changes the bytes of a KV page: the resolved model revision
    and the non-trivial DCP interleave layout. Engines that differ in either
    must not hit each other's objects.

    Args:
        vllm_config: The active vLLM configuration.
        kv_cache_config: vLLM's resolved KV cache group configuration.

    Returns:
        The decorated cache model name, or the plain model name when neither a
        revision nor a non-trivial DCP layout applies.
    """
    cache_model_name = get_dcp_decorated_model_name(vllm_config, kv_cache_config)
    revision = _revision_cache_token(vllm_config)
    if revision is None:
        return cache_model_name
    # '@' separates the model name from the chunk hash inside an object key
    # and is rejected in model_name, so the revision is percent-encoded.
    return (
        f"{cache_model_name}{_REVISION_NAMESPACE}"
        f"{urllib.parse.quote(revision, safe='')}"
    )


PYFUNCS

echo "=== 1. unpinned, stock LMCache (baseline name)"
start_lmcache 1
start_vllm "" 8002 unpinned_stock
stop_all

apply_pr6267

echo "=== 2. unpinned, with #6267"
start_lmcache 2
start_vllm "" 8002 unpinned_pr
stop_all

echo "=== 3. pinned A then pinned B, with #6267"
start_lmcache 3
start_vllm "$REV_A" 8001 A
start_vllm "$REV_B" 8002 B_after_A
ask 8001 "$OUT/3_A_first.txt"
sleep 3
ask 8002 "$OUT/3_B_after_A.txt"
stop_all
pkill -f "lmcache server" 2>/dev/null

{
  echo "== cache names"
  echo "unpinned, stock : $(cache_name unpinned_stock)"
  echo "unpinned, #6267 : $(cache_name unpinned_pr)"
  echo "pinned A, #6267 : $(cache_name A)"
  echo "pinned B, #6267 : $(cache_name B_after_A)"
  echo "== what vLLM passes as model_config.revision"
  echo "unpinned : $(revision_repr unpinned_pr)"
  echo "pinned A : $(revision_repr A)"
  echo "== LMCache store/retrieve during run 3 (expected: two stores, no retrieval)"
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/lmcache_3.log" | grep -E "Stored [0-9]+ tokens|Prefetch request completed|Retrieved [0-9]+ tokens" || echo "none"
  echo "== revision B answer after A (expected: Priya Sharma)"
  cat "$OUT/3_B_after_A.txt"
  echo "== checks"
  [ "$(cache_name unpinned_stock)" = "$(cache_name unpinned_pr)" ] && echo "PASS unpinned name unchanged" || echo "FAIL unpinned name changed"
  [ "$(cache_name A)" != "$(cache_name B_after_A)" ] && echo "PASS pinned revisions get distinct names" || echo "FAIL pinned names equal"
  sed 's/\x1b\[[0-9;]*m//g' "$OUT/lmcache_3.log" | grep -qE "Retrieved [0-9]+ tokens" && echo "FAIL B retrieved A's KV" || echo "PASS no cross-revision retrieval"
  grep -q "Priya Sharma" "$OUT/3_B_after_A.txt" && echo "PASS B answers correctly" || echo "FAIL B answer wrong"
} | tee "$OUT/summary.txt"

tar czf verify_out.tgz "$OUT"
echo "Download verify_out.tgz, then destroy the instance."
