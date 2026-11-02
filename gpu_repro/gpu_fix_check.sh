#!/usr/bin/env bash
# gpu_fix_check.sh — run after gpu_repro.sh on the same machine.
#
# Applies the proposed fix (pinned revision appended to the cache model name)
# to the INSTALLED LMCache, repeats "revision A writes, revision B reads", and
# restores the original file afterwards. Expected: revision B logs a cache
# name containing its revision, LMCache retrieves nothing for it, and it
# answers like the control ("Priya Sharma").
#
#   cd /workspace && bash /gpu_fix_check.sh 2>&1 | tee fix.log
set -uo pipefail

CONN=$(python -c "import lmcache.integration.vllm.lmcache_mp_connector as m; print(m.__file__)" 2>/dev/null | tail -n 1)
[ -f "$CONN" ] || { echo "connector not found"; exit 1; }
cp "$CONN" "$CONN.orig"
trap 'cp "$CONN.orig" "$CONN"; echo "restored original $CONN"' EXIT

python - "$CONN" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
old = "        cache_model_name = get_dcp_decorated_model_name(vllm_config, kv_cache_config)\n"
new = (old +
       "        _revision = vllm_config.model_config.revision\n"
       "        if _revision:\n"
       "            cache_model_name = f\"{cache_model_name}##lmcache-revision-v1-{_revision}\"\n")
assert s.count(old) == 1, "unexpected connector layout; send this message to Claude"
open(p, "w").write(s.replace(old, new))
print("patched", p)
PY

# Same server settings and prompt as gpu_repro.sh; only the A-then-B run.
grep -q "http-port 8089" /gpu_repro.sh || { echo "/gpu_repro.sh lacks --http-port 8089; fix it first"; exit 1; }
sed -e 's/^OUT=repro_out$/OUT=fix_out/' \
    -e '/^echo "=== 1. control/,/^stop_all$/d' \
    -e 's/repro_out.tgz/fix_out.tgz/g' \
    /gpu_repro.sh > /tmp/fix_run.sh
pkill -f "vllm serve" 2>/dev/null; pkill -f "lmcache server" 2>/dev/null; sleep 3
rm -rf fix_out
bash /tmp/fix_run.sh

echo "=== cache names logged by each server (should differ by revision)"
grep -h "cache_model_name=" fix_out/vllm_A.log fix_out/vllm_B_after_A.log \
  | sed 's/.*cache_model_name=//' | tee fix_out/cache_names.txt
echo "=== LMCache retrievals (expected: none)"
grep -E "Prefetch request completed|Retrieved [0-9]+ tokens" fix_out/lmcache_2.log \
  || echo "none"

tar czf fix_out.tgz fix_out
echo "Download fix_out.tgz (and repro_out.tgz) before destroying the instance."
