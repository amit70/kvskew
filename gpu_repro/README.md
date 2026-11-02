# End-to-end reproduction on a real vLLM + LMCache deployment

The rest of this repository measures the effect with transformers on CPU.
This folder shows the same failure through the real serving path: two vLLM
servers sharing one LMCache multiprocess (MP) server, started from the same
model ID with different `--revision` values.

## Setup

- Test model: [`amitrajula/kvskew-demo`](https://huggingface.co/amitrajula/kvskew-demo),
  one model ID with two revisions:
  - A `9f5fb381869af92331669c76bc0fab55d6ce2d63`: Qwen2-0.5B weights
  - B `6f54247043512e1226da692eec6bde78ab1f3f8a`: Qwen2.5-0.5B-Instruct weights

  Same tokenizer, layer count, KV heads and head dimension, so nothing fails
  loudly when one reads the other's KV.
- Prompt: 1,212 tokens (60 facts, then a question answered in the text: the
  manager is Priya Sharma). LMCache stores it as 4 chunks of 256 tokens.
- Greedy decoding, 48 new tokens.
- Run on 2026-11-02: one RTX 4090 (48 GB), vLLM 0.32.0, LMCache 0.5.6 from pip.

## Result

| Run | Revision B's answer | LMCache server log |
|---|---|---|
| B alone, empty cache | "Priya Sharma …" | stores 1,024 tokens |
| A first, then B (**LMCache 0.5.6 as released**) | "(over to save both and also opposite number. I do not can be such a large doming an even-lemM: According to Einkondus …" | A stores 1,024 tokens; B: `4/4 retained keys`, `Retrieved 1024 tokens` |
| A first, then B (**with the proposed fix**) | "Priya Sharma …" (identical to the control) | A and B each store 1,024 tokens; no retrieval |

Both servers log the name LMCache keys their KV on. As released, both print
`cache_model_name=amitrajula/kvskew-demo`, although vLLM was started with
different revisions. With the fix:

```
amitrajula/kvskew-demo##lmcache-revision-v1-9f5fb381869af92331669c76bc0fab55d6ce2d63
amitrajula/kvskew-demo##lmcache-revision-v1-6f54247043512e1226da692eec6bde78ab1f3f8a
```

Nothing errors in the released case. Revision B silently answers from
revision A's KV.

## Files

- `gpu_repro.sh`: the three runs (control, A then B). Writes `repro_out/`.
- `gpu_fix_check.sh`: patches the installed connector with the proposed fix
  (three lines appending the revision to the cache model name), repeats A then
  B, restores the original file. Writes `fix_out/`.
- `runs/2026-11-02_rtx4090/bug/` and `.../fixed/`: answers, server logs, and
  `hits_B_after_A.txt` with the LMCache store and retrieve lines.

## Rerun

On a Linux machine with one GPU of 24 GB or more and vLLM installed (for
example the `vllm/vllm-openai` image):

```bash
pip install lmcache
cp gpu_repro.sh gpu_fix_check.sh /          # gpu_fix_check.sh reads /gpu_repro.sh
cd /workspace                               # any writable folder
bash /gpu_repro.sh 2>&1 | tee run.log       # about 5-10 minutes
bash /gpu_fix_check.sh 2>&1 | tee fix.log   # about 5 minutes
```

On hosted images that already run a vLLM server, stop it first so the GPU is
free. The LMCache dashboard is moved to port 8089 because some images use 8080.
