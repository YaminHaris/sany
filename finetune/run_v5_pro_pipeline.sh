#!/usr/bin/env bash
# Stage 1 (rsLoRA + NEFTune SFT on Qwen2.5-Coder-7B) + Stage 2 (On-Policy RPO/DPO)
# -> GGUF -> Ollama (qwen-pentest-v5) -> Full Eval Suite.
# Fully resumable at every stage.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

BASE="unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
OLLAMA_BASE="qwen2.5:7b"
SFT_OUT="out/lora-pentest-v5b-sft"
DPO_OUT="out/lora-pentest-v5"
SFT_NAME="qwen-pentest-v5-sft"
FINAL_NAME="qwen-pentest-v5"
DATA="data/train_v5b_pro.jsonl"
MAX_LEN=3072

free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 12 ]; then
  echo "Only ${free_gb} GB free; need at least 12 GB." >&2
  exit 1
fi

if [ -f "$SFT_OUT/train_state.pt" ] && [ -f "$DATA" ]; then
  echo "== Resuming Stage 1 SFT from $SFT_OUT/train_state.pt"
else
  echo "== Building 2,800-example multi-domain SFT dataset -> $DATA"
  $PY - "$DATA" <<'PYEOF'
import random, sys
from gen_data import main, FAMILIES
from gen_data_pentest import PENTEST_FAMILIES
from gen_data_exploit import EXPLOIT_FAMILIES
from gen_data_pro import PRO_FAMILIES

out = sys.argv[1]
main(1300, 'data/_v5_general.jsonl', FAMILIES)
main(450,  'data/_v5_recon.jsonl', PENTEST_FAMILIES)
main(450,  'data/_v5_exploit.jsonl', EXPLOIT_FAMILIES)
main(600,  'data/_v5_pro.jsonl', PRO_FAMILIES)

rows = []
for f in ('data/_v5_general.jsonl', 'data/_v5_recon.jsonl', 'data/_v5_exploit.jsonl', 'data/_v5_pro.jsonl'):
    with open(f) as fh:
        rows += [l for l in fh if l.strip()]
random.Random(2026).shuffle(rows)
with open(out, 'w') as fh:
    fh.writelines(rows)
print(f'wrote {len(rows)} examples -> {out}')
PYEOF
fi

echo "== Unloading Ollama models so GPU is free"
for m in $(curl -s localhost:11434/api/ps | python3 -c 'import sys,json;print(" ".join(x["name"] for x in json.load(sys.stdin).get("models",[])))' 2>/dev/null || true); do
  curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null || true
done

echo "== Stage 1: SFT (Qwen2.5-7B-Instruct, LoRA r=16, NEFTune=5.0, lr=1.5e-4)"
$PY -u train.py \
  --base "$BASE" \
  --data "$DATA" \
  --out "$SFT_OUT" \
  --max-samples 2800 \
  --max-len "$MAX_LEN" \
  --rank 16 \
  --dropout 0.05 \
  --weight-decay 0.01 \
  --neftune 5.0 \
  --lr 1.5e-4 \
  --save-every 25 2>&1 | tee -a "$SFT_OUT.train.log"

echo "== Registering $SFT_NAME in Ollama for on-policy DPO mining"
$PY lora_to_gguf.py "$SFT_OUT" "$SFT_OUT/${SFT_NAME}-lora.gguf"
printf 'FROM %s\nADAPTER %s\n' "$OLLAMA_BASE" "$(realpath "$SFT_OUT/${SFT_NAME}-lora.gguf")" > "$SFT_OUT/Modelfile"
ollama create "$SFT_NAME" -f "$SFT_OUT/Modelfile"

echo "== Stage 2: On-Policy RPO/DPO Refinement -> $DPO_OUT"
$PY -u dpo_refine.py \
  --base "$BASE" \
  --sft-dir "$SFT_OUT" \
  --sft-ollama "$SFT_NAME" \
  --out "$DPO_OUT" \
  --max-len "$MAX_LEN" \
  --rank 16 \
  --lr 5e-6 \
  --beta 0.05 \
  --rpo-alpha 0.5 \
  --epochs 1.0 2>&1 | tee -a "$DPO_OUT.dpo.log"

echo "== Converting final $FINAL_NAME adapter to GGUF"
$PY lora_to_gguf.py "$DPO_OUT" "$DPO_OUT/${FINAL_NAME}-lora.gguf"
printf 'FROM %s\nADAPTER %s\n' "$OLLAMA_BASE" "$(realpath "$DPO_OUT/${FINAL_NAME}-lora.gguf")" > "$DPO_OUT/Modelfile"
ollama create "$FINAL_NAME" -f "$DPO_OUT/Modelfile"

echo "== Evaluating $FINAL_NAME on all suites =="
$PY run_eval.py "$FINAL_NAME" --cases-module eval_cases_exploit --out "results/${FINAL_NAME}_exploit.json" | tee "results_${FINAL_NAME}_exploit.log"
$PY run_eval.py "$FINAL_NAME" --cases-module eval_cases_pentest --out "results/${FINAL_NAME}_recon.json" | tee "results_${FINAL_NAME}_recon.log"
$PY run_eval.py "$FINAL_NAME" --cases-module eval_cases_explore --out "results/${FINAL_NAME}_explore.json" | tee "results_${FINAL_NAME}_explore.log"
$PY run_eval.py "$FINAL_NAME" --cases-module eval_cases_bugbounty --out "results/${FINAL_NAME}_bugbounty.json" | tee "results_${FINAL_NAME}_bugbounty.log"
$PY run_eval.py "$FINAL_NAME" --out "results/${FINAL_NAME}_general.json" | tee "results_${FINAL_NAME}_general.log"
$PY run_1000_bench.py "$FINAL_NAME" --stratified 5 | tee "results_${FINAL_NAME}_strat5.log"
echo "== DONE: $FINAL_NAME pipeline complete =="
