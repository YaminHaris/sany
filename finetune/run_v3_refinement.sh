#!/usr/bin/env bash
# Incremental refinement (qwen-pentest-v3) initialized from out/lora-pentest-v2:
# Teaches multi-step local network/ARP/gateway inspection (fam_lan_devices),
# multi-word fuzzy file search, df -h free-space reporting, and step-by-step math,
# mixed with replay trajectories at a gentle learning rate (lr=4e-5, 50 steps, ~34 min).
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

OUT=${OUT:-out/lora-pentest-v3}
NAME=${NAME:-qwen-pentest-v3}
DATA=${DATA:-data/train_v3_refine.jsonl}
MAX_LEN=${MAX_LEN:-3072}

if [ -f "$OUT/train_state.pt" ] && [ -f "$DATA" ]; then
  echo "== Resuming from $OUT/train_state.pt"
else
  echo "== Building 400-example v3 refinement + replay dataset -> $DATA"
  $PY - "$DATA" <<'PYEOF'
import json, random, sys
from gen_data import main, FAMILIES, fam_lan_devices, fam_find_fuzzy, fam_sysinfo, fam_chat
from gen_data_pentest import PENTEST_FAMILIES
from gen_data_exploit import EXPLOIT_FAMILIES

out = sys.argv[1]
main(280, 'data/_v3_focus.jsonl', [
    (fam_lan_devices, 12),
    (fam_find_fuzzy, 10),
    (fam_sysinfo, 3),
    (fam_chat, 3),
])
main(60, 'data/_v3_general_replay.jsonl', FAMILIES)
main(30, 'data/_v3_recon_replay.jsonl', PENTEST_FAMILIES)
main(30, 'data/_v3_exploit_replay.jsonl', EXPLOIT_FAMILIES)

rows = []
for f in ('data/_v3_focus.jsonl', 'data/_v3_general_replay.jsonl', 'data/_v3_recon_replay.jsonl', 'data/_v3_exploit_replay.jsonl'):
    with open(f) as fh:
        rows += [l for l in fh if l.strip()]
random.Random(42).shuffle(rows)
with open(out, 'w') as fh:
    fh.writelines(rows)
print(f'wrote {len(rows)} examples -> {out}')
PYEOF
fi

echo "== Unloading Ollama models so the GPU is free"
for m in $(curl -s localhost:11434/api/ps | python3 -c 'import sys,json;print(" ".join(x["name"] for x in json.load(sys.stdin).get("models",[])))'); do
  curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
done

echo "== Training $NAME (50 steps from out/lora-pentest-v2 at lr=4e-5)"
$PY train.py \
  --data "$DATA" \
  --out "$OUT" \
  --init-adapter out/lora-pentest-v2 \
  --lr 4e-5 \
  --max-samples 400 \
  --max-len "$MAX_LEN" \
  --save-every 25 2>&1 | tee -a "$OUT.train.log"

echo "== Converting adapter to GGUF"
$PY lora_to_gguf.py "$OUT" "$OUT/qwen-pentest-v3-lora.gguf"

echo "== Registering $NAME in Ollama"
printf 'FROM qwen2.5:7b\nADAPTER %s\n' "$(realpath "$OUT/qwen-pentest-v3-lora.gguf")" > "$OUT/Modelfile"
ollama create "$NAME" -f "$OUT/Modelfile"

echo "== Evaluating (exploit held-out)"
$PY run_eval.py "$NAME" --cases-module eval_cases_exploit --out "results/${NAME}_exploit.json" | tee "results_${NAME}_exploit.log"
echo "== Evaluating (recon held-out)"
$PY run_eval.py "$NAME" --cases-module eval_cases_pentest --out "results/${NAME}_recon.json" | tee "results_${NAME}_recon.log"
echo "== Evaluating (general 35-task held-out)"
$PY run_eval.py "$NAME" --out "results/${NAME}_general.json" | tee "results_${NAME}_general.log"
