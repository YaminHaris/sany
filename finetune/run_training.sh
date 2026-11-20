#!/usr/bin/env bash
# Full pipeline: probe -> train LoRA -> convert to GGUF adapter -> register in Ollama as qwen-agent -> evaluate.
# Needs ~10 GB free disk (5.5 GB base model download + headroom) and an otherwise idle GPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
OUT=${OUT:-out/lora-v1}
MAX_SAMPLES=${MAX_SAMPLES:-3000}
NAME=${NAME:-qwen-agent}
MAX_LEN=${MAX_LEN:-2560}

free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
if [ "$free_gb" -lt 10 ]; then
  echo "Only ${free_gb} GB free; need at least 10 GB. Free some space first." >&2
  exit 1
fi

echo "== Unloading Ollama models so the GPU is free"
for m in $(curl -s localhost:11434/api/ps | python3 -c 'import sys,json;print(" ".join(x["name"] for x in json.load(sys.stdin)["models"]))'); do
  curl -s localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null
done

echo "== Probe: 3 optimizer steps to check memory and speed"
$PY train.py --out "$OUT" --probe --max-samples 200 --max-len "$MAX_LEN"

echo "== Training"
$PY train.py --out "$OUT" --max-samples "$MAX_SAMPLES" --max-len "$MAX_LEN" 2>&1 | tee "$OUT.train.log"

echo "== Converting adapter to GGUF"
$PY lora_to_gguf.py "$OUT" "$OUT/qwen-agent-lora.gguf"

echo "== Registering $NAME in Ollama"
printf 'FROM qwen2.5:7b\nADAPTER %s\n' "$(realpath "$OUT/qwen-agent-lora.gguf")" > "$OUT/Modelfile"
ollama create "$NAME" -f "$OUT/Modelfile"

echo "== Evaluating"
python3 run_eval.py "$NAME" --out "results/$NAME.json" | tee "results_$NAME.log"
echo "Baseline for comparison: $(tail -1 results_base_compact.log)"
