# SANY — Local Agentic CLI & Fine-Tuned Qwen2.5-7B Pentest/Systems Model

**SANY** is a fully local, tool-calling AI terminal agent powered by a custom fine-tuned **Qwen2.5-7B-Instruct** model running via **Ollama**, paired with a patched TypeScript/Node **Gemini CLI** frontend.

---

## Architecture

```
┌──────────────────────────────┐     HTTP /api/chat     ┌──────────────────────────────┐
│  cli/ (gemini-cli-local)     │ ─────────────────────▶ │  Ollama                      │
│  - Compact system prompt     │  system + history +    │  qwen-pentest-v5 (7B QLoRA)  │
│  - 9 core local/web tools    │  compact tool schemas  │  16K context, temp 0.3       │
│  - Local web search & fetch  │ ◀───────────────────── │                              │
└──────────────────────────────┘   text or tool_call    └──────────────────────────────┘
                                                                       ▲
                                                                       │ GGUF LoRA
                                                        ┌──────────────────────────────┐
                                                        │  finetune/                   │
                                                        │  - QLoRA + DPO + GRPO RL     │
                                                        │  - Virtual sandbox simulator │
                                                        │  - 1,200-task eval suite     │
                                                        └──────────────────────────────┘
```

### Key Components

1. **`cli/` — Local Agent Runtime (Patched Gemini CLI)**
   * Bridges Gemini CLI's conversation loop to Ollama's `/api/chat` (`packages/core/src/core/localContentGenerator.ts`).
   * Uses compact tool schemas (`localToolSchemas.ts`) and a focused 25-line system prompt (`local/system.md`) tailored for 7B models.
   * Implements local web search (SearXNG / Bing fallback) and direct `web_fetch` without external cloud APIs.

2. **`finetune/` — Training, RL & Evaluation Pipeline**
   * **Constrained-Hardware QLoRA:** Trained on a 6 GB RTX 4050 Laptop GPU using Unsloth, 4-bit quantization, layer offloading, and NEFTune (`train.py`, `dpo_refine.py`).
   * **GRPO Reinforcement Learning:** Sandbox-verified RL training loop with a web control dashboard (`rl_grpo.py`, `rl_dashboard.py`).
   * **Token-Exact Template Verification:** `render.py` and `verify_render.py` validate training sequences token-for-token against Ollama's Qwen2.5 template, masking loss to assistant turns only.
   * **Cognitive Memory Layer:** `sany_memory.py` adds episodic experience replay (reflexion), session fact pinning, and `<thought>` scratchpad parsing.
   * **Deterministic Simulator & 1,200-Task Benchmark:** `sim.py`, `sandbox_env.py`, and `run_1000_bench.py` evaluate multi-step shell, file, networking, recon, and web exploitation tasks.

---

## Latest Benchmark Results (`qwen-pentest-v5`)

Evaluated on the 120-case stratified sample of the 1,200-task benchmark (`run_1000_bench.py --stratified 5`):
* **Overall Pass Rate:** **100 / 120 (83.33%)**
* **Described Instead of Acting:** **0**
* **Targeted Suites:** Bug Bounty **5/5**, Recon **10/12**, Exploit **5/8**, Explore **2/3**

---

## Quick Start

### 1. Build & Run the CLI
```bash
cd cli
npm install
npm run bundle
./local/gemini-local
```
Environment overrides: `LOCAL_LLM_MODEL` (default `qwen-pentest-v5`), `LOCAL_LLM_BASE_URL` (default `http://localhost:11434`), `LOCAL_LLM_NUM_CTX` (default `16384`).

### 2. Fine-Tuning & Evaluation
```bash
cd finetune
python3 -m venv .venv && source .venv/bin/activate
# Generate data, train QLoRA adapter, convert to GGUF, and register in Ollama:
./run_v5_polish.sh
# Run benchmark evaluation:
python run_1000_bench.py --model qwen-pentest-v5 --stratified 5
```
See [`finetune/GUIDE.md`](finetune/GUIDE.md) for detailed documentation of the training pipeline and simulator.
