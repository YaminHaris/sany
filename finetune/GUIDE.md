# How SANY works, and how to train it yourself

This guide covers everything built for SANY: what each piece does, where it is, how to run it, and which decisions matter.
Read it top to bottom once; after that the sections work as reference.

---

## 1. The big picture

SANY is two separate things that meet at Ollama:

```
 ┌────────────────────────────┐        HTTP /api/chat        ┌──────────────────────────┐
 │  The agent program (CLI)   │ ───────────────────────────▶ │  Ollama                  │
 │  ~/Development/            │   system prompt + history    │  runs the model on your  │
 │    gemini-cli-local        │   + tool list                │  GPU (qwen-agent-v3 etc) │
 │                            │ ◀─────────────────────────── │                          │
 │  - shows the UI            │   text reply OR tool call    └──────────────────────────┘
 │  - runs the tools (shell,  │                                          ▲
 │    files, web) for real    │                                          │ ollama create
 │  - asks you for approval   │                                          │ (base + adapter)
 └────────────────────────────┘                              ┌──────────────────────────┐
                                                             │  The fine-tune project   │
                                                             │  ~/Development/          │
                                                             │    qwen-agent-finetune   │
                                                             │  makes training data,    │
                                                             │  trains a LoRA adapter,  │
                                                             │  tests it in a simulator │
                                                             └──────────────────────────┘
```

**The model never touches your computer directly.** It only ever outputs text, and some of that text is a
structured "tool call" such as `run_shell_command {"command": "ls"}`. The CLI decides whether to run it (asking you first unless YOLO is on),
runs it, and sends the result back. Then the model picks the next step. This loop is called an *agent loop*.

So "making SANY better" means changing one of three things:
1. **The model** (which base model you use, and fine-tuning it). This happens in the fine-tune project.
2. **What the model sees** (system prompt, tool descriptions). These are files in `gemini-cli-local/local/`.
3. **What the tools do** (e.g. how web search works, what's allowed). This is the CLI source code.

Changes 2 and 3 are instant and free. Change 1 takes hours of GPU time. Always ask whether a problem can be fixed by 2 or 3 first.
The ports fix in v6 is an example: the model kept misreading addresses, so instead of more training data the agent now runs a `grep` that does the reading for it.

---

## 2. Part one: the agent program (`~/Development/gemini-cli-local`)

This is Google's open-source Gemini CLI (Apache 2.0 licence), exported clean from its git history and patched to talk to Ollama instead of Google.
Your original `~/Development/gemini-cli` (the broken gemini→gemma rename) was never touched.

### How you start it
`sany` (or `qagent`) → `~/.local/bin/sany` is a link to **`local/gemini-local`**, a small shell script:

```bash
export LOCAL_LLM_BASE_URL=http://localhost:11434     # where Ollama is; setting this switches the CLI to local mode
export LOCAL_LLM_MODEL=qwen-agent-v3                 # which Ollama model to use
export LOCAL_LLM_NUM_CTX=16384                       # context window (tokens the model can see at once)
export GEMINI_SYSTEM_MD=local/system.md              # our short system prompt replaces Google's long one
export GEMINI_CLI_SYSTEM_SETTINGS_PATH=local/settings.json
exec node bundle/gemini.js "$@"
```

Override any of these per run, e.g. `LOCAL_LLM_MODEL=qwen3.5:9b sany`.

### The files we added or changed

| File | What it does |
|---|---|
| `local/gemini-local` | Launcher (above). |
| `local/system.md` | **The system prompt**, the standing instructions the model sees every turn ("You are SANY…", how to work). ~2,200 characters. Edit this to change behaviour without retraining. |
| `local/settings.json` | Which tools exist (9 of them), auto-update off, sub-agents/skills off (they confused the small model and cost extra requests). |
| `packages/core/src/core/localContentGenerator.ts` | **The bridge to Ollama.** Converts the CLI's Gemini-format conversation into Ollama's `/api/chat` format and the reply back. Sets temperature 0.3 (the CLI's default 1.0 made Qwen erratic). Sends tool-call turns with empty text because Ollama's Qwen template drops tool calls when text is present. |
| `packages/core/src/core/localToolSchemas.ts` | **Short tool descriptions.** Google's descriptions were 13,000 characters and drowned a 7B model; these are 4,500. The training data uses the exact same text, so if you edit this you must regenerate data and retrain. |
| `packages/core/src/core/contentGenerator.ts` | Adds an auth type `local`, picked automatically when `LOCAL_LLM_BASE_URL` is set. |
| `packages/cli/src/config/auth.ts` | Skips the Google login check in local mode. |
| `packages/core/src/tools/localWebSearch.ts`, `web-search.ts` | Web search without Google: uses SearXNG if `SEARXNG_URL` is set, otherwise scrapes Bing's results page. |
| `packages/core/src/tools/web-fetch.ts` | In local mode, fetches the URL directly (the original asked Gemini to do it) and caps page text at 12,000 characters. |
| `packages/core/src/config/models.ts` | Footer shows `qwen-agent-v3 (local)` instead of a Gemini model name. |
| `packages/cli/src/ui/components/AsciiArt.ts`, `Header.tsx`, `UserIdentity.tsx`, `packages/cli/src/utils/windowTitle.ts` | SANY branding. |

### Rebuilding after a code change
```bash
cd ~/Development/gemini-cli-local
rm -rf bundle/docs bundle/builtin bundle/policies   # otherwise the asset copy fails with EEXIST
npm run bundle                                     # produces bundle/gemini.js, which the launcher runs
```
Edits to `local/system.md` or `local/settings.json` need no rebuild.

---

## 3. Part two: the fine-tune project (`~/Development/qwen-agent-finetune`)

### 3.1 Key ideas, in plain terms

- **Base model.** A pre-trained model like Qwen2.5-7B-Instruct. It already knows language, Linux and how to call tools; fine-tuning only *adjusts habits*.
- **Fine-tuning** = showing the model thousands of example conversations and nudging its weights so it would have produced them.
- **LoRA** (Low-Rank Adaptation). Instead of changing all 7 billion weights, we add small extra matrices (an *adapter*, ~80 MB as GGUF, 160 MB as safetensors) to each layer and train only those, about 0.5% of the size. The base model stays untouched; Ollama loads base + adapter together.
- **QLoRA.** While training, the frozen base is stored in 4-bit precision to fit in your 6 GB GPU. Part of it still doesn't fit, so `offload_layers='auto'` keeps some layers in system RAM and streams them in, which is slower but works.
- **Chat template.** The model doesn't see "messages"; it sees one long text with special markers like `<|im_start|>assistant`. Training text must be formatted *exactly* the way Ollama formats it at run time, or the model learns habits for a format it never sees. `render.py` does this formatting, and it was checked token-by-token against Ollama (`verify_render.py`).
- **Loss masking.** We only train on the *assistant's* parts (its tool calls and answers), about 4% of the tokens. The system prompt, your messages and tool outputs are context, not things to imitate.
- **Loss.** A number for how surprised the model is by the training examples. Ours drops from ~1.3 to ~0.005, which means it has nearly memorised the patterns. That's a warning sign (see §6).
- **Simulator.** Training examples and tests run against a *fake* computer (`sim.py`): a virtual filesystem, a mini shell that understands ~25 commands, and canned web pages. Nothing the model writes is ever executed for real. This keeps tests safe and repeatable, and lets us generate examples whose answers are guaranteed correct.
- **GGUF.** The file format Ollama/llama.cpp use. Training produces a safetensors adapter; `lora_to_gguf.py` converts it.

### 3.2 Where everything is

| Path | What it is |
|---|---|
| `common.py` | Shared pieces: loads the system prompt from `gemini-cli-local/local/system.md`, builds the "session context" message the CLI sends (date, folder listing), loads the tool definitions, formats tool results exactly like the CLI. |
| `spec/tools.json` | The real tool definitions, dumped from the CLI source. |
| `capture/` | Requests captured from the real CLI with a fake Ollama server (`fake_ollama.py`), used to make sure training data looks byte-for-byte like real traffic. |
| `sim.py` | The simulated computer. `Sim(cwd, files, web=, search=, downloads=, programs=, extra_shell=)`. Each tool is a method `t_<toolname>`. The mini shell is `shell()` / `shell_one()` / `filter()`. |
| `gen_data.py` | **The training data generator.** ~29 "task families" (functions named `fam_*`). Each makes one random example by playing it out in the simulator. `FAMILIES` sets how often each appears. `validate()` rejects examples with unexpected tool errors. |
| `data/train.jsonl` | The generated examples, one conversation per line (3,000 of them). |
| `render.py` | Turns a conversation into the exact text Ollama feeds Qwen2.5, plus the character ranges to train on. |
| `verify_render.py` | Checks `render.py` against Ollama's real token counts. |
| `train.py` | The training loop (Unsloth + PyTorch). |
| `lora_to_gguf.py` | Converts the trained adapter to GGUF for Ollama. Hard-codes the Qwen2 layer names. |
| `run_training.sh` | One-command pipeline: disk check → free GPU → 3-step probe → train → convert → `ollama create` → test. |
| `eval_cases.py` | **The 35 test tasks**, each a prompt, a starting filesystem and a `check` function. Wording deliberately differs from the training data. |
| `run_eval.py` | Runs a model through the test tasks in the simulator (up to 10 steps each) and prints PASS/FAIL. |
| `out/lora-vN/` | Each round's adapter (`adapter_model.safetensors`, `qwen-agent-lora.gguf`, `Modelfile`). `out/pipeline-vN.log` is each round's full log. |
| `results/` | Test results as JSON (every step of every task is in `transcript`). `results/bench/` is the base-model comparison. |
| `gen_data_v1.py … v5.py`, `sim_v1.py …`, `eval_cases_v2.py …` | Snapshots of earlier rounds, kept so any version can be reproduced. |
| `.venv/` | Python environment (6 GB: PyTorch, Unsloth, Transformers). Run Python as `.venv/bin/python`. |
| `llama.cpp/` | A partial copy of llama.cpp: only `gguf-py` (the GGUF writing library) is used. |

### 3.3 The full cycle, step by step

**Step 1: decide what to teach.** Use SANY for real, find where it fails (your pasted sessions were the best source), and write down what it *should* have done.

**Step 2: add a test first.** Add a case to `eval_cases.py` that fails today. Example (the CPU one):
```python
dict(id='cpu_hog', prompt='something is maxing out my processor, what is it?', cwd='/home/yamin', files={},
     extra_shell=[(r'\bps\b|\btop\b', PS_OUT)],          # any command containing "ps" or "top" gets this canned output
     check=lambda s, f: 'blender' in lower(f)),         # s = simulator after the run, f = final answer text
```
Run it on the current model: `python3 run_eval.py qwen-agent-v3 --only cpu_hog`. Keep the wording different from anything in `gen_data.py`, or you're testing memory, not skill.

**Step 3: write a task family** in `gen_data.py`. The pattern is always: set up random files → create a `Traj` (trajectory) → make the tool calls the ideal agent would → write the ideal answer *computed from the simulator's real outputs*, so it can't be wrong:
```python
def fam_hogs():
    procs = R.sample(PROCS, 6)                       # randomise everything: names, numbers, wording
    ...                                               # build a fake `ps` output
    prompt = pick(['what is using my cpu', 'which process is eating cpu', ...])
    t = Traj(prompt, {}, cwd=HOME, extra_shell=[(r'\bps\b|\btop\b', out)])
    t.sh(f'ps aux --sort=-%cpu | head -6', 'Show the top processes by CPU.')   # the ideal tool call
    return t.answer(f'Top CPU users right now:\n{top3}')                       # the ideal final answer
```
Then add it to `FAMILIES` with a weight, e.g. `(fam_hogs, 4)`. Weights are relative: a family with weight 8 appears twice as often as one with 4.
Print a few samples and read them before training. Every bug found this way saves a 4-hour run.

**Step 4: generate data.**
```bash
cd ~/Development/qwen-agent-finetune
.venv/bin/python gen_data.py 3000        # writes data/train.jsonl; prints how many of each family were kept/rejected
```

**Step 5: train, convert, register, test, all in one go.**
```bash
NAME=qwen-agent-v7 OUT=out/lora-v7 MAX_LEN=3072 ./run_training.sh > out/pipeline-v7.log 2>&1 &
tail -f out/pipeline-v7.log              # watch progress; Ctrl+C stops watching, not training
```
Close games and SANY first: training needs the GPU. Expect ~4 hours. The log shows `step N/375 loss … eta …m`.
When it's done, it prints the score. Compare per task with the previous version, e.g.:
```bash
diff <(cut -c1-24 results_qwen-agent-v3.log) <(cut -c1-24 results_qwen-agent-v7.log)
```

**Step 6: read the failures, not just the score.** Open `results/qwen-agent-v7.json`, find the failing task, and read its `transcript`: every call the model made and what came back. Every fix in this project came from reading transcripts.

**Step 7: switch SANY to the new version** if it's better: edit the `LOCAL_LLM_MODEL` default in `~/Development/gemini-cli-local/local/gemini-local`.

### 3.4 What `train.py` does, line by line in words

1. Load the base model in 4-bit (`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`, downloaded once into `~/.cache/huggingface`), offloading what doesn't fit.
2. Attach LoRA adapters to all 7 projection layers in every transformer block (`q,k,v,o` attention and `gate,up,down` MLP), rank 16.
3. Render each example with `render.py`, tokenise it, and mark only assistant tokens as trainable (`labels = -100` everywhere else). Drop examples longer than `--max-len`.
4. For each example: run it through the model, compute the loss, add up gradients. Every 8 examples (`--accum 8`) take one optimiser step. Learning rate warms up, then decays with a cosine curve.
5. Save the adapter every 50 steps and at the end.

### 3.5 The settings you can change, and what they do

| Setting (`train.py` flag / env var) | Now | Effect of changing it |
|---|---|---|
| `--lr` | 2e-4 | Lower (5e-5 to 1e-4) = gentler, less memorising, keeps more general ability. Higher = learns faster, forgets more. |
| `--epochs` | 1 | Passes over the data. More = more memorising. |
| `--rank` | 16 | Adapter size. Bigger can learn more but overfits more easily and uses more memory. |
| `--accum` | 8 | Examples per optimiser step. Bigger = smoother, fewer steps. |
| `MAX_LEN` / `--max-len` | 3072 | Longest example kept. **Too small silently drops examples**; v4 lost most of its new examples this way. Check `skipped_too_long=` in the log. |
| `MAX_SAMPLES` | 3000 | How many examples to train on. Time scales with this. |
| `gen_data.py N` and `FAMILIES` weights | 3000 | The *mix* matters more than the total. Adding a family takes share from the others. |
| Temperature (in `localContentGenerator.ts` and `run_eval.py`) | 0.3 | Randomness when the model answers. Lower = more consistent. |

---

## 4. Testing: how to read scores honestly

- The 35 tasks run in the simulator, one conversation each. **They can't test follow-up conversations** like "look in desktop". Only real use does that.
- A single run is noisy. v3 scored 33 with seed 7 and 34 with seed 11, same model. Run with two seeds (`--seed 7`, `--seed 11`) and compare averages. **A difference of 1 task is noise.**
- A test is only as good as its `check`. The first ports check passed a wrong answer; always read a few passing answers too.
- `results/bench/*.log` holds the base-model comparison, run with `EVAL_THINK=0|1` to switch Qwen3's thinking mode off or on.

---

## 5. Switching to a different base model

To train on another base model, these pieces must match it:

1. **`train.py --base`** → a 4-bit version on Hugging Face (Unsloth publishes them, e.g. `unsloth/Qwen3-8B-bnb-4bit`).
2. **`render.py`** → must reproduce *that model's* chat template exactly. Qwen3 and Qwen3.5 differ from Qwen2.5 (e.g. thinking blocks). Rewrite, then check with `verify_render.py` against `ollama run <model>`.
3. **`lora_to_gguf.py`** → its `PROJ` table maps layer names to llama.cpp names, and it writes architecture `'qwen2'`. Qwen3 = same names plus `qwen3`; **Qwen3.5 has different layers** (three of four use "linear attention") and needs new mappings, and Ollama may not support adapters for it at all. The fallback is merging the adapter into the model and converting the whole thing with llama.cpp's converter, which needs its full source downloaded.
4. **`run_training.sh`** → the `FROM qwen2.5:7b` line in the Modelfile becomes the new Ollama base.
5. **Thinking models** (Qwen3/3.5): decide thinking on or off; `localContentGenerator.ts` would need to send `think: false` for speed.

Benchmark results (untrained, 35 tasks, seeds 7/11): Qwen3.5 9B 31/32 (19 s/task, doesn't fit the GPU), Qwen3 8B 24/25, Qwen3.5 4B 23/28, versus our trained v3 at 33/34 (6 s/task).

---

## 6. Lessons from the six rounds

| Round | Result | Lesson |
|---|---|---|
| v1 | 25/30 | Answers that jump to a result teach mental arithmetic → it got sums wrong. **Make examples show their working.** |
| v2 | 30/30 | Targeted examples fix targeted habits fast. |
| v3 | 30/30 + 3/4 new | Your real sessions revealed failures my tests never covered. **Real usage is the best test.** |
| v4 | 32/35 | Most new examples were longer than `max-len` and silently dropped. **Check `skipped_too_long` every time.** |
| v5 | 33/35 | It learned "always open the page after searching" too well and looped when pages failed. **Every habit needs examples of when NOT to use it, and of failure.** |
| v6 | stopped | Reading address columns was beyond the model, so the tool now does it with `grep`. **Fix the environment when the model hits its limits.** |

And the general warning: training loss near zero means the model memorises these template examples. The cure is gentler training (lower `--lr`) and mixing in 20–30% general, open-licensed function-calling or chat data so it doesn't forget everything else.

---

## 7. Disk and housekeeping

- Each Ollama `qwen-agent-vN` is mostly *shared* with `qwen2.5:7b`; only the adapter (~80 MB) is unique. Remove old ones with `ollama rm qwen-agent-v4` etc.
- `out/lora-vN/` is ~240 MB each.
- `qwen2.5-coder:7b` (4.7 GB) is unused. The base-model benchmark downloaded `qwen3.5:4b`, `qwen3:8b`, `qwen3.5:9b` (15 GB together); remove the losers when you've decided.
- The 4-bit training base lives in `~/.cache/huggingface` (~5.5 GB).

---

## 8. Glossary

**Adapter**: the small trained add-on (LoRA) loaded on top of a base model. **Agent loop**: model proposes a tool call → program runs it → result goes back → repeat. **Context window**: how much text the model can see at once (16,384 tokens here). **Epoch**: one pass over the training data. **Eval**: the test suite. **GGUF**: the model file format Ollama uses. **Gradient accumulation**: adding up several examples before one weight update. **Learning rate**: how big each update is. **Loss**: how wrong the model is on training examples (lower = closer to them). **Overfitting**: memorising the training examples instead of learning the skill. **Quantization (4-bit)**: storing weights with fewer bits to save memory. **Seed**: the random number that makes a run repeatable. **Token**: a word piece; ~4 characters of English. **Tool call**: the structured request the model emits to run a tool. **Trajectory**: one full example conversation including tool calls and results.
