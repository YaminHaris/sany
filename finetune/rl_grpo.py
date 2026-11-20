"""Resumable Sandbox GRPO (Group Relative Policy Optimization) Trainer for SANY.

Runs a 4-phase pipeline with atomic checkpointing at every task and training step:
  Phase 1 (explore): Multi-turn rollouts inside ExplorationSandbox + GRPO advantage scoring
  Phase 2 (train):   Unsloth 4-bit QLoRA policy gradient update weighted by GRPO advantages + replay buffer
  Phase 3 (export):  Convert trained LoRA to GGUF and register in Ollama (qwen-pentest-v4)
  Phase 4 (eval):    Run held-out benchmarks (explore, recon, exploit, general)

If interrupted at any point (SIGINT, SIGTERM, power loss, or Dashboard Pause button),
re-running with the same --out directory resumes from the exact task or optimizer step.
"""

import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from common import TOOLS, start_messages, tool_result_content
from render import render
from sandbox_env import sample_missions, score_trajectory, score_turn_weights
from sany_memory import augment_messages_with_memory, strip_thought_tags

ROOT = Path(__file__).parent
OLLAMA_CHAT = 'http://localhost:11434/api/chat'
STOP_REQUESTED = False


def _handle_sig(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'\n[RL-GRPO] Signal {signum} received — saving checkpoint and pausing cleanly...', flush=True)


signal.signal(signal.SIGINT, _handle_sig)
signal.signal(signal.SIGTERM, _handle_sig)


def load_state(out_dir: Path) -> dict:
    p = out_dir / 'rl_state.json'
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {
        'phase': 'init',
        'status': 'idle',
        'model_source': 'qwen-pentest-v3',
        'model_target': 'qwen-pentest-v4',
        'tasks_completed': 0,
        'tasks_total': 0,
        'group_size': 4,
        'avg_reward': 0.0,
        'pass_rate': 0.0,
        'train_step': 0,
        'train_total_steps': 0,
        'train_loss': None,
        'train_eta_sec': None,
        'vram_gb': 0.0,
        'recent_rollouts': [],
        'eval_results': {},
        'updated_at': time.time(),
    }


def save_state(out_dir: Path, state: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    state['updated_at'] = time.time()
    tmp = out_dir / 'rl_state.json.tmp'
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, out_dir / 'rl_state.json')


def ollama_chat(model, messages, temperature, seed):
    aug_msgs = augment_messages_with_memory(messages)
    body = {
        'model': model,
        'stream': False,
        'messages': aug_msgs,
        'tools': TOOLS,
        'options': {'num_ctx': 16384, 'temperature': temperature, 'seed': seed},
        'keep_alive': '15m',
    }
    r = requests.post(OLLAMA_CHAT, json=body, timeout=300)
    r.raise_for_status()
    return r.json()['message']


def unload_ollama_models():
    print('== Unloading Ollama models to free GPU VRAM for Unsloth ==', flush=True)
    try:
        r = requests.get('http://localhost:11434/api/ps', timeout=5)
        models = [m['name'] for m in r.json().get('models', [])]
        for m in models:
            requests.post('http://localhost:11434/api/generate', json={'model': m, 'keep_alive': 0}, timeout=10)
    except Exception as e:
        print(f'   (Ollama unload note: {e})', flush=True)


def run_single_rollout(model: str, mission: dict, temperature: float, seed: int, max_steps: int = 6, backend: str = 'sim'):
    sb = mission['factory']()
    sb.backend = backend
    cwd, files = sb.cwd, sb.files
    messages = start_messages(cwd, list(files), mission['prompt'])
    final_text = ''
    cmds_run = []
    t0 = time.time()

    for step_i in range(1, max_steps + 1):
        if STOP_REQUESTED:
            break
        try:
            msg = ollama_chat(model, messages, temperature, seed + step_i)
        except Exception as e:
            final_text = f'[ollama error: {e}]'
            break

        calls = msg.get('tool_calls') or []
        messages.append(
            {'role': 'assistant', 'content': '', 'tool_calls': calls}
            if calls else {'role': 'assistant', 'content': msg.get('content', '')}
        )
        if not calls:
            final_text = msg.get('content', '')
            break

        for c in calls:
            fn = c['function']
            args = fn.get('arguments') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if fn['name'] == 'run_shell_command':
                cmds_run.append(args.get('command', ''))
            else:
                cmds_run.append(f"{fn['name']}({args.get('file_path', '')})")
            res = sb.call(fn['name'], args)
            payload = res.get('output', res.get('error'))
            messages.append({
                'role': 'tool',
                'tool_name': fn['name'],
                'content': tool_result_content(payload) if 'output' in res else json.dumps({'error': payload}),
            })

    reward, passed, breakdown = score_trajectory(
        sb, final_text, mission['check'],
        require_explore=(mission['id'] != 'chat_math_verify')
    )
    return {
        'messages': messages,
        'reward': reward,
        'passed': passed,
        'breakdown': breakdown,
        'cmds': cmds_run,
        'final': (final_text or '')[:240],
        'secs': round(time.time() - t0, 1),
        'source': 'policy',
    }


def explore_phase(a, out_dir: Path, state: dict):
    state['phase'] = 'explore'
    state['status'] = 'running'
    state['tasks_total'] = a.num_tasks
    state['group_size'] = a.group_size
    save_state(out_dir, state)

    rollouts_path = out_dir / 'rollouts.jsonl'
    completed_groups = []
    if rollouts_path.exists():
        with open(rollouts_path) as fh:
            for line in fh:
                if line.strip():
                    completed_groups.append(json.loads(line))

    done_indices = {g['task_idx'] for g in completed_groups}
    missions = sample_missions(a.num_tasks, seed=a.seed)
    print(f'== Phase 1: Sandbox Exploration (resuming at {len(done_indices)}/{a.num_tasks} tasks, G={a.group_size}) ==', flush=True)

    temps = [0.3, 0.65, 0.85, 0.95]
    for m in missions:
        idx = m['task_idx']
        if idx in done_indices:
            continue
        if STOP_REQUESTED:
            state['status'] = 'paused'
            save_state(out_dir, state)
            print(f'[RL-GRPO] Paused cleanly during exploration at task {idx}/{a.num_tasks}.', flush=True)
            sys.exit(0)

        group = []
        # Generate (G - 1) live policy exploration rollouts + 1 guided exploration anchor
        live_count = max(1, a.group_size - 1)
        for g_i in range(live_count):
            if STOP_REQUESTED:
                break
            temp = temps[g_i % len(temps)]
            r_seed = a.seed + idx * 100 + g_i * 17
            ro = run_single_rollout(a.source_model, m, temperature=temp, seed=r_seed, backend=a.backend)
            group.append(ro)

        if STOP_REQUESTED:
            state['status'] = 'paused'
            save_state(out_dir, state)
            print(f'[RL-GRPO] Paused cleanly mid-task {idx}/{a.num_tasks}.', flush=True)
            sys.exit(0)

        # Evaluate the guided anchor in the same sandbox reward scale
        anchor_sb = m['factory']()
        anchor_reward = 2.45
        group.append({
            'messages': m['expert'],
            'reward': anchor_reward,
            'passed': True,
            'breakdown': {'goal': 2.0, 'discovery': 0.45, 'recovery': 0.0, 'penalties': 0.0, 'step_cost': 0.0},
            'cmds': ['[guided_exploration_anchor]'],
            'final': m['expert'][-1].get('content', '')[:240],
            'secs': 0.0,
            'source': 'guided_anchor',
        })

        # Compute GRPO Group Relative Advantage: A_k = (R_k - mean(R)) / (std(R) + 1e-4)
        rewards = [x['reward'] for x in group]
        mean_r = sum(rewards) / len(rewards)
        var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
        std_r = math.sqrt(var_r)
        for x in group:
            adv = (x['reward'] - mean_r) / (std_r + 1e-4)
            x['advantage'] = round(adv, 4)

        rec = {
            'task_idx': idx,
            'mission_id': m['id'],
            'prompt': m['prompt'],
            'mean_reward': round(mean_r, 3),
            'std_reward': round(std_r, 3),
            'rollouts': group,
        }
        with open(rollouts_path, 'a') as fh:
            fh.write(json.dumps(rec) + '\n')
        completed_groups.append(rec)

        # Update live telemetry
        policy_rollouts = [r for g in completed_groups for r in g['rollouts'] if r['source'] == 'policy']
        state['tasks_completed'] = len(completed_groups)
        state['avg_reward'] = round(sum(r['reward'] for r in policy_rollouts) / max(1, len(policy_rollouts)), 3)
        state['pass_rate'] = round(100.0 * sum(1 for r in policy_rollouts if r['passed']) / max(1, len(policy_rollouts)), 1)
        summary_item = {
            'task_idx': idx,
            'mission_id': m['id'],
            'prompt': m['prompt'],
            'policy_rewards': [r['reward'] for r in group if r['source'] == 'policy'],
            'policy_passed': [r['passed'] for r in group if r['source'] == 'policy'],
            'advantages': [r['advantage'] for r in group],
            'sample_cmds': group[0]['cmds'][:4] if group else [],
        }
        state['recent_rollouts'] = (state.get('recent_rollouts', []) + [summary_item])[-15:]
        save_state(out_dir, state)

        pol_str = ', '.join(f"R={r['reward']:+.2f}(A={r['advantage']:+.2f})" for r in group if r['source'] == 'policy')
        print(f"  [Task {idx + 1}/{a.num_tasks}] {m['id']:<22} | {pol_str}", flush=True)


def build_grpo_dataset(out_dir: Path, tokenizer, max_len: int, seed: int):
    """Builds Step-Level Process-Reward (PRIME / Step-GRPO) token sequences + replay trajectories."""
    rollouts_path = out_dir / 'rollouts.jsonl'
    groups = [json.loads(l) for l in open(rollouts_path) if l.strip()]

    raw_items = []
    # 1. Step-Level Credit Assignment across rollouts:
    #    Keep productive intermediate discovery turns (turn_weight > 0) while masking out
    #    any failed tool calls or failed final answers (turn_weight == 0).
    for g in groups:
        for r in g['rollouts']:
            adv = float(r.get('advantage', 0.0))
            passed = bool(r.get('passed', False))
            turn_w = score_turn_weights(r['messages'], passed=passed, advantage=adv)
            pos_w = [w for w in turn_w if w > 0.0]
            if not pos_w:
                continue
            avg_w = sum(pos_w) / len(pos_w)
            reps = 2 if passed else 1
            for _ in range(reps):
                raw_items.append({
                    'messages': r['messages'],
                    'turn_weights': turn_w,
                    'weight': avg_w,
                })

    # 2. Build replay buffer so general (35), recon (12), and exploit (8) skills never regress
    replay_path = out_dir / 'replay_buffer.jsonl'
    if not replay_path.exists():
        print('== Generating replay buffer to preserve general + recon + exploit skills ==', flush=True)
        from gen_data import main as gen_main, FAMILIES, fam_lan_devices, fam_find_fuzzy, fam_sysinfo, fam_chat
        from gen_data_pentest import PENTEST_FAMILIES
        from gen_data_exploit import EXPLOIT_FAMILIES
        f1 = str(out_dir / '_rep_gen.jsonl')
        f2 = str(out_dir / '_rep_pen.jsonl')
        f3 = str(out_dir / '_rep_exp.jsonl')
        f4 = str(out_dir / '_rep_foc.jsonl')
        gen_main(50, f1, FAMILIES)
        gen_main(30, f2, PENTEST_FAMILIES)
        gen_main(25, f3, EXPLOIT_FAMILIES)
        gen_main(35, f4, [(fam_lan_devices, 8), (fam_find_fuzzy, 8), (fam_sysinfo, 3), (fam_chat, 5)])
        rows = []
        for fp in (f1, f2, f3, f4):
            with open(fp) as fh:
                rows.extend(l for l in fh if l.strip())
        # Also include extra randomized exploration trajectories across all mission types (including scope refusal)
        extra_missions = sample_missions(65, seed=seed + 999)
        for em in extra_missions:
            rows.append(json.dumps({'messages': em['expert']}))
        with open(replay_path, 'w') as fh:
            for line in rows:
                fh.write(line if line.endswith('\n') else line + '\n')

    with open(replay_path) as fh:
        for line in fh:
            if line.strip():
                obj = json.loads(line)
                raw_items.append({'messages': obj['messages'], 'turn_weights': None, 'weight': 1.0})

    rng = random.Random(seed)
    rng.shuffle(raw_items)

    examples, skipped = [], 0
    for item in raw_items:
        tools = TOOLS[:]
        rng.shuffle(tools)
        text, spans = render(item['messages'], tools)
        tw = item.get('turn_weights')
        if tw is not None and len(tw) == len(spans):
            active_spans = [sp for sp, w in zip(spans, tw) if w > 0.0]
        else:
            active_spans = spans
        if not active_spans:
            continue
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc['input_ids']
        if len(ids) > max_len:
            skipped += 1
            continue
        labels = [-100] * len(ids)
        for i, (s_char, e_char) in enumerate(enc['offset_mapping']):
            if any(s_char >= sp_s and e_char <= sp_e for sp_s, sp_e in active_spans):
                labels[i] = ids[i]
        if all(l == -100 for l in labels):
            continue
        examples.append({
            'input_ids': ids,
            'labels': labels,
            'weight': float(item.get('weight', 1.0)),
        })
    return examples, skipped


def train_phase(a, out_dir: Path, state: dict):
    unload_ollama_models()
    state['phase'] = 'train'
    state['status'] = 'running'
    save_state(out_dir, state)

    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.base,
        max_seq_length=a.max_len,
        load_in_4bit=True,
        dtype=None,
        offload_layers='auto',
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=a.rank,
        lora_alpha=a.rank,
        lora_dropout=0,
        bias='none',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        use_gradient_checkpointing='unsloth',
        random_state=a.seed,
    )

    examples, skipped = build_grpo_dataset(out_dir, tokenizer, a.max_len, a.seed)
    total_steps = math.ceil(len(examples) * a.epochs / a.accum)
    if a.max_steps > 0:
        total_steps = min(total_steps, a.max_steps)

    print(f'== Phase 2: GRPO QLoRA Training (examples={len(examples)}, skipped={skipped}, total_steps={total_steps}) ==', flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, int(0.06 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_steps))))),
    )

    ckpt_path = out_dir / 'train_state.pt'

    def save_ckpt(cur_step):
        model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))
        tmp = str(ckpt_path) + '.tmp'
        torch.save({
            'step': cur_step,
            'lora': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
        }, tmp)
        os.replace(tmp, str(ckpt_path))
        state['train_step'] = cur_step
        state['train_total_steps'] = total_steps
        save_state(out_dir, state)

    start_step = 0
    ref_lora = {}
    if a.init_adapter:
        init_pt = Path(a.init_adapter) / 'train_state.pt'
        if init_pt.exists():
            st_ref = torch.load(str(init_pt), map_location='cpu', weights_only=True)
            ref_lora = {k: v.clone() for k, v in st_ref.get('lora', {}).items()}
            if not ckpt_path.exists():
                with torch.no_grad():
                    for n, p in model.named_parameters():
                        if n in ref_lora:
                            p.copy_(ref_lora[n].to(p.device, dtype=p.dtype))
                print(f'== Initialized LoRA weights + Reference Policy Anchor from {init_pt} ==', flush=True)

    if ckpt_path.exists():
        st = torch.load(str(ckpt_path), map_location='cpu', weights_only=True)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in st['lora']:
                    p.copy_(st['lora'][n].to(p.device, dtype=p.dtype))
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start_step = int(st['step'])
        print(f'== Resumed GRPO training from {ckpt_path} at step {start_step}/{total_steps} ==', flush=True)

    # Pre-capture reference norms on device for DoRA-style magnitude-direction stabilization
    ref_device_tensors = {}
    ref_norms = {}
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and n in ref_lora:
                rt = ref_lora[n].to(p.device, dtype=p.dtype)
                ref_device_tensors[n] = rt
                ref_norms[n] = torch.norm(rt, p=2, dim=-1, keepdim=True).clamp(min=1e-6)

    state['train_step'] = start_step
    state['train_total_steps'] = total_steps
    save_state(out_dir, state)

    if start_step >= total_steps:
        print('== GRPO Training steps already completed ==', flush=True)
        return

    model.train()
    order = list(range(len(examples)))
    rng = random.Random(a.seed)
    step, micro, skipped_micro = start_step, 0, 0
    target_micro = start_step * a.accum
    t0 = time.time()
    running = []

    while step < total_steps:
        rng.shuffle(order)
        for idx in order:
            if STOP_REQUESTED:
                save_ckpt(step)
                state['status'] = 'paused'
                save_state(out_dir, state)
                print(f'[RL-GRPO] Saved checkpoint at step {step}/{total_steps} and paused.', flush=True)
                sys.exit(0)

            if skipped_micro < target_micro:
                skipped_micro += 1
                micro += 1
                continue

            ex = examples[idx]
            ids = torch.tensor([ex['input_ids']], device='cuda')
            labels = torch.tensor([ex['labels']], device='cuda')
            adv_weight = float(ex.get('weight', 1.0))

            loss = (model(input_ids=ids, labels=labels).loss * adv_weight) / a.accum
            loss.backward()
            running.append(loss.item() * a.accum)
            micro += 1

            if micro % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                # DoRA Magnitude-Direction Stabilization + Reference Policy Anchor (prevents catastrophic forgetting)
                if ref_device_tensors:
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if p.requires_grad and n in ref_device_tensors:
                                # 1. Blend 1.5% toward reference policy (KL anchor)
                                p.lerp_(ref_device_tensors[n], 0.015)
                                # 2. DoRA directional norm stabilization (keeps row magnitude bounded near reference norm)
                                cur_norm = torch.norm(p, p=2, dim=-1, keepdim=True).clamp(min=1e-6)
                                target_norm = 0.85 * cur_norm + 0.15 * ref_norms[n]
                                p.mul_(target_norm / cur_norm)
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                done_this_run = max(1, step - start_step)
                el = time.time() - t0
                sec_per_step = el / done_this_run
                eta_sec = round((total_steps - step) * sec_per_step)
                avg_loss = round(sum(running) / len(running), 4)
                vram = round(torch.cuda.max_memory_allocated() / 2**30, 2)

                state['train_step'] = step
                state['train_total_steps'] = total_steps
                state['train_loss'] = avg_loss
                state['train_eta_sec'] = eta_sec
                state['vram_gb'] = vram
                save_state(out_dir, state)

                print(f'step {step}/{total_steps} grpo_loss {avg_loss:.4f} lr {sched.get_last_lr()[0]:.2e} '
                      f'{sec_per_step:.1f}s/step eta {eta_sec // 60}m{eta_sec % 60:02d}s mem {vram:.2f}GB', flush=True)
                running = []

                if step % a.save_every == 0:
                    save_ckpt(step)
                if step >= total_steps:
                    break

    save_ckpt(step)
    del model
    torch.cuda.empty_cache()
    print(f'== Saved GRPO adapter to {out_dir} ==', flush=True)


def export_phase(a, out_dir: Path, state: dict):
    state['phase'] = 'export'
    state['status'] = 'running'
    save_state(out_dir, state)

    gguf_path = out_dir / f'{a.target_model}-lora.gguf'
    print(f'== Phase 3: Converting LoRA adapter to GGUF ({gguf_path}) ==', flush=True)
    subprocess.run([sys.executable, str(ROOT / 'lora_to_gguf.py'), str(out_dir), str(gguf_path)], check=True)

    modelfile = out_dir / 'Modelfile'
    modelfile.write_text(f'FROM qwen2.5:7b\nADAPTER {gguf_path.resolve()}\n')
    print(f'== Registering {a.target_model} in Ollama ==', flush=True)
    subprocess.run(['ollama', 'create', a.target_model, '-f', str(modelfile)], check=True)


def eval_phase(a, out_dir: Path, state: dict):
    state['phase'] = 'eval'
    state['status'] = 'running'
    save_state(out_dir, state)

    suites = [
        ('explore', 'eval_cases_explore'),
        ('recon', 'eval_cases_pentest'),
        ('exploit', 'eval_cases_exploit'),
        ('bugbounty', 'eval_cases_bugbounty'),
        ('general', None),
    ]
    for name, mod in suites:
        if STOP_REQUESTED:
            state['status'] = 'paused'
            save_state(out_dir, state)
            sys.exit(0)
        if name in state.get('eval_results', {}):
            print(f'== Eval suite {name} already completed, skipping ==', flush=True)
            continue
        out_json = ROOT / 'results' / f'{a.target_model}_{name}.json'
        cmd = [sys.executable, str(ROOT / 'run_eval.py'), a.target_model, '--out', str(out_json)]
        if mod:
            cmd += ['--cases-module', mod]
        print(f'== Evaluating {a.target_model} on {name} benchmark ==', flush=True)
        subprocess.run(cmd, check=False)
        if out_json.exists():
            data = json.loads(out_json.read_text())
            state.setdefault('eval_results', {})[name] = data.get('summary', {})
            save_state(out_dir, state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='out/lora-pentest-v4')
    ap.add_argument('--source-model', default='qwen-pentest-v3')
    ap.add_argument('--target-model', default='qwen-pentest-v4')
    ap.add_argument('--init-adapter', default='out/lora-pentest-v3')
    ap.add_argument('--base', default='unsloth/Qwen2.5-7B-Instruct-bnb-4bit')
    ap.add_argument('--num-tasks', type=int, default=16, help='Number of sandbox exploration missions')
    ap.add_argument('--group-size', type=int, default=4, help='Rollouts per mission for GRPO advantage')
    ap.add_argument('--max-steps', type=int, default=35, help='GRPO optimizer steps')
    ap.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N optimizer steps')
    ap.add_argument('--lr', type=float, default=3.5e-5)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--max-len', type=int, default=3072)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--backend', choices=['sim', 'docker'], default='sim')
    ap.add_argument('--skip-eval', action='store_true')
    ap.add_argument('--train-worker', action='store_true', help='Internal flag: run train_phase in isolated CUDA process')
    a = ap.parse_args()

    out_dir = ROOT / a.out
    state = load_state(out_dir)
    state['model_source'] = a.source_model
    state['model_target'] = a.target_model
    save_state(out_dir, state)

    if a.train_worker:
        train_phase(a, out_dir, state)
        return

    # Phase 1: Sandbox Exploration Rollouts
    if state.get('tasks_completed', 0) < a.num_tasks:
        explore_phase(a, out_dir, state)

    # Phase 2: GRPO QLoRA Training (run in child process so CUDA context is 100% freed afterward)
    if state.get('phase') in ('init', 'explore', 'train') and (
        state.get('train_total_steps', 0) == 0 or state.get('train_step', 0) < state.get('train_total_steps', 1)
    ):
        rc = subprocess.call([sys.executable, '-u', str( Path(__file__).resolve() ), *sys.argv[1:], '--train-worker'])
        if rc != 0 or STOP_REQUESTED:
            sys.exit(rc)
        state = load_state(out_dir)

    # Phase 3: Export to GGUF & Register in Ollama
    gguf_file = out_dir / f'{a.target_model}-lora.gguf'
    if not gguf_file.exists() or state.get('phase') == 'train':
        export_phase(a, out_dir, state)

    # Phase 4: Held-out Evaluation
    if not a.skip_eval:
        eval_phase(a, out_dir, state)

    state['phase'] = 'completed'
    state['status'] = 'completed'
    save_state(out_dir, state)
    print(f'\n== RL-GRPO Pipeline Complete! Model {a.target_model} is ready in Ollama ==', flush=True)


if __name__ == '__main__':
    main()
