"""QLoRA fine-tune of Qwen2.5-7B-Instruct on agent trajectories (6 GB GPU budget).

Usage: python train.py --data data/train.jsonl --out out/lora-v1 [--max-samples N] [--max-len 4096]
"""

import argparse
import json
import math
import os
import random
import time

import torch
from unsloth import FastLanguageModel  # must be imported before transformers

from common import TOOLS
from render import render


def build_examples(path, tokenizer, max_len, max_samples, seed):
    rows = [json.loads(l) for l in open(path)]
    random.Random(seed).shuffle(rows)
    examples, skipped = [], 0
    for r in rows:
        tools = TOOLS[:]
        random.Random(len(examples)).shuffle(tools)
        text, spans = render(r['messages'], tools)
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc['input_ids']
        if len(ids) > max_len:
            skipped += 1
            continue
        labels = [-100] * len(ids)
        for i, (a, b) in enumerate(enc['offset_mapping']):
            if any(a >= s and b <= e for s, e in spans):
                labels[i] = ids[i]
        examples.append({'input_ids': ids, 'labels': labels})
        if max_samples and len(examples) >= max_samples:
            break
    return examples, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data/train.jsonl')
    ap.add_argument('--out', default='out/lora-v1')
    ap.add_argument('--base', default='unsloth/Qwen2.5-7B-Instruct-bnb-4bit')
    ap.add_argument('--max-len', type=int, default=2560)
    ap.add_argument('--max-samples', type=int, default=0)
    ap.add_argument('--epochs', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--accum', type=int, default=8)
    ap.add_argument('--rank', type=int, default=16)
    ap.add_argument('--rslora', action='store_true', help='Use Rank-Stabilized LoRA (scales by alpha/sqrt(r)).')
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--neftune', type=float, default=0.0, help='NEFTune embedding noise alpha (e.g. 5.0).')
    ap.add_argument('--seed', type=int, default=3407)
    ap.add_argument('--save-every', type=int, default=25, help='Save adapter and resumable checkpoint every N steps.')
    ap.add_argument('--no-resume', action='store_true', help='Ignore any existing train_state.pt checkpoint in --out.')
    ap.add_argument('--init-adapter', default='', help='Initialize LoRA weights from an existing checkpoint dir (containing train_state.pt).')
    ap.add_argument('--offload', default='auto', help='offload_layers for Unsloth ("auto" streams layers that do not fit in VRAM from RAM; "none" to disable).')
    ap.add_argument('--probe', action='store_true', help='Run a few steps to measure memory and speed, then exit.')
    a = ap.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=a.base, max_seq_length=a.max_len, load_in_4bit=True, dtype=None,
        **({} if a.offload == 'none' else {'offload_layers': a.offload}))
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.rank, lora_dropout=a.dropout, bias='none',
        use_rslora=a.rslora,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        use_gradient_checkpointing='unsloth', random_state=a.seed)

    if a.neftune > 0:
        emb = model.get_input_embeddings()
        def _neftune_hook(mod, inp, out):
            if mod.training and out.dim() == 3:
                mag = a.neftune / math.sqrt(out.size(1) * out.size(2))
                return out + torch.empty_like(out).uniform_(-mag, mag)
            return out
        emb.register_forward_hook(_neftune_hook)
        print(f'enabled NEFTune noise alpha={a.neftune}', flush=True)

    examples, skipped = build_examples(a.data, tokenizer, a.max_len, a.max_samples, a.seed)
    lens = sorted(len(e['input_ids']) for e in examples)
    print(f'examples={len(examples)} skipped_too_long={skipped} tokens median={lens[len(lens) // 2]} max={lens[-1]} '
          f'trained_tokens_frac={sum(sum(l != -100 for l in e["labels"]) for e in examples) / sum(lens):.3f}', flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.weight_decay)
    total_steps = math.ceil(len(examples) * a.epochs / a.accum)
    if a.probe:
        total_steps = 3
    warmup = max(1, int(0.05 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))

    state_path = os.path.join(a.out, 'train_state.pt')

    def save_ckpt(cur_step):
        model.save_pretrained(a.out)
        tmp = state_path + '.tmp'
        torch.save({
            'step': cur_step,
            'lora': {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
            'opt': opt.state_dict(),
            'sched': sched.state_dict(),
        }, tmp)
        os.replace(tmp, state_path)

    start_step = 0
    if a.init_adapter and not os.path.exists(state_path):
        init_st_path = os.path.join(a.init_adapter, 'train_state.pt')
        st = torch.load(init_st_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in st['lora']:
                    p.copy_(st['lora'][n].to(p.device, dtype=p.dtype))
        print(f'initialized LoRA weights from {init_st_path}', flush=True)
    if not a.probe and not a.no_resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location='cpu', weights_only=True)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in st['lora']:
                    p.copy_(st['lora'][n].to(p.device, dtype=p.dtype))
        opt.load_state_dict(st['opt'])
        sched.load_state_dict(st['sched'])
        start_step = int(st['step'])
        print(f'resumed from {state_path} at step {start_step}/{total_steps}', flush=True)

    model.train()
    order = list(range(len(examples)))
    rng = random.Random(a.seed)
    step, micro, skipped_micro, t0, running = start_step, 0, 0, time.time(), []
    target_micro = start_step * a.accum
    while step < total_steps:
        rng.shuffle(order)
        for idx in order:
            if skipped_micro < target_micro:
                skipped_micro += 1
                micro += 1
                continue
            ex = examples[idx]
            ids = torch.tensor([ex['input_ids']], device='cuda')
            labels = torch.tensor([ex['labels']], device='cuda')
            loss = model(input_ids=ids, labels=labels).loss / a.accum
            loss.backward()
            running.append(loss.item() * a.accum)
            micro += 1
            if micro % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                done_this_run = max(1, step - start_step)
                el = time.time() - t0
                sec_per_step = el / done_this_run
                print(f'step {step}/{total_steps} loss {sum(running) / len(running):.4f} lr {sched.get_last_lr()[0]:.2e} '
                      f'{sec_per_step:.1f}s/step eta {(total_steps - step) * sec_per_step / 60:.0f}m '
                      f'mem {torch.cuda.max_memory_allocated() / 2**30:.2f}GB', flush=True)
                running = []
                if step % a.save_every == 0 and not a.probe:
                    save_ckpt(step)
                if step >= total_steps:
                    break
    if not a.probe:
        save_ckpt(step)
        tokenizer.save_pretrained(a.out)
        print('saved', a.out)


if __name__ == '__main__':
    main()
