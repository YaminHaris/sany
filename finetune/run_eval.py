"""Runs the held-out tasks against an Ollama model in the simulated environment.

Usage: python run_eval.py <ollama-model> [--only id1,id2] [--out results/x.json]
"""

import argparse
import json
import time
from pathlib import Path

import os

import requests

from common import TOOLS, start_messages, tool_result_content
from eval_cases import BASE_FILES, CASES, CWD
from sany_memory import augment_messages_with_memory, strip_thought_tags
from sim import Sim

OLLAMA = 'http://localhost:11434/api/chat'
MAX_STEPS = 10


def chat(model, messages, temperature, seed):
    aug_msgs = augment_messages_with_memory(messages) if os.environ.get('SANY_MEMORY', '1') != '0' else messages
    body = {
        'model': model, 'stream': False, 'messages': aug_msgs, 'tools': TOOLS,
        'options': {'num_ctx': 16384, 'temperature': temperature, 'seed': seed, 'num_predict': 1024},
        **({'think': os.environ['EVAL_THINK'] == '1'} if os.environ.get('EVAL_THINK') else {}),
        'keep_alive': '30m',
    }
    r = requests.post(OLLAMA, json=body, timeout=600)
    r.raise_for_status()
    return r.json()['message']


def run_case(model, case, temperature, seed):
    cwd, files = case.get('cwd', CWD), case.get('files', BASE_FILES)
    sim = Sim(cwd, files, web=case.get('web'), search=case.get('search'),
              downloads=case.get('downloads'), programs=case.get('programs'),
              extra_shell=case.get('extra_shell'))
    messages = start_messages(cwd, list(files), case['prompt'])
    final, errors, steps, transcript = '', 0, 0, []
    t0 = time.time()
    for steps in range(1, MAX_STEPS + 1):
        msg = chat(model, messages, temperature, seed)
        calls = msg.get('tool_calls') or []
        cleaned_content = strip_thought_tags(msg.get('content', ''))
        # Mirror the CLI adapter: tool-call turns are sent back without text.
        messages.append({'role': 'assistant', 'content': '', 'tool_calls': calls} if calls
                        else {'role': 'assistant', 'content': cleaned_content})
        transcript.append({'assistant': cleaned_content, 'calls': [c['function'] for c in calls]})
        if not calls:
            final = cleaned_content
            break
        for c in calls:
            fn = c['function']
            args = fn.get('arguments') or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            res = sim.call(fn['name'], args)
            if 'error' in res:
                errors += 1
            payload = res.get('output', res.get('error'))
            messages.append({'role': 'tool', 'tool_name': fn['name'], 'content': tool_result_content(payload) if 'output' in res else json.dumps({'error': payload})})
            transcript[-1].setdefault('results', []).append(payload[:400])
    try:
        ok = bool(case['check'](sim, final))
    except Exception:
        ok = False
    tool_calls = len(sim.log)
    if case.get('no_tools') and tool_calls:
        ok = False
    described = (not case.get('no_tools')) and tool_calls == 0
    return {
        'id': case['id'], 'pass': ok, 'steps': steps, 'tool_calls': tool_calls, 'tool_errors': errors,
        'described_instead_of_acting': described, 'secs': round(time.time() - t0, 1),
        'final': final[:600], 'transcript': transcript,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('model')
    ap.add_argument('--only')
    ap.add_argument('--out')
    ap.add_argument('--temperature', type=float, default=0.3)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--cases-module', help='import CASES/BASE_FILES/CWD from this module instead of eval_cases')
    a = ap.parse_args()
    all_cases = CASES
    if a.cases_module:
        import importlib
        global BASE_FILES, CWD
        m = importlib.import_module(a.cases_module)
        all_cases, BASE_FILES, CWD = m.CASES, m.BASE_FILES, m.CWD
    cases = [c for c in all_cases if not a.only or c['id'] in a.only.split(',')]
    results = []
    for c in cases:
        r = run_case(a.model, c, a.temperature, a.seed)
        results.append(r)
        print(f"{'PASS' if r['pass'] else 'FAIL'}  {r['id']:<18} steps={r['steps']} calls={r['tool_calls']} errs={r['tool_errors']} {r['secs']}s"
              + ('  [described instead of acting]' if r['described_instead_of_acting'] else ''), flush=True)
    n = sum(r['pass'] for r in results)
    summary = {
        'model': a.model, 'passed': n, 'total': len(results),
        'tool_errors': sum(r['tool_errors'] for r in results),
        'described_instead_of_acting': sum(r['described_instead_of_acting'] for r in results),
        'secs': round(sum(r['secs'] for r in results), 1),
    }
    print(json.dumps(summary))
    out = Path(a.out or f"results/{a.model.replace(':', '_').replace('/', '_')}.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({'summary': summary, 'results': results}, indent=1))


if __name__ == '__main__':
    main()
