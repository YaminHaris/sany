"""Interruption-Proof, Resumable 1,200-Case Bug-Bounty & Security Benchmark Runner.

Features:
  - Ignores SIGHUP so terminal disconnects cannot terminate the run.
  - Automatic retry with exponential backoff on transient Ollama/HTTP errors.
  - Per-case atomic append + fsync to JSONL (`results/<model>_1200_ckpt.jsonl`).
  - Live progress snapshot written to `results/bench_1200_status.json` after every case.
  - Can be paused/stopped and resumed at any point without losing a single completed case.
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import signal
import time

from eval_cases_1000 import CASES
from run_eval import run_case

# Ignore SIGHUP so closing a terminal never interrupts the runner
if hasattr(signal, 'SIGHUP'):
    signal.signal(signal.SIGHUP, signal.SIG_IGN)


def select_cases(all_cases: list[dict], stratified: int, category_filter: str | None) -> list[dict]:
    filtered = [
        c for c in all_cases
        if not category_filter or category_filter.lower() in c['category'].lower()
    ]
    if stratified <= 0:
        return filtered
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for c in filtered:
        if len(by_cat[c['category']]) < stratified:
            by_cat[c['category']].append(c)
    out = []
    for cat in sorted(by_cat.keys()):
        out.extend(by_cat[cat])
    return out


def load_checkpoint(ckpt_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not ckpt_path.exists():
        return done
    for line in ckpt_path.read_text(encoding='utf-8', errors='replace').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if 'id' in row:
                done[row['id']] = row
        except Exception:
            continue
    return done


def run_case_with_retry(model: str, case: dict, temperature: float, seed: int, max_retries: int = 5) -> dict:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return run_case(model, case, temperature, seed)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            last_err = e
            wait_s = min(30, 2 ** attempt)
            print(f"[1200-Bench] Transient error on {case['id']} (attempt {attempt}/{max_retries}): {e}. "
                  f"Retrying in {wait_s}s...", flush=True)
            time.sleep(wait_s)
    raise RuntimeError(f"Failed case {case['id']} after {max_retries} attempts: {last_err}")


def write_live_status(status_path: Path, model: str, target_cases: list[dict], done_map: dict[str, dict], state: str):
    completed = [done_map[c['id']] for c in target_cases if c['id'] in done_map]
    passed = sum(1 for r in completed if r['pass'])
    total_done = len(completed)
    total_target = len(target_cases)
    avg_secs = (sum(r.get('secs', 0.0) for r in completed) / total_done) if total_done else 15.0
    rem_cases = max(0, total_target - total_done)
    eta_mins = round((rem_cases * avg_secs) / 60.0, 1)

    by_cat: dict[str, dict] = {}
    for r in completed:
        cat = r.get('category', 'unknown')
        entry = by_cat.setdefault(cat, {'passed': 0, 'total': 0})
        entry['total'] += 1
        if r['pass']:
            entry['passed'] += 1

    payload = {
        'state': state,
        'pid': os.getpid(),
        'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': model,
        'completed': total_done,
        'target_total': total_target,
        'passed': passed,
        'failed': total_done - passed,
        'pass_rate_pct': round((100.0 * passed / total_done), 2) if total_done else 0.0,
        'avg_secs_per_case': round(avg_secs, 2),
        'eta_minutes': eta_mins,
        'categories': by_cat,
    }
    tmp_path = status_path.with_suffix('.tmp')
    tmp_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp_path.replace(status_path)


def print_category_report(results: list[dict]):
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[r.get('category', 'unknown')].append(r)
    print('\n' + '=' * 72)
    print(f"{'CATEGORY':<34} {'PASSED':>10} {'RATE':>10} {'AVG SECS':>12}")
    print('-' * 72)
    for cat in sorted(by_cat.keys()):
        rows = by_cat[cat]
        passed = sum(1 for r in rows if r['pass'])
        total = len(rows)
        rate = (100.0 * passed / total) if total else 0.0
        avg_s = sum(r['secs'] for r in rows) / total if total else 0.0
        print(f"{cat:<34} {f'{passed}/{total}':>10} {rate:>9.1f}% {avg_s:>11.1f}s")
    print('=' * 72)


def main():
    ap = argparse.ArgumentParser(description='Interruption-proof 1,200-case benchmark runner for SANY')
    ap.add_argument('model', nargs='?', default='qwen-pentest-v4', help='Ollama model name')
    ap.add_argument('--stratified', type=int, default=0,
                    help='Run N cases per category across all 24 categories (0 = all 1,200 cases)')
    ap.add_argument('--limit', type=int, default=0,
                    help='Max number of uncompleted cases to run in this session (0 = no limit)')
    ap.add_argument('--category', type=str, default=None,
                    help='Filter by category substring')
    ap.add_argument('--temperature', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--ckpt', type=str, default=None,
                    help='Custom checkpoint JSONL path')
    ap.add_argument('--reset', action='store_true',
                    help='Clear checkpoint file before starting')
    args = ap.parse_args()

    safe_model = args.model.replace(':', '_').replace('/', '_')
    suffix = f'_strat{args.stratified}' if args.stratified > 0 else '_1200'
    ckpt_path = Path(args.ckpt or f'results/{safe_model}{suffix}_ckpt.jsonl')
    out_path = Path(f'results/{safe_model}{suffix}_summary.json')
    status_path = Path('results/bench_1200_status.json')
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    if args.reset and ckpt_path.exists():
        ckpt_path.unlink()

    target_cases = select_cases(CASES, args.stratified, args.category)
    done_map = load_checkpoint(ckpt_path)

    print(f'[1200-Bench] Model: {args.model} | Target Cases: {len(target_cases)} '
          f'(Total Suite: {len(CASES)}) | Already Checkpointed: {len(done_map)}', flush=True)

    write_live_status(status_path, args.model, target_cases, done_map, 'running')
    ran_this_session = 0
    t_start = time.time()

    with ckpt_path.open('a', encoding='utf-8') as ckpt_f:
        for case in target_cases:
            cid = case['id']
            if cid in done_map:
                continue
            if args.limit > 0 and ran_this_session >= args.limit:
                print(f'[1200-Bench] Reached --limit {args.limit}. Stopping cleanly (resume anytime).')
                break

            res = run_case_with_retry(args.model, case, args.temperature, args.seed)
            res['category'] = case['category']
            done_map[cid] = res
            ran_this_session += 1

            ckpt_f.write(json.dumps(res) + '\n')
            ckpt_f.flush()
            os.fsync(ckpt_f.fileno())

            write_live_status(status_path, args.model, target_cases, done_map, 'running')

            completed_targets = [done_map[c['id']] for c in target_cases if c['id'] in done_map]
            passed_so_far = sum(1 for r in completed_targets if r['pass'])
            pct = 100.0 * passed_so_far / len(completed_targets)
            status = 'PASS' if res['pass'] else 'FAIL'
            print(
                f"[{len(completed_targets):>4}/{len(target_cases)}] {status}  {cid:<30} "
                f"steps={res['steps']} calls={res['tool_calls']} {res['secs']:>4.1f}s  "
                f"(Running: {passed_so_far}/{len(completed_targets)} = {pct:.1f}%)",
                flush=True,
            )

    final_results = [done_map[c['id']] for c in target_cases if c['id'] in done_map]
    is_finished = len(final_results) >= len(target_cases)
    write_live_status(status_path, args.model, target_cases, done_map, 'completed' if is_finished else 'paused')
    print_category_report(final_results)

    passed = sum(1 for r in final_results if r['pass'])
    total = len(final_results)
    summary = {
        'model': args.model,
        'completed': total,
        'target_total': len(target_cases),
        'suite_total': len(CASES),
        'passed': passed,
        'pass_rate_pct': round((100.0 * passed / total), 2) if total else 0.0,
        'tool_errors': sum(r['tool_errors'] for r in final_results),
        'described_instead_of_acting': sum(r['described_instead_of_acting'] for r in final_results),
        'session_secs': round(time.time() - t_start, 1),
        'checkpoint_file': str(ckpt_path),
    }
    print(json.dumps(summary, indent=2))
    out_path.write_text(json.dumps({'summary': summary, 'results': final_results}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
