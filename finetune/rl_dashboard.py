#!/usr/bin/env python3
"""SANY Sandbox RL (GRPO) Control Panel & Resumable Training Server.

Provides:
  - A live web UI at http://localhost:8765 with a one-click [Start / Resume] and [Pause & Save] button
  - Real-time GPU VRAM & utilization telemetry (via nvidia-smi)
  - Live Sandbox Exploration trajectory inspector (rewards, group advantages, shell commands)
  - Live GRPO QLoRA step progress bar, loss curve, ETA, and log stream
  - CLI subcommands: `ui`, `start`, `resume`, `pause`, `status`
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).parent
PY = str(ROOT / '.venv' / 'bin' / 'python')
DEFAULT_OUT = 'out/lora-pentest-v4'


def get_out_dir(out_rel=DEFAULT_OUT) -> Path:
    return ROOT / out_rel


def get_pid_file(out_dir: Path) -> Path:
    return out_dir / 'trainer.pid'


def get_log_file(out_dir: Path) -> Path:
    return out_dir / 'rl_pipeline.log'


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        # Check it's not a zombie
        stat_p = Path(f'/proc/{pid}/status')
        if stat_p.exists() and 'State:\tZ' in stat_p.read_text():
            return False
        return True
    except OSError:
        return False


def find_running_trainer(out_dir: Path):
    pid_file = get_pid_file(out_dir)
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if is_pid_running(pid):
                return pid
        except Exception:
            pass
    # Fallback pgrep check
    try:
        r = subprocess.run(['pgrep', '-f', 'rl_grpo.py'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            p = int(line.strip())
            if p != os.getpid() and is_pid_running(p):
                return p
    except Exception:
        pass
    return None


def start_or_resume_trainer(out_rel=DEFAULT_OUT, num_tasks=16, max_steps=35, backend='sim'):
    out_dir = get_out_dir(out_rel)
    out_dir.mkdir(parents=True, exist_ok=True)
    running_pid = find_running_trainer(out_dir)
    if running_pid:
        return {'ok': True, 'msg': f'Trainer already running (PID {running_pid})', 'pid': running_pid}

    # If state was already 'completed' and user clicks Resume/Run again, only run if not completed
    state_p = out_dir / 'rl_state.json'
    if state_p.exists():
        try:
            st = json.loads(state_p.read_text())
            if st.get('status') == 'paused':
                st['status'] = 'resuming'
                state_p.write_text(json.dumps(st, indent=2))
        except Exception:
            pass

    log_fh = open(get_log_file(out_dir), 'a')
    log_fh.write(f'\n=== [{time.strftime("%Y-%m-%d %H:%M:%S")}] Starting / Resuming RL-GRPO Trainer ===\n')
    log_fh.flush()

    cmd = [
        PY, '-u', str(ROOT / 'rl_grpo.py'),
        '--out', out_rel,
        '--num-tasks', str(num_tasks),
        '--max-steps', str(max_steps),
        '--backend', backend,
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    get_pid_file(out_dir).write_text(str(proc.pid))
    return {'ok': True, 'msg': f'Started / Resumed RL-GRPO Trainer (PID {proc.pid})', 'pid': proc.pid}


def pause_trainer(out_rel=DEFAULT_OUT):
    out_dir = get_out_dir(out_rel)
    pid = find_running_trainer(out_dir)
    if not pid:
        return {'ok': False, 'msg': 'No running trainer found.'}
    try:
        os.kill(pid, signal.SIGINT)
        return {'ok': True, 'msg': f'Sent graceful checkpoint-and-pause signal (SIGINT) to PID {pid}.'}
    except Exception as e:
        return {'ok': False, 'msg': str(e)}


def gpu_telemetry():
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        )
        if r.returncode == 0 and r.stdout.strip():
            parts = [x.strip() for x in r.stdout.strip().split(',')]
            return {
                'name': parts[0],
                'mem_used_mb': int(parts[1]),
                'mem_total_mb': int(parts[2]),
                'util_pct': int(parts[3]),
                'temp_c': int(parts[4]),
            }
    except Exception:
        pass
    return {'name': 'GPU', 'mem_used_mb': 0, 'mem_total_mb': 6141, 'util_pct': 0, 'temp_c': 0}


def get_full_status(out_rel=DEFAULT_OUT):
    out_dir = get_out_dir(out_rel)
    state_p = out_dir / 'rl_state.json'
    state = {}
    if state_p.exists():
        try:
            state = json.loads(state_p.read_text())
        except Exception:
            pass

    pid = find_running_trainer(out_dir)
    if pid:
        state['live_status'] = 'running'
        state['pid'] = pid
    else:
        if state.get('phase') == 'completed':
            state['live_status'] = 'completed'
        elif state.get('tasks_completed', 0) > 0 or state.get('train_step', 0) > 0:
            state['live_status'] = 'paused (resumable)'
        else:
            state['live_status'] = 'idle'
        state['pid'] = None

    log_p = get_log_file(out_dir)
    log_tail = ''
    if log_p.exists():
        try:
            lines = log_p.read_text(errors='replace').splitlines()[-45:]
            log_tail = '\n'.join(lines)
        except Exception:
            pass

    state['gpu'] = gpu_telemetry()
    state['log_tail'] = log_tail
    state['has_checkpoint'] = (out_dir / 'rollouts.jsonl').exists() or (out_dir / 'train_state.pt').exists()
    return state


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>SANY · Sandbox RL (GRPO) Trainer</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #131b2e;
    --border: #23304d;
    --text: #e2e8f0;
    --muted: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #ef4444;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 24px;
    background: var(--bg); color: var(--text);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, monospace;
  }
  .container { max-width: 1180px; margin: 0 auto; }
  .header {
    display: flex; justify-content: space-between; align-items: center;
    background: var(--panel); border: 1px solid var(--border);
    padding: 18px 24px; border-radius: 12px; margin-bottom: 20px;
  }
  .title h1 { margin: 0; font-size: 20px; letter-spacing: 0.5px; color: var(--accent); }
  .title p { margin: 4px 0 0; font-size: 13px; color: var(--muted); }
  .controls { display: flex; gap: 12px; align-items: center; }
  button {
    border: none; border-radius: 8px; padding: 11px 20px;
    font-size: 14px; font-weight: 600; cursor: pointer;
    transition: all 0.15s ease;
  }
  .btn-resume { background: var(--green); color: #052e16; }
  .btn-resume:hover { filter: brightness(1.1); }
  .btn-pause { background: var(--amber); color: #451a03; }
  .btn-pause:hover { filter: brightness(1.1); }
  button:disabled { opacity: 0.45; cursor: not-allowed; }
  .badge {
    padding: 6px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;
    text-transform: uppercase; border: 1px solid var(--border);
  }
  .badge.running { background: rgba(34,197,94,0.15); color: var(--green); border-color: var(--green); }
  .badge.paused { background: rgba(245,158,11,0.15); color: var(--amber); border-color: var(--amber); }
  .badge.completed { background: rgba(56,189,248,0.15); color: var(--accent); border-color: var(--accent); }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    padding: 16px; border-radius: 10px;
  }
  .card .label { font-size: 12px; color: var(--muted); text-transform: uppercase; }
  .card .val { font-size: 24px; font-weight: 700; margin-top: 6px; }
  .card .sub { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .bar-wrap {
    width: 100%; height: 10px; background: #090d14;
    border-radius: 999px; overflow: hidden; margin-top: 8px;
  }
  .bar-fill { height: 100%; background: linear-gradient(90deg, var(--accent), var(--green)); width: 0%; transition: width 0.3s; }
  .two-col { display: grid; grid-template-columns: 1.15fr 1fr; gap: 18px; }
  .panel-box {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }
  .panel-box h3 { margin: 0 0 12px; font-size: 15px; color: var(--accent); }
  .rollout-list { max-height: 390px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .rollout-item {
    background: #0d1322; border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px; font-size: 13px;
  }
  .rollout-top { display: flex; justify-content: space-between; font-weight: 600; margin-bottom: 4px; }
  .cmd-pill {
    display: inline-block; background: #1e293b; color: #cbd5e1;
    padding: 2px 7px; border-radius: 4px; font-family: monospace; font-size: 11px; margin: 2px 4px 2px 0;
  }
  pre.logs {
    background: #070a10; border: 1px solid var(--border);
    border-radius: 8px; padding: 12px; height: 390px; overflow-y: auto;
    font-size: 12px; line-height: 1.45; color: #cbd5e1; margin: 0;
    white-space: pre-wrap;
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div class="title">
      <h1>SANY · Sandbox Exploration RL (GRPO) Control Center</h1>
      <p id="subtitle">Model: qwen-pentest-v3 ➔ qwen-pentest-v4 · Checkpoints saved automatically after every task & step</p>
    </div>
    <div class="controls">
      <span id="statusBadge" class="badge">IDLE</span>
      <button id="btnResume" class="btn-resume" onclick="triggerAction('/api/resume')">▶ Start / Resume</button>
      <button id="btnPause" class="btn-pause" onclick="triggerAction('/api/pause')">⏸ Pause & Save</button>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="label">1. Sandbox Exploration</div>
      <div class="val" id="exploreVal">0 / 16</div>
      <div class="bar-wrap"><div class="bar-fill" id="exploreBar"></div></div>
      <div class="sub" id="exploreSub">Avg Reward: 0.00 · Pass: 0%</div>
    </div>
    <div class="card">
      <div class="label">2. GRPO QLoRA Steps</div>
      <div class="val" id="trainVal">0 / 35</div>
      <div class="bar-wrap"><div class="bar-fill" id="trainBar"></div></div>
      <div class="sub" id="trainSub">Loss: — · ETA: —</div>
    </div>
    <div class="card">
      <div class="label">RTX 4050 GPU Telemetry</div>
      <div class="val" id="gpuVal">0 / 6141 MB</div>
      <div class="bar-wrap"><div class="bar-fill" id="gpuBar"></div></div>
      <div class="sub" id="gpuSub">Util: 0% · Temp: 0°C</div>
    </div>
    <div class="card">
      <div class="label">Held-Out Benchmarks</div>
      <div class="val" id="evalVal">Pending</div>
      <div class="sub" id="evalSub">Explore · Recon · Exploit · General</div>
    </div>
  </div>

  <div class="two-col">
    <div class="panel-box">
      <h3>Live Sandbox Exploration Rollouts & GRPO Advantages</h3>
      <div class="rollout-list" id="rolloutList">
        <div style="color:var(--muted);font-size:13px;">Waiting for rollouts...</div>
      </div>
    </div>
    <div class="panel-box">
      <h3>Live Pipeline & Checkpoint Log</h3>
      <pre class="logs" id="logBox">Loading logs...</pre>
    </div>
  </div>
</div>

<script>
async function triggerAction(url) {
  const res = await fetch(url, {method: 'POST'});
  const data = await res.json();
  await refresh();
}

async function refresh() {
  try {
    const res = await fetch('/api/status');
    const st = await res.json();
    const badge = document.getElementById('statusBadge');
    badge.textContent = (st.live_status || 'idle').toUpperCase() + (st.phase ? ` (${st.phase})` : '');
    badge.className = 'badge ' + (st.live_status === 'running' ? 'running' : (st.live_status === 'completed' ? 'completed' : 'paused'));

    const isRun = st.live_status === 'running';
    const btnResume = document.getElementById('btnResume');
    btnResume.disabled = isRun;
    btnResume.textContent = st.has_checkpoint ? '▶ Resume from Checkpoint' : '▶ Start RL Training';
    document.getElementById('btnPause').disabled = !isRun;

    const tc = st.tasks_completed || 0, tt = st.tasks_total || 16;
    document.getElementById('exploreVal').textContent = `${tc} / ${tt} missions`;
    document.getElementById('exploreBar').style.width = `${Math.min(100, (tc / Math.max(1, tt)) * 100)}%`;
    document.getElementById('exploreSub').textContent = `Policy Avg Reward: ${st.avg_reward || 0} · Pass Rate: ${st.pass_rate || 0}%`;

    const sc = st.train_step || 0, stot = st.train_total_steps || 35;
    document.getElementById('trainVal').textContent = `${sc} / ${stot} steps`;
    document.getElementById('trainBar').style.width = `${Math.min(100, (sc / Math.max(1, stot)) * 100)}%`;
    const eta = st.train_eta_sec != null ? `${Math.floor(st.train_eta_sec/60)}m ${st.train_eta_sec%60}s` : '—';
    document.getElementById('trainSub').textContent = `GRPO Loss: ${st.train_loss ?? '—'} · ETA: ${eta}`;

    if (st.gpu) {
      const u = st.gpu.mem_used_mb, tot = st.gpu.mem_total_mb || 6141;
      document.getElementById('gpuVal').textContent = `${u} / ${tot} MB`;
      document.getElementById('gpuBar').style.width = `${Math.min(100, (u / tot) * 100)}%`;
      document.getElementById('gpuSub').textContent = `Util: ${st.gpu.util_pct}% · Temp: ${st.gpu.temp_c}°C`;
    }

    const ev = st.eval_results || {};
    const keys = Object.keys(ev);
    if (keys.length > 0) {
      const parts = keys.map(k => `${k}: ${ev[k].passed}/${ev[k].total}`);
      document.getElementById('evalVal').textContent = `${keys.length}/4 Done`;
      document.getElementById('evalSub').textContent = parts.join(' · ');
    }

    const rList = document.getElementById('rolloutList');
    const items = (st.recent_rollouts || []).slice().reverse();
    if (items.length > 0) {
      rList.innerHTML = items.map(it => {
        const rw = (it.policy_rewards || []).map(r => (r >= 0 ? '+' + r : r)).join(', ');
        const cmds = (it.sample_cmds || []).map(c => `<span class="cmd-pill">${c.replace(/</g,'&lt;')}</span>`).join('');
        return `<div class="rollout-item">
          <div class="rollout-top">
            <span>#${it.task_idx + 1} · ${it.mission_id}</span>
            <span style="color:var(--accent)">Rewards: [${rw}]</span>
          </div>
          <div style="color:var(--muted);margin-bottom:4px;">${it.prompt}</div>
          <div>${cmds || '<span class="cmd-pill">(direct reply)</span>'}</div>
        </div>`;
      }).join('');
    }

    const logBox = document.getElementById('logBox');
    const atBottom = logBox.scrollHeight - logBox.clientHeight <= logBox.scrollTop + 40;
    logBox.textContent = st.log_tail || 'No logs yet.';
    if (atBottom) logBox.scrollTop = logBox.scrollHeight;
  } catch (e) {}
}
setInterval(refresh, 2000);
refresh();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    out_rel = DEFAULT_OUT

    def log_message(self, format, *args):
        return  # Keep console clean

    def _send_json(self, obj, code=200):
        raw = json.dumps(obj).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/status':
            return self._send_json(get_full_status(self.out_rel))
        raw = HTML_PAGE.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/resume':
            return self._send_json(start_or_resume_trainer(self.out_rel))
        if parsed.path == '/api/pause':
            return self._send_json(pause_trainer(self.out_rel))
        return self._send_json({'ok': False, 'msg': 'Unknown endpoint'}, 404)


def main():
    ap = argparse.ArgumentParser(description='SANY Sandbox RL Control Center & Resumable Runner')
    ap.add_argument('action', nargs='?', default='status', choices=['ui', 'start', 'resume', 'pause', 'status'])
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--num-tasks', type=int, default=16)
    ap.add_argument('--max-steps', type=int, default=35)
    ap.add_argument('--backend', choices=['sim', 'docker'], default='sim')
    a = ap.parse_args()

    DashboardHandler.out_rel = a.out

    if a.action in ('start', 'resume'):
        res = start_or_resume_trainer(a.out, num_tasks=a.num_tasks, max_steps=a.max_steps, backend=a.backend)
        print(res['msg'])
    elif a.action == 'pause':
        res = pause_trainer(a.out)
        print(res['msg'])
    elif a.action == 'status':
        st = get_full_status(a.out)
        print(f"Status:       {st.get('live_status')} (Phase: {st.get('phase', 'init')}, PID: {st.get('pid')})")
        print(f"Exploration:  {st.get('tasks_completed', 0)}/{st.get('tasks_total', a.num_tasks)} missions | Avg Reward: {st.get('avg_reward', 0.0)} | Pass Rate: {st.get('pass_rate', 0.0)}%")
        print(f"GRPO Train:   {st.get('train_step', 0)}/{st.get('train_total_steps', a.max_steps)} steps | Loss: {st.get('train_loss')} | ETA: {st.get('train_eta_sec')}s")
        gpu = st.get('gpu', {})
        print(f"GPU:          {gpu.get('name')} | {gpu.get('mem_used_mb')}/{gpu.get('mem_total_mb')} MB | Util: {gpu.get('util_pct')}% | Temp: {gpu.get('temp_c')}°C")
    elif a.action == 'ui':
        srv = HTTPServer(('127.0.0.1', a.port), DashboardHandler)
        print(f'SANY RL Dashboard listening at http://localhost:{a.port}', flush=True)
        srv.serve_forever()


if __name__ == '__main__':
    main()
