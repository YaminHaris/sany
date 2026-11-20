"""Generates tool-use trajectories for fine-tuning.

Each example is played out against sim.Sim so tool results and final answers are consistent.
Wording and names are deliberately disjoint from eval_cases.py.
"""

import json
import random
import re
from pathlib import Path

from common import start_messages, tool_result_content
from sim import Sim

R = random.Random(1234)

USERS_DIRS = ['/home/yamin/work/site', '/home/yamin/Documents/uni', '/home/yamin/code/bot', '/home/yamin/Development/tracker',
              '/home/yamin/scratch', '/home/yamin/projects/shop', '/home/yamin/Downloads/batch', '/home/yamin/code/dashboard']
WORDS = ['orbit', 'maple', 'falcon', 'pixel', 'harbor', 'quartz', 'lumen', 'cedar', 'nimbus', 'echo', 'delta', 'ember', 'tundra', 'sable', 'zephyr']
ITEMS = ['apples', 'bananas', 'rice', 'eggs', 'bread', 'coffee', 'detergent', 'yogurt', 'onions', 'tea', 'pasta', 'cheese', 'lentils']
TASKS = ['water the plants', 'email the landlord', 'submit the assignment', 'renew gym membership', 'back up the laptop',
         'fix the bike tyre', 'pay electricity bill', 'call grandma', 'return library books', 'book train tickets', 'clean the desk']
FACTS = ['User prefers concise answers.', 'User\'s name is Yamin.', 'User works mostly in Python and ROS2.', 'User is based in India (IST).']


def pick(xs, k=None):
    return R.choice(xs) if k is None else R.sample(xs, k)


def word():
    return pick(WORDS)


def base_files():
    w = word()
    files = {
        'README.md': f'# {w.title()}\n\nSmall project.\n',
        f'{w}.py': 'def run():\n    return 1\n',
        'data/input.txt': '\n'.join(pick(ITEMS, 4)) + '\n',
    }
    for _ in range(R.randint(0, 3)):
        files[f'misc/{word()}_{R.randint(1, 99)}.txt'] = 'x\n'
    return files


class Traj:
    def __init__(self, prompt, files, cwd=None, **simkw):
        self.cwd = cwd or pick(USERS_DIRS)
        self.sim = Sim(self.cwd, files, **simkw)
        facts = pick(FACTS, R.randint(1, 2)) if R.random() < 0.25 else ()
        self.messages = start_messages(self.cwd, list(files), prompt, facts)
        self.n_calls = 0

    def call(self, name, **args):
        res = self.sim.call(name, args)
        self.messages.append({'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': name, 'arguments': args}}]})
        payload = res['output'] if 'output' in res else None
        content = tool_result_content(payload) if payload is not None else json.dumps({'error': res['error']})
        self.messages.append({'role': 'tool', 'tool_name': name, 'content': content})
        self.n_calls += 1
        return res

    def sh(self, command, description):
        return self.call('run_shell_command', command=command, description=description)

    def answer(self, text):
        self.messages.append({'role': 'assistant', 'content': text})
        return self.messages


# ---------------------------------------------------------------- families
def fam_count_ext():
    ext = pick(['md', 'log', 'json', 'csv', 'jpg', 'py'])
    n = R.randint(0, 6)
    files = base_files()
    for i in range(n):
        files[f'{word()}{i}.{ext}'] = 'z\n'
    prompt = pick([f'count the .{ext} files in this directory', f'How many {ext} files do I have here?',
                   f'number of *.{ext} files in the current folder?', f'tell me how many .{ext} files are in here (top level only)'])
    t = Traj(prompt, files)
    real = len([f for f in t.sim.files if '/' not in f and f.endswith('.' + ext)])
    t.sh(f'ls -1 *.{ext} 2>/dev/null | wc -l' if R.random() < 0.5 else f'find . -maxdepth 1 -name "*.{ext}" | wc -l', f'Count .{ext} files in the current directory.')
    return t.answer(pick([f'There are {real} .{ext} files in this directory.', f'{real}.', f'You have {real} `.{ext}` files here.']))


def fam_count_lines():
    name = pick(['notes.txt', 'server.log', 'list.csv', 'words.txt', 'data/input.txt'])
    n = R.randint(3, 40)
    files = base_files()
    files[name] = '\n'.join(f'line {i}' for i in range(n)) + '\n'
    t = Traj(pick([f'how long is {name} in lines?', f'Count the lines in {name}.', f'{name} - how many lines?']), files)
    t.sh(f'wc -l {name}', f'Count lines in {name}.')
    return t.answer(f'{name} has {n} lines.')


def fam_log_grep():
    level = pick(['WARN', 'ERROR', 'CRITICAL'])
    msgs = pick(['connection reset', 'out of memory', 'permission denied', 'cache miss storm', 'disk quota exceeded', 'bad gateway'], R.randint(1, 3))
    lines = []
    for m in msgs:
        lines += [f'INFO tick {R.randint(1, 99)}', f'{level} {m}']
    lines.append('INFO shutdown')
    name = pick(['service.log', 'logs/worker.log', 'var/api.log'])
    files = base_files()
    files[name] = '\n'.join(lines) + '\n'
    t = Traj(pick([f'Are there any {level} entries in {name}? What are they?', f'show me the {level.lower()} lines from {name}',
                   f'Summarize the {level} messages in {name}.']), files)
    if R.random() < 0.5:
        t.sh(f'grep -n "{level}" {name}', f'Show {level} lines in {name}.')
    else:
        t.call('grep_search', pattern=level, include_pattern=name.split('/')[-1])
    listing = '\n'.join(f'- {m}' for m in msgs)
    return t.answer(f'{len(msgs)} {level} line{"s" if len(msgs) > 1 else ""} in {name}:\n{listing}')


SYS_INFO = [
    ('node version', r'node (-v|--version)', 'node --version', 'v22.11.0', 'Node.js v22.11.0 is installed.'),
    ('which kernel am I running', r'uname', 'uname -r', '6.14.0-37-generic', 'You are running kernel 6.14.0-37-generic.'),
    ('how much RAM is free', r'\bfree\b', 'free -h', '               total        used        free      shared  buff/cache   available\nMem:            14Gi       8.1Gi       0.9Gi       0.3Gi       5.6Gi       5.9Gi',
     'You have 14 GiB total; about 5.9 GiB is available (0.9 GiB completely free, the rest is reclaimable cache).'),
    ('what is my username', r'whoami', 'whoami', 'yamin', 'Your username is `yamin`.'),
    ('git status of this repo', r'git', 'git --no-pager status --short', ' M README.md\n?? notes/', 'One modified file (README.md) and one untracked folder (notes/).'),
    ('last 3 commits', r'git', 'git --no-pager log --oneline -3', 'a1b2c3d fix login bug\n9f8e7d6 add tests\n1234abc initial commit',
     'Last 3 commits:\n- a1b2c3d fix login bug\n- 9f8e7d6 add tests\n- 1234abc initial commit'),
    ('is docker running', r'docker|systemctl', 'systemctl is-active docker', 'active', 'Yes, the docker service is active.'),
    ('what is using port 8000', r'\bss\b|lsof', 'ss -ltnp | grep :8000', 'LISTEN 0 4096 127.0.0.1:8000 0.0.0.0:* users:(("python3",pid=4412,fd=3))',
     'Port 8000 is used by a python3 process (PID 4412), listening on 127.0.0.1.'),
    ('how big is the data folder', r'\bdu\b', 'du -sh data', '2.3M\tdata', 'The data folder is 2.3 MB.'),
    ('what time is it', r'\bdate\b', 'date', 'Sat Oct 31 01:12:45 IST 2026', "It's 01:12 IST on Saturday, 31 October 2026."),
    ('which pip packages are outdated', r'pip', 'pip list --outdated --format=columns', 'Package Version Latest Type\nrequests 2.31.0 2.32.5 wheel\nnumpy 1.26.4 2.3.4 wheel',
     'Two packages are outdated: requests (2.31.0 -> 2.32.5) and numpy (1.26.4 -> 2.3.4).'),
    ('what GPU do I have', r'nvidia-smi|lspci', 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader', 'NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB',
     'You have an NVIDIA GeForce RTX 4050 Laptop GPU with 6 GB of memory.'),
]


def fam_sysinfo():
    q, rx, cmd, out, ans = pick(SYS_INFO)
    prompt = pick([q + '?', q.capitalize() + '?', 'can you check ' + q, 'quick check: ' + q])
    t = Traj(prompt, base_files(), extra_shell=[(rx, out)])
    t.sh(cmd, 'Check ' + q + '.')
    return t.answer(ans)


def fam_create_list():
    kind = R.random()
    items = pick(ITEMS if kind < 0.5 else TASKS, R.randint(2, 5))
    path = pick(['shopping.md', 'lists/groceries.md', 'todo.md', 'plans/weekend.md', 'docs/checklist.md'])
    style = pick(['checklist', 'bullet list', 'numbered list'])
    prompt = pick([f'create {path} with a {style} of: {", ".join(items)}', f'Write a {style} to {path}: {"; ".join(items)}',
                   f'put these in {path} as a {style} -> {", ".join(items)}'])
    t = Traj(prompt, base_files())
    if style == 'checklist':
        body = '\n'.join(f'- [ ] {i}' for i in items)
    elif style == 'bullet list':
        body = '\n'.join(f'- {i}' for i in items)
    else:
        body = '\n'.join(f'{k}. {i}' for k, i in enumerate(items, 1))
    title = pick(['', '# Shopping\n\n' if kind < 0.5 else '# To do\n\n'])
    t.call('write_file', file_path=path, content=title + body + '\n')
    return t.answer(pick([f'Created {path} with {len(items)} items.', f'Done — {path} now has your {style}.']))


def fam_json_edit():
    key, old, new = pick([('port', 3000, 5050), ('timeout', 30, 60), ('retries', 3, 5), ('workers', 2, 8), ('verbose', False, True), ('theme', '"light"', '"dark"')])
    name = pick(['settings.json', 'app.config.json', 'config/server.json', '.prettierrc.json'])
    other = word()
    fmt = lambda v: json.dumps(v) if not isinstance(v, str) else v
    content = '{\n  "app": "' + other + '",\n  "' + key + '": ' + fmt(old) + ',\n  "region": "ap-south-1"\n}\n'
    files = base_files()
    files[name] = content
    new_disp = new if not isinstance(new, str) else new.strip('"')
    t = Traj(pick([f'set {key} to {new_disp} in {name}', f'Change {key} in {name} to {new_disp}, nothing else.', f'update {name}: {key} = {new_disp}']), files)
    t.call('read_file', file_path=name)
    t.call('replace', file_path=name, instruction=f'Change {key} from {fmt(old)} to {fmt(new)}.',
           old_string=f'"{key}": {fmt(old)}', new_string=f'"{key}": {fmt(new)}')
    return t.answer(f'Updated {name}: `{key}` is now {fmt(new)}.')


def fam_env_edit():
    key = pick(['DEBUG', 'LOG_LEVEL', 'MAX_UPLOAD_MB', 'API_URL'])
    vals = {'DEBUG': ('false', 'true'), 'LOG_LEVEL': ('info', 'debug'), 'MAX_UPLOAD_MB': ('10', '50'), 'API_URL': ('http://localhost:8000', 'https://api.example.net')}
    old, new = vals[key]
    files = base_files()
    files['.env'] = f'APP_NAME={word()}\n{key}={old}\nTZ=Asia/Kolkata\n'
    t = Traj(pick([f'in .env set {key} to {new}', f'Change {key} in the .env file to {new}.']), files)
    t.call('read_file', file_path='.env')
    t.call('replace', file_path='.env', instruction=f'Set {key} to {new}.', old_string=f'{key}={old}', new_string=f'{key}={new}')
    return t.answer(f'Set {key}={new} in .env.')


def fam_append():
    name = pick(['journal.md', 'ideas.md', 'reading-list.md'])
    existing = [pick(TASKS) for _ in range(R.randint(1, 3))]
    files = base_files()
    files[name] = f'# {name[:-3].replace("-", " ").title()}\n\n' + '\n'.join(f'- {e}' for e in existing) + '\n'
    new = pick(TASKS)
    t = Traj(pick([f'append "{new}" to {name}', f'Add another bullet to {name}: {new}', f'add {new} to the end of {name}']), files)
    t.call('read_file', file_path=name)
    last = f'- {existing[-1]}\n'
    t.call('replace', file_path=name, instruction=f'Append a bullet for "{new}".', old_string=last if files[name].count(last) == 1 else files[name],
           new_string=(last if files[name].count(last) == 1 else files[name]) + f'- {new}\n')
    return t.answer(f'Added "- {new}" to the end of {name}.')


def fam_move():
    src = pick(['draft.txt', 'photo.jpg', 'old_notes.md', 'report_v1.pdf'])
    files = base_files()
    files[src] = 'content\n'
    if R.random() < 0.5:
        dst = pick(['final.txt', 'cover.jpg', 'notes_2025.md', 'report.pdf'])
        prompt = pick([f'rename {src} to {dst}', f'Can you rename {src} -> {dst}?'])
        cmd = f'mv {src} {dst}'
    else:
        d = pick(['archive', 'backup', 'done'])
        dst = f'{d}/{src}'
        prompt = pick([f'move {src} into a folder called {d}', f'Put {src} in {d}/ (create it if needed).'])
        cmd = f'mkdir -p {d} && mv {src} {d}/'
    t = Traj(prompt, files)
    t.sh(cmd, f'Move {src} to {dst}.')
    return t.answer(f'Moved {src} to {dst}.')


def fam_script():
    lang, path, body, out = pick([
        ('python', 'count_words.py', 'import sys\n\ntext = open(sys.argv[1]).read()\nprint(len(text.split()))\n', None),
        ('python', 'hello.py', 'print("hi there")\n', 'hi there'),
        ('bash', 'backup.sh', '#!/usr/bin/env bash\nset -euo pipefail\ntar -czf "backup-$(date +%F).tar.gz" data/\necho "backup done"\n', None),
        ('python', 'fib.py', 'a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n', '0\n1\n1\n2\n3\n5\n8\n13\n21\n34'),
    ])
    run = out is not None and R.random() < 0.6
    prompt = {
        'count_words.py': 'write a python script count_words.py that prints the number of words in a file passed as an argument',
        'hello.py': 'make hello.py that prints "hi there"' + (' and run it' if run else ''),
        'backup.sh': 'create a bash script backup.sh that tars the data folder into backup-<date>.tar.gz',
        'fib.py': 'write fib.py printing the first 10 fibonacci numbers' + (', then run it' if run else ''),
    }[path]
    t = Traj(prompt, base_files(), programs={f'python3 {path}': out} if out else None)
    t.call('write_file', file_path=path, content=body)
    if path.endswith('.sh'):
        t.sh(f'chmod +x {path}', f'Make {path} executable.')
    if run:
        t.sh(f'python3 {path}', f'Run {path}.')
        return t.answer(f'Created {path} and ran it. Output:\n```\n{out}\n```')
    return t.answer(f'Created {path}.' + (' Run it with `python3 count_words.py <file>`.' if path == 'count_words.py' else ''))


def fam_find_text():
    marker = pick(['FIXME', 'TODO', 'HACK', 'XXX'])
    files = base_files()
    notes = pick(['retry on timeout', 'validate input', 'remove debug print', 'cache results', 'use env var for key'], R.randint(1, 3))
    paths = []
    for n in notes:
        p = f'{pick(["lib", "app", "core"])}/{word()}_{len(paths)}.py'
        files[p] = f'import json\n\n# {marker}: {n}\ndef f():\n    pass\n'
        paths.append((p, n))
    t = Traj(pick([f'find all {marker} comments in the code', f'Where are the {marker}s? list them', f'list every {marker} note with its file']), files)
    t.call('grep_search', pattern=marker)
    lst = '\n'.join(f'- {p}:3 — {n}' for p, n in paths)
    return t.answer(f'Found {len(paths)} {marker} comment{"s" if len(paths) > 1 else ""}:\n{lst}')


def fam_glob():
    ext = pick(['js', 'ts', 'yaml', 'md', 'sh'])
    files = base_files()
    made = []
    for _ in range(R.randint(1, 5)):
        p = f'{pick(["src", "scripts", "conf", "docs"])}/{word()}.{ext}'
        files[p] = '\n'
        made.append(p)
    t = Traj(pick([f'show me every .{ext} file in this project', f'list the {ext} files', f'which {ext} files exist here?']), files)
    t.call('glob', pattern=f'**/*.{ext}')
    real = sorted(f for f in t.sim.files if f.endswith('.' + ext))
    return t.answer(f'{len(real)} .{ext} file{"s" if len(real) != 1 else ""}:\n' + '\n'.join(f'- {p}' for p in real))


def fam_define():
    fn = pick(['parse_date', 'load_config', 'send_email', 'resize_image', 'hash_password'])
    files = base_files()
    target = f'{pick(["utils", "helpers", "services"])}/{word()}.py'
    files[target] = f'import os\n\n\ndef {fn}(x):\n    return x\n'
    files[f'main_{word()}.py'] = f'from x import {fn}\n\n{fn}(1)\n'
    t = Traj(pick([f'where is {fn} defined?', f'Which file defines the function {fn}?', f'find the definition of {fn}']), files)
    t.call('grep_search', pattern=f'def {fn}\\(')
    return t.answer(f'`{fn}` is defined in {target} (line 4).')


def fam_download():
    host = pick(['https://raw.githubusercontent.com/torvalds/linux/master', 'https://example.net/files', 'https://data.example.org/pub'])
    fname = pick(['COPYING', 'sample.csv', 'words.txt', 'config.yaml', 'LICENSE'])
    url = f'{host}/{fname}'
    n = R.randint(5, 60)
    content = '\n'.join(f'row {i}' for i in range(n)) + '\n'
    dest_dir = pick(['downloads', 'vendor', 'data/raw', ''])
    dest = f'{dest_dir}/{fname}' if dest_dir else fname
    want_lines = R.random() < 0.5
    prompt = pick([f'download {url} into {dest_dir or "this folder"}', f'Fetch the file at {url} and save it as {dest}', f'get {url} -> {dest}'])
    if want_lines:
        prompt += pick([', then tell me how many lines it has', ' and count its lines'])
    t = Traj(prompt, base_files(), downloads={url: content})
    mk = f'mkdir -p {dest_dir} && ' if dest_dir else ''
    if R.random() < 0.15 and dest_dir:
        res = t.sh(f'curl -fsSL -o {dest} {url}', f'Download {fname}.')
        if 'Failure writing' in res.get('output', ''):
            t.sh(f'mkdir -p {dest_dir} && curl -fsSL -o {dest} {url}', f'Create {dest_dir} and download {fname}.')
    else:
        t.sh(f'{mk}curl -fsSL -o {dest} {url}', f'Download {fname} to {dest}.')
    if want_lines:
        t.sh(f'wc -l {dest}', f'Count lines in {dest}.')
        return t.answer(f'Downloaded to {dest}. It has {n} lines.')
    return t.answer(f'Downloaded {fname} to {dest}.')


PAGES = [
    ('https://www.python.org', 'Welcome to Python.org\n\nPython is a programming language that lets you work quickly and integrate systems more effectively.\n\nLatest: Python 3.14.0',
     'What does the python.org homepage say is the latest release?', 'python.org lists **Python 3.14.0** as the latest release.'),
    ('https://httpbin.org/html', 'Herman Melville - Moby-Dick\n\nAvailing himself of the mild, summer-cool weather that now reigned in these latitudes...',
     'what is the title on https://httpbin.org/html', 'The page shows "Herman Melville - Moby-Dick".'),
    ('https://www.kernel.org', 'The Linux Kernel Archives\n\nmainline: 6.18-rc3  2026-10-26\nstable: 6.17.5  2026-10-23\nlongterm: 6.12.55',
     'what is the current stable kernel on kernel.org?', 'kernel.org lists **6.17.5** as the current stable kernel (mainline is 6.18-rc3).'),
    ('https://blog.example.dev/posts/rust-vs-go', 'Rust vs Go in 2026\n\nTL;DR: pick Go for fast iteration on network services; pick Rust when you need predictable latency and memory safety without GC.',
     'summarize https://blog.example.dev/posts/rust-vs-go in one line', 'Go for fast iteration on network services; Rust when you need predictable latency and memory safety without a GC.'),
    ('https://status.example.com', 'Example Status\n\nAPI: Operational\nDashboard: Degraded performance\nWebhooks: Operational',
     'check https://status.example.com - is anything down?', 'Nothing is fully down, but the Dashboard shows **degraded performance**. API and Webhooks are operational.'),
]


def fam_fetch():
    url, text, q, ans = pick(PAGES)
    if 'http' not in q:
        q = q.replace('python.org homepage', f'python.org homepage ({url})').replace('kernel.org', f'kernel.org ({url})')
    t = Traj(q, base_files(), web={url: text})
    t.call('web_fetch', prompt=f'{url} — {pick(["extract the requested information", "read the page", "get the main content"])}')
    return t.answer(ans)


SEARCHES = [
    ('latest stable version of Kubernetes', [('Kubernetes Releases', 'https://kubernetes.io/releases/', 'The latest release is v1.34.1 (released 2026-10-08).')], 'The latest stable Kubernetes release is **v1.34.1**.'),
    ('current price of bitcoin', [('Bitcoin price today', 'https://coins.example/btc', 'BTC is trading at $71,240, up 1.2% in 24h.')], 'Bitcoin is trading around **$71,240** (up 1.2% in the last 24h), per coins.example.'),
    ('who won the most recent F1 race', [('Mexico City GP results', 'https://f1.example/results', 'Lando Norris won the Mexico City Grand Prix on Oct 26, 2026.')], 'Lando Norris won the most recent race, the Mexico City Grand Prix (Oct 26, 2026).'),
    ('USD to INR exchange rate today', [('USD/INR', 'https://fx.example/usd-inr', '1 USD = 88.42 INR as of 31 Oct 2026.')], '1 USD is about **88.42 INR** today.'),
    ('latest Ubuntu LTS release', [('Ubuntu releases', 'https://ubuntu.com/about/release-cycle', 'Ubuntu 26.04 LTS (Resolute Raccoon) was released in April 2026.')], 'The latest Ubuntu LTS is **26.04** (released April 2026).'),
    ('ROS 2 newest distribution', [('ROS 2 Distributions', 'https://docs.ros.org/en/rolling/Releases.html', 'Lyrical Luth is the newest ROS 2 distribution, released May 2026.')], 'The newest ROS 2 distribution is **Lyrical Luth** (May 2026).'),
    ('is there a holiday in Kerala on November 1', [('Kerala Piravi', 'https://holidays.example/kerala', 'Kerala Piravi (Nov 1) is a public holiday in Kerala.')], 'Yes — November 1 is Kerala Piravi, a public holiday in Kerala.'),
]


def fam_search():
    topic, results, ans = pick(SEARCHES)
    if topic.startswith(('who ', 'is ')):
        prompt = pick([topic + '?', 'search the web: ' + topic, 'can you check ' + topic + '?'])
    else:
        prompt = pick([f'what is the {topic}?', f'look up the {topic}', f'search: {topic}', f'Can you find out the {topic}?'])
    t = Traj(prompt, base_files(), search=results)
    t.call('google_web_search', query=topic)
    return t.answer(ans)


def fam_search_fetch():
    t_ = pick([
        ('the changelog for the latest FastAPI release', ('FastAPI release notes', 'https://fastapi.tiangolo.com/release-notes/', 'Release notes for FastAPI'),
         'FastAPI 0.120.2\n\nFixes\n- Fix OpenAPI schema for Annotated dependencies.\n- Fix duplicated tags in docs.', 'The latest FastAPI release is 0.120.2. It fixes the OpenAPI schema for `Annotated` dependencies and duplicated tags in the docs.'),
        ('the opening hours of the Kochi Metro', ('Kochi Metro timings', 'https://kochimetro.example/timings', 'Train timings and frequency'),
         'Kochi Metro\n\nFirst train: 6:00 AM\nLast train: 10:30 PM\nFrequency: every 7 minutes at peak', 'Kochi Metro runs from 6:00 AM to 10:30 PM, every ~7 minutes at peak.'),
    ])
    topic, (title, url, snip), page, ans = t_
    t = Traj(pick([f'find {topic}', f'look up {topic} and give me the details']), base_files(), search=[(title, url, snip)], web={url: page})
    t.call('google_web_search', query=topic.replace('the ', ''))
    t.call('web_fetch', prompt=f'{url} — extract {topic}')
    return t.answer(ans)


def fam_csv():
    rows = [(pick(ITEMS), R.randint(1, 9), R.randint(2, 60)) for _ in range(R.randint(3, 6))]
    seen, uniq = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            uniq.append(r)
    rows = uniq
    name = pick(['orders.csv', 'expenses.csv', 'inventory.csv', 'data/purchases.csv'])
    files = base_files()
    files[name] = 'product,count,unit_price\n' + '\n'.join(f'{a},{b},{c}' for a, b, c in rows) + '\n'
    total = sum(b * c for _, b, c in rows)
    top = max(rows, key=lambda r: r[1] * r[2])
    kind = R.random()
    if kind < 0.5:
        q = pick([f'what is the total of count*unit_price in {name}?', f'Sum up the spend in {name} (count times unit price).'])
        t = Traj(q, files)
        t.call('read_file', file_path=name)
        breakdown = ' + '.join(f'{b}×{c}' for _, b, c in rows)
        return t.answer(f'Total = {breakdown} = **{total}**.')
    q = pick([f'which product in {name} has the highest total value?', f'In {name}, what costs the most overall?'])
    t = Traj(q, files)
    t.call('read_file', file_path=name)
    return t.answer(f'**{top[0]}**: {top[1]} × {top[2]} = {top[1] * top[2]}, the highest in {name}.')


def fam_run_report():
    progs = {}
    files = base_files()
    for _ in range(R.randint(2, 3)):
        n = f'tools/{word()}_{R.randint(1, 9)}.py'
        out = pick(['ok', '42', 'done in 0.3s', 'ready', '3 items processed', 'hello'])
        files[n] = f'print("{out}")\n'
        progs[f'python3 {n}'] = out
    report = pick(['summary.md', 'run_log.md', 'results.md'])
    t = Traj(pick([f'run all the python scripts in tools/ and write their outputs to {report}', f'Execute every .py in tools and record each output in {report}.']),
             files, programs=progs)
    t.call('glob', pattern='tools/*.py')
    lines = []
    for p, out in progs.items():
        f = p.split()[-1]
        t.sh(p, f'Run {f}.')
        lines.append(f'- `{f}`: {out}')
    t.call('write_file', file_path=report, content='# Script outputs\n\n' + '\n'.join(lines) + '\n')
    return t.answer(f'Ran {len(progs)} scripts and wrote their outputs to {report}.')


def fam_extract_to_file():
    files = base_files()
    words_ = pick(['WARN', 'FAIL', 'DENIED'])
    lines = [f'{pick(["OK", "INFO"])} step {i}' for i in range(4)]
    bad = [f'{words_} {pick(["auth", "db", "net", "fs"])} {i}' for i in range(R.randint(1, 3))]
    allines = lines[:2] + bad + lines[2:]
    files['run.log'] = '\n'.join(allines) + '\n'
    out = pick(['problems.txt', 'warnings.txt', 'flagged.log'])
    t = Traj(pick([f'copy every {words_} line from run.log into {out}', f'extract the {words_} lines of run.log to a new file {out}']), files)
    if R.random() < 0.5:
        t.sh(f'grep "{words_}" run.log > {out}', f'Write {words_} lines to {out}.')
        t.sim.files[out] = '\n'.join(bad) + '\n'
    else:
        t.call('read_file', file_path='run.log')
        t.call('write_file', file_path=out, content='\n'.join(bad) + '\n')
    return t.answer(f'Wrote {len(bad)} {words_} line{"s" if len(bad) > 1 else ""} to {out}.')


def fam_replace_recover():
    fn_old, fn_new = pick([('calc', 'calculate'), ('get_usr', 'get_user'), ('proc', 'process_item')])
    name = f'{pick(["lib", "app"])}/{word()}.py'
    files = base_files()
    files[name] = f'import math\n\n\ndef {fn_old}(value):\n    return value * 2\n\n\nprint({fn_old}(3))\n'
    t = Traj(pick([f'rename {fn_old} to {fn_new} in {name}, including where it is called', f'In {name} change the function name {fn_old} -> {fn_new} everywhere']), files)
    t.call('replace', file_path=name, instruction=f'Rename {fn_old} to {fn_new}.', old_string=f'def {fn_old}(value) :', new_string=f'def {fn_new}(value):')
    t.call('read_file', file_path=name)
    t.call('replace', file_path=name, instruction=f'Rename {fn_old} to {fn_new} at the definition and call site.', old_string=fn_old, new_string=fn_new, allow_multiple=True)
    return t.answer(f'Renamed `{fn_old}` to `{fn_new}` in {name} (definition and call).')


def fam_wrong_path_recover():
    real = f'{pick(["docs", "notes", "config"])}/{word()}.md'
    files = base_files()
    files[real] = f'# Setup\n\nRun `make install` then `make run`.\n'
    guess = real.split('/')[-1]
    t = Traj(f'what does {guess} say?', files)
    t.call('read_file', file_path=guess)
    t.call('glob', pattern=f'**/{guess}')
    t.call('read_file', file_path=real)
    return t.answer(f'{real} says to run `make install`, then `make run`.')


def fam_shell_error_recover():
    files = base_files()
    t = Traj(pick(['install the requests package for python', 'pip install requests please']), files,
             extra_shell=[(r'^pip install requests$', 'error: externally-managed-environment\n× This environment is externally managed'),
                          (r'pip install --user requests|python3 -m pip install --user', 'Successfully installed requests-2.32.5'),
                          (r'venv', 'Successfully installed requests-2.32.5')])
    t.sh('pip install requests', 'Install the requests package.')
    t.sh('python3 -m venv .venv && .venv/bin/pip install -q requests && .venv/bin/pip show requests | head -2', 'Create a virtualenv and install requests into it.')
    return t.answer('The system Python is externally managed, so I created a virtualenv in `.venv` and installed requests 2.32.5 there. Activate it with `source .venv/bin/activate`.')


CHAT = [
    ('what is 12 * 14?', '168.'),
    ('whats 15% of 2400', '15% of 2400 is 360.'),
    ('convert 5 km to miles', '5 km is about 3.11 miles.'),
    ('hey', 'Hi! What can I help you with?'),
    ('thanks!', "You're welcome!"),
    ('explain what a git rebase does in two sentences', 'A rebase moves your branch\'s commits so they start from a new base commit, replaying them one by one. This gives a linear history instead of a merge commit.'),
    ('what does the -p flag do in mkdir?', '`mkdir -p` creates any missing parent directories and does not error if the directory already exists.'),
    ('give me a name idea for a weather app', 'How about "Skycast"?'),
    ('what is the difference between a list and a tuple in python', 'Lists are mutable (you can change them), tuples are immutable. Tuples are hashable, so they can be dict keys; lists cannot.'),
    ('how do I exit vim', 'Press `Esc`, then type `:q` and Enter (`:wq` to save and quit, `:q!` to quit without saving).'),
    ('write a haiku about rain', 'Soft rain on the roof\nthe kettle hums its answer\nmorning takes its time'),
    ('is 97 a prime number', 'Yes, 97 is prime.'),
]


def fam_chat():
    q, a = pick(CHAT)
    t = Traj(q, base_files())
    return t.answer(a)


FAMILIES = [
    (fam_count_ext, 6), (fam_count_lines, 4), (fam_log_grep, 6), (fam_sysinfo, 8), (fam_create_list, 6), (fam_json_edit, 7),
    (fam_env_edit, 4), (fam_append, 5), (fam_move, 5), (fam_script, 6), (fam_find_text, 5), (fam_glob, 5), (fam_define, 4),
    (fam_download, 8), (fam_fetch, 6), (fam_search, 8), (fam_search_fetch, 4), (fam_csv, 7), (fam_run_report, 5),
    (fam_extract_to_file, 5), (fam_replace_recover, 4), (fam_wrong_path_recover, 3), (fam_shell_error_recover, 2), (fam_chat, 8),
]


RECOVERY = {'fam_replace_recover', 'fam_wrong_path_recover', 'fam_shell_error_recover', 'fam_download'}


def validate(family, messages):
    """Reject trajectories with tool failures, except the deliberate first failure in recovery families."""
    failures = 0
    for m in messages:
        if m['role'] != 'tool':
            continue
        c = m['content']
        if 'could not interpret' in c:
            return False
        if c.startswith('{"error"') or 'No such file' in c or 'Exit Code' in c or 'Failure writing' in c or 'externally-managed' in c:
            failures += 1
    return failures == 0 or (family in RECOVERY and failures == 1)


def main(n=3000, out='data/train.jsonl'):
    Path(out).parent.mkdir(exist_ok=True)
    funcs = [f for f, w in FAMILIES for _ in range(w)]
    counts = {}
    with open(out, 'w') as fh:
        rejected = {}
        written = 0
        while written < n:
            f = pick(funcs)
            msgs = f()
            if not validate(f.__name__, msgs):
                rejected[f.__name__] = rejected.get(f.__name__, 0) + 1
                continue
            counts[f.__name__] = counts.get(f.__name__, 0) + 1
            fh.write(json.dumps({'family': f.__name__, 'messages': msgs}) + '\n')
            written += 1
    print('kept', json.dumps(counts))
    print('rejected', json.dumps(rejected))


if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
