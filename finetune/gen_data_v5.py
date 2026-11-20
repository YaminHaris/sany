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
    scoped = R.random() < 0.5
    folder = pick(['lib', 'app', 'core', 'pkg', 'backend'])
    for n in notes:
        p = f'{folder if scoped else pick(["lib", "app", "core"])}/{word()}_{len(paths)}.py'
        files[p] = f'import json\n\n# {marker}: {n}\ndef f():\n    pass\n'
        paths.append((p, n))
    if scoped:
        files[f'docs/{word()}.md'] = f'# Notes\n\n{marker} list lives in the code.\n'
        t = Traj(pick([f'find the {marker} comments in the {folder} folder', f'list every {marker} under {folder}/ with file names',
                       f'what {marker}s are in {folder}?']), files)
        args = {'pattern': marker, 'dir_path': folder}
        if R.random() < 0.4:
            args['include_pattern'] = '*.py'
        t.call('grep_search', **args)
    else:
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


SAVE_FACTS = [
    ('latest stable Python version', 'Python 3.14.0 is now available', 'https://www.python.org/downloads/release/python-3140/', 'Python Release Python 3.14.0', '3.14.0'),
    ('current LTS version of Node.js', 'Node.js 24.11.0 (LTS) is the recommended version for most users.', 'https://nodejs.org/en/download', 'Download Node.js', '24.11.0'),
    ('latest Go release', 'go1.25.3 (released 2026-10-07) includes security fixes.', 'https://go.dev/doc/devel/release', 'Release History - The Go Programming Language', '1.25.3'),
    ('newest PostgreSQL version', 'PostgreSQL 18.1 released with bug fixes.', 'https://www.postgresql.org/about/news/', 'PostgreSQL News', '18.1'),
    ('latest Docker Engine version', 'Docker Engine 29.0.1 release notes.', 'https://docs.docker.com/engine/release-notes/', 'Docker Engine release notes', '29.0.1'),
]


def fam_search_save():
    topic, snip, url, title, val = pick(SAVE_FACTS)
    out = pick(['version.txt', 'latest.txt', 'notes/version.txt', 'ver.txt'])
    files = base_files()
    if out.startswith('notes/'):
        files['notes/README.md'] = '# notes\n'
    prompt = pick([f'look up the {topic} online and save only the number to {out}', f'search for the {topic} and write it into {out}',
                   f'find the {topic} on the web and put just the version in {out}'])
    broken = R.random() < 0.35
    t = Traj(prompt, files, search=[(title, url, snip)], web={} if broken else {url: f'{title}\n\n{snip}'})
    t.call('google_web_search', query=topic)
    if R.random() < 0.5:
        res = t.call('web_fetch', prompt=f'{url} — confirm the {topic}')
        if '404' in res.get('output', ''):
            t.call('write_file', file_path=out, content=val + '\n')
            return t.answer(f'The page itself returned 404, but the search result says {val}. Wrote `{val}` to {out}.')
    t.call('write_file', file_path=out, content=val + '\n')
    return t.answer(f'Wrote `{val}` to {out}.')


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
        steps = '\n'.join(f'- {a}: {b} × {c} = {b * c}' for a, b, c in rows)
        return t.answer(f'{steps}\n\nTotal = ' + ' + '.join(str(b * c) for _, b, c in rows) + f' = **{total}**.')
    q = pick([f'which product in {name} has the highest total value?', f'In {name}, what costs the most overall?'])
    t = Traj(q, files)
    t.call('read_file', file_path=name)
    steps = '\n'.join(f'- {a}: {b} × {c} = {b * c}' for a, b, c in rows)
    return t.answer(f'{steps}\n\n**{top[0]}** has the highest total ({top[1] * top[2]}).')


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


# ---------------------------------------------------------------- v3 families: fuzzy find, ports, search+read, hogs
HOME = '/home/yamin'
SUBJECTS = ['DBMS', 'OS', 'CN', 'maths', 'DSA', 'compiler', 'AI', 'TOC', 'signals', 'economics']

FIND_TOPICS = [
    dict(asks=['my endsem notes', 'the end sem notes', 'my end-semester study material'], p1=['*endsem*', '*end*sem*'], p2=['*final*exam*', '*semester*exam*'],
         hits1=['Documents/Sem{n}/EndSem_{s}.pdf', 'Downloads/endsem_{s}_notes.docx'], dir1='Documents/Sem{n}/End-Sem Notes',
         hits2=['Documents/Sem{n}/Final_Exam_{s}.pdf', 'Downloads/{s} semester exam prep.pdf'], alt='final exam'),
    dict(asks=['my midsem notes', 'the mid sem papers', 'my mid-semester notes'], p1=['*midsem*', '*mid*sem*'], p2=['*midterm*', '*internal*'],
         hits1=['Documents/Sem{n}/MidSem_{s}.pdf', 'Downloads/midsem-{s}.pdf'], dir1='Documents/Sem{n}/Mid Sem',
         hits2=['Documents/Sem{n}/Midterm_{s}.pdf', 'Documents/internals/{s}_internal_1.pdf'], alt='midterm or internals'),
    dict(asks=['my resume', 'my latest CV file', 'the resume I made'], p1=['*resume*'], p2=['*cv*', '*biodata*'],
         hits1=['Documents/Resume_2026.pdf', 'Documents/jobs/yamin_resume.docx'], dir1='Documents/Resume',
         hits2=['Documents/CV_Yamin.pdf', 'Downloads/Yamin_CV_v3.pdf'], alt='CV'),
    dict(asks=['my internship offer letter', 'the offer letter from the internship'], p1=['*offer*'], p2=['*internship*', '*appointment*'],
         hits1=['Documents/Offer_Letter_{s}.pdf', 'Downloads/offer-letter.pdf'], dir1='Documents/Offers',
         hits2=['Documents/Internship_Confirmation.pdf', 'Downloads/appointment_letter.pdf'], alt='internship confirmation or appointment letter'),
    dict(asks=['my bank statements', 'the bank statement pdfs'], p1=['*statement*'], p2=['*bank*', '*hdfc*', '*sbi*'],
         hits1=['Documents/finance/Statement_Oct2026.pdf', 'Downloads/acct_statement_sep.pdf'], dir1='Documents/Statements',
         hits2=['Documents/finance/HDFC_Sep2026.pdf', 'Downloads/SBI_passbook_2026.pdf'], alt='bank name (HDFC, SBI)'),
    dict(asks=['my mini project report', 'the project report I wrote'], p1=['*report*'], p2=['*mini*project*', '*project*doc*'],
         hits1=['Documents/college/Project_Report_final.pdf', 'Documents/college/report_draft.docx'], dir1='Documents/college/Reports',
         hits2=['Documents/college/MiniProject_Documentation.pdf', 'Downloads/project-doc-v2.docx'], alt='mini project documentation'),
]


EXCL = "-not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/site-packages/*' -not -path '*/*venv*/*' -not -path '*/__pycache__/*'"
CORE = {'*endsem*': '*sem*', '*midsem*': '*sem*', '*resume*': '*job*', '*offer*': '*intern*', '*statement*': '*finance*', '*report*': '*college*'}


def _expr(pats):
    if len(pats) == 1:
        return f"-iname '{pats[0]}'"
    return '\\( ' + ' -o '.join(f"-iname '{p}'" for p in pats) + ' \\)'


def _find(root, pats, kind=''):
    return f"find {root} {kind}{_expr(pats)} {EXCL} 2>/dev/null | head -50"


def _noise(t_):
    files = {
        'Documents/todo.txt': 'x\n', 'Pictures/IMG_2041.jpg': 'x', 'Music/playlist.m3u': 'x\n', 'Downloads/setup.deb': 'x',
        '.cache/thumbnails/a.png': 'x', '.local/share/recently-used.xbel': 'x\n', 'Development/app/README.md': '# app\n',
    }
    for _ in range(R.randint(2, 5)):
        files[f'Documents/{pick(["college", "uni", "misc"])}/{pick(SUBJECTS)}_assignment_{R.randint(1, 5)}.pdf'] = 'x'
    # dependency junk whose names match the search words; the excludes must hide it
    words = [p.strip('*').split('*')[0] for p in t_['p1'] + t_['p2'] + [CORE[t_['p1'][0]]]]
    for w in words:
        for _ in range(R.randint(2, 6)):
            proj = pick(['E-Drone', 'bot', 'gods-eye', 'ml-lab', 'site'])
            files[pick([f'Development/{proj}/venv/lib/python3.12/site-packages/{word()}/{w}{pick(["aphore", "antic", "_utils", "ion", "er"])}.py',
                        f'Development/{proj}/node_modules/{word()}/lib/{w}{pick(["ion", "ector", "_x"])}.js',
                        f'Development/{proj}/.venv/lib/python3.12/site-packages/{w}lib/__init__.py'])] = 'x'
    return files


def _paths(res):
    return [l for l in res['output'].removeprefix('Output: ').split('\n') if l.startswith('/')]


def _bullets(ps):
    return '\n'.join(f'- {p}' for p in ps)


def fam_find_fuzzy():
    t_ = pick(FIND_TOPICS)
    n, s = R.randint(3, 8), pick(SUBJECTS)
    fmt = lambda p: p.format(n=n, s=s)
    files = _noise(t_)
    ask = pick(t_['asks'])
    prompt = pick([f'find {ask}', f'where are {ask}?', f'i want to find {ask}', f'can you locate {ask}', f'search my laptop for {ask}', f'can you search for {ask}'])
    core = CORE[t_['p1'][0]]
    mode = R.random()
    if mode < 0.35:  # direct hit
        use_dir = R.random() < 0.4
        root_pref = pick(['', 'Desktop/'])
        if use_dir:
            d = fmt(t_['dir1'])
            d = (root_pref + d.split('/', 1)[1]) if root_pref else d
            inside = [f'{d}/{x}_unit{k}.pdf' for k, x in enumerate(pick(SUBJECTS, R.randint(2, 4)), 1)]
            for f in inside:
                files[f] = 'x'
        else:
            for h in pick(t_['hits1'], R.randint(1, 2)):
                h = fmt(h)
                files[(root_pref + h.split('/', 1)[1]) if root_pref else h] = 'x'
        t = Traj(prompt, files, cwd=HOME)
        res = t.sh(_find('~', t_['p1']), 'Search names in the home folder (case-insensitive), skipping hidden and dependency folders.')
        out = _paths(res)
        if use_dir:
            t.sh(f'ls -1 "{HOME}/{d}"', 'List the matching folder.')
            return t.answer(f'Found the folder **{HOME}/{d}**:\n' + '\n'.join(f'- {f.rsplit("/", 1)[1]}' for f in sorted(inside)))
        return t.answer('Found:\n' + _bullets(out))
    if mode < 0.55 or (0.55 <= mode < 0.7 and core != '*sem*'):  # broaden in the same turn
        for h in pick(t_['hits2'], R.randint(1, 2)):
            files[fmt(h)] = 'x'
        t = Traj(prompt, files, cwd=HOME)
        t.sh(_find('~', t_['p1']), 'Search names in the home folder (case-insensitive), skipping hidden and dependency folders.')
        out = _paths(t.sh(_find('~', t_['p2']), f'Nothing matched; try related names ({t_["alt"]}).'))
        return t.answer(f'Nothing is named like "{t_["p1"][0].strip("*")}", but these look like it:\n' + _bullets(out))
    if mode < 0.7:  # only a broad folder search finds it
        top = pick(['Desktop', 'Documents'])
        folder = pick([f'{top}/SEM{n}', f'{top}/Notes Sem {n}', f'{top}/semester-{n}', f'{top}/{n}th_Semester_Notes'])
        for x in pick(SUBJECTS, R.randint(2, 3)):
            files[f'{folder}/{x}/{pick(["papers", "notes", "material"])}.pdf'] = 'x'
        t = Traj(prompt, files, cwd=HOME)
        t.sh(_find('~', t_['p1'] + t_['p2']), 'Search names in the home folder (case-insensitive), skipping hidden and dependency folders.')
        res = t.sh(_find('~', [core], '-type d '), f'No file names matched; look for folders named like {core}.')
        out = _paths(res)
        t.sh(f'ls -1 "{out[0]}"', 'Look inside the best-matching folder.')
        return t.answer(f'No file is named like "{t_["p1"][0].strip("*")}", but this folder looks right:\n' + _bullets(out) +
                        f'\n\nInside {out[0]} there are subject folders — tell me which subject and I\'ll open it.')
    if mode < 0.85:  # user points to a location after a miss
        loc = pick(['Desktop', 'Documents', 'Downloads'])
        hit = f'{loc}/{pick(["college", "old", "misc", "SEM" + str(n)])}/{s} {pick(["final", "endterm", "exam"])} {pick(["prep", "material"])}.pdf'
        files[hit] = 'x'
        t = Traj(prompt, files, cwd=HOME)
        t.sh(_find('~', t_['p1']), 'Search names in the home folder (case-insensitive), skipping hidden and dependency folders.')
        t.answer(f'Nothing in your home folder is named like "{t_["p1"][0].strip("*")}". It might be named differently (for example {t_["alt"]}). '
                 'Want me to try other names, or is there a folder I should look in?')
        t.messages.append({'role': 'user', 'content': pick([f'look in {loc.lower()}', f'check the {loc} folder', f'try {loc.lower()}', f'its probably in {loc.lower()}'])})
        res = t.sh(f"find ~/{loc} -type f \\( -iname '*.pdf' -o -iname '*.docx' -o -iname '*.pptx' -o -iname '*.md' \\) {EXCL} 2>/dev/null | head -50",
                   f'List documents in ~/{loc}, since the name may not contain the search words.')
        out = _paths(res)
        best = [p for p in out if s in p]
        return t.answer(f'Documents in ~/{loc}:\n' + _bullets(out) + (f'\n\nThe most likely match is **{best[0]}**.' if best else ''))
    # not in home: check other mounted drives
    usb = R.random() < 0.5
    usb_path = f'/media/yamin/{pick(["PENDRIVE", "KINGSTON", "SANDISK"])}/{pick(["sem", "college"])}{n}/{fmt(pick(t_["hits1"])).rsplit("/", 1)[1]}'
    t = Traj(prompt, files, cwd=HOME, extra_shell=[(r'find /media', usb_path if usb else ''),
                                                    (r'lsblk', 'NAME        SIZE MOUNTPOINT\nnvme0n1   476.9G\n├─nvme0n1p1  100M /boot/efi\n└─nvme0n1p2 476.8G /' +
                                                     ('\nsda        29.3G\n└─sda1      29.3G ' + usb_path.rsplit('/', 2)[0] if usb else ''))])
    t.sh(_find('~', t_['p1'] + t_['p2']), 'Search names in the home folder (case-insensitive), skipping hidden and dependency folders.')
    t.answer(f'Nothing in your home folder matches "{t_["p1"][0].strip("*")}" or {t_["alt"]}. Is it maybe on a USB drive, or in a folder I should check?')
    t.messages.append({'role': 'user', 'content': pick(['what if its not in my home? can you look in other places', 'check other drives', 'maybe its on my pendrive', 'look outside home'])})
    t.sh('lsblk -o NAME,SIZE,MOUNTPOINT', 'See which drives are mounted.')
    t.sh(f"find /media /mnt /run/media {_expr(t_['p1'] + t_['p2'])} 2>/dev/null | head -50", 'Search mounted drives.')
    if usb:
        return t.answer(f'Found it on the USB drive:\n- {usb_path}')
    return t.answer('Only your laptop\'s own disk is mounted, and nothing on it matches. If the notes are on a USB drive or phone, plug it in and I\'ll search it.')


PORTS = {22: ('sshd', 'SSH'), 53: ('systemd-resolve', 'DNS resolver'), 631: ('cupsd', 'CUPS printing'), 5432: ('postgres', 'PostgreSQL'),
         3000: ('node', 'dev server'), 8080: ('python3', 'web server'), 11434: ('ollama', 'Ollama'), 9050: ('tor', 'Tor'),
         27036: ('steam', 'Steam'), 1716: ('kdeconnectd', 'KDE Connect'), 6379: ('redis-server', 'Redis'), 8000: ('python3', 'web server'),
         3306: ('mysqld', 'MySQL'), 5353: ('avahi-daemon', 'mDNS'), 8118: ('privoxy', 'Privoxy'), 9040: ('tor', 'Tor transparent proxy')}


def ss_table(entries):
    rows = ['State   Recv-Q  Send-Q   Local Address:Port    Peer Address:Port  Process']
    for addr, port, proc, pid in entries:
        users = f'users:(("{proc}",pid={pid},fd={R.randint(3, 140)}))' if proc else ''
        peer = '[::]:*' if addr.startswith('[') else ('*:*' if addr == '*' else '0.0.0.0:*')
        rows.append(f'LISTEN  0       {R.choice([128, 511, 4096])}    {addr + ":" + str(port):<22} {peer:<18} {users}'.rstrip())
    return '\n'.join(rows)


def make_ports():
    chosen = R.sample(list(PORTS), R.randint(5, 10))
    chosen += [R.randint(32768, 60999) for _ in range(R.randint(0, 3))]  # random ephemeral listeners
    entries, info = [], {}
    for p in chosen:
        public = R.random() < 0.35
        addr = pick(['0.0.0.0', '*']) if public else '127.0.0.1'
        proc = PORTS[p][0] if p in PORTS and R.random() < 0.6 else None
        pid = R.randint(800, 300000)
        entries.append((addr, p, proc, pid))
        if R.random() < 0.3:
            entries.append(('[::]' if public else '[::1]', p, proc, pid))
        info[p] = (public, proc)
    R.shuffle(entries)
    return entries, info


def describe_ports(info, network_q):
    def line(p):
        public, proc = info[p]
        label = PORTS[p][1] if p in PORTS else None
        bits = [x for x in [label, f'`{proc}`' if proc else None] if x]
        return f'- **{p}**' + (f' — {", ".join(bits)}' if bits else ' — unknown process (run `sudo ss -ltnp` to see it)')
    pub = sorted(p for p in info if info[p][0])
    loc = sorted(p for p in info if not info[p][0])
    parts = [f'{len(info)} TCP ports are listening on this machine (IPv4 and IPv6 lines for the same port count once).']
    if pub:
        parts.append('**Reachable from other devices** (bound to all interfaces):\n' + '\n'.join(line(p) for p in pub))
    else:
        parts.append('None are reachable from other devices; all are bound to localhost.')
    if loc:
        parts.append('**Local only** (127.0.0.1 / ::1):\n' + '\n'.join(line(p) for p in loc))
    if network_q:
        parts.append('This covers only this laptop. To see other devices on your network, use `nmap` (e.g. `nmap -sn 192.168.1.0/24`).')
    return '\n\n'.join(parts)


def read_lines(entries):
    seen, rows = set(), []
    for addr, port, proc, _ in entries:
        scope = 'all interfaces' if addr in ('0.0.0.0', '*', '[::]') else 'localhost only'
        dup = ' (same port again)' if port in seen else ''
        seen.add(port)
        rows.append(f'- `{addr}:{port}` → {scope}{dup}')
    return 'Reading each line:\n' + '\n'.join(rows)


def fam_ports():
    entries, info = make_ports()
    network_q = R.random() < 0.4
    prompt = pick(['find open ports in my network', 'scan my network for open ports', 'what ports are open on my network?']) if network_q else \
        pick(['what ports are open?', 'which ports is my laptop listening on', 'list the listening ports', 'show open ports on this pc'])
    t = Traj(prompt, {}, cwd=HOME, extra_shell=[(r'\bss\b|netstat', ss_table(entries))])
    t.sh(pick(['ss -ltnp', 'ss -tlnp']), 'Show listening TCP ports and their processes.')
    return t.answer(read_lines(entries) + '\n\n' + describe_ports(info, network_q))


EVENTS = [
    ('Nimbus Cloud outage', 'nimbuscloud.example/status/incident-1029', 'Nimbus Cloud incident report',
     'On 29 October 2026 Nimbus Cloud was down for about 6 hours in its Asia region. The cause was a faulty configuration push to its load balancers. '
     'Roughly 12,000 customer sites went offline. Nimbus rolled back the change and promised staged rollouts and service credits.',
     'Nimbus Cloud\'s Asia region was down for about **6 hours** on 29 Oct 2026 after a faulty configuration push to its load balancers. '
     'About 12,000 customer sites went offline; Nimbus rolled the change back and is offering service credits and moving to staged rollouts.'),
    ('Quartz Pay data leak', 'news.example/tech/quartz-pay-leak', 'Quartz Pay confirms data leak',
     'Quartz Pay said an exposed backup server leaked names and phone numbers of 2.1 million users. No card numbers or passwords were included. '
     'The server was public for three weeks before a researcher reported it. Users are advised to watch for phishing messages.',
     'An exposed backup server at Quartz Pay leaked the **names and phone numbers of 2.1 million users** for about three weeks before a researcher reported it. '
     'No card numbers or passwords were exposed. Users should watch for phishing messages.'),
    ('Falcon Rail strike', 'transit.example/falcon-rail-strike', 'Falcon Rail workers strike',
     'Falcon Rail drivers began a 48-hour strike on 30 October over shift lengths and pay. About 70% of services were cancelled. '
     'Talks with the union are set to resume on 3 November.',
     'Falcon Rail drivers started a **48-hour strike** on 30 October over shift lengths and pay, cancelling about 70% of services. Union talks resume on 3 November.'),
    ('Harbor Linux sudo bug', 'security.example/advisories/harbor-sudo', 'Harbor Linux sudo vulnerability advisory',
     'A bug in the patched sudo shipped by Harbor Linux 12 let local users gain root by passing a long --chdir argument. It is tracked as HLSA-2026-117. '
     'Fixed packages (sudo 1.9.17-3) were released on 27 October; users should run a system update.',
     'Harbor Linux 12\'s sudo had a bug (**HLSA-2026-117**) that let local users get root via a long `--chdir` argument. Fixed in **sudo 1.9.17-3** (27 Oct); run a system update.'),
    ('Cedar Valley flooding', 'weather.example/cedar-valley-floods', 'Cedar Valley floods: what we know',
     'Heavy rain caused the Cedar River to burst its banks on 28 October. About 3,400 people were evacuated and two bridges closed. '
     'Water levels started falling on 30 October, and relief camps remain open.',
     'Heavy rain made the Cedar River burst its banks on 28 Oct. About **3,400 people were evacuated** and two bridges closed. Levels began falling on 30 Oct; relief camps are still open.'),
    ('Lumen OS 5 launch', 'blog.lumenos.example/lumen-5', 'Introducing Lumen OS 5',
     'Lumen OS 5 was released on 25 October with a new tiling window manager, Wayland by default and kernel 6.17. '
     'It drops 32-bit support. Upgrades from Lumen 4 are available through the software centre.',
     'Lumen OS 5 came out on 25 Oct with a new **tiling window manager**, Wayland by default and kernel 6.17. It drops 32-bit support; Lumen 4 users can upgrade from the software centre.'),
]


def fam_search_read():
    name, url_path, title, page, ans = pick(EVENTS)
    url = 'https://' + url_path
    results = [(title, url, page.split('. ')[0][:90] + '…'),
               (f'{name} - discussion', f'https://forum.example/t/{R.randint(1000, 9999)}', 'Thread discussing the news…'),
               (f'{name}: live updates', f'https://live.example/{R.randint(100, 999)}', 'Latest updates…')]
    broken = R.random() < 0.2
    web = {results[1][1]: f'{name} discussion\n\n{page}'} if broken else {url: f'{title}\n\n{page}'}
    prompt = pick([f'search online about the {name}', f'what happened with the {name}? look it up', f'tell me about the {name} from the web',
                   f'look up the {name} news', f'search the {name}'])
    t = Traj(prompt, {}, cwd=HOME, search=results, web=web)
    t.call('google_web_search', query=name)
    res = t.call('web_fetch', prompt=f'{url} — summarize what happened')
    src = url
    if '404' in res.get('output', ''):
        t.call('web_fetch', prompt=f'{results[1][1]} — summarize what happened')
        src = results[1][1]
    return t.answer(f'{ans}\n\nSource: {src}')


PROCS = ['firefox', 'chrome', 'code', 'steam', 'ollama', 'python3', 'node', 'java', 'gnome-shell', 'plasmashell', 'Xwayland', 'discord', 'spotify', 'obs', 'tracker-miner-fs']


def fam_hogs():
    by = pick(['cpu', 'mem'])
    procs = R.sample(PROCS, 6)
    vals = sorted((round(R.uniform(0.5, 95), 1) for _ in procs), reverse=True)
    other = [round(R.uniform(0.1, 12), 1) for _ in procs]
    hdr = 'USER         PID %CPU %MEM    VSZ   RSS TTY      STAT START   TIME COMMAND'
    rows = []
    for p, v, o in zip(procs, vals, other):
        cpu, mem = (v, o) if by == 'cpu' else (o, v)
        rows.append(f'yamin    {R.randint(1000, 300000):>7} {cpu:>4} {mem:>4} {R.randint(100000, 9000000):>7} {R.randint(10000, 900000):>6} ?        Sl   10:{R.randint(10, 59)}   {R.randint(0, 90)}:{R.randint(10, 59)} /usr/bin/{p}')
    out = '\n'.join([hdr] + rows[:5])
    prompt = pick(['what is using my cpu', 'why is my laptop so slow? check what is hogging the cpu', 'which process is eating cpu']) if by == 'cpu' else \
        pick(['what is using all my ram', 'which apps use the most memory', 'check memory hogs'])
    t = Traj(prompt, {}, cwd=HOME, extra_shell=[(r'\bps\b|\btop\b', out)])
    t.sh(f'ps aux --sort=-%{by} | head -6', f'Show the top processes by {"CPU" if by == "cpu" else "memory"}.')
    unit = 'CPU' if by == 'cpu' else 'memory'
    top3 = '\n'.join(f'{i}. **{p}** — {v}% {unit}' for i, (p, v) in enumerate(zip(procs[:3], vals[:3]), 1))
    return t.answer(f'Top {unit} users right now:\n{top3}')


FAMILIES = [
    (fam_count_ext, 6), (fam_count_lines, 4), (fam_log_grep, 6), (fam_sysinfo, 8), (fam_create_list, 6), (fam_json_edit, 7),
    (fam_env_edit, 4), (fam_append, 5), (fam_move, 5), (fam_script, 6), (fam_find_text, 5), (fam_glob, 5), (fam_define, 4),
    (fam_download, 8), (fam_fetch, 6), (fam_search, 8), (fam_search_fetch, 4), (fam_search_save, 5), (fam_csv, 7), (fam_run_report, 5),
    (fam_extract_to_file, 5), (fam_replace_recover, 4), (fam_wrong_path_recover, 3), (fam_shell_error_recover, 2), (fam_chat, 8),
    (fam_find_fuzzy, 8), (fam_ports, 5), (fam_search_read, 6), (fam_hogs, 4),
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
