"""Held-out evaluation tasks. Phrasings and file names here are NOT used by the data generator."""

import json
import re

CWD = '/home/yamin/projects/daily'

BASE_FILES = {
    'a.txt': 'alpha\n', 'b.txt': 'bravo\n', 'c.txt': 'charlie\n',
    'config.json': '{\n  "name": "demo",\n  "port": 8080,\n  "debug": false\n}\n',
    'sales.csv': 'item,qty,price\nwidget,3,10\ngadget,2,25\ndoohickey,5,4\n',
    'proj/sum.py': 'print(sum(range(1, 11)))\n',
    'proj/greet.py': 'print("hello from greet")\n',
    'notes.md': '# Notes\n\n- call the plumber\n- TODO: renew passport\n',
    'src/app.py': 'import os\n\n# TODO: add logging\ndef main():\n    print("app")\n\n# TODO: handle errors\n',
    'src/util.py': 'def add(a, b):\n    return a + b\n',
    'logs/app.log': 'INFO start\nERROR disk full\nINFO retry\nERROR timeout\nINFO done\n',
    'requirements.txt': 'requests==2.31.0\nflask==3.0.0\n',
}

NODE_GITIGNORE = '\n'.join(['# Logs', 'logs', '*.log', 'npm-debug.log*', 'node_modules/', 'dist/', '.env', 'coverage/', '.cache/', 'build/', '.DS_Store', '*.tgz']) + '\n'  # 12 lines
EXAMPLE_PAGE = 'Example Domain\n\nThis domain is for use in documentation examples without needing permission. Avoid use in operations.\n\nLearn more'


def lower(s):
    return (s or '').lower()


def used(sim, tool):
    return any(n == tool for n, _ in sim.log)


def shell_cmds(sim):
    return [a.get('command', '') for n, a in sim.log if n == 'run_shell_command']


CASES = [
    # ---------- shell / info ----------
    dict(id='count_txt', prompt='how many text files (.txt) are sitting in this folder right now?',
         check=lambda s, f: re.search(r'\b4\b', f) is not None),  # a,b,c + requirements.txt
    dict(id='count_lines_log', prompt='How many lines are in logs/app.log?',
         check=lambda s, f: re.search(r'\b5\b', f) is not None),
    dict(id='errors_in_log', prompt='Check logs/app.log and tell me how many ERROR lines there are and what they say.',
         check=lambda s, f: re.search(r'\b2\b|two', lower(f)) and 'disk full' in lower(f) and 'timeout' in lower(f)),
    dict(id='python_version', prompt='what version of python is installed?',
         extra_shell=[(r'python3? (--version|-V)', 'Python 3.12.3')],
         check=lambda s, f: '3.12.3' in f),
    dict(id='git_branch', prompt='Which git branch am I on?',
         extra_shell=[(r'git (--no-pager )?(branch|rev-parse|status|symbolic-ref)', 'On branch feature/login\nnothing to commit, working tree clean')],
         check=lambda s, f: 'feature/login' in f),
    dict(id='disk_space', prompt='How much free disk space do I have on / ?',
         extra_shell=[(r'\bdf\b', 'Filesystem      Size  Used Avail Use% Mounted on\n/dev/nvme0n1p2  469G  428G   19G  96% /')],
         check=lambda s, f: '19' in f),
    # ---------- files ----------
    dict(id='create_todo', prompt='Make me a todo list at notes/todo.md as a markdown checklist with these items: buy milk, call mom, pay rent.',
         check=lambda s, f: 'notes/todo.md' in s.files and all(x in lower(s.files['notes/todo.md']) for x in ['buy milk', 'call mom', 'pay rent']) and '[ ]' in s.files['notes/todo.md']),
    dict(id='edit_port', prompt='Set the port in config.json to 9090 and leave everything else alone.',
         check=lambda s, f: _json_ok(s, 'config.json', lambda j: j.get('port') == 9090 and j.get('name') == 'demo' and j.get('debug') is False)),
    dict(id='edit_debug', prompt='turn debug on in config.json',
         check=lambda s, f: _json_ok(s, 'config.json', lambda j: j.get('debug') is True and j.get('port') == 8080)),
    dict(id='append_note', prompt='Add "- book dentist appointment" as a new bullet at the end of notes.md.',
         check=lambda s, f: 'book dentist appointment' in s.files.get('notes.md', '') and 'renew passport' in s.files.get('notes.md', '')),
    dict(id='rename_file', prompt='rename b.txt to beta.txt',
         check=lambda s, f: 'beta.txt' in s.files and 'b.txt' not in s.files and s.files['beta.txt'] == 'bravo\n'),
    dict(id='add_requirement', prompt='Add numpy version 1.26.4 to requirements.txt, pinned the same way as the others.',
         check=lambda s, f: 'numpy==1.26.4' in s.files.get('requirements.txt', '') and 'flask==3.0.0' in s.files.get('requirements.txt', '')),
    dict(id='write_script', prompt='Write a python script hello.py that prints "Hello, Yamin!".',
         check=lambda s, f: 'hello.py' in s.files and 'print' in s.files['hello.py'] and 'Hello, Yamin!' in s.files['hello.py']),
    # ---------- search in files ----------
    dict(id='find_todos', prompt='Find every TODO comment in the src folder and list them with file names.',
         check=lambda s, f: 'add logging' in lower(f) and 'handle errors' in lower(f)),
    dict(id='find_py_files', prompt='list all the python files in this project',
         check=lambda s, f: all(x in f for x in ['sum.py', 'greet.py', 'app.py', 'util.py'])),
    dict(id='where_defined', prompt='In which file is the function add defined?',
         check=lambda s, f: 'util.py' in f),
    # ---------- downloads ----------
    dict(id='download_count', prompt='Grab https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore, save it into downloads/Node.gitignore, and tell me how many lines it has.',
         downloads={'https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore': NODE_GITIGNORE},
         check=lambda s, f: s.files.get('downloads/Node.gitignore') == NODE_GITIGNORE and re.search(r'\b12\b', f) is not None),
    dict(id='download_simple', prompt='Download https://example.org/data/report.csv to the current folder.',
         downloads={'https://example.org/data/report.csv': 'a,b\n1,2\n'},
         check=lambda s, f: any(k.endswith('report.csv') for k in s.files)),
    # ---------- web ----------
    dict(id='fetch_heading', prompt='What is the main heading on https://example.com ?',
         web={'https://example.com': EXAMPLE_PAGE},
         check=lambda s, f: 'example domain' in lower(f) and used(s, 'web_fetch')),
    dict(id='search_node_lts', prompt='What is the newest Node.js LTS major version right now?',
         search=[('Node.js Releases', 'https://nodejs.org/en/about/previous-releases', 'Node.js 24 "Krypton" is the Active LTS release. Node.js 22 is Maintenance LTS. Node.js 25 is Current.'),
                 ('Node.js — Download', 'https://nodejs.org/en/download', 'Get Node.js v24.11.0 (LTS) for Linux.')],
         check=lambda s, f: '24' in f and used(s, 'google_web_search')),
    dict(id='search_weather', prompt="Look up today's weather forecast for Kochi.",
         search=[('Kochi Weather Forecast', 'https://weather.example/kochi', 'Saturday: Thunderstorms, high 31°C, low 24°C, 80% chance of rain.')],
         check=lambda s, f: '31' in f and used(s, 'google_web_search')),
    dict(id='search_then_fetch', prompt='Find the official Python 3.13 "What\'s New" page and tell me one headline feature from it.',
         search=[("What's New In Python 3.13", 'https://docs.python.org/3/whatsnew/3.13.html', 'Summary of release highlights...')],
         web={'https://docs.python.org/3/whatsnew/3.13.html': "What's New In Python 3.13\n\nSummary - Release Highlights\n\nA new and improved interactive interpreter.\nAn experimental free-threaded build mode (PEP 703).\nA preliminary JIT compiler (PEP 744)."},
         check=lambda s, f: used(s, 'web_fetch') and any(x in lower(f) for x in ['interactive interpreter', 'free-threaded', 'free threaded', 'jit', 'gil'])),
    # ---------- data ----------
    dict(id='csv_total', prompt='What is the total revenue in sales.csv (qty × price, summed)?',
         check=lambda s, f: re.search(r'\b100\b', f) is not None),
    dict(id='csv_top', prompt='Which item in sales.csv brought in the most money?',
         check=lambda s, f: 'gadget' in lower(f)),
    # ---------- multi-step ----------
    dict(id='run_report', prompt='Run each python file in proj/ and save a report.md listing every file name and exactly what it printed.',
         programs={'python3 proj/sum.py': '55', 'python3 proj/greet.py': 'hello from greet'},
         check=lambda s, f: 'report.md' in s.files and all(x in s.files['report.md'] for x in ['sum.py', '55', 'greet.py', 'hello from greet'])),
    dict(id='errors_to_file', prompt='Pull all the ERROR lines out of logs/app.log into a new file errors.txt.',
         check=lambda s, f: 'errors.txt' in s.files and 'disk full' in s.files['errors.txt'] and 'timeout' in s.files['errors.txt'] and 'INFO' not in s.files['errors.txt']),
    dict(id='search_save', prompt='Search the web for the current stable Rust version and write just the version number into rust_version.txt.',
         search=[('Rust Blog — Announcing Rust 1.91.0', 'https://blog.rust-lang.org/2026/10/30/Rust-1.91.0', 'The Rust team is happy to announce a new version of Rust, 1.91.0.')],
         check=lambda s, f: '1.91' in s.files.get('rust_version.txt', '')),
    # ---------- error recovery ----------
    dict(id='replace_recover', prompt='In src/util.py, rename the function add to add_numbers.',
         check=lambda s, f: 'def add_numbers(a, b):' in s.files.get('src/util.py', '') and 'def add(' not in s.files.get('src/util.py', '')),
    # ---------- no tool needed ----------
    dict(id='chat_math', prompt='quick one: what is 17 times 23?', no_tools=True,
         check=lambda s, f: '391' in f),
    dict(id='chat_advice', prompt='Give me a one-sentence tip for staying focused while studying.', no_tools=True,
         check=lambda s, f: len(f.strip()) > 10),
]


def _json_ok(sim, path, pred):
    try:
        return pred(json.loads(sim.files.get(path, '')))
    except Exception:
        return False
