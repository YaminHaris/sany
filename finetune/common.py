"""Shared pieces: the exact prompt/tool format the CLI sends to the model."""

import json
from pathlib import Path

ROOT = Path(__file__).parent
SYSTEM_PROMPT = Path('/home/yamin/Development/gemini-cli-local/local/system.md').read_text()
ALL_TOOLS = json.loads((ROOT / 'spec' / 'tools.json').read_text())
TOOL_NAMES = [t['function']['name'] for t in ALL_TOOLS]

SESSION_CONTEXT = """<session_context>
This is the Gemini CLI. We are setting up the context for our chat.
Today's date is {date} (formatted according to the user's locale).
My operating system is: linux
The project's temporary directory is: /home/{user}/.gemini/tmp/{proj}
- **Workspace Directories:**
  - {cwd}
- **Directory Structure:**

Showing up to 200 items (files + folders).

{cwd}/
{tree}

</session_context>"""


def render_tree(files):
    """files: list of relative paths. Renders the CLI's directory listing style."""
    dirs, top = {}, []
    for f in sorted(files):
        parts = f.split('/')
        if len(parts) == 1:
            top.append(parts[0])
        else:
            dirs.setdefault(parts[0], []).append('/'.join(parts[1:]))
    entries = [(n, None) for n in top] + [(d + '/', sub) for d, sub in sorted(dirs.items())]
    lines = []
    for i, (name, sub) in enumerate(entries):
        last = i == len(entries) - 1
        lines.append(('└───' if last else '├───') + name)
        if sub:
            for j, s in enumerate(sub):
                pre = '    ' if last else '│   '
                lines.append(pre + ('└───' if j == len(sub) - 1 else '├───') + s)
    return '\n'.join(lines)


def session_message(cwd, files, date='Saturday, October 31, 2026', user='yamin'):
    proj = cwd.rstrip('/').split('/')[-1]
    return SESSION_CONTEXT.format(date=date, user=user, proj=proj, cwd=cwd, tree=render_tree(files))


def tool_result_content(output):
    """How the CLI's adapter serialises a functionResponse into an Ollama tool message."""
    return json.dumps({'output': output})


MEMORY_BLOCK = """

---

<loaded_context>
<global_context>
--- Context from: {rel}/.gemini/GEMINI.md ---
## Gemini Added Memories
{facts}--- End of Context from: {rel}/.gemini/GEMINI.md ---
</global_context>
</loaded_context>"""


def system_prompt(cwd='/home/yamin/projects/daily', facts=()):
    depth = len([p for p in cwd.strip('/').split('/') if p])
    rel = '/'.join(['..'] * depth) + '/home/yamin'
    fact_text = ''.join(f'- {f}\n' for f in facts)
    return SYSTEM_PROMPT.rstrip('\n') + MEMORY_BLOCK.format(rel=rel, facts=fact_text)


def start_messages(cwd, files, prompt, facts=()):
    return [
        {'role': 'system', 'content': system_prompt(cwd, facts)},
        {'role': 'user', 'content': session_message(cwd, files)},
        {'role': 'user', 'content': prompt},
    ]


def _load_compact():
    import re
    src = Path('/home/yamin/Development/gemini-cli-local/packages/core/src/core/localToolSchemas.ts').read_text()
    return json.loads(src[src.index('> = ') + 4:].rstrip().rstrip(';'))


def compact_tools(tools):
    """Mirror of toOllamaTools() in localContentGenerator.ts."""
    compact = _load_compact()
    out = []
    for t in tools:
        f = t['function']
        c = compact.get(f['name'])
        if not c:
            out.append(t)
            continue
        params = dict(f['parameters'])
        props = {}
        for k, d in c['params'].items():
            if k in params.get('properties', {}):
                props[k] = {**params['properties'][k], 'description': d}
        params['properties'] = props
        params['required'] = [r for r in params.get('required', []) if r in props]
        out.append({'type': 'function', 'function': {'name': f['name'], 'description': c['description'], 'parameters': params}})
    return out


TOOLS = compact_tools(ALL_TOOLS)
