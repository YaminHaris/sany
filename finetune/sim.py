"""Simulated tool environment. Model-written commands are interpreted, never executed."""

import fnmatch
import posixpath
import re
import shlex


def split_pipes(cmd):
    """Split on | outside quotes (|| is left alone)."""
    parts, cur, q, i = [], '', None, 0
    while i < len(cmd):
        ch = cmd[i]
        if q:
            q = None if ch == q else q
        elif ch in '\'"':
            q = ch
        elif ch == '|' and cmd[i + 1:i + 2] != '|' and cmd[i - 1:i] != '|':
            parts.append(cur.strip()); cur = ''; i += 1; continue
        cur += ch; i += 1
    parts.append(cur.strip())
    return parts


class Sim:
    def __init__(self, cwd, files, web=None, search=None, downloads=None, programs=None, extra_shell=None):
        self.cwd = cwd.rstrip('/')
        self.files = dict(files)          # relative path -> content
        self.web = web or {}              # url -> page text (web_fetch)
        self.search = search or []        # list of (title, url, snippet)
        self.downloads = downloads or {}  # url -> content (curl/wget)
        self.programs = programs or {}    # 'python3 proj/x.py' style -> stdout
        self.extra_shell = extra_shell or []  # [(regex, output)]
        self.dirs = set()
        self.log = []

    # ---------- path helpers ----------
    def rel(self, p):
        p = p.strip().strip('"').strip("'")
        if p == '~':
            p = '/home/yamin'
        if p.startswith('~/'):
            p = '/home/yamin/' + p[2:]
        if p.startswith('/'):
            if p == self.cwd:
                return ''
            if p.startswith(self.cwd + '/'):
                p = p[len(self.cwd) + 1:]
            else:
                return None
        p = posixpath.normpath(p)
        return '' if p == '.' else p

    def is_dir(self, r):
        return r == '' or r in self.dirs or any(f.startswith(r + '/') for f in self.files)

    def listdir(self, r):
        prefix = '' if r == '' else r + '/'
        out = set()
        for f in list(self.files) + list(self.dirs):
            if f.startswith(prefix) and f != r:
                rest = f[len(prefix):]
                head = rest.split('/')[0]
                out.add(head + '/' if ('/' in rest or f in self.dirs and rest == head) else head)
        return sorted(out)

    # ---------- tools ----------
    def call(self, name, args):
        self.log.append((name, args))
        fn = getattr(self, 't_' + name, None)
        if fn is None:
            return {'error': f'Tool "{name}" not found in registry.'}
        try:
            return fn(**args)
        except TypeError as e:
            return {'error': f'Invalid parameters for {name}: {e}'}

    def t_read_file(self, file_path=None, start_line=None, end_line=None, **_):
        if not file_path:
            return {'error': "params must have required property 'file_path'"}
        r = self.rel(file_path)
        if r is None or r not in self.files:
            return {'error': f'File not found: {file_path}'}
        lines = self.files[r].split('\n')
        if start_line or end_line:
            s = max(1, int(start_line or 1))
            e = int(end_line or len(lines))
            if e < 1:
                return {'error': 'end_line must be at least 1'}
            return {'output': '\n'.join(lines[s - 1:e])}
        return {'output': self.files[r]}

    def t_write_file(self, file_path=None, content=None, **_):
        if not file_path or content is None:
            return {'error': "params must have required properties 'file_path' and 'content'"}
        r = self.rel(file_path)
        if r is None:
            return {'error': f'Path is outside the workspace: {file_path}'}
        existed = r in self.files
        self.files[r] = content
        return {'output': f'Successfully {"overwrote" if existed else "created and wrote to new"} file: {self.cwd}/{r}.'}

    def t_replace(self, file_path=None, old_string=None, new_string=None, instruction=None, allow_multiple=False, **_):
        if not file_path or old_string is None or new_string is None:
            return {'error': "params must have required properties 'file_path', 'old_string', 'new_string'"}
        r = self.rel(file_path)
        if r is None or r not in self.files:
            return {'error': f'File not found: {file_path}'}
        c = self.files[r]
        n = c.count(old_string)
        if old_string == '' or n == 0:
            return {'error': f'Failed to edit, 0 occurrences found for old_string in {file_path}. Ensure you are matching the exact text, including whitespace. Use read_file to verify.'}
        if n > 1 and not allow_multiple:
            return {'error': f'Failed to edit, expected 1 occurrence but found {n}.'}
        self.files[r] = c.replace(old_string, new_string)
        return {'output': f'Successfully modified file: {self.cwd}/{r} ({n} replacements).'}

    def t_list_directory(self, dir_path='.', **_):
        r = self.rel(dir_path or '.')
        if r is None or not self.is_dir(r):
            return {'error': f'Directory not found: {dir_path}'}
        items = self.listdir(r)
        return {'output': f'Directory listing for {self.cwd}/{r}:\n' + '\n'.join(('[DIR] ' + i[:-1]) if i.endswith('/') else i for i in items)}

    def t_glob(self, pattern=None, dir_path=None, **_):
        if not pattern:
            return {'error': "params must have required property 'pattern'"}
        base = self.rel(dir_path) if dir_path else ''
        pats = [pattern] + ([pattern[3:]] if pattern.startswith('**/') else [])
        hits = []
        for f in sorted(self.files):
            rf = f[len(base) + 1:] if base and f.startswith(base + '/') else (f if not base else None)
            if rf is None:
                continue
            if any(fnmatch.fnmatchcase(rf, p) for p in pats):
                hits.append(f'{self.cwd}/{f}')
        if not hits:
            return {'output': f'No files found matching pattern "{pattern}"'}
        return {'output': f'Found {len(hits)} file(s) matching "{pattern}":\n' + '\n'.join(hits)}

    def t_grep_search(self, pattern=None, dir_path=None, include_pattern=None, **_):
        if not pattern:
            return {'error': "params must have required property 'pattern'"}
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return {'error': f'Invalid regex: {e}'}
        out = []
        base = self.rel(dir_path).rstrip('/') + '/' if dir_path and self.rel(dir_path) not in ('', '.') else ''
        for f in sorted(self.files):
            if base and not f.startswith(base):
                continue
            if include_pattern and not fnmatch.fnmatch(posixpath.basename(f), include_pattern.replace('**/', '')):
                continue
            for i, line in enumerate(self.files[f].split('\n'), 1):
                if rx.search(line):
                    out.append(f'{f}:{i}: {line}')
        if not out:
            return {'output': f'No matches found for pattern "{pattern}".'}
        return {'output': f'Found {len(out)} matches:\n' + '\n'.join(out)}

    def t_web_fetch(self, prompt=None, **_):
        if not prompt:
            return {'error': "params must have required property 'prompt'"}
        m = re.search(r'https?://[^\s)"\']+', prompt)
        if not m:
            return {'error': 'The prompt must contain at least one valid URL (starting with http:// or https://).'}
        url = m.group(0).rstrip('.,')
        for u, text in self.web.items():
            if url.rstrip('/') == u.rstrip('/'):
                return {'output': text}
        return {'output': f'Request failed with status 404\nResponse: Not Found ({url})'}

    def t_google_web_search(self, query=None, **_):
        if not query:
            return {'error': "params must have required property 'query'"}
        if not self.search:
            return {'output': f'No search results found for query: "{query}"'}
        lines = [f'[{i + 1}] {t}\n    {u}\n    {s}' for i, (t, u, s) in enumerate(self.search)]
        return {'output': f'Web search results for "{query}":\n\n' + '\n\n'.join(lines) + '\n\nUse web_fetch on a result URL to read the full page.'}

    def t_run_shell_command(self, command=None, description=None, dir_path=None, is_background=False, **_):
        if not command:
            return {'error': "params must have required property 'command'"}
        out = self.shell(command)
        if out is None:
            return {'output': f'Output: bash: simulated environment could not interpret: {command}\nExit Code: 127'}
        return {'output': 'Output: ' + (out if out.strip() else '(empty)')}

    # ---------- tiny shell ----------
    def shell(self, command):
        self._quiet = False
        segs = split_pipes(command)
        for rx, out in self.extra_shell:
            if len(segs) > 1 and re.search(rx, segs[0]) and not re.search(rx, '|'.join(segs[1:])):
                data = out  # canned program output, then apply the rest of the pipeline
                for seg in segs[1:]:
                    data = self.filter(seg, data)
                    if data is None:
                        return None
                return data
            if re.search(rx, command):
                return out
        outs = []
        for part in re.split(r'\s*(?:&&|;|\n)\s*', command.strip()):
            if not part:
                continue
            o = self.shell_one(part)
            if o is None:
                return None
            outs.append(o)
        return '\n'.join(o for o in outs if o)

    def expand(self, pat):
        r = self.rel(pat)
        if r is None:
            return []
        return sorted(f for f in self.files if fnmatch.fnmatchcase(f, r))

    def shell_one(self, cmd):
        cmd = re.sub(r'^cd\s+\S+\s*$', '', cmd)
        if not cmd:
            return ''
        quiet = bool(re.search(r'2>\s*/dev/null|2>&1', cmd))
        cmd = re.sub(r'\s*2>\s*/dev/null|\s*2>&1', '', cmd)
        redirect = re.search(r'\s*(>>?)\s*([^\s|&;>]+)\s*$', cmd)
        if redirect and not cmd.lstrip().startswith(('python', 'node')):
            out = self.shell_one(cmd[:redirect.start()])
            if out is None:
                return None
            r = self.rel(redirect.group(2))
            text = out + ('\n' if out and not out.endswith('\n') else '')
            self.files[r] = (self.files.get(r, '') + text) if redirect.group(1) == '>>' else text
            return ''
        if quiet:
            self._quiet = True
        for prog, out in self.programs.items():
            if cmd.strip() == prog or re.fullmatch(r'(python3?|bash|sh|node)\s+(\./)?' + re.escape(prog.split()[-1]), cmd.strip()):
                return out
        pipes = split_pipes(cmd)
        if len(pipes) > 1:
            first = self.shell_one(pipes[0])
            if first is None:
                return None
            data = first
            for p in pipes[1:]:
                data = self.filter(p, data)
                if data is None:
                    return None
            return data
        try:
            argv = shlex.split(cmd)
        except ValueError:
            return None
        if not argv:
            return ''
        c, a = argv[0], argv[1:]
        flags = [x for x in a if x.startswith('-')]
        args = [x for x in a if not x.startswith('-')]
        if c == 'mkdir':
            for d in args:
                r = self.rel(d)
                if r:
                    self.dirs.add(r)
            return ''
        if c == 'chmod':
            return ''
        if c in ('ls', 'dir'):
            if not args:
                return '\n'.join(i.rstrip('/') for i in self.listdir(''))
            res = []
            for x in args:
                if any(ch in x for ch in '*?['):
                    hits = self.expand(x)
                    if not hits and not getattr(self, '_quiet', False):
                        return f"ls: cannot access '{x}': No such file or directory"
                    res += hits
                else:
                    r = self.rel(x)
                    if r in self.files:
                        res.append(r)
                    elif r is not None and self.is_dir(r):
                        res += [i.rstrip('/') for i in self.listdir(r)]
                    else:
                        return f"ls: cannot access '{x}': No such file or directory"
            return '\n'.join(res)
        if c == 'cat':
            res = []
            for x in args:
                fs = self.expand(x) if any(ch in x for ch in '*?') else [self.rel(x)]
                for r in fs:
                    if r not in self.files:
                        return f'cat: {x}: No such file or directory'
                    res.append(self.files[r])
            return '\n'.join(res)
        if c == 'wc':
            res, total = [], 0
            for x in args:
                for r in (self.expand(x) if any(ch in x for ch in '*?') else [self.rel(x)]):
                    if r not in self.files:
                        return f'wc: {x}: No such file or directory'
                    n = self.files[r].count('\n')
                    total += n
                    res.append(f'{n} {r}')
            if len(res) > 1:
                res.append(f'{total} total')
            return '\n'.join(res)
        if c in ('curl', 'wget'):
            url = next((x for x in args if x.startswith('http')), None)
            if not url or url not in self.downloads:
                return f'{c}: (22) The requested URL returned error: 404'
            out_path = None
            for i, x in enumerate(a):
                if x in ('-o', '-O', '--output', '--output-document') and i + 1 < len(a) and not a[i + 1].startswith('http'):
                    out_path = a[i + 1]
                elif re.fullmatch(r'-[a-zA-Z]*o', x) and i + 1 < len(a):
                    out_path = a[i + 1]
                elif x.startswith('--output=') or x.startswith('--output-document='):
                    out_path = x.split('=', 1)[1]
            if c == 'wget' and not out_path:
                prefix = next((a[i + 1] for i, x in enumerate(a) if x == '-P' and i + 1 < len(a)), '')
                out_path = posixpath.join(prefix, url.rsplit('/', 1)[-1])
            if c == 'curl' and '-O' in a and not out_path:
                out_path = url.rsplit('/', 1)[-1]
            if not out_path:
                return self.downloads[url]
            r = self.rel(out_path)
            parent = posixpath.dirname(r)
            if parent and not self.is_dir(parent) and not ('--create-dirs' in a or c == 'wget'):
                return f'curl: (23) Failure writing output to destination, passed 0 returned 0\nWarning: Failed to open the file {out_path}: No such file or directory'
            self.files[r] = self.downloads[url]
            return ''
        if c in ('mv', 'cp') and len(args) == 2:
            src, dst = self.rel(args[0]), self.rel(args[1])
            if src not in self.files:
                return f"{c}: cannot stat '{args[0]}': No such file or directory"
            if self.is_dir(dst) and dst not in self.files:
                dst = posixpath.join(dst, posixpath.basename(src))
            self.files[dst] = self.files[src]
            if c == 'mv':
                del self.files[src]
            return ''
        if c == 'rm':
            for x in args:
                for r in (self.expand(x) if any(ch in x for ch in '*?') else [self.rel(x)]):
                    self.files.pop(r, None)
            return ''
        if c == 'touch':
            for x in args:
                self.files.setdefault(self.rel(x), '')
            return ''
        if c == 'echo':
            return ' '.join(args)
        if c == 'pwd':
            return self.cwd
        if c == 'find':
            names, excludes, depth, ftype, base_arg = [], [], 99, None, None
            i = 0
            while i < len(a):
                x = a[i]
                if x in ('-name', '-iname') and i + 1 < len(a):
                    names.append((a[i + 1], x == '-iname')); i += 2; continue
                if x in ('-not', '!') and i + 2 < len(a) and a[i + 1] in ('-path', '-name', '-ipath'):
                    excludes.append(a[i + 2]); i += 3; continue
                if x == '-maxdepth' and i + 1 < len(a):
                    depth = int(a[i + 1]); i += 2; continue
                if x == '-type' and i + 1 < len(a):
                    ftype = a[i + 1]; i += 2; continue
                if x in ('-o', '-or', '(', ')', '-print'):
                    i += 1; continue
                if x.startswith('-'):
                    return None
                if base_arg is None:
                    base_arg = x
                i += 1
            base = self.rel(base_arg) if base_arg else ''
            if base is None:
                return f'find: \'{base_arg}\': No such file or directory'
            if base_arg in (None, '.'):
                shown = lambda f: './' + f
            else:
                root = self.cwd if base == '' else self.cwd + '/' + base
                shown = lambda f: root + '/' + (f[len(base) + 1:] if base else f)
            def ok(f):
                rel_f = f[len(base) + 1:] if base else f
                if base and not f.startswith(base + '/'):
                    return False
                if rel_f.count('/') >= depth:
                    return False
                bn = posixpath.basename(f)
                if names and not any(fnmatch.fnmatch(bn.lower(), n.lower()) if ci else fnmatch.fnmatchcase(bn, n) for n, ci in names):
                    return False
                full = './' + rel_f
                return not any(fnmatch.fnmatch(full, e) or fnmatch.fnmatch('/' + f, e) for e in excludes)
            dirs = {posixpath.dirname(f) for f in self.files} | set(getattr(self, 'dirs', ()))
            dirs = {d for x in dirs for d in [x] + [x.rsplit('/', k)[0] for k in range(1, x.count('/') + 1)] if d}
            cands = (set() if ftype == 'd' else set(self.files)) | (set() if ftype == 'f' else dirs)
            return '\n'.join(shown(f) for f in sorted(cands) if ok(f))
        if c in ('head', 'tail'):
            n = 10
            for i, x in enumerate(a):
                if x == '-n' and i + 1 < len(a):
                    n = int(a[i + 1])
                elif re.fullmatch(r'-\d+', x):
                    n = int(x[1:])
            files = [x for x in args if not x.isdigit()]
            if not files:
                return None
            r = self.rel(files[0])
            if r not in self.files:
                return f'{c}: cannot open {files[0]}'
            lines = self.files[r].split('\n')
            return '\n'.join(lines[:n] if c == 'head' else lines[-n:])
        if c == 'grep':
            pat = args[0] if args else ''
            res = []
            targets = args[1:] or []
            if '-r' in flags or '-rn' in flags or '-R' in flags or not targets:
                targets_files = sorted(self.files)
            else:
                targets_files = [r for x in targets for r in (self.expand(x) if any(ch in x for ch in '*?') else [self.rel(x)])]
            for r in targets_files:
                for i, line in enumerate(self.files.get(r, '').split('\n'), 1):
                    if re.search(pat, line, re.I if '-i' in flags else 0):
                        res.append(f'{r}:{i}:{line}' if len(targets_files) > 1 or '-n' in ''.join(flags) else line)
            if '-c' in flags:
                return str(len(res))
            return '\n'.join(res)
        return None

    def filter(self, p, data):
        lines = [l for l in data.split('\n') if l != ''] if data else []
        try:
            argv = shlex.split(p)
        except ValueError:
            return None
        c, a = argv[0], argv[1:]
        if c == 'wc' and '-l' in a:
            return str(len(lines))
        if c == 'grep':
            pat = [x for x in a if not x.startswith('-')]
            fl = ''.join(x[1:] for x in a if x.startswith('-') and not x.startswith('--'))
            rx = re.compile(pat[0] if pat else '', re.I if 'i' in fl else 0)
            keep = [l for l in lines if bool(rx.search(l)) != ('v' in fl)]
            return str(len(keep)) if 'c' in fl else '\n'.join(keep)
        if c == 'head':
            n = int(a[a.index('-n') + 1]) if '-n' in a else 10
            return '\n'.join(lines[:n])
        if c == 'tail':
            n = int(a[a.index('-n') + 1]) if '-n' in a else 10
            return '\n'.join(lines[-n:])
        if c == 'sort':
            return '\n'.join(sorted(lines, reverse='-r' in a))
        return None
