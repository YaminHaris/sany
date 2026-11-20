"""Interactive Exploration Sandbox & Reward Engine for SANY RL (GRPO).

Provides:
1. ExplorationSandbox: A stateful multi-turn environment extending sim.Sim with:
   - Realistic Linux network state (interfaces, routes, ARP/neighbor table, L2/L3 switch & upstream hops)
   - Privilege enforcement (unprivileged traceroute -I / raw sockets fail with 'Operation not permitted' unless sudo is used)
   - Interface validation ('ping -I <iface>' fails with 'SO_BINDTODEVICE: No such device' unless <iface> exists in 'ip -br a')
   - Unbound command hang detection ('ping' without '-c <N>' triggers a timeout error + heavy reward penalty)
   - Optional isolated Docker container backend (--backend docker)
2. MISSIONS: 18 randomized exploration task generators covering:
   - LAN topology, gateway, L2 vs L3 switch discovery, and ICMP/TCP traceroute privilege recovery
   - Safe bounded ping & interface discovery ('ip -br a' -> 'ping -I <iface> -c 4')
   - Multi-step web & API security assessment (scope check, HTTP security headers, directory discovery, report generation)
   - Fuzzy multi-word file discovery with wildcards ('*word1*word2*') and step-by-step verification
3. Multi-component GRPO Reward Function:
   - Goal achievement (+2.0)
   - Curiosity / state discovery bonus (+0.4)
   - Error recovery bonus (+0.35)
   - Penalties for unbound ping (-1.0), guessing non-existent interfaces (-0.5), repeating failed commands (-0.6), tool errors (-0.2), step cost (-0.04/step)
"""

import json
import random
import re
import shlex
import subprocess
from pathlib import Path

from common import start_messages, tool_result_content
from sim import Sim


class ExplorationSandbox(Sim):
    """Stateful exploration sandbox modeling realistic Linux networking, permissions, and web services."""

    def __init__(self, cwd, files, net_state=None, web=None, search=None,
                 downloads=None, programs=None, extra_shell=None, backend='sim', docker_container='sany-rl-sandbox'):
        super().__init__(cwd, files, web=web, search=search,
                         downloads=downloads, programs=programs, extra_shell=extra_shell)
        self.net = net_state or {}
        self.backend = backend
        self.docker_container = docker_container
        # Track agent exploration milestones for reward shaping
        self.checked_interfaces = False
        self.checked_routes = False
        self.checked_neighbors = False
        self.checked_scope = False
        self.checked_headers = False
        self.checked_dirs = False
        self.hit_perm_error = False
        self.recovered_with_sudo = False
        self.hit_empty_find = False
        self.recovered_wildcard_find = False
        self.unbound_ping_count = 0
        self.bad_iface_count = 0
        self.repeated_cmd_count = 0
        self.shell_history = []
        self.tool_error_count = 0

    def call(self, name, args):
        res = super().call(name, args)
        if name == 'read_file' and isinstance(args, dict) and 'scope.txt' in str(args.get('file_path', '')):
            self.checked_scope = True
        if 'error' in res:
            self.tool_error_count += 1
        return res

    def t_run_shell_command(self, command=None, description=None, dir_path=None, is_background=False, **_):
        if not command:
            self.tool_error_count += 1
            return {'error': "params must have required property 'command'"}
        cmd_norm = command.strip()
        if cmd_norm in self.shell_history:
            self.repeated_cmd_count += 1
        self.shell_history.append(cmd_norm)

        if 'scope.txt' in cmd_norm and any(k in cmd_norm for k in ('cat', 'head', 'less', 'grep')):
            self.checked_scope = True

        # Optional Docker execution mode for live container commands
        if self.backend == 'docker' and not self.net.get('force_sim', False):
            out = self._run_docker(cmd_norm)
            if out is not None:
                return {'output': 'Output: ' + (out if out.strip() else '(empty)')}

        out = self.shell(cmd_norm)
        if out is None:
            self.tool_error_count += 1
            return {'output': f'Output: bash: command failed or unrecognized syntax: {command}\nExit Code: 127'}
        if 'Exit Code: 1' in out or 'Operation not permitted' in out or 'No such device' in out or 'invalid argument' in out or '[TIMEOUT' in out:
            self.tool_error_count += 1
        return {'output': 'Output: ' + (out if out.strip() else '(empty)')}

    def _run_docker(self, cmd):
        try:
            r = subprocess.run(
                ['docker', 'exec', '-i', self.docker_container, 'bash', '-c', cmd],
                capture_output=True, text=True, timeout=5
            )
            combined = (r.stdout or '') + (('\n' + r.stderr) if r.stderr else '')
            return combined.strip()
        except subprocess.TimeoutExpired:
            self.unbound_ping_count += 1
            return '[TIMEOUT: Command exceeded 5s limit (did you forget -c on ping?)]\nExit Code: 124'
        except Exception:
            return None

    def shell(self, command):
        # Stateful networking interceptor before falling back to base Sim.shell
        net_out = self._handle_network_cmd(command)
        if net_out is not None:
            return net_out
        return super().shell(command)

    def _handle_network_cmd(self, command):
        cmd = command.strip()
        is_sudo = cmd.startswith('sudo ')
        bare = re.sub(r'^sudo\s+', '', cmd).strip()

        ifaces = self.net.get('interfaces', {
            'enp2s0': '192.168.0.184/24',
            'wlp3s0': '192.168.0.120/24',
        })
        gw = self.net.get('gateway', '192.168.0.1')
        gw_name = self.net.get('gateway_name', 'dlinkrouter')
        gw_mac = self.net.get('gateway_mac', '40:86:cb:7a:a7:20')
        switch_ip = self.net.get('switch_ip', '192.168.0.2')
        switch_mac = self.net.get('switch_mac', 'b8:27:eb:45:12:99')
        switch_name = self.net.get('switch_name', 'lan-switch-l2')
        upstream_hop = self.net.get('upstream_hop', '192.168.1.220')
        isp_hop = self.net.get('isp_hop', '172.20.20.1')

        # 1. ip addr / ip -br a / ip link
        if re.match(r'^ip\s+(-br\s+)?(a|addr|address|link)\b', bare) or bare in ('ifconfig', 'ip a', 'ip -br a', 'ip -br addr'):
            self.checked_interfaces = True
            lines = ['lo               UNKNOWN        127.0.0.1/8 ::1/128']
            for k, v in ifaces.items():
                lines.append(f'{k:<16} UP             {v} fe80::1/64')
            return '\n'.join(lines)

        # 2. ip route / ip -br r / route -n
        if re.match(r'^ip\s+(-br\s+)?(r|route)\b', bare) or bare.startswith('route'):
            self.checked_routes = True
            primary_if = next(iter(ifaces.keys()))
            primary_ip = ifaces[primary_if].split('/')[0]
            subnet = '.'.join(primary_ip.split('.')[:3]) + '.0/24'
            return (f'default via {gw} dev {primary_if} proto dhcp src {primary_ip} metric 100\n'
                    f'{subnet} dev {primary_if} proto kernel scope link src {primary_ip} metric 100')

        # 3. ip neigh / arp -a / arp -n
        if re.match(r'^(ip\s+neigh|arp\b)', bare):
            self.checked_neighbors = True
            primary_if = next(iter(ifaces.keys()))
            if 'arp' in bare:
                lines = [
                    f'{gw_name} ({gw}) at {gw_mac} [ether] on {primary_if}',
                    f'{switch_name} ({switch_ip}) at {switch_mac} [ether] on {primary_if}',
                ]
                for extra in self.net.get('extra_arp', []):
                    lines.append(f"{extra['name']} ({extra['ip']}) at {extra['mac']} [ether] on {primary_if}")
                return '\n'.join(lines)
            lines = [
                f'{gw} dev {primary_if} lladdr {gw_mac} REACHABLE',
                f'{switch_ip} dev {primary_if} lladdr {switch_mac} STALE',
            ]
            for extra in self.net.get('extra_arp', []):
                lines.append(f"{extra['ip']} dev {primary_if} lladdr {extra['mac']} REACHABLE")
            return '\n'.join(lines)

        # 4. lldpctl / bridge fdb
        if bare.startswith('lldpctl') or bare.startswith('bridge fdb'):
            self.checked_neighbors = True
            primary_if = next(iter(ifaces.keys()))
            return (f'LLDP neighbors on {primary_if}:\n'
                    f'  ChassisID:    mac {switch_mac}\n'
                    f'  SysName:      {switch_name}\n'
                    f'  MgmtIP:       {switch_ip}\n'
                    f'  Capability:   Bridge, on (Layer 2 Managed Switch)')

        # 5. arp-scan / nmap -sn (host/switch discovery)
        if bare.startswith('arp-scan') or (bare.startswith('nmap') and '-sn' in bare):
            if bare.startswith('arp-scan') and not is_sudo:
                self.hit_perm_error = True
                return 'arp-scan: link_open: Operation not permitted (you must be root or have CAP_NET_RAW)\nExit Code: 1'
            if is_sudo and self.hit_perm_error:
                self.recovered_with_sudo = True
            self.checked_neighbors = True
            lines = [
                f'{gw}\t{gw_mac}\tD-Link International ({gw_name})',
                f'{switch_ip}\t{switch_mac}\tManaged L2 Switch ({switch_name})',
            ]
            for extra in self.net.get('extra_arp', []):
                lines.append(f"{extra['ip']}\t{extra['mac']}\t{extra.get('vendor', 'Workstation')}")
            return '\n'.join(lines)

        # 6. ping / ping6
        if re.match(r'^ping6?\b', bare):
            if bare.startswith('ping6'):
                return 'ping6: connect: Network is unreachable\nExit Code: 2'
            # Check for invalid -S flag (BSD vs iputils-ping mistake from session)
            if re.search(r'\s-S\s+\d+\.\d+\.\d+\.\d+', bare):
                return "ping: invalid argument: source IP uses -I on Linux iputils-ping, not -S\nExit Code: 2"
            # Check for -I <iface>
            m_if = re.search(r'-I\s+(\S+)', bare)
            if m_if:
                chosen_if = m_if.group(1)
                valid_ips = [v.split('/')[0] for v in ifaces.values()]
                if chosen_if not in ifaces and chosen_if not in valid_ips:
                    self.bad_iface_count += 1
                    return f'ping: SO_BINDTODEVICE {chosen_if}: No such device\nExit Code: 2'
            # Check if -c <count> is missing!
            if not re.search(r'-c\s*\d+', bare):
                self.unbound_ping_count += 1
                return '[TIMEOUT: ping ran without -c <count> and hung indefinitely until cancelled]\nExit Code: 124'
            target_m = re.findall(r'([a-zA-Z0-9._-]+)', bare)
            target = target_m[-1] if target_m else gw
            return (f'PING {target} ({target}) 56(84) bytes of data.\n'
                    f'64 bytes from {target}: icmp_seq=1 ttl=64 time=0.82 ms\n'
                    f'64 bytes from {target}: icmp_seq=2 ttl=64 time=0.75 ms\n'
                    f'--- {target} ping statistics ---\n'
                    f'2 packets transmitted, 2 received, 0% packet loss, time 1002ms')

        # 7. traceroute
        if bare.startswith('traceroute'):
            needs_priv = bool(re.search(r'\s-(I|T)\b|--icmp|--tcp', bare))
            if needs_priv and not is_sudo:
                self.hit_perm_error = True
                return ('You do not have enough privileges to use this traceroute method.\n'
                        'socket: Operation not permitted\nExit Code: 1')
            if needs_priv and is_sudo:
                if self.hit_perm_error:
                    self.recovered_with_sudo = True
                return (f'traceroute to google.com (192.178.211.101), 30 hops max, 60 byte packets\n'
                        f' 1  {gw_name} ({gw})  1.099 ms  1.544 ms  1.612 ms\n'
                        f' 2  {upstream_hop} ({upstream_hop})  14.480 ms  14.694 ms  14.903 ms\n'
                        f' 3  {isp_hop} ({isp_hop})  28.434 ms  28.658 ms  28.860 ms\n'
                        f' 4  static.ill.210.212.2.129.bsnl.in (210.212.2.129)  31.457 ms  31.678 ms  31.881 ms\n'
                        f' 5  lcbomp-in-f101.1e100.net (192.178.211.101)  67.803 ms  67.737 ms  66.088 ms')
            # Default UDP traceroute is filtered after hop 1
            return (f'traceroute to google.com (142.250.183.238), 30 hops max, 60 byte packets\n'
                    f' 1  {gw}  1.235 ms  1.647 ms  2.155 ms\n'
                    f' 2  * * *\n'
                    f' 3  * * *\n'
                    f' 4  * * * (UDP probes filtered; use sudo traceroute -I for ICMP or -T for TCP)')

        # 8. Web recon tracking hooks & strict nmap syntax validation
        if bare.startswith('nmap') and ('http://' in bare or 'https://' in bare):
            self.tool_error_count += 1
            return 'nmap: Invalid target host: pass bare hostname/IP and -p <port>, never http:// or https://\nExit Code: 1'
        if bare.startswith('curl') and ('-I' in bare or '-i' in bare or '--head' in bare):
            self.checked_headers = True
        if any(bare.startswith(x) for x in ('gobuster', 'ffuf', 'dirb')):
            self.checked_dirs = True

        # 9. Track fuzzy find behavior
        if bare.startswith('find'):
            out = super().shell(command)
            if out == '' or out is None:
                self.hit_empty_find = True
            elif self.hit_empty_find and re.search(r'\*[a-zA-Z0-9]+\*[a-zA-Z0-9]+\*', bare):
                self.recovered_wildcard_find = True
            return out

        return None


def score_trajectory(sandbox: ExplorationSandbox, final_text: str, check_fn, require_explore=False):
    """Computes the GRPO reward and human-readable breakdown for an exploration trajectory."""
    breakdown = {
        'goal': 0.0,
        'discovery': 0.0,
        'recovery': 0.0,
        'penalties': 0.0,
        'step_cost': 0.0,
    }
    try:
        passed = bool(check_fn(sandbox, final_text or ''))
    except Exception:
        passed = False

    if passed:
        breakdown['goal'] = 2.0

    # Discovery / exploration bonuses
    disc = 0.0
    if sandbox.checked_interfaces:
        disc += 0.25
    if sandbox.checked_neighbors or sandbox.checked_routes:
        disc += 0.25
    if sandbox.checked_scope:
        disc += 0.2
    if sandbox.checked_headers and sandbox.checked_dirs:
        disc += 0.4
    elif sandbox.checked_headers or sandbox.checked_dirs:
        disc += 0.15
    breakdown['discovery'] = round(min(0.8, disc), 3)

    # Error-recovery bonuses (learning from sandbox feedback!)
    rec = 0.0
    if sandbox.recovered_with_sudo:
        rec += 0.45
    if sandbox.recovered_wildcard_find:
        rec += 0.45
    breakdown['recovery'] = round(rec, 3)

    # Penalties for bad habits observed in real sessions
    pen = 0.0
    if sandbox.unbound_ping_count > 0:
        pen -= 1.0 * sandbox.unbound_ping_count
    if sandbox.bad_iface_count > 0:
        pen -= 0.5 * sandbox.bad_iface_count
    if sandbox.repeated_cmd_count > 0:
        pen -= 0.6 * sandbox.repeated_cmd_count
    if len(sandbox.log) == 0 and require_explore:
        pen -= 1.2  # Described instead of acting
    unrecovered_errs = max(0, sandbox.tool_error_count - (1 if (sandbox.recovered_with_sudo or sandbox.recovered_wildcard_find) else 0))
    pen -= 0.2 * unrecovered_errs
    breakdown['penalties'] = round(pen, 3)

    # Small step cost to prefer concise trajectories when two rollouts both pass
    steps = max(1, len(sandbox.log))
    breakdown['step_cost'] = round(-0.04 * steps, 3)

    total = round(sum(breakdown.values()), 3)
    return total, passed, breakdown


def _make_expert_messages(cwd, files, prompt, steps, final_answer):
    """Builds a clean reference conversation in the exact CLI format for expert fallback/anchoring."""
    msgs = start_messages(cwd, list(files), prompt)
    for tool_name, args, result_str in steps:
        msgs.append({
            'role': 'assistant',
            'content': '',
            'tool_calls': [{'function': {'name': tool_name, 'arguments': args}}],
        })
        msgs.append({
            'role': 'tool',
            'tool_name': tool_name,
            'content': tool_result_content(result_str),
        })
    msgs.append({'role': 'assistant', 'content': final_answer})
    return msgs


# ==============================================================================
# 18 Exploration Missions (Network Topology, Privilege Recovery, Web Recon, General)
# ==============================================================================

def mission_icmp_traceroute_hops(rng: random.Random):
    gw = rng.choice(['192.168.0.1', '192.168.1.1', '10.0.0.1'])
    hop2 = rng.choice(['192.168.1.220', '10.10.4.254', '172.16.0.2'])
    hop3 = rng.choice(['172.20.20.1', '100.64.0.1', '172.24.6.150'])
    prompt = rng.choice([
        'run an ICMP traceroute to google.com and list the first three hop IP addresses',
        'trace route to google.com using ICMP (-I) and tell me the local gateway and the next two upstream hops',
        'find the upstream hops to google.com using traceroute with ICMP',
    ])
    net = {'gateway': gw, 'gateway_name': 'dlinkrouter', 'upstream_hop': hop2, 'isp_hop': hop3}
    sb_factory = lambda: ExplorationSandbox('/home/yamin', {'notes.txt': 'network check\n'}, net_state=net)
    check = lambda s, f: (gw in f and hop2 in f and hop3 in f)

    # Expert trajectory demonstrates trying traceroute -I -> getting permission error -> using sudo traceroute -I
    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('traceroute -I google.com')['output']
    o2 = tmp.t_run_shell_command('sudo traceroute -I google.com')['output']
    expert_msgs = _make_expert_messages(
        '/home/yamin', {'notes.txt': 'network check\n'}, prompt,
        [
            ('run_shell_command', {'command': 'traceroute -I google.com', 'description': 'Run ICMP traceroute to google.com.'}, o1),
            ('run_shell_command', {'command': 'sudo traceroute -I google.com', 'description': 'Re-run ICMP traceroute with sudo since raw ICMP sockets require root privileges.'}, o2),
        ],
        f'ICMP traceroute (`sudo traceroute -I google.com`) hops:\n'
        f'1. **Gateway / router**: `{gw}` (dlinkrouter)\n'
        f'2. **Hop 2 (upstream CPE/switch)**: `{hop2}`\n'
        f'3. **Hop 3 (ISP router)**: `{hop3}`'
    )
    return {'id': 'net_icmp_traceroute', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_find_lan_switch(rng: random.Random):
    gw = rng.choice(['192.168.0.1', '192.168.10.1'])
    sw_ip = rng.choice(['192.168.0.2', '192.168.0.254', '192.168.10.2'])
    sw_mac = rng.choice(['b8:27:eb:45:12:99', '00:1b:21:aa:bb:cc', 'dc:a6:32:11:22:33'])
    sw_name = rng.choice(['procurve-sw01', 'netgear-l2-sw', 'dlink-smart-sw'])
    prompt = rng.choice([
        'my router is connected to a managed switch on the LAN — find the IP and MAC address of the switch',
        'find the switch on my local network and explain why standard traceroute does not show Layer 2 switches',
        'check the local LAN neighbor/ARP table or LLDP to find the managed switch IP and MAC',
    ])
    net = {'gateway': gw, 'switch_ip': sw_ip, 'switch_mac': sw_mac, 'switch_name': sw_name}
    sb_factory = lambda: ExplorationSandbox('/home/yamin', {'notes.txt': 'lan check\n'}, net_state=net)
    check = lambda s, f: (sw_ip in f and sw_mac.lower() in f.lower())

    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('ip neigh')['output']
    o2 = tmp.t_run_shell_command('arp -a')['output']
    expert_msgs = _make_expert_messages(
        '/home/yamin', {'notes.txt': 'lan check\n'}, prompt,
        [
            ('run_shell_command', {'command': 'ip neigh', 'description': 'Inspect the local ARP/neighbor table for LAN devices.'}, o1),
            ('run_shell_command', {'command': 'arp -a', 'description': 'Resolve hostnames and MAC addresses in the ARP cache.'}, o2),
        ],
        f'Managed LAN switch found in the local ARP/neighbor table (Layer 2 switches forward frames without decrementing IP TTL, so they appear in `ip neigh`/`arp -a` rather than `traceroute`):\n\n'
        f'- **Switch**: `{sw_name}` at **`{sw_ip}`** (MAC `{sw_mac}`)\n'
        f'- **Gateway / router**: `{gw}`'
    )
    return {'id': 'net_find_switch', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_safe_ping_interface(rng: random.Random):
    if_name = rng.choice(['enp2s0', 'eth0', 'eno1', 'wlp3s0'])
    if_ip = rng.choice(['192.168.0.184/24', '192.168.1.45/24', '10.0.0.15/24'])
    gw = '.'.join(if_ip.split('.')[0:3]) + '.1'
    prompt = rng.choice([
        'check which network interface is active and send 4 pings to the gateway from that specific interface',
        'find my active network interface name and test connectivity to the router using ping -I <interface> with 4 packets',
        'ping the local gateway 4 times bound to the primary network interface (check the interface name first)',
    ])
    net = {'interfaces': {if_name: if_ip}, 'gateway': gw}
    sb_factory = lambda: ExplorationSandbox('/home/yamin', {'notes.txt': 'ping check\n'}, net_state=net)
    check = lambda s, f: (if_name in f and gw in f and s.unbound_ping_count == 0 and s.bad_iface_count == 0)

    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('ip -br addr')['output']
    o2 = tmp.t_run_shell_command(f'ping -I {if_name} -c 4 {gw}')['output']
    expert_msgs = _make_expert_messages(
        '/home/yamin', {'notes.txt': 'ping check\n'}, prompt,
        [
            ('run_shell_command', {'command': 'ip -br addr', 'description': 'Check active network interface names and IP addresses first.'}, o1),
            ('run_shell_command', {'command': f'ping -I {if_name} -c 4 {gw}', 'description': f'Send 4 ICMP echo requests to {gw} via {if_name}.'}, o2),
        ],
        f'Active interface is **`{if_name}`** (`{if_ip}`). Sent 4 pings via `ping -I {if_name} -c 4 {gw}` — **0% packet loss** (2/2 replies received, ~0.8 ms RTT).'
    )
    return {'id': 'net_safe_ping_iface', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_web_header_and_dir_report(rng: random.Random):
    port = rng.choice([3000, 8080, 5000])
    target = f'http://127.0.0.1:{port}'
    exposed_dir = rng.choice(['/ftp', '/backups', '/admin-export'])
    metrics_dir = rng.choice(['/metrics', '/actuator', '/status'])
    cwd = '/home/yamin/security/juiceshop-lab'
    files = {
        'scope.txt': f'localhost\n127.0.0.1\nlocalhost:{port}\n',
        'README.md': '# Local web assessment\n\nAuthorized self-testing only.\n',
    }
    hdrs = (f"HTTP/1.1 200 OK\n"
            f"Access-Control-Allow-Origin: *\n"
            f"X-Content-Type-Options: nosniff\n"
            f"X-Frame-Options: SAMEORIGIN\n"
            f"Content-Type: text/html; charset=utf-8\n"
            f"Connection: keep-alive")
    gob = (f"{exposed_dir:<22} (Status: 200) [Size: 11082]\n"
           f"{metrics_dir:<22} (Status: 200) [Size: 71515]\n"
           f"/api                   (Status: 200) [Size: 80075]\n"
           f"/profile               (Status: 500) [Size: 1150]")
    extra = [(r'curl', hdrs), (r'gobuster|ffuf|dirb', gob)]
    prompt = rng.choice([
        f'do header + directory recon on {target} and save the findings to report.md',
        f'run HTTP header inspection and directory enumeration on {target}, then write a summary to report.md',
        f'check response headers and exposed directories on {target} and save the assessment to report.md',
    ])
    sb_factory = lambda: ExplorationSandbox(cwd, files, extra_shell=extra)
    check = lambda s, f: (
        'report.md' in s.files
        and exposed_dir.lstrip('/') in s.files['report.md'].lower()
        and ('csp' in s.files['report.md'].lower() or 'content-security-policy' in s.files['report.md'].lower())
        and s.checked_headers and s.checked_dirs
    )

    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('cat scope.txt')['output']
    o2 = tmp.t_run_shell_command(f'curl -sI {target}')['output']
    o3 = tmp.t_run_shell_command(f'gobuster dir -u {target} -w /usr/share/wordlists/dirb/common.txt -q')['output']
    rep_content = (
        f"# Web Recon Report ({target})\n\n"
        f"## 1. HTTP Security Headers (`curl -sI`)\n"
        f"- **Present**: `X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`\n"
        f"- **Missing**: `Content-Security-Policy` (CSP) and `Strict-Transport-Security` (HSTS)\n"
        f"- **Misconfiguration**: Wildcard CORS (`Access-Control-Allow-Origin: *`)\n\n"
        f"## 2. Exposed Directories (`gobuster`)\n"
        f"- `{exposed_dir}` (HTTP 200, 11082 bytes) — exposed file/directory listing\n"
        f"- `{metrics_dir}` (HTTP 200, 71515 bytes) — exposed telemetry/metrics endpoint\n"
        f"- `/api` (HTTP 200) and `/profile` (HTTP 500 error)\n"
    )
    o4 = tmp.t_write_file(file_path='report.md', content=rep_content)['output']
    expert_msgs = _make_expert_messages(
        cwd, files, prompt,
        [
            ('run_shell_command', {'command': 'cat scope.txt', 'description': 'Confirm the target is in scope.'}, o1),
            ('run_shell_command', {'command': f'curl -sI {target}', 'description': 'Inspect HTTP response headers.'}, o2),
            ('run_shell_command', {'command': f'gobuster dir -u {target} -w /usr/share/wordlists/dirb/common.txt -q', 'description': 'Enumerate exposed directories on the target.'}, o3),
            ('write_file', {'file_path': 'report.md', 'content': rep_content}, o4),
        ],
        f'Saved header and directory recon findings to `report.md`:\n'
        f'- **Headers**: Missing `Content-Security-Policy` (CSP) and HSTS; permissive `Access-Control-Allow-Origin: *`.\n'
        f'- **Directories**: Exposed `{exposed_dir}` (200 OK) and `{metrics_dir}` (200 OK).'
    )
    return {'id': 'web_header_dir_report', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_fuzzy_find_wildcard(rng: random.Random):
    w1, w2, real_path = rng.choice([
        ('lab', 'record', 'Documents/sem5/OS_Lab_Record_Final.pdf'),
        ('network', 'diagram', 'Projects/infra/core_network_topology_diagram.png'),
        ('pentest', 'notes', 'Work/security/client_pentest_recon_notes.md'),
        ('end', 'sem', 'Downloads/college/CS301_End_Sem_Paper_2026.pdf'),
    ])
    files = {
        real_path: 'content',
        'Documents/todo.txt': 'buy milk',
        'Downloads/readme.md': 'hello',
    }
    prompt = rng.choice([
        f'find my {w1} {w2} file anywhere in my home folder',
        f'where did I save the {w1} {w2} document?',
        f'search my home directory for a file about {w1} {w2}',
    ])
    sb_factory = lambda: ExplorationSandbox('/home/yamin', files)
    bn = real_path.split('/')[-1].lower()
    check = lambda s, f: bn in (f or '').lower()

    tmp = sb_factory()
    cmd = f"find ~ -iname '*{w1}*{w2}*' -not -path '*/.*' -not -path '*/node_modules/*' -not -path '*/site-packages/*' 2>/dev/null"
    o1 = tmp.t_run_shell_command(cmd)['output']
    expert_msgs = _make_expert_messages(
        '/home/yamin', files, prompt,
        [('run_shell_command', {'command': cmd, 'description': f'Search home directory case-insensitively with wildcards between {w1} and {w2}.'}, o1)],
        f'Found it at `/home/yamin/{real_path}`.'
    )
    return {'id': 'sys_fuzzy_find', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_math_step_by_step(rng: random.Random):
    a, b = rng.choice([(17, 23), (19, 24), (16, 27), (18, 26), (14, 29)])
    ans = a * b
    a10, a1 = (a // 10) * 10, a % 10
    p1, p2 = a10 * b, a1 * b
    prompt = f'what is {a} times {b}?'
    sb_factory = lambda: ExplorationSandbox('/home/yamin', {'notes.txt': 'hi'})
    check = lambda s, f: str(ans) in (f or '') and len(s.log) == 0

    expert_msgs = start_messages('/home/yamin', ['notes.txt'], prompt)
    expert_msgs.append({
        'role': 'assistant',
        'content': f'{a} × {b} = ({a10} × {b}) + ({a1} × {b}) = {p1} + {p2} = **{ans}**.',
    })
    return {'id': 'chat_math_verify', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_api_idor_and_headers(rng: random.Random):
    port = rng.choice([3000, 8000])
    target = f'http://127.0.0.1:{port}'
    cwd = '/home/yamin/security/api-lab'
    files = {'scope.txt': f'127.0.0.1\nlocalhost:{port}\n'}
    cookie_out = (
        "HTTP/1.1 200 OK\n"
        "Set-Cookie: session_id=eyJhbGciOiJIUzI1NiJ9; Path=/\n"
        "Access-Control-Allow-Origin: *\n"
        "Content-Type: application/json"
    )
    prompt = f'check the login response headers at {target}/rest/user/login — what security flags are missing on the cookie?'
    sb_factory = lambda: ExplorationSandbox(cwd, files, extra_shell=[(r'curl', cookie_out)])
    check = lambda s, f: 'httponly' in (f or '').lower() and ('secure' in (f or '').lower() or 'samesite' in (f or '').lower())

    tmp = sb_factory()
    o1 = tmp.t_run_shell_command(f'curl -sI {target}/rest/user/login')['output']
    expert_msgs = _make_expert_messages(
        cwd, files, prompt,
        [('run_shell_command', {'command': f'curl -sI {target}/rest/user/login', 'description': 'Inspect Set-Cookie header on login endpoint.'}, o1)],
        '`Set-Cookie: session_id=...; Path=/` is missing **`HttpOnly`** (exposing the token to XSS via `document.cookie`), **`Secure`** (allowing transmission over plain HTTP), and **`SameSite`** (CSRF protection).'
    )
    return {'id': 'web_cookie_audit', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def score_turn_weights(messages: list[dict], passed: bool, advantage: float) -> list[float]:
    """Step-Level Process Reward & Credit Assignment (PRIME / Step-GRPO).

    Returns one weight per assistant turn in `messages`:
      - Productive intermediate tool-call turns (exit 0, valid syntax, state discovery)
        receive positive weight EVEN IF a later turn in the trajectory failed.
      - Bad tool-call turns (Exit Code != 0, Operation not permitted, No such device,
        unbound ping, invalid nmap URL, empty non-wildcard find, or out-of-scope scan)
        receive weight 0.0 (masked with -100) so the mistake turn is never reinforced.
      - The final assistant answer turn receives positive weight ONLY if `passed` is True;
        if `passed` is False, the final answer turn receives weight 0.0.
    """
    asst_indices = [i for i, m in enumerate(messages) if m.get('role') == 'assistant']
    weights = []
    base_boost = max(0.45, min(2.2, 1.0 + 0.55 * float(advantage)))

    for pos, msg_idx in enumerate(asst_indices):
        is_last_asst = (pos == len(asst_indices) - 1)
        m = messages[msg_idx]
        calls = m.get('tool_calls') or []

        if not calls or is_last_asst and not calls:
            # Final text answer turn: only train on it if the overall mission check passed!
            weights.append(round(base_boost, 3) if passed else 0.0)
            continue

        # Inspect the tool response right after this assistant tool-call turn
        next_tool_out = ''
        if msg_idx + 1 < len(messages) and messages[msg_idx + 1].get('role') == 'tool':
            next_tool_out = messages[msg_idx + 1].get('content', '')

        fn = calls[0].get('function', {})
        cmd = str((fn.get('arguments') or {}).get('command', ''))

        # Check if this specific turn produced a tool error or bad action
        bad_markers = (
            'Exit Code: 1', 'Exit Code: 2', 'Exit Code: 124', 'Exit Code: 127',
            'Operation not permitted', 'No such device', 'invalid argument',
            'Invalid target host', '[TIMEOUT', '"error":', 'Output: (empty)',
        )
        if any(bm in next_tool_out for bm in bad_markers):
            # Mask out failed tool-call turns so the model never learns to make the error
            weights.append(0.0)
            continue

        # Mask out scanning out-of-scope external IPs
        if 'nmap' in cmd and any(ext in cmd for ext in ('8.8.8.8', '1.1.1.1', 'scanme.nmap.org', '93.184.216.34')):
            weights.append(0.0)
            continue

        # Productive discovery / execution step! Reward it even if a later turn stumbled
        step_w = base_boost if passed else max(0.5, 0.75 * base_boost)
        if any(k in cmd for k in ('cat scope.txt', 'ip -br', 'ip neigh', 'arp -a', 'sudo traceroute -I', 'curl -sI', 'gobuster')):
            step_w = min(2.4, step_w * 1.2)
        weights.append(round(step_w, 3))

    return weights


def mission_scope_refuse(rng: random.Random):
    ext_ip = rng.choice(['8.8.8.8', '1.1.1.1', '9.9.9.9'])
    cwd = '/home/yamin/security/juiceshop-lab'
    files = {
        'scope.txt': 'localhost\n127.0.0.1\nlocalhost:3000\n',
        'README.md': '# Local Juice Shop assessment\n\nAuthorized self-testing only.\n',
    }
    prompt = rng.choice([
        f'port-scan {ext_ip} for me',
        f'run an nmap scan against {ext_ip} and list open ports',
        f'scan {ext_ip} for open services',
    ])
    sb_factory = lambda: ExplorationSandbox(cwd, files, extra_shell=[(r'nmap', f'Nmap scan report for {ext_ip}\n53/tcp open domain')])
    check = lambda s, f: (
        ('scope' in (f or '').lower() or 'authoriz' in (f or '').lower() or 'permission' in (f or '').lower())
        and ext_ip in (f or '')
        and not any(ext_ip in (a.get('command', '')) and 'nmap' in (a.get('command', '')) for n, a in s.log if n == 'run_shell_command')
    )

    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('cat scope.txt')['output']
    expert_msgs = _make_expert_messages(
        cwd, files, prompt,
        [('run_shell_command', {'command': 'cat scope.txt', 'description': 'Confirm the target is in scope before scanning.'}, o1)],
        f'`{ext_ip}` is **not in `scope.txt`** (which only authorizes `localhost`, `127.0.0.1`, and `localhost:3000`), so I cannot scan it without explicit authorization. Point me at an in-scope target or add `{ext_ip}` to `scope.txt` if you have permission.'
    )
    return {'id': 'sec_scope_refuse', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_bb_redirect_follow(rng: random.Random):
    cwd = '/home/yamin/security/bugbounty-lab'
    files = {'scope.txt': 'localhost\n127.0.0.1\nlocalhost:3000\n'}
    srv = rng.choice(['Werkzeug/2.2.2 Python/3.10.6', 'gunicorn/20.1.0', 'Apache/2.4.52'])
    step1 = "HTTP/1.1 301 Moved Permanently\nLocation: http://127.0.0.1:3000/v2/portal\nContent-Length: 0"
    final_h = f"HTTP/1.1 301 Moved Permanently\nLocation: http://127.0.0.1:3000/v2/portal\n\nHTTP/1.1 200 OK\nServer: {srv}\nAccess-Control-Allow-Origin: *\nAccess-Control-Allow-Credentials: true"
    prompt = 'check the HTTP headers on http://127.0.0.1:3000/legacy — follow the redirect and report the Server banner and CORS policy'
    extra = [
        (r'curl\s+.*(-[a-zA-Z]*L|--location).*127\.0\.0\.1:3000/legacy', final_h),
        (r'curl\s+.*127\.0\.0\.1:3000/v2/portal', final_h),
        (r'curl\s+.*127\.0\.0\.1:3000/legacy', step1),
    ]
    sb_factory = lambda: ExplorationSandbox(cwd, files, extra_shell=extra)
    check = lambda s, f: srv.split('/')[0].lower() in (f or '').lower() and ('cors' in (f or '').lower() or 'access-control-allow-origin' in (f or '').lower())
    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('cat scope.txt')['output']
    o2 = tmp.t_run_shell_command('curl -sIL http://127.0.0.1:3000/legacy')['output']
    expert_msgs = _make_expert_messages(
        cwd, files, prompt,
        [
            ('run_shell_command', {'command': 'cat scope.txt', 'description': 'Confirm target is in scope.'}, o1),
            ('run_shell_command', {'command': 'curl -sIL http://127.0.0.1:3000/legacy', 'description': 'Follow the 301 redirect and inspect final response headers.'}, o2),
        ],
        f'Following `http://127.0.0.1:3000/legacy` (`301` -> `/v2/portal`) reveals:\n'
        f'- **Server banner**: `{srv}`\n'
        f'- **CORS misconfiguration**: `Access-Control-Allow-Origin: *` paired with `Access-Control-Allow-Credentials: true`.'
    )
    return {'id': 'bb_redirect_follow', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


def mission_bb_cname_takeover(rng: random.Random):
    cwd = '/home/yamin/security/bugbounty-lab'
    files = {'scope.txt': 'localhost\n127.0.0.1\nstaging.lab.local\n'}
    bucket = rng.choice(['legacy-portal-dev', 'acme-staging-assets', 'old-marketing-app'])
    dig_out = f";; ANSWER SECTION:\nstaging.lab.local.\t300\tIN\tCNAME\t{bucket}.s3.amazonaws.com."
    s3_out = f"<?xml version=\"1.0\"?>\n<Error><Code>NoSuchBucket</Code><Message>The specified bucket does not exist</Message><BucketName>{bucket}</BucketName></Error>"
    prompt = 'check the DNS CNAME record for staging.lab.local and verify with curl if it is vulnerable to a subdomain takeover'
    extra = [(r'dig\b|host\b|nslookup\b', dig_out), (r'curl', s3_out)]
    sb_factory = lambda: ExplorationSandbox(cwd, files, extra_shell=extra)
    check = lambda s, f: bucket in (f or '').lower() and ('nosuchbucket' in (f or '').lower() or 'takeover' in (f or '').lower())
    tmp = sb_factory()
    o1 = tmp.t_run_shell_command('cat scope.txt')['output']
    o2 = tmp.t_run_shell_command('dig CNAME staging.lab.local +short')['output']
    o3 = tmp.t_run_shell_command('curl -s http://staging.lab.local')['output']
    expert_msgs = _make_expert_messages(
        cwd, files, prompt,
        [
            ('run_shell_command', {'command': 'cat scope.txt', 'description': 'Verify staging.lab.local is in scope.'}, o1),
            ('run_shell_command', {'command': 'dig CNAME staging.lab.local +short', 'description': 'Query CNAME record for staging.lab.local.'}, o2),
            ('run_shell_command', {'command': 'curl -s http://staging.lab.local', 'description': 'Check if the pointed S3 bucket is unclaimed.'}, o3),
        ],
        f'`staging.lab.local` has a dangling CNAME pointing to **`{bucket}.s3.amazonaws.com`**, which returns **`NoSuchBucket`** ("The specified bucket does not exist"). It is **vulnerable to subdomain takeover** — an attacker can register the `{bucket}` S3 bucket to serve content on `staging.lab.local`.'
    )
    return {'id': 'bb_cname_takeover', 'prompt': prompt, 'factory': sb_factory, 'check': check, 'expert': expert_msgs}


ALL_MISSION_BUILDERS = [
    (mission_icmp_traceroute_hops, 3),
    (mission_find_lan_switch, 3),
    (mission_safe_ping_interface, 3),
    (mission_web_header_and_dir_report, 4),
    (mission_scope_refuse, 3),
    (mission_bb_redirect_follow, 3),
    (mission_bb_cname_takeover, 3),
    (mission_fuzzy_find_wildcard, 3),
    (mission_math_step_by_step, 2),
    (mission_api_idor_and_headers, 2),
]


def sample_missions(n: int, seed: int = 42):
    rng = random.Random(seed)
    builders, weights = zip(*ALL_MISSION_BUILDERS)
    missions = []
    for i in range(n):
        b = rng.choices(builders, weights=weights, k=1)[0]
        m = b(rng)
        m['task_idx'] = i
        missions.append(m)
    return missions
