"""Held-out evaluation suite for Sandbox Exploration, LAN Switch/Router Discovery, and Error Recovery.

Tests the exact failure modes from the user's real SANY terminal session:
1. ICMP traceroute privilege escalation recovery (traceroute -I -> Operation not permitted -> sudo traceroute -I)
2. Finding a managed switch on the LAN via ARP/neighbor inspection
3. Checking active network interfaces (ip -br a) before running bounded ping (-c 4 -I <iface>)
4. Multi-step web header + directory recon report generation
"""

import re

CWD = '/home/yamin'
BASE_FILES = {
    'scope.txt': 'localhost\n127.0.0.1\n192.168.0.0/24\n',
    'notes.md': '# Local network & lab assessment\n',
}


def lower(s):
    return (s or '').lower()


def shell_cmds(sim):
    return [a.get('command', '') for n, a in sim.log if n == 'run_shell_command']


IP_ADDR_OUT = (
    "lo               UNKNOWN        127.0.0.1/8 ::1/128\n"
    "enp2s0           UP             192.168.0.184/24 fe80::c34e:26f1:d4f7:8db7/64\n"
    "wlp3s0           UP             192.168.0.120/24 fe80::238e:1379:8b2c:53cd/64"
)

ARP_OUT = (
    "dlinkrouter (192.168.0.1) at 40:86:cb:7a:a7:20 [ether] on enp2s0\n"
    "procurve-switch (192.168.0.254) at b8:27:eb:45:12:99 [ether] on enp2s0"
)

NEIGH_OUT = (
    "192.168.0.1 dev enp2s0 lladdr 40:86:cb:7a:a7:20 REACHABLE\n"
    "192.168.0.254 dev enp2s0 lladdr b8:27:eb:45:12:99 REACHABLE"
)

TRACEROUTE_SUDO_OUT = (
    "traceroute to google.com (192.178.211.101), 30 hops max, 60 byte packets\n"
    " 1  dlinkrouter (192.168.0.1)  1.099 ms  1.544 ms  1.612 ms\n"
    " 2  192.168.1.220 (192.168.1.220)  27.480 ms  27.694 ms  27.903 ms\n"
    " 3  172.20.20.1 (172.20.20.1)  28.434 ms  28.658 ms  28.860 ms"
)

TRACEROUTE_UNPRIV_OUT = (
    "You do not have enough privileges to use this traceroute method.\n"
    "socket: Operation not permitted"
)

PING_OK_OUT = (
    "PING 192.168.0.1 (192.168.0.1) 56(84) bytes of data.\n"
    "64 bytes from 192.168.0.1: icmp_seq=1 ttl=64 time=0.61 ms\n"
    "--- 192.168.0.1 ping statistics ---\n"
    "4 packets transmitted, 4 received, 0% packet loss"
)


CASES = [
    dict(
        id='ex_icmp_trace_sudo',
        prompt='run an ICMP traceroute to google.com and report the first three hop IPs',
        files=BASE_FILES, cwd=CWD,
        extra_shell=[
            (r'^sudo\s+traceroute\s+.*-I', TRACEROUTE_SUDO_OUT),
            (r'^traceroute\s+.*-I', TRACEROUTE_UNPRIV_OUT),
        ],
        check=lambda s, f: '192.168.0.1' in f and '192.168.1.220' in f and '172.20.20.1' in f,
    ),
    dict(
        id='ex_lan_switch_arp',
        prompt='my router is connected to a managed switch on the LAN — find the IP and MAC address of the switch',
        files=BASE_FILES, cwd=CWD,
        extra_shell=[
            (r'ip\s+neigh', NEIGH_OUT),
            (r'arp', ARP_OUT),
            (r'ip\s+(-br\s+)?a', IP_ADDR_OUT),
        ],
        check=lambda s, f: '192.168.0.254' in f and 'b8:27:eb:45:12:99' in lower(f),
    ),
    dict(
        id='ex_bounded_ping_if',
        prompt='check which network interface has 192.168.0.184 and send 4 pings to 192.168.0.1 from that interface',
        files=BASE_FILES, cwd=CWD,
        extra_shell=[
            (r'ip\s+(-br\s+)?a|ifconfig', IP_ADDR_OUT),
            (r'ping\s+.*-c\s*\d+', PING_OK_OUT),
        ],
        check=lambda s, f: (
            'enp2s0' in f
            and any('ping' in c and '-c' in c and 'enp2s0' in c for c in shell_cmds(s))
            and not any('ping' in c and '-c' not in c for c in shell_cmds(s))
        ),
    ),
]
