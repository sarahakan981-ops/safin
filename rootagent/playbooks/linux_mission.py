from dataclasses import dataclass, field
from typing import List

from .base_playbook import BasePlaybook, Phase


@dataclass
class LinuxRootMissionPlaybook(BasePlaybook):
    """Standing mission for the LinuxTerminalAgent.

    Exercises the FULL capability set discovered so far:
      - every installed binary via linux_exec (1318+ executables)
      - root-only tooling via linux_exec(sudo=True) (NOPASSWD sudo)
      - self-sufficient tool discovery (which_tool / list_installed_tools)
      - autonomous missing-tool installation (linux_install)
    """

    name: str = "linux_mission"
    description: str = "Linux Root Mission — full system toolset, no limits"
    mode: str = "agent"
    max_loops: int = 40

    phases: List[Phase] = field(default_factory=lambda: [
        Phase(
            name="Boot & Capability Audit",
            objective=(
                "Discover exactly what this machine can do. Run "
                "`sudo_status` once, then `uname -a`, `id`, and a quick "
                "`list_installed_tools` sample (filter=scan). Record findings."
            ),
            techniques=[
                "sudo_status → learn elevation path",
                "linux_exec: uname -a; id; df -h / | tail -1",
                "list_installed_tools(filter=scan|crack|exploit)",
            ],
        ),
        Phase(
            name="Root Tool Proof",
            objective=(
                "Prove elevated capability on real root-only tooling: "
                "iptables listing, a raw-socket SYN scan with nmap -sS, and "
                "reading a root-only file. All via linux_exec(sudo=true)."
            ),
            techniques=[
                "linux_exec(sudo=true): iptables -L -n | head -5",
                "linux_exec(sudo=true): nmap -sS -p 1-1024 <target>",
                "linux_exec(sudo=true): stat + read one /etc/shadow line header",
            ],
        ),
        Phase(
            name="Tool Assurance",
            objective=(
                "Verify every tool needed by later phases exists; install any "
                "missing one via linux_install without asking permission."
            ),
            techniques=[
                "which_tool(names=[nmap, gobuster, nikto, hydra, sqlmap, sqlmap])",
                "linux_install(packages=[...]) — auto-elevated via sudo",
                "re-verify with which_tool after install",
            ],
        ),
        Phase(
            name="Primary Mission",
            objective=(
                "Execute the operator's primary mission against the target "
                "using the full terminal toolset, aggressively iterating "
                "scan → enumerate → probe → verify until complete."
            ),
            techniques=[
                "nmap service discovery",
                "gobuster / nikto web surface mapping",
                "service-specific probing (smbclient, curl, dig, whois)",
                "all commands terminal-first via linux_exec",
            ],
        ),
        Phase(
            name="Vulnerability Analysis & Exploit Development",
            objective=(
                "From everything phase 4 revealed, pick the single most "
                "promising weakness, CONFIRM it manually (curl/raw socket/"
                "malformed input), then AUTHOR a real exploit script with "
                "write_file (loot/exploits/), RUN it with python_exec, and "
                "ITERATE on real output until it achieves its objective "
                "(RCE, dump, bypass, auth escalation)."
            ),
            techniques=[
                "manual verification of each suspected weakness before coding",
                "write_file: PoC script with target/usage header",
                "python_exec / linux_exec(sudo=true) to run and study output",
                "iterate script on observed behavior — payloads, offsets, paths",
                "raw sockets / http.client for precise control",
                "save working exploit + minimal reproduction + impact",
            ],
        ),
        Phase(
            name="Wrap-up",
            objective=(
                "Summarize every command run, key outputs, root operations "
                "performed, and artifacts saved to the loot directory."
            ),
            techniques=[
                "call finish only when every phase step is done",
                "report exact outputs — never fabricated",
            ],
        ),
    ])
