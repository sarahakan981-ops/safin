"""Linux agent tools: unrestricted terminal access to every installed CLI.

These tools are automatically registered by the tool loader and are
available to every agent. They complement the generic `terminal` tool with:

  - ``linux_exec``          – run any shell command (full Linux toolset)
  - ``which_tool``          – check availability/path of any binaries
  - ``list_installed_tools``– enumerate every executable on the system
  - ``linux_install``       – install missing tools via the system package
                              manager (apt/apk/dnf/pacman) or pip/pipx

All execution flows through the active ``Runtime`` (Local or Docker), so the
tools work identically on the host and inside a container.
"""

import shlex
import os  # noqa: F401  (used inside _build_sudo_command)
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from ..registry import ToolSchema, register_tool

if TYPE_CHECKING:
    from ...runtime import CommandResult, Runtime

# Cap tool output fed back to the LLM so huge listings do not blow the context
_MAX_OUTPUT_CHARS = 15_000

#: Environment variable holding the operator's sudo password. Optional: when
#: passwordless sudo is configured this is never needed.
SUDO_PASSWORD_ENV = "ROOTAGENT_SUDO_PASSWORD"


def _build_sudo_command(command: str) -> str:
    """Wrap *command* so it runs with root privileges inside the runtime shell.

    Resolution order (each attempt handled by the caller):
      1. Already root  -> run directly
      2. Passwordless sudo (NOPASSWD, or cached credentials) -> ``sudo -n``
      3. ``ROOTAGENT_SUDO_PASSWORD`` in the environment -> ``sudo -S`` with
         the password piped in safely (quoted, never echoed, never logged)
    """
    sudo_password = os.environ.get(SUDO_PASSWORD_ENV)
    if sudo_password:
        return (
            f"echo {shlex.quote(sudo_password)} | "
            f"sudo -S -p '' bash -c {shlex.quote(command)}"
        )
    return f"sudo -n bash -c {shlex.quote(command)}"


async def _run_elevated(
    runtime: "Runtime", command: str, timeout: int = 300
) -> tuple:
    """Run *command* as root. Returns (result, how) where `how` explains the
    elevation path used ("direct", "sudo -n", "sudo -S" or "denied")."""
    # Already root? (very common inside Docker containers)
    id_res = await runtime.execute_command("id -u", timeout=10)
    if id_res.stdout.strip() == "0":
        result = await runtime.execute_command(command, timeout=timeout)
        return result, "direct (already root)"

    # Try sudo non-interactively first (NOPASSWD or cached timestamp)
    result = await runtime.execute_command(
        _build_sudo_command(command), timeout=timeout
    )
    if result.success:
        return result, "sudo"

    # Failed — report precisely why so the agent/operator can act
    stderr = result.stderr.lower()
    if "password" in stderr or "credentials" in stderr:
        return (
            result,
            "DENIED: sudo needs a password. Either configure passwordless sudo "
            f"(`sudo visudo` -> `<user> ALL=(ALL) NOPASSWD: ALL`) or set "
            f"{SUDO_PASSWORD_ENV} in .env and restart the agent.",
        )
    return result, "sudo (failed)"


def _format_result(result: "CommandResult") -> str:
    """Format a CommandResult into a compact, LLM-friendly string."""
    lines: List[str] = [f"exit_code: {result.exit_code}"]
    if result.stdout:
        lines.append(f"stdout:\n{result.stdout}")
    if result.stderr:
        lines.append(f"stderr:\n{result.stderr}")
    if not result.stdout and not result.stderr:
        lines.append("(no output)")

    text = "\n".join(lines)
    if len(text) > _MAX_OUTPUT_CHARS:
        cut = len(text) - _MAX_OUTPUT_CHARS
        text = text[:_MAX_OUTPUT_CHARS] + f"\n... [output truncated, {cut} chars cut]"
    return text


@register_tool(
    name="linux_exec",
    description=(
        "Execute ANY shell command on the Linux system with full access to every "
        "installed tool (nmap, sqlmap, hydra, gobuster, curl, python, gcc, ...). "
        "Use this for anything available on the terminal."
    ),
    schema=ToolSchema(
        properties={
            "command": {
                "type": "string",
                "description": "The full shell command to execute (supports pipes, redirection, and chaining)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 300)",
                "default": 300,
            },
            "working_dir": {
                "type": "string",
                "description": "Directory to run the command in (optional)",
            },
            "sudo": {
                "type": "boolean",
                "description": (
                    "Run as root via sudo (default: false). Uses passwordless "
                    "sudo automatically; uses ROOTAGENT_SUDO_PASSWORD from "
                    ".env when a password is required."
                ),
            },
        },
        required=["command"],
    ),
    category="linux",
)
async def linux_exec(arguments: dict, runtime: "Runtime") -> str:
    """Execute an arbitrary Linux shell command through the runtime."""
    command = arguments["command"]
    timeout = arguments.get("timeout", 300)
    working_dir = arguments.get("working_dir")
    as_root = bool(arguments.get("sudo", False))

    if working_dir:
        command = f"cd {shlex.quote(working_dir)} && {command}"

    if as_root:
        result, how = await _run_elevated(runtime, command, timeout)
        return f"[elevated via {how}]\n" + _format_result(result)

    result = await runtime.execute_command(command, timeout=timeout)
    return _format_result(result)


@register_tool(
    name="which_tool",
    description=(
        "Check whether one or more Linux tools/binary names are installed and get "
        "their absolute paths. Example: names=['nmap', 'hydra']"
    ),
    schema=ToolSchema(
        properties={
            "names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of binary names to look up",
            },
        },
        required=["names"],
    ),
    category="linux",
)
async def which_tool(arguments: dict, runtime: "Runtime") -> str:
    """Resolve binary paths using `command -v` inside the runtime."""
    names: List[str] = arguments.get("names", [])
    if not names:
        return "No tool names provided."

    quoted = " ".join(shlex.quote(n) for n in names)
    # Pure POSIX: works in dash/sh, on the host and inside containers alike
    script = (
        f"for t in {quoted}; do "
        'p=$(command -v "$t" 2>/dev/null); '
        'if [ -n "$p" ]; then echo "$t FOUND $p"; '
        'else echo "$t MISSING"; fi; done'
    )
    result = await runtime.execute_command(script)
    return _format_result(result)


@register_tool(
    name="list_installed_tools",
    description=(
        "List every executable binary available on the system (scans all "
        "directories in PATH). Optionally filter by a substring."
    ),
    schema=ToolSchema(
        properties={
            "filter": {
                "type": "string",
                "description": "Only list tools whose name contains this substring (optional)",
            },
        },
        required=[],
    ),
    category="linux",
)
async def list_installed_tools(arguments: dict, runtime: "Runtime") -> str:
    """Enumerate all executables found in PATH inside the runtime."""
    pattern = arguments.get("filter")
    filter_clause = ""
    if pattern:
        safe = shlex.quote(f"*{pattern}*")
        filter_clause = f" case $b in ({safe}) ) [ -x $f ] && echo $b ;; esac; return"

    # Enumerate executables in PATH without spawning a process per file
    script = (
        'OLDIFS=$IFS; IFS=:; for d in $PATH; do '
        '[ -d "$d" ] || continue; '
        'for f in "$d"/*; do '
        '[ -x "$f" ] || continue; '
        'b=${f##*/}; '
        'case "$b" in *[!A-Za-z0-9_.+_-]*) ;; *) printf \'%s\\n\' "$b";; esac; '
        'done; done; IFS=$OLDIFS'
    )

    if pattern:
        # Wrap the enumeration with the filter (grep based, simplest and POSIX)
        script = f"({script}) 2>/dev/null | sort -u | grep -F {shlex.quote(pattern)}"

    result = await runtime.execute_command("(" + script + ") 2>/dev/null | sort -u")
    text = _format_result(result)

    if pattern and ("exit_code: 1" in text.splitlines()[0]):
        return f"No installed tools matching filter: {pattern}"
    return f"Available Linux tools (filter={pattern or 'none'}):\n{text}"


@register_tool(
    name="linux_install",
    description=(
        "Install a missing Linux tool using the system package manager "
        "(auto-detects apt/apk/dnf/pacman) or pip/pipx for Python packages."
    ),
    schema=ToolSchema(
        properties={
            "packages": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Package names to install (e.g. ['seclists', 'nmap'])",
            },
            "manager": {
                "type": "string",
                "description": (
                    "Force a specific manager: apt, apk, dnf, pacman, pip, pipx. "
                    "Leave empty to auto-detect."
                ),
            },
            "update_cache": {
                "type": "boolean",
                "description": "Refresh the package index before installing (default: false)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 600)",
                "default": 600,
            },
        },
        required=["packages"],
    ),
    category="linux",
)
async def linux_install(arguments: dict, runtime: "Runtime") -> str:
    """Install packages with the best available package manager."""
    packages: List[str] = arguments.get("packages", [])
    manager: Optional[str] = arguments.get("manager")
    update_cache: bool = bool(arguments.get("update_cache", False))
    timeout: int = arguments.get("timeout", 600)

    if not packages:
        return "No packages provided."

    pkgs = " ".join(shlex.quote(p) for p in packages)

    # Detect the current user (needed for sudo decisions)
    id_res = await runtime.execute_command("id -u")
    is_root = id_res.stdout.strip() == "0"

    # Auto-detect the package manager
    detect_script = (
        "for m in apt-get apt apk dnf yum pacman pipx pip; do "
        'command -v "$m" >/dev/null 2>&1 && { echo "$m"; break; }; done'
    )
    if manager:
        chosen = manager
    else:
        detect = await runtime.execute_command(detect_script)
        chosen = detect.stdout.strip().splitlines()[0] if detect.stdout.strip() else ""

    if not chosen:
        return "No supported package manager found (apt/apk/dnf/pacman/pip/pipx)."

    sudo = "" if is_root else "sudo "

    commands = {
        "apt-get": lambda: (
            (f"{sudo}apt-get update && " if update_cache else "")
            + f"DEBIAN_FRONTEND=noninteractive {sudo}apt-get install -y {pkgs}"
        ),
        "apt": lambda: (
            (f"{sudo}apt update && " if update_cache else "")
            + f"DEBIAN_FRONTEND=noninteractive {sudo}apt install -y {pkgs}"
        ),
        "apk": lambda: (
            (f"{sudo}apk update && " if update_cache else "")
            + f"{sudo}apk add --no-cache {pkgs}"
        ),
        "dnf": lambda: (
            (f"{sudo}dnf makecache && " if update_cache else "")
            + f"{sudo}dnf install -y {pkgs}"
        ),
        "yum": lambda: (
            (f"{sudo}yum makecache && " if update_cache else "")
            + f"{sudo}yum install -y {pkgs}"
        ),
        "pacman": lambda: (
            (f"{sudo}pacman -Sy && " if update_cache else "")
            + f"{sudo}pacman -S --noconfirm {pkgs}"
        ),
        "pip": lambda: f"{sudo}{chosen} install {pkgs}",
        "pipx": lambda: f"{chosen} install {pkgs}",
    }

    builder = commands.get(chosen)
    if builder is None:
        return f"Unsupported package manager: {chosen}"

    command = builder()
    if is_root:
        result = await runtime.execute_command(command, timeout=timeout)
        how = " [root]"
    elif "sudo " in command:
        # Only system-package installs need elevation (pipx runs as user).
        # Note: the command may START with env vars (DEBIAN_FRONTEND=...) so
        # we check for sudo anywhere, not just as the first token.
        result, how2 = await _run_elevated(runtime, command)
        how = f" [{how2}]"
    else:
        result = await runtime.execute_command(command, timeout=timeout)
        how = " [user]"

    status = "OK" if result.success else "FAILED"
    return f"Install [{status}]{how} via {chosen}:\n{_format_result(result)}"


__all__ = [
    "linux_exec",
    "which_tool",
    "list_installed_tools",
    "linux_install",
    "sudo_status",
    "python_exec",
    "write_file",
    "read_file",
]


@register_tool(
    name="python_exec",
    description=(
        "Run a Python snippet (output to stdout/stderr is returned). Use for "
        "exploit prototyping, socket/raw-HTTP craft, data parsing, forensics "
        "and anything bash is clumsy at. Runs `python3 <tmpfile>`."
    ),
    schema=ToolSchema(
        properties={
            "code": {
                "type": "string",
                "description": "Python source code to execute (full script, use __main__-style flow)",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default: 60)",
                "default": 60,
            },
            "keep_file": {
                "type": "boolean",
                "description": "Keep the generated script file for reuse (default: true)",
            },
        },
        required=["code"],
    ),
    category="linux",
)
async def python_exec(arguments: dict, runtime: "Runtime") -> str:
    """Write a Python script to disk and execute it via the runtime."""
    import time
    import uuid

    from ...workspaces.utils import get_loot_base

    code: str = arguments["code"]
    timeout: int = arguments.get("timeout", 60)
    keep: bool = bool(arguments.get("keep_file", True))

    exploits_dir = get_loot_base() / "exploits"
    exploits_dir.mkdir(parents=True, exist_ok=True)
    script = exploits_dir / f"pyexec_{uuid.uuid4().hex[:8]}.py"
    script.write_text(code, encoding="utf-8")

    try:
        result = await runtime.execute_command(f"python3 {shlex.quote(str(script))}", timeout=timeout)
        out = _format_result(result)
        return f"[script: {script}]\n{out}"
    finally:
        if not keep:
            try:
                script.unlink()
            except OSError:
                pass


@register_tool(
    name="write_file",
    description=(
        "Write a full file to disk (create/overwrite). Perfect for authoring "
        "exploit scripts, payloads, wordlists and reports. Default location "
        "is the loot/exploits directory."
    ),
    schema=ToolSchema(
        properties={
            "content": {"type": "string", "description": "Full file content"},
            "path": {
                "type": "string",
                "description": "Absolute or workspace-relative target path (optional; default loot/exploits/exploit_<ts>.py)",
            },
        },
        required=["content"],
    ),
    category="linux",
)
async def write_file(arguments: dict, runtime: "Runtime") -> str:
    """Create/overwrite a file. Falls back to loot/exploits/ by default."""
    import time

    from ...workspaces.utils import get_loot_base

    content: str = arguments["content"]
    path: Optional[str] = arguments.get("path")

    if path:
        target = Path(path).expanduser()
    else:
        exploits = get_loot_base() / "exploits"
        exploits.mkdir(parents=True, exist_ok=True)
        target = exploits / f"exploit_{int(time.time())}.py"

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"written: {target} ({len(content)} chars)"


@register_tool(
    name="read_file",
    description=(
        "Read a file from disk (source code, creds dumps, tool configs, "
        "target files). Returns up to max_chars characters."
    ),
    schema=ToolSchema(
        properties={
            "path": {"type": "string", "description": "File path to read"},
            "max_chars": {
                "type": "integer",
                "description": "Max characters to return (default 12000)",
                "default": 12000,
            },
            "offset": {
                "type": "integer",
                "description": "Skip first N chars before reading (default 0)",
                "default": 0,
            },
        },
        required=["path"],
    ),
    category="linux",
)
async def read_file(arguments: dict, runtime: "Runtime") -> str:
    """Read a file (truncated to max_chars from the given offset)."""
    path = Path(arguments["path"]).expanduser()
    max_chars = int(arguments.get("max_chars", 12000))
    offset = int(arguments.get("offset", 0))

    if not path.is_file():
        return f"NOT A FILE: {path}"

    import time as _t

    text = path.read_text(errors="replace")
    if offset:
        text = text[offset:]
    chunk = text[:max_chars]
    more = len(text) - len(chunk)
    suffix = f"\n... [+{more} more chars — use offset={offset + len(chunk)} to continue]" if more > 0 else ""
    return f"{path} ({len(text)} chars):\n{chunk}{suffix}"


@register_tool(
    name="sudo_status",
    description=(
        "Check root/sudo capability: are we root, is passwordless sudo "
        "available, and which elevation path will linux_exec(sudo=true) use."
    ),
    schema=ToolSchema(properties={}, required=[]),
    category="linux",
)
async def sudo_status(arguments: dict, runtime: "Runtime") -> str:
    """Report the exact elevation path available to the agent."""
    import os

    id_res = await runtime.execute_command("id -u", timeout=10)
    if id_res.stdout.strip() == "0":
        return ("sudo_status: ROOT — every command runs directly with full "
                "privileges. No elevation needed.")

    probe = await runtime.execute_command("sudo -n true 2>&1", timeout=10)
    has_passwordless = probe.success

    lines = [
        f"current user: not root (uid!={id_res.stdout.strip()})",
        f"passwordless sudo: {'AVAILABLE' if has_passwordless else 'NOT configured'}",
    ]

    sudo_password = os.environ.get(SUDO_PASSWORD_ENV)
    if sudo_password:
        lines.append(
            f"{SUDO_PASSWORD_ENV}: set — elevated commands will pipe this "
            "password to sudo -S automatically"
        )
    elif has_passwordless:
        lines.append(f"{SUDO_PASSWORD_ENV}: not needed (passwordless sudo)")
    else:
        lines.append(
            f"{SUDO_PASSWORD_ENV}: not set — sudo=true commands will FAIL until "
            "the operator does ONE of:\n"
            "  1. sudo visudo  ->  add: <user> ALL=(ALL) NOPASSWD: ALL\n"
            f"  2. add {SUDO_PASSWORD_ENV}=<sudo-password> to .env and restart"
        )
    return "\n".join(lines)
