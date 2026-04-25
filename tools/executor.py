"""
tools/executor.py
Unified execution layer for WE-Hybrid MCP.

Supports three execution modes, tried in order:

  1. ControlMaster (recommended for EXPANSE/DUO)
     User keeps one `ssh expanse` terminal open with ControlMaster enabled.
     MCP server piggybacks on that socket — no TOTP per-command.
     Set up ~/.ssh/config (see CLAUDE.md) then call:
       connect_to_cluster(host="expanse", use_control_master=True)

  2. Paramiko SSH (clusters without MFA, or key-only auth)
     Full SSH via paramiko + SFTP. Works transparently when no 2FA is needed.
       connect_to_cluster(host="login.expanse.sdsc.edu", username="ssonti",
                          key_file="~/.ssh/id_expanse")

  3. Local (default when not connected)
     subprocess + pathlib. Used for laptop/template-only mode.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import paramiko
    _PARAMIKO_OK = True
except ImportError:
    _PARAMIKO_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class Executor:
    """
    Unified local/SSH executor.

    All public methods work identically regardless of connection mode —
    callers never need to branch on connection state.
    """

    def __init__(self):
        # paramiko mode
        self._ssh:      Optional["paramiko.SSHClient"] = None
        self._sftp:     Optional["paramiko.SFTPClient"] = None

        # ControlMaster mode
        self._cm_host:  Optional[str] = None   # SSH alias, e.g. "expanse"
        self._cm_user:  Optional[str] = None
        self._cm_socket: Optional[str] = None  # ControlPath socket

        # Shared
        self._host:     Optional[str] = None
        self._username: Optional[str] = None
        self._setup_cmd: str = ""

    # ── ControlMaster connect ─────────────────────────────────────────────────

    def connect_control_master(
        self,
        host: str,
        username: Optional[str] = None,
        control_path: Optional[str] = None,
        setup_cmd: str = "",
    ) -> tuple[bool, str]:
        """
        Connect using an existing SSH ControlMaster socket.

        This is the recommended method for EXPANSE (DUO/TOTP clusters).
        The user must have an active `ssh expanse` session in another terminal
        with ControlMaster enabled in ~/.ssh/config.

        host:         SSH host alias as defined in ~/.ssh/config (e.g. "expanse")
                      OR hostname if ControlPath is given explicitly.
        username:     SSH username (optional if set in ~/.ssh/config)
        control_path: Path to control socket. If omitted, auto-detected from
                      ~/.ssh/config or uses the default pattern.
        setup_cmd:    Shell commands prepended to every remote command,
                      e.g. "module load amber; conda activate westpa2"
        """
        if control_path is None:
            control_path = self._find_control_socket(host, username)

        if control_path is None:
            return False, (
                f"No active ControlMaster socket found for {host}.\n\n"
                "To enable ControlMaster on EXPANSE:\n"
                "  1. Add to ~/.ssh/config:\n\n"
                "       Host expanse\n"
                "           HostName login.expanse.sdsc.edu\n"
                "           User ssonti\n"
                "           IdentityFile ~/.ssh/id_expanse\n"
                "           ControlMaster auto\n"
                "           ControlPath ~/.ssh/cm-%r@%h:%p\n"
                "           ControlPersist 8h\n\n"
                "  2. In a separate terminal, run:\n"
                "       ssh expanse\n"
                "     (enter TOTP once — socket stays open for 8 hours)\n\n"
                "  3. Call connect_to_cluster again — no TOTP needed."
            )

        # Verify the socket is live
        check = subprocess.run(
            ["ssh", "-S", control_path, "-O", "check", host],
            capture_output=True, text=True, timeout=10,
        )
        if check.returncode != 0:
            return False, (
                f"ControlMaster socket found at {control_path} but is not responding.\n"
                f"Open a new terminal and run: ssh {host}\n"
                f"(stderr: {check.stderr.strip()})"
            )

        self._cm_host   = host
        self._cm_socket = control_path
        self._cm_user   = username
        self._host      = host
        self._username  = username
        self._setup_cmd = setup_cmd.strip()

        return True, f"Connected to {host} via ControlMaster ({control_path})"

    def _find_control_socket(self, host: str, username: Optional[str]) -> Optional[str]:
        """Try to find an active ControlMaster socket for the given host."""
        # Common patterns
        candidates = []
        home = Path.home()

        if username:
            candidates += [
                str(home / f".ssh/cm-{username}@{host}:22"),
                str(home / f".ssh/cm-{username}@{host}"),
                f"/tmp/ssh-cm-{username}@{host}",
            ]
        candidates += [
            str(home / f".ssh/cm-%r@{host}:%p"),   # literal — won't match, but try
            str(home / f".ssh/cm-*@{host}:22"),
        ]

        # Glob for actual sockets
        import glob
        for pattern in [
            str(home / f".ssh/cm-*@{host}:*"),
            str(home / f".ssh/cm-*@{host}"),
            f"/tmp/ssh-cm-*@{host}*",
        ]:
            matches = glob.glob(pattern)
            for m in matches:
                if Path(m).exists():
                    return m

        # Try reading ControlPath from ssh config
        try:
            result = subprocess.run(
                ["ssh", "-G", host],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                if line.startswith("controlpath "):
                    cp = line.split(None, 1)[1].strip()
                    # Expand %r, %h, %p
                    cp = cp.replace("%r", username or "").replace("%h", host).replace("%p", "22")
                    cp = str(Path(cp).expanduser())
                    if Path(cp).exists():
                        return cp
        except Exception:
            pass

        return None

    # ── Paramiko SSH connect ──────────────────────────────────────────────────

    def connect(
        self,
        host: str,
        username: str,
        key_file: Optional[str] = None,
        password: Optional[str] = None,
        port: int = 22,
        setup_cmd: str = "",
        timeout: float = 30.0,
    ) -> tuple[bool, str]:
        """
        Open an SSH connection via paramiko.
        Works best on clusters without MFA. For EXPANSE (DUO/TOTP),
        use connect_control_master() instead.
        """
        if not _PARAMIKO_OK:
            return False, (
                "paramiko is not installed. Run:\n"
                "  pip install paramiko\n"
                "then restart the MCP server."
            )

        if self.connected:
            self.disconnect()

        if key_file:
            key_file = str(Path(key_file).expanduser().resolve())

        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            connect_kwargs: dict = dict(
                hostname=host,
                port=port,
                username=username,
                timeout=timeout,
                allow_agent=True,
                look_for_keys=True,
            )
            if key_file and Path(key_file).exists():
                connect_kwargs["key_filename"] = key_file
            if password:
                connect_kwargs["password"] = password

            client.connect(**connect_kwargs)

        except paramiko.AuthenticationException as e:
            return False, (
                f"SSH authentication failed for {username}@{host}: {e}\n\n"
                "If this cluster uses DUO/TOTP (like EXPANSE), use ControlMaster instead:\n"
                "  1. Add ControlMaster config to ~/.ssh/config (see CLAUDE.md)\n"
                "  2. Run `ssh expanse` in a terminal to create the socket\n"
                "  3. Call connect_to_cluster with use_control_master=True"
            )
        except paramiko.SSHException as e:
            return False, f"SSH error connecting to {host}: {e}"
        except OSError as e:
            return False, f"Cannot reach {host}:{port} — {e}\nCheck VPN / network connectivity."
        except Exception as e:
            return False, f"Unexpected error: {e}"

        # Verify
        try:
            _, stdout, _ = client.exec_command("echo ok", timeout=10)
            if stdout.read().decode().strip() != "ok":
                client.close()
                return False, "SSH connected but test command failed."
        except Exception as e:
            client.close()
            return False, f"SSH connection test failed: {e}"

        self._ssh       = client
        self._sftp      = client.open_sftp()
        self._host      = host
        self._username  = username
        self._setup_cmd = setup_cmd.strip()

        return True, f"Connected to {username}@{host}:{port} via paramiko"

    # ── Disconnect ────────────────────────────────────────────────────────────

    def disconnect(self):
        """Close all connections."""
        if self._sftp:
            try: self._sftp.close()
            except Exception: pass
        if self._ssh:
            try: self._ssh.close()
            except Exception: pass
        self._ssh       = None
        self._sftp      = None
        self._cm_host   = None
        self._cm_socket = None
        self._cm_user   = None
        self._host      = None
        self._username  = None
        self._setup_cmd = ""

    # ── Connection state ──────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        if self._cm_socket is not None:
            # ControlMaster: check socket exists
            return Path(self._cm_socket).exists()
        if self._ssh is not None:
            transport = self._ssh.get_transport()
            return transport is not None and transport.is_active()
        return False

    @property
    def _connection_type(self) -> str:
        if self._cm_socket:
            return "controlmaster"
        if self._ssh:
            return "paramiko"
        return "local"

    def mode(self) -> str:
        return "ssh" if self.connected else "local"

    def connection_info(self) -> dict:
        if self.connected:
            return {
                "mode":            "ssh",
                "type":            self._connection_type,
                "host":            self._host,
                "username":        self._username,
                "setup_cmd":       self._setup_cmd,
                "control_socket":  self._cm_socket,
            }
        return {"mode": "local", "host": None, "username": None}

    # ── Command execution ─────────────────────────────────────────────────────

    def run_command(
        self,
        cmd: str,
        cwd: Optional[str] = None,
        timeout: int = 300,
        env: Optional[dict] = None,
    ) -> tuple[int, str, str]:
        """Run a shell command. Returns (returncode, stdout, stderr)."""
        if self._cm_socket:
            return self._run_via_cm(cmd, cwd=cwd, timeout=timeout)
        if self._ssh:
            return self._run_via_paramiko(cmd, cwd=cwd, timeout=timeout)
        return self._run_local(cmd, cwd=cwd, timeout=timeout, env=env)

    def _run_via_cm(self, cmd: str, cwd: Optional[str], timeout: int) -> tuple[int, str, str]:
        """Execute via ControlMaster socket using ssh subprocess."""
        full_cmd = ""
        if self._setup_cmd:
            full_cmd += f"{self._setup_cmd}; "
        if cwd:
            full_cmd += f"cd {cwd}; "
        full_cmd += cmd

        ssh_cmd = [
            "ssh",
            "-S", self._cm_socket,
            "-o", "BatchMode=yes",
            self._cm_host,
            full_cmd,
        ]
        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True, text=True, timeout=timeout,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Remote command timed out after {timeout}s"
        except Exception as e:
            return 1, "", f"ControlMaster execution error: {e}"

    def _run_via_paramiko(self, cmd: str, cwd: Optional[str], timeout: int) -> tuple[int, str, str]:
        """Execute via paramiko SSH."""
        full_cmd = ""
        if self._setup_cmd:
            full_cmd += f"{self._setup_cmd}; "
        if cwd:
            full_cmd += f"cd {cwd!r}; "
        full_cmd += cmd

        try:
            _, stdout_chan, stderr_chan = self._ssh.exec_command(full_cmd, timeout=timeout)
            stdout_chan.channel.settimeout(timeout)
            stdout = stdout_chan.read().decode("utf-8", errors="replace")
            stderr = stderr_chan.read().decode("utf-8", errors="replace")
            rc     = stdout_chan.channel.recv_exit_status()
            return rc, stdout, stderr
        except Exception as e:
            return 1, "", f"Paramiko execution error: {e}"

    def _run_local(self, cmd: str, cwd: Optional[str], timeout: int, env: Optional[dict]) -> tuple[int, str, str]:
        """Execute locally via subprocess."""
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                cwd=cwd, timeout=timeout, env=merged_env,
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return 1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            return 1, "", f"Local execution error: {e}"

    # ── Binary detection ──────────────────────────────────────────────────────

    def which(self, binary: str) -> Optional[str]:
        if self.connected:
            rc, stdout, _ = self.run_command(f"which {binary} 2>/dev/null", timeout=60)
            path = stdout.strip()
            return path if rc == 0 and path else None
        return shutil.which(binary)

    def has_binary(self, binary: str) -> bool:
        return self.which(binary) is not None

    # ── File system operations ────────────────────────────────────────────────

    def make_dir(self, path: str):
        if self.connected:
            self.run_command(f"mkdir -p {path}", timeout=30)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)

    def write_file(self, path: str, content: str, mode: int = 0o644):
        """Write text content to local or remote path."""
        if self._cm_socket:
            # ControlMaster: use sftp subprocess or heredoc via ssh
            self._write_via_cm(path, content, mode)
        elif self._sftp:
            buf = io.BytesIO(content.encode("utf-8"))
            self._sftp.putfo(buf, path)
            self._sftp.chmod(path, mode)
        else:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            p.chmod(mode)

    def _write_via_cm(self, remote_path: str, content: str, mode: int):
        """Write a file to the remote host via sftp subprocess using the CM socket."""
        import tempfile, base64

        # Write content to a temp local file, then sftp it
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False, encoding='utf-8') as f:
            f.write(content)
            tmp_path = f.name

        try:
            sftp_batch = f"put {tmp_path} {remote_path}\nchmod {oct(mode)[2:]} {remote_path}\n"
            sftp_cmd = [
                "sftp",
                "-o", f"ControlPath={self._cm_socket}",
                "-b", "-",
                self._cm_host,
            ]
            subprocess.run(
                sftp_cmd,
                input=sftp_batch, capture_output=True, text=True, timeout=60,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def read_file(self, path: str) -> str:
        if self.connected:
            rc, stdout, stderr = self.run_command(f"cat {path}", timeout=30)
            if rc != 0:
                raise FileNotFoundError(f"Cannot read remote file {path}: {stderr}")
            return stdout
        return Path(path).read_text(encoding="utf-8")

    def file_exists(self, path: str) -> bool:
        if self.connected:
            rc, _, _ = self.run_command(f"test -e {path}", timeout=10)
            return rc == 0
        return Path(path).exists()

    def list_files(self, path: str) -> list[str]:
        if self.connected:
            rc, stdout, _ = self.run_command(f"ls {path}", timeout=10)
            if rc != 0:
                return []
            return [l for l in stdout.splitlines() if l]
        p = Path(path)
        return [x.name for x in p.iterdir()] if p.is_dir() else []

    def get_capabilities(self) -> dict:
        """Check which cluster tools are available on the target."""
        tools = ("tleap", "cpptraj", "squeue", "pmemd.cuda")
        if self.connected:
            # Run a single command with modules loaded once to avoid per-call timeout
            checks = "; ".join(f"echo {t}=$(which {t} 2>/dev/null)" for t in tools)
            rc, stdout, _ = self.run_command(checks, timeout=60)
            caps = {}
            for line in stdout.splitlines():
                if "=" in line:
                    name, _, path = line.partition("=")
                    name = name.strip()
                    if name in tools:
                        caps[name] = bool(path.strip())
            for t in tools:
                caps.setdefault(t, False)
            return caps
        return {t: shutil.which(t) is not None for t