"""Sync project files to Tencent Cloud server. Auth: SSH key or password."""

import os
import sys
import glob
from getpass import getpass

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_project_root)

try:
    from paramiko import SSHClient, AutoAddPolicy, RSAKey
except ImportError:
    print("paramiko not installed. Run: pip install paramiko")
    sys.exit(1)

SERVER = os.environ.get("EPF_SERVER", "150.158.150.98")
USER = os.environ.get("EPF_USER", "ubuntu")
REMOTE_BASE = "/opt/epf-agent"

# Auto-discover all project files (exclude .git, venv, __pycache__, .idea, etc.)
EXCLUDE_DIRS = {'.git', 'venv', '__pycache__', '.idea', '.claude', 'node_modules',
                'chroma_db', 'chroma_db_v2', '.pytest_cache', '__MACOSX'}
EXCLUDE_EXTS = {'.pyc', '.pyo', '.sqlite3', '.db'}

def _collect_files():
    """Walk the project root and collect all file paths relative to project root."""
    files = []
    for root, dirs, filenames in os.walk(_project_root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        rel_dir = os.path.relpath(root, _project_root)
        for f in filenames:
            _, ext = os.path.splitext(f)
            if ext in EXCLUDE_EXTS:
                continue
            rel_path = os.path.join(rel_dir, f).replace(os.sep, '/')
            if rel_path.startswith('.'):
                continue
            files.append(rel_path)
    return files

RESTART_CMD = (
    "sudo systemctl restart epf-agent && "
    "sleep 2 && "
    "sudo systemctl status epf-agent --no-pager"
)


def _load_ssh_key():
    """Try to load SSH key from default locations."""
    for key_path in [
        os.path.expanduser("~/.ssh/id_rsa"),
        os.path.expanduser("~/.ssh/id_ed25519"),
    ]:
        if os.path.exists(key_path):
            try:
                return RSAKey.from_private_key_file(key_path)
            except Exception:
                pass
    return None


def main():
    pkey = _load_ssh_key()
    password = None

    if pkey:
        print(f"[Sync] Using SSH key for {USER}@{SERVER}")
    else:
        password = os.environ.get("SERVER_PASSWORD", "").strip()
        if password:
            print(f"[Sync] Using SERVER_PASSWORD env var for {USER}@{SERVER}")
        else:
            password = getpass(f"Password for {USER}@{SERVER}: ")

    print(f"Connecting to {SERVER}...")
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(AutoAddPolicy())

    try:
        ssh.connect(SERVER, username=USER, password=password, pkey=pkey, timeout=15)
    except Exception as e:
        print(f"Connection failed: {e}")
        sys.exit(1)

    sftp = ssh.open_sftp()

    files = _collect_files()
    print(f"Found {len(files)} files to sync")
    for local_path in files:
        local_full = os.path.join(_project_root, local_path)

        remote_full = f"{REMOTE_BASE}/{local_path.replace(os.sep, '/')}"

        # Ensure remote directory exists
        remote_dir = os.path.dirname(remote_full)
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            ssh.exec_command(f"mkdir -p {remote_dir}")

        try:
            sftp.put(local_full, remote_full)
            size_kb = os.path.getsize(local_full) / 1024
            print(f"  OK  {local_path} ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  FAIL {local_path}: {e}")

    sftp.close()

    print(f"\nRestarting service...")
    stdin, stdout, stderr = ssh.exec_command(RESTART_CMD)
    out = stdout.read().decode()
    err = stderr.read().decode()
    print(out)
    if err:
        print(err)

    ssh.close()
    print("Done. Visit http://150.158.150.98")


if __name__ == "__main__":
    main()
