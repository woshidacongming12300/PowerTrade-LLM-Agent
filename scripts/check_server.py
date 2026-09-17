"""Check if server has new code, rebuild if needed."""
import os
import sys
from getpass import getpass
from paramiko import SSHClient, AutoAddPolicy, RSAKey

HOST = os.environ.get("EPF_SERVER", "").strip()
USER = os.environ.get("EPF_USER", "ubuntu")

if not HOST:
    print("目标服务器未配置。请先设置环境变量：")
    print("  Windows:  set EPF_SERVER=<服务器地址或 IP>")
    print("  Linux:    export EPF_SERVER=<服务器地址或 IP>")
    print("可选：EPF_USER（默认 ubuntu）")
    sys.exit(1)

# Find SSH key
pkey = None
for kp in [os.path.expanduser("~/.ssh/id_rsa"), os.path.expanduser("~/.ssh/id_ed25519")]:
    if os.path.exists(kp):
        try:
            pkey = RSAKey.from_private_key_file(kp)
            break
        except Exception:
            pass

password = None
if not pkey:
    password = os.environ.get("SERVER_PASSWORD", "").strip()

ssh = SSHClient()
ssh.set_missing_host_key_policy(AutoAddPolicy())

try:
    ssh.connect(HOST, username=USER, password=password, pkey=pkey, timeout=15)
except Exception:
    if not pkey:
        password = getpass(f"Password for {USER}@{HOST}: ")
        ssh.connect(HOST, username=USER, password=password, pkey=pkey, timeout=15)
    else:
        raise

# 1. Check files on host
print("[1/3] Checking files on host...")
stdin, stdout, stderr = ssh.exec_command(
    "ls /opt/epf-agent/algorithms/weather_data.py /opt/epf-agent/algorithms/lstm_epf_multivariate.pth 2>&1"
)
out = stdout.read().decode().strip()
err = stderr.read().decode().strip()
print(f"  {out or err}")

if "cannot access" in out.lower() or "No such file" in out:
    print("\n  Files not synced! Run: python scripts/sync_to_server.py")
    ssh.close()
    sys.exit(1)

# 2. Check systemd service path for correct Python
print("\n[2/3] Checking service configuration...")
stdin, stdout, stderr = ssh.exec_command(
    "cat /etc/systemd/system/epf-agent.service 2>/dev/null || "
    "cat /lib/systemd/system/epf-agent.service 2>/dev/null || "
    "echo NO_SERVICE_FILE"
)
out = stdout.read().decode().strip()
print(f"  {out[:300]}")

# 3. Restart service
print("\n[3/3] Restarting service...")
# First quick resync of the critical file
stdin, stdout, stderr = ssh.exec_command("cat > /tmp/epf_fix.py")
with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "algorithms", "epf_predictor.py")) as f:
    ssh.exec_command(f"cat > /opt/epf-agent/algorithms/epf_predictor.py << 'FIXEOF'\n{f.read()}\nFIXEOF")

stdin, stdout, stderr = ssh.exec_command(
    "sudo systemctl restart epf-agent && "
    "sleep 3 && "
    "sudo systemctl status epf-agent --no-pager"
)
out = stdout.read().decode()
err = stderr.read().decode()
print(out[-800:] if len(out) > 800 else out)
if err:
    print("STDERR:", err[-300:] if len(err) > 300 else err)

if "RESTART_OK" in out:
    print(f"\n[DONE] Server updated! Open http://{HOST} and press Ctrl+F5")
else:
    print("\n[FAIL] Something went wrong. Check the output above.")

ssh.close()
