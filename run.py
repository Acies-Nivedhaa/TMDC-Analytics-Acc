# run.py
import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
VENV = ROOT / ".venv"
IS_WIN = os.name == "nt"

PY_IN_VENV = VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")
PIP_ARGS = [str(PY_IN_VENV), "-m", "pip"]
STREAMLIT_CMD = [str(PY_IN_VENV), "-m", "streamlit", "run", "app.py"]

def sh(cmd, **kw):
    print(" ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=ROOT, **kw)

def ensure_venv():
    if PY_IN_VENV.exists():
        return
    print("Creating virtual environment in .venv ...")
    sh([sys.executable, "-m", "venv", str(VENV)])

def install_deps():
    sh(PIP_ARGS + ["install", "--upgrade", "pip", "wheel"])
    req = ROOT / "requirements.txt"
    if not req.exists():
        sys.exit("requirements.txt not found at repo root.")
    sh(PIP_ARGS + ["install", "-r", str(req)])

def run_app():
    # Forward any extra args to Streamlit (optional)
    extra = sys.argv[1:]
    sh(STREAMLIT_CMD + extra)

if __name__ == "__main__":
    ensure_venv()
    install_deps()
    run_app()
