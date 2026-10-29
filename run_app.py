"""Convenience launcher (used by Visual Studio's Start button): runs `streamlit run app.py`
with the CURRENT Python interpreter. It never installs anything."""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

if __name__ == "__main__":
    if importlib.util.find_spec("streamlit") is None:
        sys.exit("Streamlit is not available in the selected Python environment. "
                 "Select the environment that already has your libraries, then try again.")
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py")], cwd=str(ROOT)))
