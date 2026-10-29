"""Root conftest: makes the project root importable (config, src) when running `pytest` from any IDE."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
