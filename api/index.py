"""Vercel entrypoint: the Safety Mode demo API (safety_mode/webapp.py) as a Python serverless function."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from safety_mode.webapp import app  # noqa: E402,F401
