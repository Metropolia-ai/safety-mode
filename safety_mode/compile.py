"""Entry point for `python -m safety_mode.compile`; the code is in cli.py."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
