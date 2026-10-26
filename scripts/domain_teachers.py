#!/usr/bin/env python3
"""Use the existing multidomain Python overlay; no dependency installation."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_teachers.cli import main

if __name__ == "__main__":
    main()
