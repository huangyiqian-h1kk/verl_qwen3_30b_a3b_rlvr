#!/usr/bin/env python3
"""Project entry point (also usable with Python -I)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_mopd.cli import main

if __name__ == "__main__":
    main()
