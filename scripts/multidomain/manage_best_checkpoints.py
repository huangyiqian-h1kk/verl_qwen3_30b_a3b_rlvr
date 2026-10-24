#!/usr/bin/env python3
"""Rescue and retain the two 0390 setups' best models without changing training."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from d4_training_continuation.best_archiver import main

if __name__ == "__main__":
    main()
