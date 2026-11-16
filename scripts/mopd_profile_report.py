#!/usr/bin/env python3
"""Print JSON evidence for one or more MOPD profile attempts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_mopd.profile_report import report

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("attempt", nargs="+")
args = parser.parse_args()
print(json.dumps([report(p) for p in args.attempt], ensure_ascii=False, indent=2))
