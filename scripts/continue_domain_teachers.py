#!/usr/bin/env python3
"""Register IF70 test, Math step50 continuation, and the explicit CP80 choice."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_teachers import followup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status"))
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    result = (followup.launch if args.action == "start" else followup.report)(
        args.campaign
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
