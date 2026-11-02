#!/usr/bin/env python3
"""Prepare frozen DeepMath held-outs and evaluate base/Math20/Math50 with the guardian."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from d4_teachers import controller, deepmath_data, deepmath_evaluation
from d4_teachers.source import load_campaign


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("discover", "prepare", "start", "run", "status", "report")
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        help="Raw DeepMath parquet or light jsonl.gz; repeatable",
    )
    parser.add_argument(
        "--holdout",
        type=Path,
        action="append",
        help="Existing DeepMath holdout parquet; repeatable",
    )
    parser.add_argument("--exclude-probe", type=Path, action="append")
    parser.add_argument(
        "--size",
        type=int,
        default=256,
        help="Questions per split, stratified by frozen DeepMath L8/L9 counts",
    )
    args = parser.parse_args()
    c = load_campaign(args.campaign)
    if args.action == "discover":
        result = deepmath_data.discover(c)
    elif args.action in ("prepare", "run"):
        inputs = None
        if any((args.input, args.holdout, args.exclude_probe)):
            inputs = deepmath_data.discover(c)
            for name, value in (
                ("raw", args.input),
                ("holdout", args.holdout),
                ("exclude_probe", args.exclude_probe),
            ):
                if value is not None:
                    inputs[name] = [str(p.resolve()) for p in value]
        manifest = deepmath_data.prepare(c, inputs=inputs, size=args.size)
        result = dict(
            status="DEEPMATH_DATA_PREPARED",
            identity=manifest["identity"],
            splits={
                s: {k: v for k, v in info.items() if k != "ids"}
                for s, info in manifest["splits"].items()
            },
            audit=manifest["audit"],
            model_weights_loaded=False,
            gpu_jobs_submitted=0,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if args.action == "prepare":
            return
        result = deepmath_evaluation.launch(c["path"])
    elif args.action == "start":
        result = deepmath_evaluation.launch(c["path"])
    elif args.action == "status":
        state = controller.status(c["path"])
        result = {
            k: state[k] for k in ("guardian_lock_held", "heartbeat_stale", "fenced")
        }
        result["heartbeat"] = {
            k: (state["heartbeat"] or {}).get(k) for k in ("at_utc", "host", "pid")
        }
        result["tasks"] = {}
        for task in deepmath_evaluation.TARGETS:
            entry = state["states"].get(task, {})
            row = {
                k: entry.get(k)
                for k in (
                    "status",
                    "phase",
                    "waiting_for",
                    "error",
                    "no_progress_retries",
                )
            }
            row["job_id"] = (entry.get("job") or {}).get("job_id")
            run = Path(c["path"]) / "runs" / task
            progress = sorted(
                (run / "evaluations").glob("heldout_*/progress.json"),
                key=lambda p: p.stat().st_mtime_ns,
            )
            if progress:
                row["progress"] = json.loads(progress[-1].read_text())
            row["receipt_exists"] = (run / "heldout.json").is_file()
            result["tasks"][task] = row
    else:
        result = deepmath_evaluation.report(c["path"])
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
