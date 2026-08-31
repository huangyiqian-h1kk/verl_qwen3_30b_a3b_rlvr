#!/usr/bin/env python3
"""Write an atomic step-0 pointer for later snapshot/probing workflows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--system-prompt-file", required=True)
    parser.add_argument("--verl-src", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    prompt_path = Path(args.system_prompt_file).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    commit = subprocess.run(
        ["git", "-C", str(Path(args.verl_src).resolve()), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "step": 0,
        "kind": "starting_model_pointer",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_on": os.uname().nodename,
        "experiment_name": args.experiment_name,
        "model_key": args.model_key,
        "model_path": str(model_path),
        "model_config_sha256": sha256_file(model_path / "config.json"),
        "system_prompt_path": str(prompt_path),
        "system_prompt_sha256": sha256_file(prompt_path),
        "enable_thinking": False,
        "verl_commit": commit,
        "note": (
            "No weights are duplicated for step 0. Use model_path with the exact "
            "system prompt and enable_thinking=false."
        ),
    }
    destination = output_dir / "baseline_step_0.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, destination)
    print(f"[PASS] wrote {destination}")


if __name__ == "__main__":
    main()
