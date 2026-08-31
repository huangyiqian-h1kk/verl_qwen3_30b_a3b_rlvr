#!/usr/bin/env python3
"""Create or verify an immutable run fingerprint before auto-resume.

``target_training_steps`` is deliberately not part of the fingerprint, so a
20-step production burn-in can be extended to the final budget without
starting a different run. All algorithmic and data settings must remain fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--item", action="append", default=[], help="immutable KEY=VALUE")
    parser.add_argument("--hash-file", action="append", default=[], help="immutable LABEL=PATH")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path = checkpoint_dir / "run_fingerprint.json"

    values: dict[str, str] = {}
    for raw in args.item:
        if "=" not in raw:
            raise SystemExit(f"[FAIL] invalid --item {raw!r}; expected KEY=VALUE")
        key, value = raw.split("=", 1)
        if not key or key in values:
            raise SystemExit(f"[FAIL] invalid or duplicate fingerprint key: {key!r}")
        values[key] = value

    files: dict[str, dict[str, str]] = {}
    for raw in args.hash_file:
        if "=" not in raw:
            raise SystemExit(f"[FAIL] invalid --hash-file {raw!r}; expected LABEL=PATH")
        label, raw_path = raw.split("=", 1)
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise SystemExit(f"[FAIL] fingerprint file missing: {path}")
        files[label] = {"path": str(path), "sha256": sha256_file(path)}

    current = {
        "schema_version": 1,
        "immutable_values": dict(sorted(values.items())),
        "immutable_files": dict(sorted(files.items())),
    }

    if fingerprint_path.exists():
        stored = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        comparable = {
            "schema_version": stored.get("schema_version"),
            "immutable_values": stored.get("immutable_values"),
            "immutable_files": stored.get("immutable_files"),
        }
        if comparable != current:
            print("[FAIL] run fingerprint mismatch; refusing unsafe auto-resume.")
            print("--- stored ---")
            print(json.dumps(comparable, indent=2, ensure_ascii=False))
            print("--- current ---")
            print(json.dumps(current, indent=2, ensure_ascii=False))
            raise SystemExit(2)
        print(f"[PASS] run fingerprint matches {fingerprint_path}")
        return

    payload = {
        **current,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_on": os.uname().nodename,
    }
    temporary = fingerprint_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, fingerprint_path)
    print(f"[PASS] created run fingerprint {fingerprint_path}")


if __name__ == "__main__":
    main()
