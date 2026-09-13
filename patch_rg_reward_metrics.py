#!/usr/bin/env python3
"""Fix nullable reward diagnostics and the matching preflight check in place.

Run from the project root: python patch_rg_reward_metrics.py
Only Python's standard library is required. No dataset or verl source is changed.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text and old not in text:
        return text
    if text.count(old) != 1:
        raise ValueError(f"{label}: expected one matching statement; no files changed")
    return text.replace(old, new, 1)


def patch_reward(text: str) -> str:
    text = replace_once(
        text,
        '"score_error": error,',
        '"score_error": error or "",  # verl skips strings, but tries to average None.',
        "reward score_error",
    )
    # Optional descriptive fields must also remain strings on every sample.
    for key in ("rg_category", "rg_tier"):
        text = replace_once(
            text,
            f'"{key}": extra.get("{key}"),',
            f'"{key}": str(extra.get("{key}") or ""),',
            f"reward {key}",
        )
    # Preserve the word_sorting/Fraction tests and update only their no-error contract.
    text = re.sub(
        r'\b((?:result|wrong)\["score_error"\])\s+is\s+None\b',
        r'\1 == ""',
        text,
    )
    return text


def patch_preflight(text: str) -> str:
    # Accept both the previous representation and the new string-only diagnostics.
    # A nonempty error message still fails the oracle replay check.
    old = 'result.get("score_error") is None'
    new = 'result.get("score_error") in (None, "")'
    return replace_once(text, old, new, "preflight score_error check")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    changes = []
    for relative, transform in (
        ("rewards/unicode_reasoning_gym_reward_v2.py", patch_reward),
        ("scripts/preflight_reasoning_gym_training.py", patch_preflight),
    ):
        path = root / relative
        if not path.is_file():
            raise SystemExit(f"[FAIL] missing {path}; run from the repository root")
        original = path.read_text(encoding="utf-8")
        try:
            updated = transform(original)
            compile(updated, str(path), "exec")
        except (ValueError, SyntaxError) as exc:
            raise SystemExit(f"[FAIL] {relative}: {exc}") from exc
        changes.append((path, original, updated))

    # Validate both files before writing either one; keep one backup per changed file.
    for path, original, updated in changes:
        if original == updated:
            print(f"[PASS] already patched: {path.relative_to(root)}")
            continue
        backup = path.with_name(path.name + ".before_reward_metrics.bak")
        if not backup.exists():
            shutil.copy2(path, backup)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        try:
            shutil.copymode(path, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[PASS] patched: {path.relative_to(root)}")
    print("[PASS] reward metrics patch complete")


if __name__ == "__main__":
    main()
