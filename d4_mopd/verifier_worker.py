"""Launched with the old Python -I; isolate its imports from the new verl stack."""
import asyncio
import contextlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys


def main():
    manifest = json.loads(Path(sys.argv[1]).read_text())
    root = Path(manifest["manifest_path"]).parent / "verifier"
    sys.path[:0] = [str(root / "runtime"), str(root)]
    with contextlib.redirect_stdout(sys.stderr):
        module = importlib.import_module("md_reward_roundoff_0390")
        payload = json.load(sys.stdin)
        if payload.get("import_only"):
            result = {"status": "PASS", "module": module.__file__, "python": sys.executable}
        else:
            result = module.compute_score(**payload)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
            score = result.get("score") if isinstance(result, dict) else result
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                raise RuntimeError("Non-finite or missing verifier score")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
