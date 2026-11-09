"""Evaluation-only scoring in the original verifier environment, never an RL loss."""
import asyncio
import json
from pathlib import Path


def json_scalar(value):
    # Native reward-loop metadata can contain NumPy scalars/arrays.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError("Unsupported verifier payload type: " + type(value).__name__)


async def compute_score(data_source, solution_str, ground_truth, extra_info=None, manifest_path=None, **kwargs):
    from d4_teachers.common import read
    extra_info = extra_info or {}
    if extra_info.get("split") == "train":
        raise RuntimeError("Training must bypass the task scorer entirely")
    manifest = read(manifest_path)
    worker = Path(__file__).with_name("verifier_worker.py")
    p = await asyncio.create_subprocess_exec(manifest["verifier_python"], "-I", str(worker), manifest_path,
                                             stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                             stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(p.communicate(json.dumps(dict(
            data_source=data_source, solution_str=solution_str, ground_truth=ground_truth,
            extra_info=extra_info), default=json_scalar).encode()), timeout=180)
    except BaseException:
        if p.returncode is None:
            p.kill()
        await p.wait()
        raise
    if p.returncode:
        raise RuntimeError("Frozen verifier failed: " + stderr.decode(errors="replace")[-3000:])
    result = json.loads(stdout)
    result = dict(result) if isinstance(result, dict) else {"score": result}
    result.update(mopd_source=extra_info["source"], mopd_domain=data_source,
                  mopd_sample_id=extra_info["sample_id"])
    return result
