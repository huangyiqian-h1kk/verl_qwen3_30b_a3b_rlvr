"""Read phase timing and sampled physical GPU use without importing GPU libraries."""
import csv
import json
from pathlib import Path
import time

from d4_teachers.common import need, read


def memory_summary(folder, expected_nodes=None, max_age_seconds=None):
    result = []
    paths = sorted(Path(folder).glob("gpu_memory_*.csv"))
    if expected_nodes is not None:
        need({p.name for p in paths} == {f"gpu_memory_{i}.csv" for i in range(expected_nodes)},
             "Physical GPU sampling is missing for part of the allocation")
    for path in paths:
        if max_age_seconds is not None:
            need(0 <= time.time() - path.stat().st_mtime <= max_age_seconds,
                 "Physical GPU sampling stopped before the attempt completed: " + path.name)
        devices = {}
        incomplete = 0
        with path.open(newline="") as stream:
            for fields in csv.reader(stream):
                if len(fields) != 6:
                    incomplete += 1
                    continue
                timestamp, index, uuid, used, total, utilization = [s.strip() for s in fields]
                try:
                    used, total = float(used), float(total)
                except ValueError:
                    incomplete += 1
                    continue
                need(0 <= used <= total and total > 0, "Invalid physical GPU memory sample")
                value = devices.setdefault(uuid, dict(index=int(index), uuid=uuid, samples=0,
                    peak_sampled_used_MiB=used, total_MiB=total, first_sample_utc=timestamp,
                    peak_sample_utc=timestamp))
                need(value["index"] == int(index) and value["total_MiB"] == total,
                     "Physical GPU identity changed inside an attempt")
                value["samples"] += 1
                value["last_sample_utc"] = timestamp
                if used > value["peak_sampled_used_MiB"]:
                    value.update(peak_sampled_used_MiB=used, peak_sample_utc=timestamp)
        if expected_nodes is not None:
            need(len(devices) == 8 and {v["index"] for v in devices.values()} == set(range(8)),
                 "Physical GPU sampling did not cover all eight GPUs on " + path.name)
        result.append(dict(file=str(path), devices=sorted(devices.values(), key=lambda r: r["index"]),
                           ignored_non_sample_rows=incomplete))
    return dict(scope="Physical device memory from nvidia-smi; 250 ms samples, not exact instantaneous peaks",
                sample_interval_ms=250, nodes=result)


def report(folder):
    folder = Path(folder).resolve()
    request = read(folder / "request.json")
    from .execution import profile
    result = dict(attempt=str(folder), arm=request["arm"], phase=request["phase"],
                  execution_profile=request.get("execution_profile", "dual_node"))
    result["result"] = read(folder / "result.json") if (folder / "result.json").exists() else None
    result["memory"] = memory_summary(folder, profile(result["execution_profile"])["physical_nodes"])
    events = folder / "phase_events.jsonl"
    stages = {}
    if events.exists():
        for line in events.read_text().splitlines():
            event = json.loads(line)
            if event["event"] == "FINISH":
                stage = stages.setdefault(event["stage"], dict(count=0, total_seconds=0.0, max_seconds=0.0))
                elapsed = event["elapsed_seconds"]
                stage["count"] += 1
                stage["total_seconds"] += elapsed
                stage["max_seconds"] = max(stage["max_seconds"], elapsed)
    result["phase_timings"] = stages
    result["completed_updates"] = [read(p) for p in sorted(folder.glob("update_*.json"),
                                                         key=lambda p: int(p.stem.split("_")[-1]))]
    result["comparison_note"] = ("Compare update durations after initialization; report validation/save/restore separately. "
                                 "Dual-node generation and teacher scoring overlap. No performance threshold is assumed.")
    return result
