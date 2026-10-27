"""Prepare once, CPU preflight, then directly run four independent formal teachers."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from . import DOMAINS
from .common import active, allocated_storage, is_fenced, lock, need, now, read, write
from .config import DEFAULT, load
from .source import prepare, load_campaign, make_job
from . import controller


def preflight(path):
    c = load_campaign(path)
    job = make_job(c, "math", "train")
    folder = Path(job["folder"])
    py = str(Path(c["contract"]["overlay"]) / "bin/python")
    env = dict(
        os.environ, PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"
    )
    log = folder / "cpu_preflight.log"
    with log.open("w") as stream:
        subprocess.run(
            [
                py,
                "-I",
                str(folder / "entry.py"),
                "cpu-checks",
                "--request",
                str(folder / "request.json"),
            ],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    proof = read(folder / "cpu_preflight.json")
    write(Path(path) / "preflight.json", dict(proof, job=str(folder), log=str(log)))
    return proof


def start(path, background):
    c = load_campaign(path)
    need(not is_fenced(path), "Campaign stopped; explicit reenable is required")
    proof = read(Path(path) / "preflight.json")
    need(
        proof["status"] == "PASS" and proof["campaign_identity"] == c["identity"],
        "Run CPU preflight first",
    )
    if active(Path(path) / "guardian.lock"):
        return controller.status(path)
    if not background:
        return controller.guardian(path)
    with lock(Path(path) / "launch.lock"):
        if active(Path(path) / "guardian.lock"):
            return controller.status(path)
        with (Path(path) / "guardian.log").open("a") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    str(Path(path) / "control.py"),
                    "guardian",
                    "--campaign",
                    str(path),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        write(
            Path(path) / "guardian_launch.json",
            dict(pid=process.pid, at_utc=now(), log=str(Path(path) / "guardian.log")),
        )
        # A PID is informational only. Shared-filesystem flock is the authority.
        return dict(
            status="LAUNCHED", pid=process.pid, log=str(Path(path) / "guardian.log")
        )


def enqueue(path, domain, phase):
    c = load_campaign(path)
    with lock(Path(path) / "control.lock", wait=60):
        need(not is_fenced(path, domain), "Teacher stopped; use reenable explicitly")
        state = read(controller.state_path(c, domain))
        need(
            not state.get("job") and not state.get("pending"),
            "Teacher already owns a job/intent",
        )
        controller.artifact(c, domain, "train" if phase == "export" else "export")
        state.update(
            status="READY", phase=phase, no_progress_retries=0, last_progress=0
        )
        write(controller.state_path(c, domain), state)
    return start(path, True)


def collect(path):
    c = load_campaign(path, verify=False)
    root = Path(path)
    write(root / "storage.json", allocated_storage(root))
    target = root / ("teacher_diagnostics_" + str(time.time_ns()) + ".tar.gz")
    with tarfile.open(target, "w:gz") as tar:
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.is_symlink() or file == target:
                continue
            parts = file.relative_to(root).parts
            if any(
                p
                in (
                    "checkpoints",
                    "best_weights",
                    "artifacts",
                    "export_candidate",
                    "__pycache__",
                    ".venv-multidomain",
                )
                or p.startswith((".staging_", ".pin_"))
                for p in parts
            ):
                continue
            if file.suffix in (
                ".json",
                ".jsonl",
                ".log",
                ".txt",
                ".yaml",
                ".yml",
                ".py",
                ".sh",
                ".pbs",
            ):
                tar.add(file, arcname=str(file.relative_to(root)), recursive=False)
    return dict(
        path=str(target), bytes=target.stat().st_size, campaign=c["policy"]["campaign"]
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=(
            "prepare",
            "preflight",
            "start",
            "resume",
            "guardian",
            "status",
            "stop",
            "reenable",
            "export-best",
            "evaluate",
            "collect",
            "storage",
            "repair-retries",
        ),
    )
    p.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="Shared filesystem campaign directory",
    )
    p.add_argument("--config", type=Path, default=DEFAULT)
    p.add_argument("--domain", choices=(*DOMAINS, "base"))
    p.add_argument("--max-concurrent", type=int, choices=range(1, 5))
    p.add_argument("--background", action="store_true")
    a = p.parse_args(argv)
    need(
        a.max_concurrent is None or a.action == "prepare",
        "Set --max-concurrent at prepare time",
    )
    need(
        a.domain is None or a.action in ("stop", "reenable", "export-best", "evaluate"),
        "--domain is supported for stop, reenable, export-best and evaluate",
    )
    a.campaign = a.campaign.resolve()
    if a.action == "prepare":
        policy = load(a.config)
        if a.max_concurrent is not None:
            policy["max_concurrent_jobs"] = a.max_concurrent
        c = prepare(a.campaign, policy)
        result = dict(
            status="PREPARED", identity=c["identity"], campaign=str(a.campaign)
        )
    elif a.action == "preflight":
        result = preflight(a.campaign)
    elif a.action in ("start", "resume"):
        result = start(a.campaign, a.background)
    elif a.action == "guardian":
        result = controller.guardian(a.campaign)
    elif a.action in ("export-best", "evaluate"):
        need(a.domain in DOMAINS, "Select one teacher with --domain")
        result = enqueue(
            a.campaign, a.domain, "export" if a.action == "export-best" else "test"
        )
    elif a.action in ("stop", "reenable"):
        result = getattr(controller, a.action)(a.campaign, a.domain)
    elif a.action == "status":
        result = controller.status(a.campaign)
    elif a.action == "storage":
        result = allocated_storage(a.campaign)
    elif a.action == "repair-retries":
        from .retry_repair import repair

        result = repair(a.campaign)
    else:
        result = collect(a.campaign)
    print(json.dumps(result, ensure_ascii=False, indent=2))
