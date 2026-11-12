"""Diagnosis must expose earlier engine errors without touching experiment state."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/inspect_mopd_failures.py"
spec = importlib.util.spec_from_file_location("failure_inspection", SCRIPT)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path):
    root = tmp_path / "campaign"
    write_json(root / "manifest.json", {"identity": "test"})
    states = {}
    for arm in ("domain_mix", "domain_only"):
        folder = root / "qualification" / arm / "attempts" / "current"
        folder.mkdir(parents=True)
        mix = arm == "domain_mix"
        error = "Engine core initialization failed" if mix else "1"
        trace = "Traceback\nRuntimeError: engine failed" if mix else "Traceback\nSystemExit: 1"
        write_json(folder / "result.json", {"status": "FAIL", "error": error, "traceback": trace})
        states["qualification:" + arm] = {"status": "BLOCKED", "phase": "qualification", "arm": arm,
            "job": {"folder": str(folder), "job_id": "fixture-mix" if mix else "fixture-only"},
            "pbs": {"job_state": "F", "Exit_status": 1 if mix else 271}}
        log = ("init\n" * 100 + "ERROR EngineCore_DP0\nTraceback (most recent call last):\n"
               + "ERROR worker traceback frame\n" * 90
               + "(Worker_TP0 pid=123) ImportError: ROOT_CAUSE_FROM_ENGINE\n"
               + "cleanup\n" * 800
               + "RuntimeError: Engine core initialization failed. See root cause above.\n")
        (folder / "pbs.log").write_text(log if mix else "Received SIGTERM\nSystemExit: 1\n")
    write_json(root / "states.json", states)
    return root


def fingerprints(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def run_script(root, target):
    # Support the documented pipe mode, with no installed project or scheduler.
    return subprocess.run([sys.executable, "-I", "-", "--manifest", str(root / "manifest.json"),
                           "--output", str(target)], input=SCRIPT.read_text(), text=True,
                          capture_output=True, env=dict(os.environ, PATH=""), timeout=10)


def test_isolated_report_exposes_inner_failure_and_271_evidence_without_mutation(tmp_path):
    root = fixture(tmp_path)
    before = fingerprints(root)
    target = tmp_path / "report.txt"
    result = run_script(root, target)
    assert result.returncode == 0, result.stderr
    text = target.read_text()
    assert "ImportError: ROOT_CAUSE_FROM_ENGINE" in text
    assert "Engine core initialization failed" in text
    assert "SystemExit: 1" in text and "PBS=271，同时 result.json=FAIL" in text
    assert "fixture-mix" in text and "fixture-only" in text
    assert "RESULT_REPORT=" in result.stdout
    assert fingerprints(root) == before


def test_large_logs_keep_first_error_and_tail_and_disclose_unread_middle(tmp_path):
    log = tmp_path / "pbs.log"
    log.write_text("RuntimeError: EARLY_ROOT\n" + "padding\n" * 5000 + "SystemExit: FINAL_EXIT\n")
    text = inspector.log_excerpt(log, scan_bytes=1024)
    assert "EARLY_ROOT" in text and "FINAL_EXIT" in text
    assert "扫描未覆盖中段" in text and "本次读取 bytes=1024" in text


@pytest.mark.parametrize("inside", [False, True])
def test_report_cannot_overwrite_or_write_into_campaign(tmp_path, inside):
    root = fixture(tmp_path)
    target = root / "unexpected.txt" if inside else tmp_path / "existing.txt"
    if not inside:
        target.write_text("keep this report")
    before = fingerprints(root)
    result = run_script(root, target)
    assert result.returncode != 0
    assert fingerprints(root) == before
    assert not target.exists() if inside else target.read_text() == "keep this report"


def test_missing_corrupt_and_outside_paths_do_not_hide_other_jobs(tmp_path):
    root = fixture(tmp_path)
    states = json.loads((root / "states.json").read_text())
    mix = Path(states["qualification:domain_mix"]["job"]["folder"])
    (mix / "result.json").write_text("{unfinished")
    external = tmp_path / "not_campaign"
    write_json(external / "result.json", {"error": "MUST_NOT_COLLECT"})
    states["qualification:domain_only"]["previous_job"] = {"folder": str(external), "job_id": "old"}
    write_json(root / "states.json", states)
    text, _ = inspector.build_report(root / "manifest.json")
    assert "ROOT_CAUSE_FROM_ENGINE" in text and "SystemExit: 1" in text
    assert "inspection_error" in text and "Refusing a path outside" in text
    assert "MUST_NOT_COLLECT" not in text
