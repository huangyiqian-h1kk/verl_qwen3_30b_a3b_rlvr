"""Install a versioned Math compatibility fix for final tests, never training."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import uuid

from .common import (
    below,
    digest,
    file_sha,
    files,
    is_fenced,
    lock,
    need,
    now,
    read,
    verify_files,
    write,
)
from d4_training_continuation.scheduler import PBS, finished

POLICY = "math-test-and-typeguard-v1"
OLD_POLICY = "math-verify-0.9-isolated-exact-product-timeout-zero-v2"
NEW_POLICY = OLD_POLICY + "+and-typeguard-v1"
ERROR = "AttributeError: 'Equality' object has no attribute '_unsorted_args'"
PINNED = {
    "__init__.py": "1cf74f0405b5172268a00baa9462911a20977cf80ea28653102ad8513594d379",
    "backend.py": "51618d090a98b149e0d761fbdf9155e2b74a219fef7e36219584dfcceebd1c4c",
    "backend_worker.py": "151132ce991e559ceee4a1d887e84eb61f795176092d934eb062dd70cd164d15",
    "exact_math.py": "3ce669961d08f876ddee1f3270709c6f64c98232873049317ce877752abed77c",
    "math_score.py": "02e1eb26d2add79f4def0504f06e832b3e7546db0c7b7a0afbf446431dc1c3ea",
    "selfcheck.py": "914f796b61cb5f15bd3b438de02143923e001ba08865665f54f84bd2b790005d",
    "verifier_backend.zip": "067bb4557182f7b3159006dbaa11b449b617b175784a15e84b7895f65aea8393",
}


def revision_directory(c, identity):
    need(
        len(identity) == 64 and all(x in "0123456789abcdef" for x in identity),
        "Invalid verifier revision",
    )
    target = Path(c["path"]) / "test_verifier_revisions" / identity
    need(
        target.resolve() == target and not target.is_symlink(),
        "Verifier revision path changed",
    )
    return target


def prepare_revision(c):
    source = Path(c["path"]) / "template/runtime/d4_db_mathfix"
    for name, expected in PINNED.items():
        need(
            not (source / name).is_symlink() and file_sha(source / name) == expected,
            "Unsupported frozen Math source: " + name,
        )
    root = Path(c["path"]) / "test_verifier_revisions"
    root.mkdir(exist_ok=True)
    stage = root / (".staging_" + uuid.uuid4().hex)
    package = stage / "runtime/d4_db_mathfix"
    try:
        shutil.copytree(
            source, package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        worker = package / "backend_worker.py"
        text = worker.read_text()
        needle = "        activate()\n"
        need(text.count(needle) == 1, "Unknown isolated activation entry")
        worker.write_text(
            text.replace(
                needle,
                needle
                + "        from d4_db_mathfix.and_compat import install\n        install()\n",
            )
        )
        init = package / "__init__.py"
        need(init.read_text().count(OLD_POLICY) == 1, "Unknown frozen scoring policy")
        init.write_text(init.read_text().replace(OLD_POLICY, NEW_POLICY))
        shutil.copy2(
            Path(__file__).with_name("math_and_compat.py"), package / "and_compat.py"
        )
        payload = dict(
            policy=POLICY,
            campaign_identity=c["identity"],
            parent_verifier=c["contract"]["verifier_identity"],
            source_files=files(source),
            files=files(package),
        )
        revision = dict(payload, identity=digest(payload))
        target = revision_directory(c, revision["identity"])
        write(stage / "manifest.json", revision)
        if target.exists():
            need(
                read(target / "manifest.json") == revision, "Verifier manifest changed"
            )
            verify_files(target / "runtime/d4_db_mathfix", revision["files"])
        else:
            stage.rename(target)
        return revision
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def worker_score(python, package, expected, generated):
    result = subprocess.run(
        [str(python), "-I", str(Path(package) / "backend_worker.py")],
        input=json.dumps(dict(expected=expected, generated=generated, seconds=5)),
        capture_output=True,
        text=True,
        timeout=25,
    )
    need(result.returncode == 0, "Verifier process failed: " + result.stderr[-2000:])
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    final = [row for row in rows if row.get("status") != "phase"]
    need(len(final) == 1, "Invalid verifier protocol")
    return final[0]


def diagnostic(path):
    record = read(path)
    expected = hashlib.sha256(
        json.dumps(
            [record["expected_answer"], record["generated_text"]], ensure_ascii=False
        ).encode()
    ).hexdigest()
    need(record.get("input_sha256") == expected, "Diagnostic input checksum changed")
    need(
        record.get("policy") == OLD_POLICY
        and record.get("status") == "INFRASTRUCTURE_ERROR"
        and record.get("resolved") is False
        and record.get("attempts")
        and record["attempts"][-1].get("error") == ERROR
        and record["attempts"][-1].get("phase") == "comparison",
        "Diagnostic is not the known And/Equality comparison failure",
    )
    return record


def preflight(c, revision, path):
    record = diagnostic(path)
    target = revision_directory(c, revision["identity"])
    proof_path = target / "cpu_preflight.json"
    if proof_path.exists():
        proof = read(proof_path)
        need(
            proof["status"] == "PASS"
            and proof["revision"] == revision["identity"]
            and proof["diagnostic_sha256"] == file_sha(path),
            "Preflight binding changed",
        )
        return proof
    python = Path(c["contract"]["overlay"]) / "bin/python"
    original = Path(c["path"]) / "template/runtime/d4_db_mathfix"
    before = worker_score(
        python, original, record["expected_answer"], record["generated_text"]
    )
    need(
        before.get("status") == "error" and before.get("error") == ERROR,
        "Original pinned worker did not reproduce the reported failure: "
        + repr(before),
    )
    package = target / "runtime/d4_db_mathfix"
    after = worker_score(
        python, package, record["expected_answer"], record["generated_text"]
    )
    need(
        after.get("status") == "ok" and after.get("score") in (0.0, 1.0),
        "Corrected worker did not score the original sample: " + repr(after),
    )
    # The original full checker includes correct/incorrect products and Gamma pairs.
    command = (
        "import sys;sys.path.insert(0," + repr(str(target / "runtime")) + ");"
        "from d4_db_mathfix.selfcheck import main;main()"
    )
    check = subprocess.run(
        [str(python), "-I", "-c", command], capture_output=True, text=True, timeout=240
    )
    (target / "selfcheck.log").write_text(check.stdout + check.stderr)
    need(
        check.returncode == 0,
        "Pinned Math selfcheck failed; see " + str(target / "selfcheck.log"),
    )
    cases = (
        ("chain_correct", "x=1", r"\boxed{x=1=1}", 1.0),
        ("chain_incorrect", "x=2", r"\boxed{x=1=1}", 0.0),
        ("inequality_correct", "x>1", r"\boxed{x>1}", 1.0),
        ("inequality_incorrect", "x>2", r"\boxed{x>1}", 0.0),
    )
    results = {}
    for name, gold, pred, score in cases:
        value = worker_score(python, package, gold, pred)
        need(
            value.get("status") == "ok" and value.get("score") == score,
            "And regression failed: " + name + " " + repr(value),
        )
        results[name] = value
    proof = dict(
        status="PASS",
        revision=revision["identity"],
        diagnostic_sha256=file_sha(path),
        input_sha256=record["input_sha256"],
        original=before,
        corrected=after,
        regressions=results,
        pinned_selfcheck="PASS",
        at_utc=now(),
        model_weights_loaded=False,
        gpu_jobs_submitted=0,
    )
    write(proof_path, proof)
    return proof


def active_revision(c):
    pointer = Path(c["path"]) / "test_verifier_revision.json"
    if not pointer.exists():
        return None
    selected = read(pointer)
    target = revision_directory(c, selected["identity"])
    revision = read(target / "manifest.json")
    need(
        revision["identity"] == selected["identity"]
        and digest({k: v for k, v in revision.items() if k != "identity"})
        == revision["identity"]
        and revision["policy"] == POLICY
        and revision["campaign_identity"] == c["identity"]
        and revision["parent_verifier"] == c["contract"]["verifier_identity"],
        "Test verifier revision binding changed",
    )
    need(
        file_sha(target / "cpu_preflight.json") == selected["preflight_sha256"],
        "Test verifier preflight changed",
    )
    proof = read(target / "cpu_preflight.json")
    need(
        proof["status"] == "PASS" and proof["revision"] == revision["identity"],
        "Test verifier preflight is missing",
    )
    verify_files(target / "runtime/d4_db_mathfix", revision["files"])
    return revision


def apply_to_job(c, folder):
    revision = active_revision(c)
    if revision is None:
        return None
    source = revision_directory(c, revision["identity"]) / "runtime/d4_db_mathfix"
    target = Path(folder) / "runtime/d4_db_mathfix"
    need(files(target) == revision["source_files"], "Job has another Math source")
    shutil.copytree(
        source,
        target,
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    need(files(target) == revision["files"], "Test verifier copy changed")
    return revision["identity"]


def verify_failed_test(c, domain, path, pbs):
    from . import controller
    from .source import validate_request

    need(not is_fenced(c["path"], domain), "Explicit stop fence is present")
    state = read(controller.state_path(c, domain))
    need(
        state["phase"] == "test" and state.get("job") and not state.get("pending"),
        "Repair requires an owned failed final-test job",
    )
    job = state["job"]
    folder = below(job["folder"], Path(c["path"]) / "runs" / domain / "jobs")
    request = validate_request(folder / "request.json")
    need(
        request["role"] == "test"
        and request["domain"] == domain
        and request["campaign_identity"] == c["identity"],
        "Failed test belongs to another run",
    )
    record = diagnostic(path)
    need(record["pbs_job_id"] == job["job_id"], "Diagnostic belongs to another PBS job")
    originals = list(
        (folder / "math_diagnostics").glob(record["input_sha256"][:16] + "_*.json")
    )
    need(
        any(file_sha(p) == file_sha(path) for p in originals),
        "Diagnostic is not from this job",
    )
    ids = read(Path(c["path"]) / "data/manifest.json")["domains"]["math"]["test"]["ids"]
    need(record["sample_id"] in ids, "Diagnostic is not a frozen Math test sample")
    row = pbs.observe(job)
    need(
        finished(row) and str(row.get("Exit_status")) == "1",
        "Test termination is not confirmed",
    )
    need(not is_fenced(c["path"], domain), "Stop fence appeared during observation")
    return state, row


def repair(path, domain, diagnostic_path, pbs=None, wait_seconds=20):
    from . import controller, retry_repair
    from .source import load_campaign

    root, diagnostic_path = Path(path).resolve(), Path(diagnostic_path).resolve()
    pbs = pbs or PBS()
    with lock(root / "math_test_repair.lock"):
        c = load_campaign(root)
        need(not is_fenced(root, domain), "Explicit stop fence is present")
        revision = prepare_revision(c)
        state = read(controller.state_path(c, domain))
        receipt_path = root / "repairs" / POLICY / (domain + ".json")
        if state.get("test_verifier_revision") == revision["identity"]:
            need(
                active_revision(c) == revision and receipt_path.exists(),
                "Incomplete installed repair",
            )
            return dict(
                status="ALREADY_INSTALLED",
                revision=revision["identity"],
                guardian=retry_repair.repair(root, pbs=pbs, wait_seconds=wait_seconds),
            )
        before, _ = verify_failed_test(c, domain, diagnostic_path, pbs)
        proof = preflight(c, revision, diagnostic_path)

        def install(bound):
            current, observation = verify_failed_test(
                bound, domain, diagnostic_path, pbs
            )
            need(current["job"] == before["job"], "Test job changed during preflight")
            receipt = dict(
                campaign_identity=c["identity"],
                revision=revision["identity"],
                before=current,
                observation=observation,
                preflight=proof,
                diagnostic_sha256=file_sha(diagnostic_path),
                at_utc=now(),
            )
            if receipt_path.exists():
                previous = read(receipt_path)
                need(
                    previous["revision"] == revision["identity"]
                    and previous["before"]["job"] == current["job"],
                    "Conflicting test repair receipt",
                )
            else:
                write(receipt_path, receipt)
            write(
                root / "test_verifier_revision.json",
                dict(
                    identity=revision["identity"],
                    preflight_sha256=file_sha(
                        revision_directory(c, revision["identity"])
                        / "cpu_preflight.json"
                    ),
                ),
            )
            need(not is_fenced(root, domain), "Stop fence appeared during installation")
            if current.get("error"):
                current["previous_error"] = dict(
                    message=current.pop("error"),
                    observed_utc=current.get("observed_utc"),
                )
            current.update(
                status="READY",
                job=None,
                previous_job=current["job"],
                test_verifier_revision=revision["identity"],
                recovered_utc=now(),
            )
            write(controller.state_path(c, domain), current)
            return dict(
                domain=domain,
                status="REARMED_TEST",
                previous_job=before["job"]["job_id"],
            )

        guardian = retry_repair.repair(
            root, pbs=pbs, wait_seconds=wait_seconds, install_under_lease=install
        )
        return dict(
            status="MATH_TEST_REPAIR_INSTALLED",
            revision=revision["identity"],
            scope="final tests only; training and validation unchanged",
            preflight=proof,
            guardian=guardian,
            cache="new verifier identity; old evaluation journals preserved, not reused",
        )
