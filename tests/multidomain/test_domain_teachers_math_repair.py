"""Actual pinned parser/worker regressions and guarded final-test deployment."""

import getpass
import hashlib
from importlib import metadata, util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from d4_teachers import controller, exporting, math_test_repair as repair, source
from d4_teachers.common import (
    active,
    digest,
    file_sha,
    files,
    model_inventory,
    read,
    write,
)
from d4_training_continuation.scheduler import PBS
import test_domain_teachers_retry_repair as retry_fixtures

ORIGINAL = Path(__file__).parent / "fixtures/math_test_repair/original"
GOLD = r"\Gamma(1/m)"
PRED = r"\boxed{a=\Gamma(1/m)=\Gamma(1/m)}"


class MathRepair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive_dir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.archive_dir.cleanup)
        cls.archive = Path(cls.archive_dir.name) / "verifier_backend.zip"
        dependencies = {
            "math-verify": ("math_verify", "0.9.0"),
            "latex2sympy2_extended": ("latex2sympy2_extended", "1.11.0"),
            "sympy": ("sympy", "1.14.0"),
            "mpmath": ("mpmath", "1.3.0"),
            "antlr4-python3-runtime": ("antlr4", "4.13.2"),
        }
        with zipfile.ZipFile(cls.archive, "w", zipfile.ZIP_DEFLATED) as archive:
            for distribution, (module, expected) in dependencies.items():
                if metadata.version(distribution) != expected:
                    raise RuntimeError(
                        "Tests require pinned " + distribution + "==" + expected
                    )
                package = Path(util.find_spec(module).origin).parent
                info = Path(metadata.distribution(distribution)._path)
                for directory in (package, info):
                    for file in sorted(directory.rglob("*")):
                        if (
                            file.is_file()
                            and "__pycache__" not in file.parts
                            and file.suffix != ".pyc"
                        ):
                            archive.write(file, file.relative_to(directory.parent))

    def setUp(self):
        # Reuse the existing real-process/PBS-executable fixture without inheriting
        # its test methods. All scheduler mutations are recorded and inspectable.
        self.harness = retry_fixtures.RetryRepair(methodName="runTest")
        self.harness.setUp()
        self.addCleanup(self.harness.doCleanups)
        h = self.harness
        self.c, self.root, self.campaign = h.c, h.root, h.campaign
        package = self.campaign / "template/runtime/d4_db_mathfix"
        shutil.copytree(ORIGINAL, package, dirs_exist_ok=True)
        shutil.copy2(self.archive, package / "verifier_backend.zip")
        backend = package / "backend.py"
        backend.write_text(
            backend.read_text().replace(
                repair.PINNED["verifier_backend.zip"], file_sha(self.archive)
            )
        )
        # Production uses its original immutable archive; this fixture packs the
        # same five versions from installed distributions, with a different ZIP hash.
        pins = patch.dict(
            repair.PINNED,
            {
                "backend.py": file_sha(backend),
                "verifier_backend.zip": file_sha(self.archive),
            },
        )
        pins.start()
        self.addCleanup(pins.stop)
        self.c["contract"]["template_files"] = files(self.campaign / "template")
        self.c["identity"] = digest(self.c["contract"])
        write(self.campaign / "campaign.json", self.c)
        write(
            self.campaign / "preflight.json",
            dict(status="PASS", campaign_identity=self.c["identity"]),
        )
        cp_run = self.campaign / "runs/conversational_pivot"
        model = cp_run / "artifacts/hf_step_50"
        model.mkdir(parents=True)
        write(model / "config.json", {"model_type": "fixture"})
        (model / "weights.safetensors").write_bytes(b"test-exported-weights")
        inventory = model_inventory(model)
        self.export = dict(
            status="PASS",
            path=str(model),
            step=50,
            files=inventory,
            weights_identity=digest(inventory),
            selected=dict(weights_identity="selected", score=0.40625),
            load_proof=dict(
                status="PASS",
                all_weights_loaded=True,
                finite_forward=True,
                selected_identity="selected",
            ),
            lineage="cp",
            verifier_identity=self.c["contract"]["verifier_identity"],
        )
        write(cp_run / "export.json", self.export)
        rows = {}
        for i, domain in enumerate(controller.TASKS[:-1], 1):
            job = dict(
                source.make_job(
                    self.c,
                    domain,
                    "test" if domain == "conversational_pivot" else "train",
                ),
                job_id=f"{i}.pbs1",
            )
            h.jobs[domain] = job
            state = read(controller.state_path(self.c, domain))
            state.update(status="RUNNING", job=job, pending=None, phase="train")
            row = dict(
                Job_Owner=getpass.getuser() + "@login",
                queue="R9920261000",
                Job_Name=job["job_name"],
                Submit_arguments=str(Path(job["folder"]) / "stage.pbs"),
                job_state="R",
            )
            if domain in ("math", "if"):
                state.update(status="DONE", phase="test", previous_job=job, job=None)
            if domain == "conversational_pivot":
                state.update(
                    status="BLOCKED_APPLICATION",
                    phase="test",
                    error="And parser failure",
                )
                row.update(job_state="F", Exit_status=1)
            write(controller.state_path(self.c, domain), state)
            rows[job["job_id"]] = row
        write(self.root / "pbs_rows.json", rows)
        job = h.jobs["conversational_pivot"]
        checksum = hashlib.sha256(
            json.dumps([GOLD, PRED], ensure_ascii=False).encode()
        ).hexdigest()
        record = dict(
            policy=repair.OLD_POLICY,
            expected_answer=GOLD,
            generated_text=PRED,
            input_sha256=checksum,
            pbs_job_id=job["job_id"],
            sample_id="test_math_0",
            status="INFRASTRUCTURE_ERROR",
            resolved=False,
            attempts=[dict(status="error", phase="comparison", error=repair.ERROR)],
        )
        self.diagnostic = (
            Path(job["folder"]) / "math_diagnostics" / (checksum[:16] + "_fixture.json")
        )
        write(self.diagnostic, record)

    def record_process(self, popen):
        def record(*args, **kwargs):
            child = popen(*args, **kwargs)
            self.harness.children.append(child)
            return child

        return record

    def test_original_worker_crashes_and_corrected_worker_scores_positive_and_negative(
        self,
    ):
        revision = repair.prepare_revision(self.c)
        proof = repair.preflight(self.c, revision, self.diagnostic)
        self.assertEqual(proof["original"]["error"], repair.ERROR)
        self.assertEqual(proof["corrected"]["status"], "ok")
        self.assertEqual(proof["corrected"]["score"], 1.0)
        self.assertEqual(proof["pinned_selfcheck"], "PASS")
        self.assertEqual(proof["regressions"]["chain_correct"]["score"], 1.0)
        self.assertEqual(proof["regressions"]["chain_incorrect"]["score"], 0.0)
        self.assertEqual(self.harness.calls(), [])

    def test_real_repair_restarts_guardian_rearms_only_test_and_is_repeatable(self):
        h = self.harness
        old = h.start_old()
        before = {
            name: files(self.campaign / name)
            for name in ("template", "data", "runs/logic_algorithmic/checkpoints")
        }
        campaign_sha = file_sha(self.campaign / "campaign.json")
        original_requests = {
            d: file_sha(Path(j["folder"]) / "request.json") for d, j in h.jobs.items()
        }
        with patch.object(
            subprocess, "Popen", side_effect=self.record_process(subprocess.Popen)
        ):
            result = repair.repair(
                self.campaign, "conversational_pivot", self.diagnostic, wait_seconds=10
            )
            self.assertEqual(old.wait(timeout=5), -15)
            self.assertEqual(result["guardian"]["status"], "REPAIRED_GUARDIAN_RUNNING")
            state = read(controller.state_path(self.c, "conversational_pivot"))
            self.assertEqual(state["status"], "QUEUED")
            job = state["job"]
            request = source.validate_request(Path(job["folder"]) / "request.json")
            self.assertEqual(request["role"], "test")
            self.assertEqual(request["evaluation_model"]["step"], 50)
            self.assertEqual(request["verifier_identity"], result["revision"])
            self.assertEqual(request["test_verifier_revision"], result["revision"])
            self.assertEqual(
                read(controller.state_path(self.c, "logic_algorithmic"))["job"],
                h.jobs["logic_algorithmic"],
            )
            repeated = repair.repair(
                self.campaign, "conversational_pivot", self.diagnostic, wait_seconds=2
            )
            self.assertEqual(repeated["status"], "ALREADY_INSTALLED")
            self.assertEqual(
                repeated["guardian"]["controller_runtime"]["runtime"]["pid"],
                result["guardian"]["controller_runtime"]["runtime"]["pid"],
            )
            self.assertEqual(
                read(controller.state_path(self.c, "conversational_pivot"))["job"], job
            )
            self.assertEqual([x[0] for x in h.calls()].count("qsub"), 1)
            # A later confirmed administrator interruption must retry the same
            # final-test phase and retain the fixed verifier, even with no progress.
            rows = read(self.root / "pbs_rows.json")
            rows[job["job_id"]].update(
                job_state="F", Exit_status=271, comment="terminated by root@login1"
            )
            write(self.root / "pbs_rows.json", rows)
            h.wait_for(
                lambda: (
                    (
                        read(controller.state_path(self.c, "conversational_pivot")).get(
                            "job"
                        )
                        or {}
                    ).get("job_id")
                    not in (None, job["job_id"])
                )
            )
            continued = read(controller.state_path(self.c, "conversational_pivot"))
            retried = source.validate_request(
                Path(continued["job"]["folder"]) / "request.json"
            )
            self.assertEqual(retried["role"], "test")
            self.assertEqual(retried["verifier_identity"], result["revision"])
            self.assertEqual(continued["previous_job"], job)
        self.assertTrue(active(self.campaign / "guardian.lock"))
        self.assertEqual([x[0] for x in h.calls()].count("qsub"), 2)
        self.assertFalse(any(x[0] == "qdel" for x in h.calls()))
        self.assertEqual(file_sha(self.campaign / "campaign.json"), campaign_sha)
        for name, value in before.items():
            self.assertEqual(files(self.campaign / name), value)
        self.assertEqual(
            {
                d: file_sha(Path(j["folder"]) / "request.json")
                for d, j in h.jobs.items()
            },
            original_requests,
        )
        self.assertEqual(
            read(self.campaign / "runs/conversational_pivot/export.json"), self.export
        )
        # Subsequent test retries/base tests get the fix; training retains original scorer.
        later = source.make_job(self.c, "conversational_pivot", "test")
        base = source.make_job(self.c, "base", "test")
        train = source.make_job(self.c, "logic_algorithmic", "train")
        for pending in (later, base):
            self.assertEqual(
                source.validate_request(Path(pending["folder"]) / "request.json")[
                    "verifier_identity"
                ],
                result["revision"],
            )
        tr = source.validate_request(Path(train["folder"]) / "request.json")
        self.assertEqual(
            tr["verifier_identity"], self.c["contract"]["verifier_identity"]
        )
        self.assertNotIn("test_verifier_revision", tr)

    def test_wrong_job_running_job_stop_or_other_error_cannot_rearm(self):
        record = read(self.diagnostic)
        with self.subTest("another job"):
            write(self.diagnostic, dict(record, pbs_job_id="foreign.pbs1"))
            with self.assertRaisesRegex(RuntimeError, "another PBS job"):
                repair.verify_failed_test(
                    self.c, "conversational_pivot", self.diagnostic, PBS()
                )
        write(self.diagnostic, record)
        rows = read(self.root / "pbs_rows.json")
        jid = self.harness.jobs["conversational_pivot"]["job_id"]
        rows[jid]["job_state"] = "R"
        write(self.root / "pbs_rows.json", rows)
        with self.assertRaisesRegex(RuntimeError, "not confirmed"):
            repair.verify_failed_test(
                self.c, "conversational_pivot", self.diagnostic, PBS()
            )
        write(self.campaign / "runs/conversational_pivot/STOP.json", {})
        with self.assertRaisesRegex(RuntimeError, "stop fence"):
            repair.repair(self.campaign, "conversational_pivot", self.diagnostic)
        write(
            self.diagnostic,
            dict(record, attempts=[dict(error="other failure", phase="comparison")]),
        )
        with self.assertRaisesRegex(RuntimeError, "not the known"):
            repair.diagnostic(self.diagnostic)
        self.assertFalse((self.campaign / "test_verifier_revision.json").exists())
        self.assertFalse(any(x[0] in ("qsub", "qdel") for x in self.harness.calls()))

    def test_failed_guarded_install_restarts_other_teachers_guardian(self):
        from d4_teachers import retry_repair

        old = self.harness.start_old()

        def refused(c):
            raise RuntimeError("lost PBS observation")

        with patch.object(
            subprocess, "Popen", side_effect=self.record_process(subprocess.Popen)
        ):
            with self.assertRaisesRegex(RuntimeError, "guardian restart attempted"):
                retry_repair.repair(self.campaign, install_under_lease=refused)
        self.assertEqual(old.wait(timeout=5), -15)
        self.harness.wait_for(lambda: active(self.campaign / "guardian.lock"))
        self.assertEqual(
            read(controller.state_path(self.c, "conversational_pivot"))["job"],
            self.harness.jobs["conversational_pivot"],
        )
        self.assertEqual([x[0] for x in self.harness.calls()].count("qsub"), 0)

    def test_changed_revision_or_preflight_is_rejected(self):
        revision = repair.prepare_revision(self.c)
        folder = repair.revision_directory(self.c, revision["identity"])
        write(
            folder / "cpu_preflight.json",
            dict(status="PASS", revision=revision["identity"]),
        )
        write(
            self.campaign / "test_verifier_revision.json",
            dict(
                identity=revision["identity"],
                preflight_sha256=file_sha(folder / "cpu_preflight.json"),
            ),
        )
        self.assertEqual(repair.active_revision(self.c), revision)
        proof = read(folder / "cpu_preflight.json")
        write(folder / "cpu_preflight.json", dict(proof, status="FAIL"))
        with self.assertRaisesRegex(RuntimeError, "preflight changed"):
            repair.active_revision(self.c)
        write(folder / "cpu_preflight.json", proof)
        (folder / "runtime/d4_db_mathfix/and_compat.py").write_text("# corrupt")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            repair.active_revision(self.c)

    def test_preflight_failure_leaves_job_and_guardian_untouched(self):
        from d4_teachers import retry_repair

        original = read(controller.state_path(self.c, "conversational_pivot"))
        with (
            patch.object(
                repair, "preflight", side_effect=RuntimeError("regression failed")
            ),
            patch.object(retry_repair, "repair") as upgrade,
        ):
            with self.assertRaisesRegex(RuntimeError, "regression failed"):
                repair.repair(self.campaign, "conversational_pivot", self.diagnostic)
        upgrade.assert_not_called()
        self.assertEqual(
            read(controller.state_path(self.c, "conversational_pivot")), original
        )
        self.assertFalse((self.campaign / "test_verifier_revision.json").exists())

    def test_new_verifier_uses_a_distinct_evaluation_cache_key(self):
        from d4_teachers.evaluation import Journal
        from d4_training_continuation.validation_journal import instrument
        from test_domain_teachers import EvaluationRecovery

        fixture = EvaluationRecovery(methodName="runTest")
        runtime = dict(rows=16, batches=2, ids=[str(i) for i in range(16)])
        old = self.root / "old_test"
        first = fixture.validator(16, fail=8)
        first.cont_journal = Journal(old, dict(verifier="original"), runtime)
        with patch("builtins.print"), self.assertRaisesRegex(RuntimeError, "walltime"):
            instrument(first._validate)()
        preserved = files(old)
        with self.assertRaisesRegex(RuntimeError, "binding changed"):
            Journal(old, dict(verifier="corrected"), runtime)
        restarted = fixture.validator(16)
        restarted.cont_journal = Journal(
            self.root / "corrected_test", dict(verifier="corrected"), runtime
        )
        with patch("builtins.print"):
            instrument(restarted._validate)()
        self.assertEqual(restarted.generated, [0, 8])
        self.assertEqual(restarted.cont_journal.finish()[0]["reused_batches"], 0)
        self.assertEqual(files(old), preserved)

    def test_mixed_test_verifiers_are_visible_and_not_called_comparable(self):
        for domain in controller.TASKS[:-1]:
            write(self.campaign / "runs" / domain / "export.json", self.export)
        receipts = {}
        for domain in controller.TASKS:
            path = self.campaign / "runs" / domain / "test.json"
            write(path, {})
            receipts[path] = dict(
                split="test",
                binding=dict(
                    weights=self.export["weights_identity"],
                    verifier="fixed"
                    if domain == "conversational_pivot"
                    else "original",
                ),
                scores={d: dict(n=1, mean=0.5) for d in controller.TASKS[:-1]},
            )
        with patch(
            "d4_teachers.evaluation.read_receipt",
            side_effect=lambda p: receipts[Path(p)],
        ):
            result = exporting.manifest(self.c)
        self.assertTrue(result["tests_complete"])
        self.assertFalse(result["comparable_tests_complete"])
        self.assertEqual(
            result["test_verifier_identities"]["conversational_pivot"], "fixed"
        )
