"""Frozen held-out construction and actual guardian/request contracts; fake GPU/PBS."""

import gzip
import json
import getpass
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_domain_teachers as fixtures
import test_domain_teachers_checkpoint_evaluation as extra_fixtures
from test_training_continuation import checkpoint
from d4_teachers import controller, deepmath_data as data, deepmath_evaluation as deep
from d4_teachers import checkpoint_evaluation as extra, evaluation, selection, source
from d4_teachers.common import digest, files, read, write
from multidomain.template import render


class Tokenizer:
    chat_template = "synthetic test tokenizer"
    special_tokens_map = {}

    def get_vocab(self):
        return {"test": 1}

    def apply_chat_template(self, msgs, **kwargs):
        return list(json.dumps(msgs, sort_keys=True).encode())


def raw_question(i):
    return f"Compute the integer result of the mathematical expression 271 * {i} + 2."


def row_question(row, question):
    row.setdefault("sample_id", digest(question))
    msgs = [
        {"role": "system", "content": "Solve without tools; final answer in a box."},
        {
            "role": "user",
            "content": "Question:\n" + question + "\nPut the result in \\boxed{}.",
        },
    ]
    row.update(
        messages_json=json.dumps(msgs),
        tools_json="[]",
        verifier_json=json.dumps({"source": "math", "expected_answer": "2"}),
    )
    ids = render(Tokenizer(), msgs, [])
    row.update(
        prompt_tokens=len(ids),
        prompt_token_hash=digest(ids),
        generation_max_tokens=65536 - len(ids),
    )
    return row


class DeepMathFixture(unittest.TestCase):
    def setUp(self):
        self.h = extra_fixtures.CheckpointEvaluation(methodName="runTest")
        original_rows = fixtures.DataAndSampling.rows

        def dataset(instance):
            rows = original_rows(instance)
            for split, rr in rows.items():
                for i, row in enumerate([r for r in rr if r["domain"] == "math"]):
                    number = (
                        i
                        if split == "train"
                        else 1000 + i
                        if split == "validation"
                        else 2000 + i
                    )
                    row_question(row, raw_question(number))
                    row["source"] = (
                        "deepmath_l8"
                        if split == "train" and i < 4
                        else "deepmath_l9"
                        if split == "train" and i == 4
                        else "math"
                    )
            return rows

        with patch.object(fixtures.DataAndSampling, "rows", dataset):
            self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.c, self.root, self.pbs = self.h.c, self.h.campaign, self.h.pbs
        run = self.root / "runs/math"
        (run / "best.json").unlink()
        cp = checkpoint(
            run / "checkpoints/global_step_20",
            20,
            lineage=read(run / "run.json")["lineage"],
        )
        selection.select(
            dict(
                role="train",
                run_dir=str(run),
                domain="math",
                lineage=cp["lineage"],
                verifier_identity=self.c["contract"]["verifier_identity"],
            ),
            cp,
            dict(
                step=20,
                split="validation",
                scores={"math": {"mean": 0.25}},
                path="old",
                identity="old",
            ),
        )
        self.h.export("math")
        extra.install(self.c)
        for task in ("base", "cp_step80"):
            self.h.change(task, status="DONE", phase="test")
        self.inputs = {"raw": [], "holdout": [], "exclude_probe": []}
        self.raw = [
            dict(
                question=raw_question(i),
                final_answer=str(271 * i + 2),
                difficulty=8 if i % 5 != 4 else 9,
            )
            for i in range(120)
        ]
        self.raw += [
            dict(question=raw_question(i), final_answer=str(271 * i + 2), difficulty=8)
            for i in (1000, 2000)
        ]
        path = self.root.parent / "deepmath_l89_fixture.jsonl.gz"
        with gzip.open(path, "wt") as stream:
            stream.write("\n".join(json.dumps(r) for r in self.raw))
        self.inputs["raw"] = [str(path)]
        hold = self.root.parent / "deepmath_holdout.jsonl"
        hold.write_text("\n".join(json.dumps(r) for r in self.raw[30:80]))
        self.inputs["holdout"] = [str(hold)]

    def prepare(self):
        return data.prepare(
            self.c,
            self.inputs,
            size=8,
            tokenizer=Tokenizer(),
            score_reference=lambda answer: dict(status="ok", score=1.0),
        )

    def registered(self):
        self.prepare()
        deep.install(self.c)

    def rows(self):
        import pyarrow.parquet as pq

        return {
            s: pq.read_table(data.directory(self.c) / (s + ".parquet")).to_pylist()
            for s in data.SPLITS
        }

    def finish(self, task, score):
        p = deep.plan(self.c)
        model = deep.evaluation_model(self.c, task)
        rows = [r for rs in self.rows().values() for r in rs]
        ids = [r["sample_id"] for r in rows]
        binding = dict(
            campaign=self.c["identity"],
            domain="math",
            split="heldout",
            step=model["step"],
            weights=model["weights_identity"],
            verifier=p["verifier_identity"],
            deepmath_evaluation=p["identity"],
        )
        runtime = dict(ids=ids, dataset_sha256=digest(rows), rows=len(rows), batches=2)
        folder = (
            self.root
            / "runs"
            / task
            / "evaluations"
            / ("heldout_" + str(model["step"]) + "_fixture")
        )
        checker = evaluation.Journal(folder, binding, runtime)
        manifest = checker.identity
        checksums = {}
        samples = []
        from d4_teachers.common import file_sha

        for k in (0, 1):
            subset = ids[8 * k : 8 * k + 8]
            vals = [float(score(i + 8 * k)) for i in range(8)]
            samples.extend(zip(subset, vals))
            payload = dict(
                batch=k + 1,
                rows=8,
                input_digest=digest(subset),
                extras={},
                lists=dict(
                    sample_uids=subset,
                    sample_scores=vals,
                    sample_inputs=["question"] * 8,
                    sample_outputs=["answer"] * 8,
                    sample_gts=["answer"] * 8,
                    sample_turns=[],
                    data_source_lst=[["math"] * 8],
                ),
            )
            name = f"batch_{k + 1:06d}.json"
            write(folder / name, dict(payload=payload, sha256=digest(payload)))
            checksums[name] = file_sha(folder / name)
        receipt = dict(
            status="PASS",
            split="heldout",
            step=model["step"],
            binding=binding,
            identity=digest(dict(binding=binding, runtime=runtime)),
            path=str(folder / "receipt.json"),
            scores=evaluation.statistics(samples, rows),
            split_scores=deep.split_statistics(samples, rows),
            journal=dict(
                path=str(folder),
                rows=len(rows),
                batches=2,
                batch_files=checksums,
                manifest_sha256=digest(manifest),
            ),
        )
        receipt["content_sha256"] = digest(receipt)
        write(folder / "receipt.json", receipt)
        write(self.root / "runs" / task / "heldout.json", receipt)
        return receipt


class Preparation(DeepMathFixture):
    def test_old_unicode_holdout_and_question_only_probe(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = self.root.parent / "deepmath_validation.parquet"
        old = [
            dict(
                prompt=[
                    dict(role="system", content="Use 《reasoning》 and 《answer》"),
                    dict(role="user", content=r["question"]),
                ],
                reward_model=dict(ground_truth=r["final_answer"]),
                extra_info=dict(difficulty_level=r["difficulty"], topic="algebra"),
            )
            for r in self.raw[30:80]
        ]
        pq.write_table(pa.Table.from_pylist(old), path)
        self.inputs["holdout"] = [str(path)]
        probe = self.root.parent / "probe.jsonl"
        probe.write_text(json.dumps(dict(question=raw_question(90))) + "\n")
        self.inputs["exclude_probe"] = [str(probe)]
        records, _, _ = data.input_records(self.inputs)
        held = [r for r in records if r["kind"] == "holdout"]
        self.assertEqual(held[0]["question"], self.raw[30]["question"])
        self.assertEqual(held[0]["preferred_split"], "validation")
        manifest = self.prepare()
        self.assertEqual(manifest["audit"]["exclusions"]["prior_probe"], 1)
        for split, rows in self.rows().items():
            self.assertTrue(
                all("《reasoning》" not in r["messages_json"] for r in rows)
            )
            if split == "test":
                self.assertTrue(
                    all(
                        json.loads(r["provenance_json"])["prior_partition"] == "raw"
                        for r in rows
                    )
                )

    def test_exact_scaffold_strata_disjointness_and_frozen_parent(self):
        before = files(self.root / "data")
        m = self.prepare()
        self.assertEqual(files(self.root / "data"), before)
        rr = self.rows()
        for split in data.SPLITS:
            self.assertEqual(len(rr[split]), 8)
            self.assertEqual(
                m["splits"][split]["sources"], {"deepmath_l8": 6, "deepmath_l9": 2}
            )
            for row in rr[split]:
                self.assertEqual(
                    data.messages(row)[0]["content"],
                    "Solve without tools; final answer in a box.",
                )
                self.assertTrue(data.last_user(row).startswith("Question:\n"))
                origin = json.loads(row["provenance_json"])
                self.assertEqual(origin["prior_partition"], "holdout")
                self.assertNotIn("《reasoning》", row["messages_json"])
        self.assertFalse(
            set(m["splits"]["validation"]["ids"]) & set(m["splits"]["test"]["ids"])
        )
        self.assertGreaterEqual(
            m["audit"]["exclusions"]["frozen_train_or_existing_eval"], 8
        )
        self.assertEqual(self.prepare(), m)

    def test_normalized_cross_source_overlap_conflicts_and_probe_excluded(self):
        rr, _, _ = data.input_records(self.inputs)
        train = [
            row_question(
                dict(domain="math", source="math"), raw_question(90).replace(" ", "  ")
            )
        ]
        template, _, _ = data.infer_template(
            [
                row_question(dict(source="deepmath_l8"), raw_question(i))
                for i in range(3)
            ],
            rr,
        )
        conflict = dict(rr[80], answer="different")
        probe = dict(rr[91], kind="exclude_probe")
        result, audit = data.remove_training_and_existing_eval(
            rr + [conflict, probe], {"train": train}, template
        )
        keys = {r["key"] for r in result}
        for i in (80, 90, 91):
            self.assertNotIn(data.question_key(raw_question(i)), keys)
        self.assertEqual(audit["exclusions"]["prior_probe"], 1)
        self.assertEqual(audit["exclusions"]["conflicting_reference_answers"], 1)

    def test_named_old_splits_and_reference_ineligibility(self):
        rr, _, _ = data.input_records(self.inputs)
        rr = [
            dict(r, key=data.question_key(r["question"]))
            for r in rr
            if r["kind"] == "raw"
        ]
        for i, r in enumerate(rr):
            r["preferred_split"] = "validation" if i % 2 == 0 else "test"

        def convert(r):
            if r["origin"]["row"] == 0:
                return None
            return dict(
                sample_id=r["key"],
                difficulty_level=r["level"],
                expected=r["preferred_split"],
            )

        selected, audit = data.choose(rr, 8, 42, {8: 4, 9: 1}, convert)
        for split in data.SPLITS:
            self.assertTrue(all(r["expected"] == split for r in selected[split]))

    def test_changed_renderer_and_dataset_fail_before_any_gpu_submission(self):
        bad = Tokenizer()
        bad.apply_chat_template = lambda *a, **k: [999]
        with self.assertRaisesRegex(RuntimeError, "renderer/tokenizer"):
            data.prepare(
                self.c,
                self.inputs,
                size=8,
                tokenizer=bad,
                score_reference=lambda x: dict(status="ok", score=1),
            )
        self.assertFalse((self.root / deep.PLAN).exists())
        self.prepare()
        (data.directory(self.c) / "validation.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            deep.install(self.c)
        self.assertFalse(self.pbs.submissions)


class Guardian(DeepMathFixture):
    def test_real_guardian_handoff_and_repeat_preserve_existing_work(self):
        from d4_teachers import retry_repair

        self.prepare()
        self.h.export("math_step50")
        self.h.change("math_step50", status="DONE", phase="test")
        write(
            self.root / "preflight.json",
            dict(status="PASS", campaign_identity=self.c["identity"]),
        )
        area = self.root.parent / "pbs"
        binary = area / "bin"
        binary.mkdir(parents=True)
        write(area / "rows.json", {})
        queue = "Queue: R9920261000\n    queue_type = Execution\n    enabled = True\n    started = True\n    default_chunk.node_group = nii\n    acl_group_enable = True\n    acl_groups = gcg51557\n    resources_available.ncpus = 1920"
        program = (
            "import json,os,sys\nfrom pathlib import Path\n"
            "root=Path(os.environ['DEEPMATH_TEST_PBS'])\nname=Path(sys.argv[0]).name\n"
            "rows=json.loads((root/'rows.json').read_text())\n"
            "with (root/'calls.jsonl').open('a') as f:f.write(json.dumps([name]+sys.argv[1:])+'\\n')\n"
            "if name=='qstat':\n"
            f"    print({queue!r} if '-Qf' in sys.argv else json.dumps({{'Jobs':rows}}))\n"
            "elif name=='qsub':\n"
            "    request=json.loads((Path(sys.argv[1]).parent/'request.json').read_text())\n"
            "    jid=str(900000+len(rows))+'.pbs1'\n"
            "    rows[jid]=dict(Job_Owner=os.environ['DEEPMATH_TEST_USER']+'@login',queue='R9920261000',Job_Name=request['job_name'],Submit_arguments=sys.argv[1],job_state='Q')\n"
            "    (root/'rows.json').write_text(json.dumps(rows))\n    print(jid)\n"
            "else:raise RuntimeError('Unexpected scheduler mutation')\n"
        )
        for name in ("qstat", "qsub", "qdel"):
            path = binary / name
            path.write_text("#!" + sys.executable + "\n" + program)
            path.chmod(0o700)
        children = []

        def cleanup():
            for child in children:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=5)

        self.addCleanup(cleanup)
        popen = subprocess.Popen

        def track(*a, **kw):
            child = popen(*a, **kw)
            children.append(child)
            return child

        env = dict(
            PATH=str(binary) + os.pathsep + os.environ["PATH"],
            DEEPMATH_TEST_PBS=str(area),
            DEEPMATH_TEST_USER=getpass.getuser(),
        )
        with (
            patch.dict(os.environ, env),
            patch.object(subprocess, "Popen", side_effect=track),
        ):
            first = retry_repair.repair(self.root)
            self.assertTrue(first["controller_runtime"]["first_tick_completed"])
            logic = read(controller.state_path(self.c, "logic_algorithmic"))["job"]
            frozen = files(self.root / "template")
            result = deep.launch(self.root)
            self.assertTrue(
                result["guardian"]["controller_runtime"]["first_tick_completed"], result
            )
            for task in deep.TARGETS:
                self.assertEqual(result["guardian"]["states"][task]["status"], "QUEUED")
            self.assertEqual(
                read(controller.state_path(self.c, "logic_algorithmic"))["job"], logic
            )
            again = deep.launch(self.root)
            self.assertEqual(
                again["guardian"]["controller_runtime"]["runtime"]["pid"],
                result["guardian"]["controller_runtime"]["runtime"]["pid"],
            )
            calls = [
                json.loads(line)
                for line in (area / "calls.jsonl").read_text().splitlines()
            ]
            self.assertEqual(sum(c[0] == "qsub" for c in calls), 4)
            self.assertFalse(any(c[0] == "qdel" for c in calls))
            self.assertEqual(files(self.root / "template"), frozen)

    def test_actual_validator_resumes_across_split_boundary_and_never_selects(self):
        from omegaconf import OmegaConf
        import numpy as np

        self.registered()
        job = source.make_job(self.c, "deepmath_math20", "test")
        request = source.validate_request(Path(job["folder"]) / "request.json")
        rows = [r for rr in self.rows().values() for r in rr]

        class Rows:
            def __init__(self):
                self.rows = rows

            def __len__(self):
                return len(rows)

        harness = fixtures.EvaluationRecovery(methodName="runTest")

        def runner(fail=None):
            obj = harness.validator(len(rows), fail=fail)
            obj.config = OmegaConf.create(json.loads(json.dumps(obj.config)))
            obj.config.actor_rollout_ref.rollout.val_kwargs.temperature = 0
            obj.config.data = dict(validation_shuffle=False, val_batch_size=8)
            obj.teacher_request = request
            obj.global_steps = 20
            obj.val_dataset = Rows()
            obj.current_checkpoint = lambda step: None
            obj.weights_identity = lambda cp: request["evaluation_model"][
                "weights_identity"
            ]
            obj._dump_generations = lambda **kwargs: None
            for i, batch in enumerate(obj.val_dataloader):
                batch["uid"] = np.array(
                    [r["sample_id"] for r in rows[i * 8 : i * 8 + 8]], dtype=object
                )
            return obj

        with (
            patch.object(
                evaluation, "dependency_versions", return_value={"fixture": "1"}
            ),
            patch("builtins.print"),
            patch.object(selection, "select") as select,
        ):
            first = runner(fail=8)
            with self.assertRaisesRegex(RuntimeError, "walltime"):
                evaluation.validate(first, first._validate)
            self.assertFalse((Path(request["run_dir"]) / "heldout.json").exists())
            second = runner()
            evaluation.validate(second, second._validate)
            receipt, _, _ = deep.validate_result(self.c, "deepmath_math20")
            self.assertEqual(receipt["journal"]["reused_batches"], 1)
            self.assertEqual(second.generated, [8])
            for split in data.SPLITS:
                self.assertEqual(
                    receipt["split_scores"][split]["math"], dict(n=8, mean=1.0)
                )
            third = runner()
            evaluation.validate(third, third._validate)
            self.assertEqual(third.generated, [])
            select.assert_not_called()
        self.assertFalse((Path(request["run_dir"]) / "test.json").exists())

    def test_register_wait_export_and_preserve_existing_jobs_and_four_job_cap(self):
        self.prepare()
        states = controller.tick(self.c, self.pbs)
        old = {t: states[t]["job"] for t in ("logic_algorithmic", "math_step50")}
        before = {
            k: v for k, v in files(self.root / "runs/math").items() if k != "state.json"
        }
        deep.install(self.c)
        self.assertEqual(deep.install(self.c)["status"], "ALREADY_REGISTERED")
        states = controller.tick(self.c, self.pbs)
        self.assertEqual(len(self.pbs.submissions), 4)
        for t, j in old.items():
            self.assertEqual(states[t]["job"], j)
        self.assertFalse(states["deepmath_math50"]["job"])
        self.assertEqual(
            {
                k: v
                for k, v in files(self.root / "runs/math").items()
                if k != "state.json"
            },
            before,
        )
        self.assertFalse(self.pbs.cancellations)
        self.h.change("deepmath_base", status="DONE", job=None)
        states = controller.tick(self.c, self.pbs)
        self.assertEqual(states["deepmath_math50"]["status"], "WAITING_MODEL")
        self.h.export("math_step50")
        states = controller.tick(self.c, self.pbs)
        self.assertEqual(states["deepmath_math50"]["status"], "QUEUED")
        self.assertEqual(len(self.pbs.submissions), 5)

    def test_real_request_binds_both_splits_weights_and_isolated_runtime(self):
        self.registered()
        job = source.make_job(self.c, "deepmath_math20", "test")
        request = source.validate_request(Path(job["folder"]) / "request.json")
        self.assertEqual(request["evaluation_model"]["step"], 20)
        self.assertEqual(request["domain"], "math")
        self.assertEqual(set(request["heldout_manifest"]["splits"]), set(data.SPLITS))
        self.assertNotIn("checkpoint_evaluation", request)
        out = subprocess.run(
            [sys.executable, "-I", str(Path(job["folder"]) / "entry.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(out.returncode, 0, out.stderr)
        subprocess.run(
            ["bash", "-n", str(Path(job["folder"]) / "stage.pbs")], check=True
        )
        with self.assertRaisesRegex(RuntimeError, "evaluation only"):
            source.make_job(self.c, "deepmath_math20", "train")
        with self.assertRaisesRegex(RuntimeError, "evaluation only"):
            source.make_job(self.c, "deepmath_math20", "export")
        path = deep.runtime_directory(self.c, deep.plan(self.c)) / "evaluation.py"
        path.write_text("changed")
        with self.assertRaisesRegex(RuntimeError, "changed"):
            source.make_job(self.c, "deepmath_base", "test")

    def test_271_retries_keep_identity_and_cached_batches_no_retry_cap(self):
        self.registered()
        self.h.change("math_step50", status="DONE", job=None)
        self.h.change("logic_algorithmic", status="DONE", job=None)
        states = controller.tick(self.c, self.pbs)
        first = source.validate_request(
            Path(states["deepmath_base"]["job"]["folder"]) / "request.json"
        )
        progress = self.root / "runs/deepmath_base/evaluations/heldout_0_progress"
        write(progress / "progress.json", dict(completed_rows=8))
        write(progress / "batch_000001.json", dict(cached="keep"))
        before = files(progress)
        for i in range(6):
            old = states["deepmath_base"]["job"]
            self.pbs.rows[old["job_id"]] = dict(job_state="F", Exit_status=271)
            states = controller.tick(self.c, self.pbs)
            req = source.validate_request(
                Path(states["deepmath_base"]["job"]["folder"]) / "request.json"
            )
            for key in (
                "deepmath_evaluation",
                "heldout_manifest",
                "evaluation_model",
                "verifier_identity",
            ):
                self.assertEqual(req[key], first[key])
        self.assertEqual(files(progress), before)
        self.assertGreater(states["deepmath_base"]["no_progress_retries"], 3)
        self.assertEqual(states["deepmath_math50"]["status"], "WAITING_MODEL")

    def test_unknown_and_ambiguous_submit_are_never_duplicated(self):
        self.registered()
        self.pbs.ambiguous = True
        states = controller.tick(self.c, self.pbs)
        before = len(self.pbs.submissions)
        self.pbs.ambiguous = False
        controller.tick(self.c, self.pbs)
        self.assertEqual(len(self.pbs.submissions), before)
        self.assertEqual(states["deepmath_base"]["status"], "BLOCKED_UNKNOWN")

    def test_registration_crash_publishes_no_partial_tasks(self):
        self.prepare()
        original = deep.write

        def fail(path, value):
            if str(path).endswith("deepmath_math20/state.json"):
                raise OSError("crash")
            return original(path, value)

        with patch.object(deep, "write", side_effect=fail), self.assertRaises(OSError):
            deep.install(self.c)
        self.assertNotIn("deepmath_base", controller.tasks(self.c))
        deep.install(self.c)
        self.assertIn("deepmath_base", controller.tasks(self.c))

    def test_stopped_tasks_and_non_retryable_failures_do_not_resubmit(self):
        self.registered()
        write(self.root / "runs/deepmath_math20/STOP.json", {})
        states = controller.tick(self.c, self.pbs)
        self.assertEqual(states["deepmath_math20"]["status"], "STOPPED")
        job = states["deepmath_base"]["job"]
        self.pbs.rows[job["job_id"]] = dict(job_state="F", Exit_status=265)
        states = controller.tick(self.c, self.pbs)
        self.assertEqual(states["deepmath_base"]["status"], "BLOCKED_APPLICATION")

    def test_completion_checks_splits_and_paired_report(self):
        self.registered()
        self.h.export("math_step50")
        self.finish("deepmath_base", lambda i: i % 2)
        self.finish("deepmath_math20", lambda i: 1)
        self.finish("deepmath_math50", lambda i: 1)
        for task in deep.TARGETS:
            controller.artifact(self.c, task, "test")
        report = deep.report(self.root)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["paired_comparisons"]), 18)
        self.assertTrue(
            all(r["mean_difference"] >= 0 for r in report["paired_comparisons"])
        )
        path = self.root / "runs/deepmath_math50/heldout.json"
        receipt = read(path)
        receipt["split_scores"]["test"]["math"]["mean"] = 0
        receipt["content_sha256"] = digest(
            {k: v for k, v in receipt.items() if k != "content_sha256"}
        )
        write(path, receipt)
        with self.assertRaisesRegex(RuntimeError, "split metrics"):
            controller.artifact(self.c, "deepmath_math50", "test")

    def test_missing_model_or_changed_selection_fails_before_guardian_handoff(self):
        self.prepare()
        selected = read(self.root / "runs/math_step50/best.json")
        selected["step"] = 45
        write(self.root / "runs/math_step50/best.json", selected)
        with patch("d4_teachers.retry_repair.repair") as repair:
            with self.assertRaises(RuntimeError):
                deep.launch(self.root)
        repair.assert_not_called()


if __name__ == "__main__":
    unittest.main()
