#!/usr/bin/env python3
"""Collect D4 learning history, including every attempt; Python stdlib only.

Reads experiment files and optionally qstat. Writes one analysis directory and
archive. Never imports training code, loads checkpoints, or changes PBS jobs.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import tarfile
import tempfile

BASE = "qwen30b_d4_10k_uniform_ctx64k_seed42_v1"
RUNS = {"random_single": "", "domain_single": "_db_v1",
        "random_dual": "_n2_once_v1", "domain_dual": "_db_n2_once_v1",
        "random_continuation": "_continuation_v1", "domain_continuation": "_db_continuation_v1"}
VERSION = "2026-10-22.history.2"
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
NUMBER = r"[-+]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf(?:inity)?)"
SCALAR_WRAPPER = re.compile(r"\b(?:np|numpy)\.(?:float(?:16|32|64|128)?|u?int(?:8|16|32|64)?)\(\s*(" + NUMBER + r")\s*\)", re.I)
METRIC = re.compile(r"(?:^|\s+-\s+)([^:\s][^:]*?):\s*(" + NUMBER + r")(?=\s+-\s+|\s*$|\s*\[repeated)", re.I)
STEP = re.compile(r"\bstep:\s*(\d+)\s+-\s+")
SAMPLE = re.compile(r"sampling step\s+(\d+):\s*domain=([^,\s]+)")
JOB_ID = re.compile(r"\d+(?:\.[A-Za-z0-9_.-]+)?\Z")
PRUNE = {"checkpoints", "source", "runtime", "wandb", ".git", "__pycache__",
         "huggingface", "actor", "optimizer", "data", "hf_home", "generations", "raw_generations"}
TEXT_SUFFIXES = {".json", ".yaml", ".yml", ".pbs", ".txt", ".log", ".jsonl", ".out", ".err"}


def now():
    return datetime.now(timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def emit(handle, value):
    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def numeric(value, prefix=""):
    """Only recorded numeric scalars. Never reinterpret text length as tokens."""
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            key = prefix + str(k)
            if finite(v):
                out[key] = v
            elif isinstance(v, dict):
                out.update(numeric(v, key + "/"))
    return out


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return str(value) if isinstance(value, float) and not math.isfinite(value) else value


def object_or_empty(value):
    return value if isinstance(value, dict) else {}


def parse_metric(line):
    line = ANSI.sub("", line).strip()
    # Pinned verl prints numpy scalar reprs. Normalize only a numeric grammar;
    # never eval log text or import numpy on the login node.
    line = SCALAR_WRAPPER.sub(lambda m: m[1], line)
    match = STEP.search(line)
    if not match:
        return None
    values, nonfinite = {}, []
    for k, v in METRIC.findall(line[match.end():]):
        k, n = k.strip(), float(v)
        if math.isfinite(n):
            values[k] = n
        else:
            values[k] = None
            nonfinite.append(k)
    if not values:
        return None
    return {"step": int(match[1]), "metrics": values, "nonfinite_keys": nonfinite}


def is_update(row):
    keys = row["metrics"]
    return row["step"] > 0 and any(k in keys for k in
        ("actor/pg_loss", "actor/grad_norm", "timing_s/update_actor", "actor/policy_loss"))


def small_json(path):
    try:
        with path.open("rb") as f:
            raw = f.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024:
            raise ValueError("JSON larger than 4 MiB; retained in inventory")
        return json.loads(raw)
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc)}


def command(argv, root, timeout=20):
    try:
        p = subprocess.run(argv, cwd=root, capture_output=True, text=True,
                           errors="replace", timeout=timeout, check=False)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def pbs_job(root, jid):
    result = command(["qstat", "-xf", "-F", "json", jid], root)
    decoder = json.JSONDecoder()
    text = result.get("stdout", "")
    if result.get("returncode") == 0:
        for m in re.finditer(r"\{", text):
            try:
                d, _ = decoder.raw_decode(text[m.start():])
                if not isinstance(d, dict):
                    continue
                for key, job in d.get("Jobs", {}).items():
                    if key == jid or ("." not in jid and key.split(".")[0] == jid):
                        keep = ("Job_Name", "queue", "job_state", "Exit_status", "comment", "qtime",
                                "stime", "obittime", "Resource_List", "resources_used", "run_count")
                        return {"job_id": key, **{k: job[k] for k in keep if k in job}}
            except (ValueError, TypeError, AttributeError):
                continue
    return {"job_id": jid, "query_error": result.get("error") or result.get("stderr") or "PBS history unavailable"}


def seconds(value):
    try:
        h, m, s = str(value).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except (ValueError, TypeError):
        return None


def files_below(folder, max_depth=4, prune=PRUNE):
    if not folder.is_dir() or folder.is_symlink():
        return
    for directory, dirs, names in os.walk(folder, followlinks=False):
        directory = Path(directory)
        depth = len(directory.relative_to(folder).parts)
        dirs[:] = sorted(d for d in dirs if d not in prune and not d.startswith(".")
                         and not (directory / d).is_symlink()) if depth < max_depth else []
        for name in sorted(names):
            p = directory / name
            if p.is_file() and not p.is_symlink():
                yield p


def slim(value, limit=12000):
    """Bound representative text; original character counts accompany it."""
    if isinstance(value, str):
        return value if len(value) <= limit else {"preview": value[:limit], "characters": len(value), "truncated": True}
    if isinstance(value, dict):
        return {k: slim(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [slim(v, limit) for v in value[:20]]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


class Collector:
    def __init__(self, root, output, include_pbs=True, full_generations=False):
        self.root, self.output = Path(root).resolve(), Path(output).resolve()
        self.include_pbs, self.full_generations = include_pbs, full_generations
        self.manifest, self.jobs, self.metrics, self.events, self.checkpoints = [], [], [], [], []
        self.generation_stats, self.warnings = [], []
        self.step_records, self.continuation = [], {}
        self.phase_cache = {}
        self.generation_seen = set()
        self.seen, self.metric_copies = set(), {}
        self.archive = None

    def relative(self, path):
        return str(path.relative_to(self.root))

    def capture(self, path, parser=None):
        """Read a fixed byte prefix, hash and parse that exact snapshot, then archive."""
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path in self.seen:
            return
        try:
            path.resolve().relative_to(self.root)
            self.seen.add(path)
            stat = path.stat()
            if stat.st_size > 8 * 1024 * 1024 and path.suffix not in {".log", ".out", ".err", ".jsonl"}:
                self.manifest.append({"path": self.relative(path), "bytes": stat.st_size, "skipped": "metadata exceeds 8 MiB"})
                return
            h = hashlib.sha256()
            with path.open("rb") as source, tempfile.TemporaryFile() as snap:
                remaining = stat.st_size
                while remaining:
                    block = source.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    snap.write(block)
                    h.update(block)
                    remaining -= len(block)
                size = snap.tell()
                digest = h.hexdigest()
                entry = {"path": self.relative(path), "bytes": size, "sha256": digest,
                         "size_at_open": stat.st_size, "mtime_ns": stat.st_mtime_ns}
                if remaining or path.stat().st_size != stat.st_size or path.stat().st_mtime_ns != stat.st_mtime_ns:
                    entry["changed_during_collection"] = True
                self.manifest.append(entry)
                if parser:
                    snap.seek(0)
                    parser(snap, digest)
                snap.seek(0)
                info = tarfile.TarInfo("evidence/" + self.relative(path))
                info.size, info.mtime, info.mode = size, int(stat.st_mtime), 0o600
                self.archive.addfile(info, snap)
        except (OSError, ValueError) as exc:
            self.warnings.append({"path": str(path), "error": str(exc)})

    def log(self, path, context):
        def parse(handle, digest):
            # Exact copies in the same job contribute once, but remain archived.
            copy_key = (context["role"], context["job_folder"], digest)
            if copy_key in self.metric_copies:
                self.manifest[-1]["metrics_duplicate_of"] = self.metric_copies[copy_key]
                return
            self.metric_copies[copy_key] = self.relative(path)
            domain, segment, previous = {}, 0, None
            for number, raw in enumerate(handle, 1):
                if not raw.endswith(b"\n"):
                    self.warnings.append({"path": self.relative(path), "line": number, "note": "unterminated final log line not parsed"})
                    continue
                for part, line in enumerate(raw.decode("utf-8", "replace").split("\r")):
                    clean = ANSI.sub("", line).strip()
                    sample = SAMPLE.search(clean)
                    if sample:
                        domain[int(sample[1])] = sample[2]
                    row = parse_metric(clean)
                    origin = dict(context, source=self.relative(path), line=number, part=part)
                    if row:
                        if previous is not None and row["step"] < previous:
                            segment += 1
                        previous = row["step"]
                        self.metrics.append(dict(origin, segment=segment, domain=domain.get(row["step"]), **row))
                    elif STEP.search(clean):
                        self.warnings.append(dict(origin, note="unparsed metric line; consult full log"))
                    if re.search(r"sampling step|Initial validation|Dumped generations|checkpoint|resume|verifier|parse_error|timeout|math.*(?:PASS|FAIL)", clean, re.I):
                        self.events.append(dict(origin, text=clean[:5000]))
        self.capture(path, parse)

    def phase_context(self, folder, phase, context):
        key = (str(folder), phase)
        if key not in self.phase_cache:
            path = folder / phase / "phase.json"
            self.phase_cache[key] = small_json(path) if path.is_file() else {}
        execution = self.phase_cache[key]
        execution = execution if isinstance(execution, dict) else {}
        return dict(context, phase=phase, source_step=execution.get("source_step"),
                    target_world=execution.get("target_world"), execution_id=execution.get("execution_id"))

    def step_record(self, path, context):
        def parse(handle, checksum):
            value = json.load(handle)
            if (not isinstance(value, dict) or type(value.get("step")) is not int
                    or value["step"] != int(path.stem.removeprefix("step_"))):
                self.warnings.append({"path": self.relative(path), "note": "step JSON identity mismatch"})
                return
            complete = value.get("actor_update_completed") is True and value.get("weights_synchronized") is True and value.get("status") == "PASS"
            metrics = numeric({k: v for k, v in value.items() if k not in ("actor_metrics", "step")}, "batch/")
            # Preserve scalar keys. List-valued actor samples get a separately
            # named arithmetic mean/count, never silently equated with console metrics.
            actor = value.get("actor_metrics", {})
            if isinstance(actor, dict):
                metrics.update(numeric(actor, "actor_recorded/"))
                for key, samples in actor.items():
                    if isinstance(samples, list) and samples and all(finite(x) for x in samples):
                        metrics["actor_samples/" + key + "/mean"] = statistics.mean(samples)
                        metrics["actor_samples/" + key + "/count"] = len(samples)
            usage = "formal" if context.get("stage") == "train" and context.get("job_role") in ("formal", "legacy_single") else "unclassified"
            if context.get("job_role") == "candidate":
                phase, step = context["phase"], value["step"]
                if phase in ("A", "B", "C", "D") or (phase == "E" and context.get("source_step") == step - 1):
                    usage = "qualification"
                elif phase == "E" and context.get("handoff_committed") and step > context["handoff_step"]:
                    usage = "formal_after_handoff"
                elif phase == "E":
                    usage = "catch_up_or_uncommitted"
            self.step_records.append(dict(context, source=self.relative(path), sha256=checksum,
                step=value["step"], domain=value.get("domain"), finished_utc=value.get("finished_utc"),
                completed_update=complete, step_usage=usage, metrics=metrics, record=json_safe(value)))
        self.capture(path, parse)

    def generations(self, path, context, stats_handle, sample_handle):
        """Scan all existing rows; keep all numeric scalars and bounded examples."""
        if path in self.generation_seen:
            return
        self.generation_seen.add(path)
        groups, examples, count, bad, h = {}, Counter(), 0, 0, hashlib.sha256()
        origin = dict(context, source=self.relative(path))
        try:
            size = path.stat().st_size
            with path.open("rb") as f:
                remaining = size
                while remaining > 0:
                    raw = f.readline(remaining)
                    if not raw:
                        break
                    remaining -= len(raw)
                    h.update(raw)
                    if not raw.endswith(b"\n"):
                        bad += 1
                        continue
                    try:
                        record = json.loads(raw)
                        if not isinstance(record, dict):
                            raise ValueError("not an object")
                    except ValueError:
                        bad += 1
                        continue
                    count += 1
                    step = record.get("step")
                    if step is None and path.stem.isdigit():
                        step = int(path.stem)
                    extra = record.get("extra_info")
                    extra = extra if isinstance(extra, dict) else {}
                    domain = record.get("domain", record.get("data_source", record.get("source", extra.get("domain", extra.get("source", "unknown")))))
                    domain = str(domain)
                    payload = numeric(record)
                    output = record.get("output", record.get("completion", record.get("response")))
                    input_value = record.get("input", record.get("prompt", record.get("sample_id")))
                    if isinstance(output, str):
                        payload["output_characters"] = len(output)
                    input_hash = hashlib.sha256(json.dumps(input_value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
                    emit(stats_handle, dict(origin, row=count, step=step, domain=domain,
                                            input_sha256=input_hash, scalars=payload))
                    for key, value in payload.items():
                        if key not in ("step", "global_step"):
                            group = groups.setdefault((str(step), domain, key), [0, 0., value, value])
                            group[0] += 1
                            group[1] += (value - group[1]) / group[0]
                            group[2], group[3] = min(group[2], value), max(group[3], value)
                    score = record.get("score", record.get("reward"))
                    bucket = "unknown" if not finite(score) else "zero" if score == 0 else "one" if score == 1 else "partial"
                    sample_key = (str(step), domain, bucket)
                    if examples[sample_key] < 2:
                        emit(sample_handle, dict(origin, row=count, step=step, domain=domain, bucket=bucket,
                                                 input_sha256=input_hash, record=slim(record)))
                        examples[sample_key] += 1
            for (step, domain, key), values in sorted(groups.items()):
                self.generation_stats.append(dict(origin, step=step, domain=domain, metric=key,
                    count=values[0], mean=values[1], min=values[2], max=values[3]))
            self.manifest.append({"path": self.relative(path), "bytes_scanned": size - remaining,
                "sha256_scanned_prefix": h.hexdigest(), "json_rows": count, "invalid_or_partial_rows": bad,
                "representation": "all numeric rows plus representative samples",
                "full_text_archived": self.full_generations})
            if bad:
                self.warnings.append(dict(origin, invalid_or_partial_rows=bad))
            if self.full_generations:
                self.capture(path)
        except OSError as exc:
            self.warnings.append(dict(origin, error=str(exc)))

    def checkpoint_metadata(self, run, role, lineage=None):
        base = run / "checkpoints"
        tracker = base / "latest_checkpointed_iteration.txt"
        self.capture(tracker)
        for cp in sorted(base.glob("global_step_*")):
            if cp.is_symlink() or not cp.is_dir():
                continue
            item = {"role": role, "setup": role.split("_")[0], "lineage": lineage,
                    "run_dir": self.relative(run), "path": self.relative(cp), "metadata_only_not_validated": True}
            try:
                item["step"] = int(cp.name.removeprefix("global_step_"))
            except ValueError:
                continue
            item["records"] = {}
            for pattern in ("*.json", "actor/*.json"):
                for p in cp.glob(pattern):
                    if p.is_file() and not p.is_symlink() and p.stat().st_size <= 4 * 1024 * 1024:
                        item["records"][str(p.relative_to(cp))] = small_json(p)
                        self.capture(p)
            self.checkpoints.append(item)

    def collect_run(self, role, suffix, stats_handle, sample_handle):
        run = self.root / "outputs/multidomain" / (BASE + suffix)
        if not run.is_dir():
            self.warnings.append({"role": role, "missing_run": str(run)})
            return
        print("[0390] Collect " + role + " (all jobs)", file=sys.stderr, flush=True)
        setup = role.split("_")[0]
        for name in ("state.json", "handoff_commit.json"):
            path = run / "control" / name
            if role.endswith("_continuation") and path.is_file():
                def parse_control(handle, checksum, name=name):
                    value = json.load(handle)
                    if not isinstance(value, dict):
                        raise ValueError("control record is not a JSON object")
                    self.continuation.setdefault(setup, {})[name] = value
                self.capture(path, parse_control)
        job_contexts = {}
        for folder in sorted((run / "jobs").glob("*")):
            if not folder.is_dir() or folder.is_symlink():
                continue
            req = small_json(folder / "request.json")
            req = req if isinstance(req, dict) else {}
            jid_path = folder / "job_id.txt"
            try:
                jid = jid_path.read_text().strip() if jid_path.is_file() else req.get("job_id")
            except OSError as exc:
                self.warnings.append({"path": str(jid_path), "error": str(exc)})
                jid = req.get("job_id")
            stage = str(req.get("stage") or ("train" if folder.name.endswith("_train") else "unknown"))
            commit_path = folder / "committed.json"
            commit = small_json(commit_path) if commit_path.is_file() else {}
            commit = commit if isinstance(commit, dict) else {}
            committed = (bool(req.get("identity")) and commit.get("identity") == req["identity"]
                         and object_or_empty(commit.get("job")).get("job_id") == jid
                         and bool(req.get("lineage"))
                         and object_or_empty(commit.get("authority")).get("lineage") == req["lineage"]
                         and type(object_or_empty(commit.get("checkpoint")).get("step")) is int)
            context = {"role": role, "setup": setup, "job_id": jid, "job_folder": self.relative(folder), "stage": stage,
                       "job_role": req.get("role", "legacy_single" if role.endswith("_single") else "legacy_candidate"),
                       "lineage": req.get("lineage"), "allocation_nodes": req.get("expected_nodes"),
                       "handoff_committed": committed, "handoff_step": commit.get("checkpoint", {}).get("step") if committed else None}
            job_contexts[folder] = context
            self.jobs.append(dict(context, request=req))
            for path in files_below(folder):
                if path.suffix == ".py" and re.search("reward|math|verif|trainer|sampl|agent_loop", str(path.relative_to(folder)), re.I):
                    self.capture(path)
                if path.suffix not in TEXT_SUFFIXES:
                    continue
                if path.name in ("node_selection_snapshot.json", "scheduler_snapshot.json"):
                    continue
                if path.suffix in (".log", ".out", ".err"):
                    self.log(path, self.phase_context(folder, str(path.parent.relative_to(folder)), context))
                elif re.fullmatch(r"step_\d+\.json", path.name):
                    self.step_record(path, self.phase_context(folder, str(path.parent.relative_to(folder)), context))
                else:
                    self.capture(path)
            # Frozen reward/sampler/trainer source versions explain metric changes.
            for directory in (folder / "runtime", folder / "reward_runtime", folder / "d4_reward_fixed", folder / "md_reward_roundoff_0390"):
                for path in files_below(directory, max_depth=3):
                    if path.suffix == ".py" and re.search("reward|math|verif|trainer|sampl|agent_loop", str(path.relative_to(folder)), re.I):
                        self.capture(path)
        for path in run.iterdir():
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                self.capture(path)
        for sub in ("reports", "monitor", "managed_switch", "controller", "control", "handoff"):
            for path in files_below(run / sub, max_depth=2):
                if path.suffix in TEXT_SUFFIXES:
                    self.capture(path)
        self.checkpoint_metadata(run, role)
        # Include phase-local validation dumps as well as the shared run outputs.
        generation_roots = [(run / name, {"role": role, "setup": setup}) for name in ("generations", "raw_generations")]
        for lineage in sorted((run / "lineages").glob("*")):
            if lineage.is_dir() and not lineage.is_symlink():
                self.checkpoint_metadata(lineage, role, lineage.name)
                # Initial-validation/recovery summaries may live at the lineage root.
                for path in lineage.iterdir():
                    if path.is_file() and path.suffix in TEXT_SUFFIXES:
                        self.capture(path)
                for sub in ("reports", "validation"):
                    for path in files_below(lineage / sub, max_depth=2):
                        if path.suffix in TEXT_SUFFIXES:
                            self.capture(path)
                for name in ("generations", "raw_generations"):
                    generation_roots.append((lineage / name, {"role": role, "setup": setup, "lineage": lineage.name,
                                                             "run_dir": self.relative(lineage)}))
        for folder in sorted((run / "jobs").glob("*")):
            if folder.is_dir() and not folder.is_symlink():
                for phase in (".", "first_update", "restored_update", "train", "A", "B", "C", "D", "E"):
                    for name in ("generations", "raw_generations"):
                        generation_roots.append((folder / phase / name, self.phase_context(folder, phase, job_contexts[folder])))
        for directory, context in generation_roots:
            for path in files_below(directory, max_depth=4, prune=set()):
                if path.suffix == ".jsonl":
                    self.generations(path, context, stats_handle, sample_handle)

    def enrich_metrics(self):
        by_step = defaultdict(list)
        for row in self.step_records:
            by_step[(row["role"], row["job_folder"], row["step"])].append(row)
        for row in self.metrics:
            matches = by_step.get((row["role"], row["job_folder"], row["step"]), [])
            if len(matches) == 1:
                proof = matches[0]
                for key in ("phase", "source_step", "target_world", "execution_id", "step_usage"):
                    row[key] = proof.get(key)
                row["domain"] = row.get("domain") or proof.get("domain")
                row["step_record_source"] = proof["source"]

    @staticmethod
    def formal_attempt(row):
        return row.get("stage") == "train" and (row.get("job_role") in ("legacy_single", "formal")
            or (row.get("job_role") == "candidate" and row.get("handoff_committed")))

    def setup_summary(self):
        result = {}
        for setup in ("random", "domain"):
            rows = [r for r in self.metrics if r.get("setup") == setup and self.formal_attempt(r)]
            records = [r for r in self.step_records if r.get("setup") == setup and self.formal_attempt(r) and r["completed_update"]]
            updates = sorted({r["step"] for r in rows if is_update(r)} | {r["step"] for r in records})
            control = self.continuation.get(setup, {})
            state = control.get("state.json", {})
            result[setup] = dict(max_completed_step_on_formal_attempts=max(updates, default=None),
                observed_formal_attempt_steps=updates,
                validation_steps=sorted({r["step"] for r in rows if any(k.startswith(("val-", "val/")) for k in r["metrics"])}),
                current_job=object_or_empty(state.get("active")).get("job_id"), phase=state.get("phase"),
                promoted=state.get("promoted"), deadline_utc=state.get("deadline_utc"),
                handoff_checkpoint_step=object_or_empty(control.get("handoff_commit.json", {}).get("checkpoint")).get("step"),
                note="All formal attempts and committed candidate ancestry; repeated/replayed steps are NOT a resolved final trajectory.")
        return result

    def summarize(self):
        summary = {"version": VERSION, "created_utc": now(), "setups": self.setup_summary(), "roles": {}, "interpretation": [
            "All attempts are retained. A repeated global step is not another unit of net progress.",
            "Single and dual curves are separate; a qualification candidate is not automatically formal training.",
            "Maximum logged update step is evidence of a logged update, not proof of a durable checkpoint.",
            "Checkpoint metadata is inventoried without opening tensors or validating shard completeness.",
            "Metric endpoints/min/max include observed attempts; disputed steps require lineage review.",
            "PBS allocated hours include startup, validation and failed work; cput is never treated as walltime.",
            "Missing historical PBS jobs make allocated-time totals incomplete. No unrecorded metric is invented.",
            "Generation samples are convenience examples, not a random performance estimate.",
            "metrics.jsonl contains console metrics; step_records.jsonl separately retains completed-step evidence and actor samples.",
            "Candidate allocation uses two nodes even during eight-rank phases; target_world records the active training ranks.",
            "Committed qualification steps are accepted ancestry, not new training steps to add to console records.",
            "Binary W&B history is inventoried only; no online sync." ]}
        for role in RUNS:
            rows = [r for r in self.metrics if r["role"] == role and r["stage"] == "train"]
            updates = [r for r in rows if is_update(r)]
            steps = sorted({r["step"] for r in updates})
            attempts = defaultdict(set)
            for r in updates:
                attempts[r["step"]].add((r["job_folder"], r["source"], r["segment"]))
            by_key = defaultdict(list)
            for row in rows:
                for key, value in row["metrics"].items():
                    if finite(value):
                        by_key[key].append((row["step"], value))
            endpoints = {}
            for key, pairs in sorted(by_key.items()):
                first, last = min(s for s, v in pairs), max(s for s, v in pairs)
                endpoints[key] = {"first_step": first, "first_values": sorted({v for s, v in pairs if s == first}),
                    "last_step": last, "last_values": sorted({v for s, v in pairs if s == last}),
                    "min_observed": min(v for s, v in pairs), "max_observed": max(v for s, v in pairs)}
            jobs = [j for j in self.jobs if j["role"] == role and j["stage"] == "train" and j.get("job_id")]
            unique = {j["job_id"]: j for j in jobs}
            allocated, gpu_hours, known, gpu_known = 0., 0., 0, 0
            for job in unique.values():
                p = job.get("pbs", {})
                used = seconds(p.get("resources_used", {}).get("walltime"))
                if used is not None:
                    allocated += used / 3600
                    known += 1
                    gpus = p.get("Resource_List", {}).get("ngpus")
                    if str(gpus).isdigit():
                        gpu_hours += used * int(gpus) / 3600
                        gpu_known += 1
            # A job's pbs.log is its canonical stdout. Phase/tee mirrors must not
            # inflate accounting. Without it, use a sole metric source only.
            by_job = defaultdict(list)
            for r in updates:
                by_job[r["job_folder"]].append(r)
            time_rows, ambiguous_time_jobs = [], []
            for folder, job_rows in by_job.items():
                primary = [r for r in job_rows if r["source"] == folder + "/pbs.log"]
                sources = {r["source"] for r in job_rows}
                if primary:
                    time_rows.extend(primary)
                elif len(sources) == 1:
                    time_rows.extend(job_rows)
                else:
                    ambiguous_time_jobs.append(folder)
            durations = [r["metrics"]["timing_s/step"] for r in time_rows if finite(r["metrics"].get("timing_s/step"))]
            completed = [r for r in self.step_records if r["role"] == role and r["stage"] == "train" and r["completed_update"]]
            summary["roles"][role] = {"job_count": len(unique), "logged_update_steps": steps,
                "max_structured_completed_step_all_attempts": max((r["step"] for r in completed), default=None),
                "structured_completed_step_records": len(completed),
                "max_logged_update_step": max(steps, default=None), "unique_logged_update_step_count": len(steps),
                "missing_update_steps_1_to_max": [s for s in range(1, max(steps, default=0)+1) if s not in set(steps)],
                "repeated_step_candidates": {str(s): len(a) for s, a in attempts.items() if len(a) > 1},
                "validation_steps": sorted({r["step"] for r in rows if any(k.startswith(("val-", "val/")) for k in r["metrics"])}),
                "metric_endpoints_all_attempts": endpoints,
                "logged_step_seconds_all_attempts": {"count": len(durations), "sum": sum(durations),
                    "median": statistics.median(durations) if durations else None,
                    "ambiguous_source_jobs_excluded": ambiguous_time_jobs},
                "allocated_job_hours_known": allocated, "allocated_gpu_hours_known": gpu_hours,
                "pbs_walltime_coverage": {"known": known, "jobs": len(unique), "gpu_known": gpu_known}}
        return summary

    def run(self):
        self.output.mkdir(parents=True, exist_ok=False)
        archive_path = self.output.with_suffix(".tar.gz")
        if archive_path.exists():
            raise FileExistsError(archive_path)
        started = now()
        with tarfile.open(archive_path, "w:gz", compresslevel=3) as archive:
            self.archive = archive
            with (self.output / "generation_rows.jsonl").open("w") as stats, (self.output / "output_samples.jsonl").open("w") as samples:
                for role, suffix in RUNS.items():
                    self.collect_run(role, suffix, stats, samples)
            self.enrich_metrics()
            ids = sorted({j["job_id"] for j in self.jobs if isinstance(j.get("job_id"), str) and JOB_ID.fullmatch(j["job_id"])})
            if self.include_pbs:
                print(f"[0390] PBS accounting for {len(ids)} job IDs (history may have expired)", file=sys.stderr, flush=True)
                with ThreadPoolExecutor(max_workers=2) as pool:
                    queried = dict(zip(ids, pool.map(lambda jid: pbs_job(self.root, jid), ids)))
                for job in self.jobs:
                    job["pbs"] = queried.get(job.get("job_id"), {})
            # Inventory binary W&B history without loading a dependency or syncing runs.
            wandb = []
            for path in (self.root / "wandb").glob("*run-*/run-*.wandb"):
                if path.is_file():
                    wandb.append({"path": self.relative(path), "bytes": path.stat().st_size})
            dump(self.output / "wandb_inventory.json", wandb)
            summary = self.summarize()
            for name, values in (("metrics.jsonl", self.metrics), ("training_events.jsonl", self.events),
                                 ("step_records.jsonl", self.step_records)):
                with (self.output / name).open("w") as f:
                    for value in values:
                        emit(f, value)
            with (self.output / "metrics_long.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                columns = ["setup", "role", "job_id", "job_folder", "job_role", "stage", "phase", "lineage", "allocation_nodes",
                           "target_world", "execution_id", "handoff_committed", "handoff_step", "step_usage", "segment", "step", "domain"]
                writer.writerow(columns + ["metric", "value", "source", "line"])
                for r in self.metrics:
                    for key, value in r["metrics"].items():
                        writer.writerow([r.get(k) for k in columns] + [key, value, r["source"], r["line"]])
            with (self.output / "step_metrics_long.csv").open("w", newline="") as f:
                writer = csv.writer(f)
                columns = ["setup", "role", "job_id", "job_folder", "job_role", "phase", "lineage", "allocation_nodes",
                           "target_world", "source_step", "execution_id", "handoff_committed", "handoff_step", "step_usage",
                           "step", "domain", "finished_utc", "completed_update"]
                writer.writerow(columns + ["metric", "value", "source"])
                for r in self.step_records:
                    for key, value in r["metrics"].items():
                        writer.writerow([r.get(k) for k in columns] + [key, value, r["source"]])
            for name, value in (("summary.json", summary), ("jobs.json", self.jobs), ("checkpoints.json", self.checkpoints),
                                ("generation_summary.json", self.generation_stats), ("manifest.json", self.manifest), ("warnings.json", self.warnings),
                                ("continuation_status.json", self.continuation)):
                dump(self.output / name, value)
            dump(self.output / "collection.json", {"version": VERSION, "started_utc": started, "finished_utc": now(), "root": str(self.root),
                 "host": socket.gethostname(), "non_atomic_snapshot": True, "full_generations": self.full_generations,
                 "git_head": command(["git", "rev-parse", "HEAD"], self.root),
                 "git_status": command(["git", "--no-optional-locks", "status", "--short", "--untracked-files=no"], self.root)})
            lines = ["# D4 training history", "", "All observed attempts; metric endpoints are not a resolved final lineage.", "",
                     "| Setup | Max completed step on formal attempts | Validation steps | Current job | Phase |", "|---|---:|---|---|---|"]
            for setup, r in summary["setups"].items():
                lines.append(f"| {setup} | {r['max_completed_step_on_formal_attempts']} | {r['validation_steps']} | {r['current_job']} | {r['phase']} |")
            lines += ["",
                     "| Role | Max logged update step | Distinct update steps | Validation steps | PBS allocated hours (known jobs) |", "|---|---:|---:|---|---:|"]
            for role, r in summary["roles"].items():
                coverage = r["pbs_walltime_coverage"]
                lines.append(f"| {role} | {r['max_logged_update_step']} | {r['unique_logged_update_step_count']} | {r['validation_steps']} | {r['allocated_job_hours_known']:.2f} ({coverage['known']}/{coverage['jobs']}) |")
            lines += ["", "See summary.json for all metric endpoints and missing/repeated steps; metrics_long.csv and metrics.jsonl retain every attempt.", "",
                      "generation_rows.jsonl contains all recorded numeric output fields; output_samples.jsonl contains bounded examples.", "",
                      "Evidence contains full job logs, configuration/recovery records and checkpoint metadata. Tensor weights are not read.", "",
                      "Check warnings.json and manifest.json for missing/partial data. Unknown token counts are not inferred from text.", ""]
            (self.output / "SUMMARY.md").write_text("\n".join(lines))
            for path in sorted(self.output.iterdir()):
                archive.add(path, arcname="analysis/" + path.name, recursive=False)
        print((self.output / "SUMMARY.md").read_text())
        print("RESULT_BUNDLE=" + str(archive_path))
        print("ANALYSIS_DIR=" + str(self.output))
        return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--output-dir", type=Path)
    ap.add_argument("--no-pbs", action="store_true", help="Skip scheduler accounting; preserve logs and metrics")
    ap.add_argument("--full-generations", action="store_true", help="Also archive full generation JSONL text; can be large")
    args = ap.parse_args()
    root = args.root.resolve()
    if not (root / "outputs/multidomain").is_dir():
        ap.error("--root must contain outputs/multidomain")
    output = args.output_dir or root / "outputs/analysis" / ("d4_training_history_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    Collector(root, output, not args.no_pbs, args.full_generations).run()


if __name__ == "__main__":
    main()
