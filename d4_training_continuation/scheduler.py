"""PBS observations, identity-bound cancellation and at-most-once submission."""
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from . import QUEUE
from .common import atomic_text, need, now, read, write
from .capacity import attributes, free_hosts, queue_pool_profile

ID = re.compile(r"[0-9]+(?:\.[A-Za-z0-9._-]+)?")


class Unknown(RuntimeError):
    pass


def terminal(row):
    return row.get("job_state") in ("F", "C") and re.fullmatch(r"-?\d+", str(row.get("Exit_status", ""))) is not None


def retryable(row):
    if not terminal(row):
        return False
    code = str(row["Exit_status"])
    comment = str(row.get("comment", "")).lower()
    return code == "271" or (code == "-29" and "walltime" in comment and "exceed" in comment)


def validate_job(job, row):
    need(ID.fullmatch(job["job_id"]), "Invalid PBS job ID")
    need(row.get("Job_Owner", "").split("@")[0] == getpass.getuser(), "PBS owner mismatch")
    need(row.get("queue") == QUEUE and row.get("Job_Name") == job["job_name"], "PBS queue/name mismatch")
    args = str(row.get("Submit_arguments", "")).replace("\n", "").replace("\t", "")
    need(args == str(Path(job["folder"]) / "stage.pbs"), "PBS submission path mismatch")
    return row


class PBS:
    def command(self, args, timeout=20):
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        env["LC_ALL"] = "C"
        try:
            p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise Unknown(str(e)) from e
        if p.returncode:
            raise Unknown(f"{args}: exit={p.returncode}; stdout={p.stdout[:400]!r}; stderr={p.stderr[:400]!r}")
        return p.stdout

    def jobs(self, history=False, job_id=None):
        try:
            args = ["qstat", "-xf" if history else "-f", "-F", "json"]
            if job_id:
                args.append(job_id)
            payload = json.loads(self.command(args))
            need(isinstance(payload, dict) and ("Jobs" in payload or "pbs_server" in payload), "Invalid PBS JSON")
            need(isinstance(payload.get("Jobs", {}), dict), "Invalid Jobs mapping")
            return payload.get("Jobs", {})
        except (ValueError, RuntimeError) as e:
            raise Unknown(str(e)) from e

    def observe(self, job):
        rows = self.jobs(history=True, job_id=job["job_id"])
        if job["job_id"] not in rows:
            raise Unknown("PBS omitted registered job " + job["job_id"])
        return validate_job(job, rows[job["job_id"]])

    def capacity(self):
        # Bounded retry, kept distinct from observing whether a job has ended.
        errors = []
        for _ in range(2):
            try:
                q = self.command(["qstat", "-Qf", QUEUE])
                need(re.search(r"^Queue:\s*" + QUEUE + r"\s*$", q, re.M), "Wrong queue")
                queue = attributes(q)
                nodes = json.loads(self.command(["pbsnodes", "-av", "-F", "json"]))
                return dict(free_hosts(nodes, None, queue), at_utc=now())
            except (Unknown, ValueError, RuntimeError) as e:
                errors.append(str(e))
        return dict(free_slots=None, reason="capacity_unknown", errors=errors, at_utc=now())

    def recover(self, folder, job_name):
        folder = Path(folder)
        known, attempted = folder / "job_id.txt", folder / "qsub_attempted.json"
        if known.exists():
            jid = known.read_text().strip()
            need(ID.fullmatch(jid), "Invalid durable submission ID")
            return dict(job_id=jid, folder=str(folder), job_name=job_name)
        if not attempted.exists():
            return None
        stdout = folder / "qsub.stdout.txt"
        if stdout.exists() and ID.fullmatch(stdout.read_text().strip()):
            jid = stdout.read_text().strip()
            job = dict(job_id=jid, folder=str(folder), job_name=job_name)
            self.observe(job)
        else:
            matches = [jid for jid, row in self.jobs(history=True).items()
                       if row.get("Submit_arguments") == str(folder / "stage.pbs")
                       and row.get("Job_Name") == job_name
                       and row.get("Job_Owner", "").split("@")[0] == getpass.getuser()]
            if len(matches) != 1:
                raise Unknown("Ambiguous qsub; will not submit again: " + str(folder))
            jid = matches[0]
            job = dict(job_id=jid, folder=str(folder), job_name=job_name)
            self.observe(job)
        atomic_text(known, jid + "\n")
        return job

    def submit(self, folder, job_name):
        folder = Path(folder)
        recovered = self.recover(folder, job_name)
        if recovered:
            return recovered
        # No node availability gate, job-count limit, user hold, or host blacklist.
        q = self.command(["qstat", "-Qf", QUEUE])
        need(re.search(r"^Queue:\s*" + QUEUE + r"\s*$", q, re.M), "Wrong queue")
        queue = attributes(q)
        queue_pool_profile(queue)
        need(queue.get("enabled", "").lower() == "true" and queue.get("started", "").lower() == "true",
             "Reserved queue is disabled")
        write(folder / "qsub_attempted.json", dict(at_utc=now(), script=str(folder / "stage.pbs")))
        try:
            output = self.command(["qsub", str(folder / "stage.pbs")], timeout=45)
        except Exception as e:
            write(folder / "qsub_error.json", dict(at_utc=now(), error=str(e), result="AMBIGUOUS_NO_RESUBMIT"))
            raise
        atomic_text(folder / "qsub.stdout.txt", output)
        need(ID.fullmatch(output.strip()), "Ambiguous qsub output; no blind retry")
        atomic_text(folder / "job_id.txt", output.strip() + "\n")
        return dict(job_id=output.strip(), folder=str(folder), job_name=job_name)

    def cancel(self, job, journal):
        row = self.observe(job)
        if terminal(row):
            return True
        need(row.get("job_state") in ("Q", "H", "R", "E", "B", "S"), "Unknown PBS cancellation state")
        # Reissuing qdel for the SAME verified job after an ambiguous return is safe.
        write(journal, dict(job=job, at_utc=now(), action="qdel", status="ATTEMPTED"))
        try:
            output = self.command(["qdel", job["job_id"]])
            write(journal, dict(job=job, at_utc=now(), status="RETURNED", stdout=output))
        except Unknown as e:
            write(journal, dict(job=job, at_utc=now(), status="UNKNOWN", error=str(e)))
        return False  # Always re-observe terminal before releasing a writer lease.


def render_pbs(root, folder, nodes, job_name):
    need(nodes in (1, 2) and re.fullmatch(r"0390_[a-z0-9_]+", job_name), "Invalid PBS request")
    q = shlex.quote
    root, folder = Path(root), Path(folder)
    py = q(str(root / ".venv-multidomain/bin/python"))
    return "\n".join([
        "#!/bin/bash", "#PBS -P gcg51557", "#PBS -q " + QUEUE, "#PBS -v RTYPE=rt_HF",
        "#PBS -l select=" + str(nodes), "#PBS -l walltime=24:00:00", "#PBS -N " + job_name,
        "#PBS -r n", "#PBS -c n", "#PBS -j oe", "#PBS -k oe", "set -euo pipefail",
        "exec > >(tee -a " + q(str(folder / "pbs.log")) + ") 2>&1",
        "export PYTHONUNBUFFERED=1", "cd " + q(str(root)),
        "if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi",
        "module load gcc/13.2.0 cuda/12.8/12.8.1 cudnn/9.10/9.10.2 nccl/2.29/2.29.7-1",
        # Each node sources the existing, contract-bound environment in node.sh.
        # The outer allocation probe uses only stdlib and the project interpreter.
        "module load hpcx/2.20 nccl/2.29/2.29.7-1",
        "awk '!seen[$0]++' \"$PBS_NODEFILE\" > " + q(str(folder / "hosts")),
        "[[ $(wc -l < " + q(str(folder / "hosts")) + ") -eq " + str(nodes) + " ]]",
        py + " -I " + q(str(folder / "entry.py")) + " allocation --request " + q(str(folder / "request.json")),
        "mpirun -np " + str(nodes) + " --map-by ppr:1:node --bind-to none --hostfile " + q(str(folder / "hosts"))
        + " bash " + q(str(folder / "node.sh")) + " " + q(str(root)) + " " + q(str(folder / "request.json"))
        + " " + q(str(folder)), ""])
