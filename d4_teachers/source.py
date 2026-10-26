"""Freeze the deployed data/verifiers and this implementation before submitting."""

from pathlib import Path
import shlex
import shutil
import uuid
from . import DOMAINS, VERSION
from .common import (
    digest,
    file_sha,
    files,
    is_fenced,
    lock,
    model_inventory,
    need,
    read,
    verify_files,
    write,
)
from .config import teacher_config
from . import data
from d4_training_continuation.scheduler import render_pbs
from d4_training_continuation.source import MANAGER_SHA, AUDIT_SHA

REPO = Path(__file__).resolve().parents[1]


def copy_bound(source, target, expected):
    source, target = Path(source), Path(target)
    need(
        source.is_file() and file_sha(source) == expected,
        "Pinned source missing/changed: " + str(source),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    need(file_sha(target) == expected, "Snapshot copy changed")


def prepare(campaign, policy):
    campaign = Path(campaign).resolve()
    with lock(campaign / "prepare.lock"):
        if (campaign / "campaign.json").exists():
            c = load_campaign(campaign)
            need(c["policy"] == policy, "Campaign already has another immutable policy")
            return c
        intent_path = campaign / "prepare_intent.json"
        if intent_path.exists():
            need(
                read(intent_path)["policy"] == policy,
                "Incomplete prepare belongs to another policy",
            )
        else:
            need(
                not (campaign / "template").exists(),
                "Unowned template directory already exists",
            )
            write(intent_path, dict(policy=policy))
        parent = Path(policy["parent_root"])
        source_request = parent / policy["source_request"]
        old = read(source_request)
        need(
            digest(old["contract"]) == old["identity"],
            "Parent request contract mismatch",
        )
        source = Path(old["job_dir"])
        need(
            source_request.resolve() == source / "request.json",
            "Parent request path changed",
        )
        need(
            Path(old["data_dir"]) == parent / policy["parent_data"],
            "Unexpected parent dataset",
        )
        parent_manifest = read(Path(old["data_dir"]) / "manifest.json")
        need(
            parent_manifest["config_sha256"] == digest(old["contract"]["base_config"]),
            "Parent data configuration changed",
        )
        for split in data.SPLITS:
            name = split + ".parquet"
            need(
                file_sha(Path(old["data_dir"]) / name)
                == parent_manifest["files"][name],
                "Frozen parent split changed",
            )
        dataset = data.prepare(old["data_dir"], campaign / "data")
        template = campaign / "template"
        if template.exists():
            # No campaign.json means this directory was never eligible for qsub.
            shutil.rmtree(template)
        template.mkdir()
        for relative, expected in old["contract"]["root_files"].items():
            need(
                relative.split("/")[0] in ("multidomain", "config", "scripts"),
                "Unexpected parent source",
            )
            copy_bound(parent / relative, template / "runtime" / relative, expected)
        snapshot = old["contract"]["snapshot_files"]
        selected = {
            n: h
            for n, h in snapshot.items()
            if n.startswith(
                ("audit_source/", "runtime/d4_db_mathfix/", "md_reward_roundoff_0390/")
            )
        }
        need(
            "runtime/d4_db_mathfix/verifier_backend.zip" in selected,
            "Isolated Math backend is not pinned",
        )
        for relative, expected in selected.items():
            copy_bound(source / relative, template / relative, expected)
        need(
            file_sha(template / "audit_source/state_audit.py") == AUDIT_SHA,
            "Unknown optimizer audit policy",
        )
        need(
            file_sha(
                Path(old["config"]["verl_src"])
                / "verl/utils/checkpoint/megatron_checkpoint_manager.py"
            )
            == MANAGER_SHA,
            "Installed checkpoint manager changed",
        )
        for package in ("d4_teachers", "d4_training_continuation"):
            shutil.copytree(
                REPO / package,
                template / "runtime" / package,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        need(
            (parent / "compat/sitecustomize.py").is_file(),
            "Parent NumPy checkpoint compatibility alias missing",
        )
        shutil.copytree(
            parent / "compat",
            template / "runtime/compat",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        base = Path(old["config"]["training"]["model_path"])
        need(
            base.is_dir() and (base / "config.json").is_file(),
            "Base HF model is unavailable",
        )
        base_files = model_inventory(base)
        contract = dict(
            version=VERSION,
            policy=policy,
            base_config=old["config"],
            source_request=str(source_request),
            source_identity=old["identity"],
            source_request_sha256=file_sha(source_request),
            source_job=str(source),
            parent_data_config=old["contract"]["base_config"],
            parent_data_manifest=parent_manifest,
            data_digest=digest(dataset),
            template_files=files(template),
            base_model=str(base),
            base_inventory=base_files,
            base_identity=digest(base_files),
            verifier_identity=digest(selected),
            overlay=str((parent / ".venv-multidomain").resolve()),
        )
        c = dict(
            contract=contract,
            identity=digest(contract),
            policy=policy,
            path=str(campaign),
        )
        (campaign / "control.py").write_text(
            "from pathlib import Path\nimport sys\n"
            "sys.path.insert(0,str(Path(__file__).resolve().parent/'template/runtime'))\n"
            "from d4_teachers.cli import main\nmain()\n"
        )
        for domain in (*DOMAINS, "base"):
            write(
                campaign / "runs" / domain / "run.json",
                dict(
                    domain=domain,
                    lineage=uuid.uuid4().hex,
                    campaign_identity=c["identity"],
                ),
            )
            write(
                campaign / "runs" / domain / "state.json",
                dict(
                    status="READY",
                    phase="test" if domain == "base" else "train",
                    job=None,
                    pending=None,
                    no_progress_retries=0,
                    last_progress=0,
                ),
            )
        write(campaign / "campaign.json", c)
        return c


def load_campaign(path, verify=True):
    path = Path(path).resolve()
    c = read(path / "campaign.json")
    need(
        c["path"] == str(path)
        and digest(c["contract"]) == c["identity"]
        and c["policy"] == c["contract"]["policy"],
        "Campaign identity changed",
    )
    if verify:
        verify_files(path / "template", c["contract"]["template_files"])
        need(
            digest(data.verify(path / "data")) == c["contract"]["data_digest"],
            "Dataset manifest changed",
        )
    return c


def make_job(c, domain, role):
    campaign = Path(c["path"])
    need(
        domain in (*DOMAINS, "base") and role in ("train", "export", "test"),
        "Unknown teacher role",
    )
    need(
        not is_fenced(campaign, domain),
        "Teacher is durably stopped; use reenable explicitly",
    )
    need(domain != "base" or role == "test", "Base is evaluation only")
    run = campaign / "runs" / domain
    lineage = read(run / "run.json")["lineage"]
    folder = run / "jobs" / (role + "_" + uuid.uuid4().hex)
    verify_files(campaign / "template", c["contract"]["template_files"])
    shutil.copytree(campaign / "template", folder)
    runtime = folder / "runtime"
    (runtime / ".venv-multidomain").symlink_to(
        c["contract"]["overlay"], target_is_directory=True
    )
    reward = folder / "md_reward_roundoff_0390/__init__.py"
    before = repr(str(Path(c["contract"]["source_job"]) / "math_diagnostics"))
    text = reward.read_text()
    need(text.count(before) == 1, "Unknown Math diagnostics binding")
    reward.write_text(text.replace(before, repr(str(folder / "math_diagnostics"))))
    cfg = teacher_config(
        c["contract"]["base_config"],
        c["policy"],
        domain if domain != "base" else "math",
        runtime,
    )
    entry = (
        "from pathlib import Path\nimport sys\nhere=Path(__file__).resolve().parent\n"
        "sys.path.insert(0,str(here))\nsys.path.insert(0,str(here/'runtime'))\n"
        f"sys.path.insert(1,{cfg['verl_src']!r})\n"
        "sys.path.insert(2,str(here/'runtime/compat'))\n"
        "import sitecustomize\n"
        "from d4_teachers.runtime import main\nmain()\n"
    )
    (folder / "entry.py").write_text(entry)
    node = (runtime / "scripts/multidomain/node.sh").read_text()
    before = '"$MD_PYTHON" -m multidomain.stage --request "$MD_REQUEST"'
    need(node.count(before) == 1, "Unknown node entry point")
    node = node.replace(
        before,
        '"$MD_PYTHON" -I '
        + shlex.quote(str(folder / "entry.py"))
        + ' execute --request "$MD_REQUEST"',
    )
    (folder / "node.sh").write_text(node)
    name = (
        "0390_dt_"
        + dict(
            math="math",
            **{"if": "if"},
            conversational_pivot="cp",
            logic_algorithmic="la",
            base="base",
        )[domain]
    )
    (folder / "stage.pbs").write_text(render_pbs(runtime, folder, 1, name))
    contract = dict(
        version=VERSION,
        campaign_identity=c["identity"],
        domain=domain,
        role=role,
        stage="train",
        campaign_dir=str(campaign),
        run_dir=str(run),
        job_dir=str(folder),
        root=str(runtime),
        job_name=name,
        expected_nodes=1,
        lineage=lineage,
        policy=c["policy"],
        config=cfg,
        data_dir=str(campaign / "data" / (domain if domain != "base" else "math")),
        data_manifest=read(campaign / "data/manifest.json"),
        base_model=c["contract"]["base_model"],
        base_identity=c["contract"]["base_identity"],
        verifier_identity=c["contract"]["verifier_identity"],
        reward=dict(import_path="pkg://md_reward_roundoff_0390"),
        snapshot_files=files(folder),
    )
    if role == "export":
        from .selection import validate_best

        need(
            read(run / "training_complete.json")["status"] == "PASS",
            "Teacher is not finalized",
        )
        contract["selected"] = validate_best(read(run / "best.json"))
    if role == "test":
        from .exporting import evaluation_model

        contract["evaluation_model"] = evaluation_model(c, domain)
    write(
        folder / "request.json",
        dict(contract, contract=contract, identity=digest(contract)),
    )
    return dict(folder=str(folder), job_name=name)


def validate_request(path):
    path = Path(path).resolve()
    r = read(path)
    need(digest(r["contract"]) == r["identity"], "Teacher request contract mismatch")
    need(
        all(r[k] == v for k, v in r["contract"].items()),
        "Teacher request fields changed",
    )
    need(path == Path(r["job_dir"]) / "request.json", "Teacher request moved")
    c = load_campaign(r["campaign_dir"], verify=False)
    need(c["identity"] == r["campaign_identity"], "Teacher belongs to another campaign")
    verify_files(path.parent, r["snapshot_files"])
    need(
        digest(data.verify(Path(r["campaign_dir"]) / "data"))
        == c["contract"]["data_digest"],
        "Data changed",
    )
    if r["role"] == "test":
        from .exporting import evaluation_model

        need(
            evaluation_model(c, r["domain"]) == r["evaluation_model"],
            "Test model changed since submission",
        )
    return r
