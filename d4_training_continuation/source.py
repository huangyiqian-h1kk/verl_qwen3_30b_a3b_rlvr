"""Freeze existing, accepted setup semantics; change only continuation machinery."""
import copy
from pathlib import Path
import shlex
import shutil
import uuid
from . import VERSION
from .common import below, digest, files, need, paths, read, sha, verify_files, write
from .scheduler import render_pbs
from .save_policy import POLICY, training_config, check_request

AUDIT_SHA = "59fddccb464acba7b2f58ed6113ca44fc74ea6af9e3a5ec65a54706b41d3fe8a"
MANAGER_SHA = "f132242b728458748e9f0cf0b60375e2f2a4ea420c5b035e85054bfa8163a3b8"


def topology_config(original, nodes):
    need(nodes in (1, 2), "Unsupported training topology")
    cfg = copy.deepcopy(original)
    cfg["training"]["train_nodes"] = nodes
    cfg.setdefault("resolved", {}).update(pbs_nodes_for_training_job=nodes, train_gpus=8 * nodes,
                                           training_rollout_replicas=2 * nodes)
    cfg.setdefault("scheduler", {})["walltime"] = "24:00:00"
    return cfg


def inspect(root, kind):
    p = paths(root, kind)
    state = read(p["old_control"] / "state.json")
    need(state.get("phase") in ("WAIT_READY", "WAIT_CHECKPOINT", "WAIT_CHECK"),
         "Legacy handoff is already underway; cannot blindly migrate " + kind)
    for field in ("pending_job", "pending_check"):
        need(not state.get(field), "Resolve old submission journal first: " + field)
    for name in ("decision.json", "qdel.json", "matched.json", "superseded.json"):
        need(not (p["old_control"] / name).exists(), "Legacy handoff has durable mutation evidence: " + name)
    monitor = p["monitor"] / "pending_submission.json"
    need(not monitor.exists(), "Resolve pending single-node submission before migration")
    need(isinstance(read(p["monitor"] / "state.json"), dict), "Missing legacy monitor state")
    single, dual = state["single"], state["dual"]
    need(Path(single["folder"]).parent == p["single"] / "jobs", "Wrong legacy single run")
    need(Path(dual["folder"]).parent == p["old"] / "jobs", "Wrong candidate run")
    folder = Path(dual["folder"])
    request = read(folder / "request.json")
    need(request["identity"] == state["identity"], "Candidate and controller identity differ")
    need(digest(request["contract"]) == request["identity"], "Old candidate contract mismatch")
    verify_files(folder, request["snapshot_files"])
    acceptance = read(p["old_control"] / "acceptance.json")
    need(acceptance["status"] == "PASS" and acceptance["identity"] == request["info"]["identity"],
         "Missing bound common check PASS")
    for name, expected in acceptance["evidence"].items():
        need(sha(name) == expected, "Common check evidence changed: " + name)
    packages = request["contract"].get("packages") or {
        "d4_two_node_resumefix": request["info"]["identity_inputs"]["package"],
        "d4_managed_switch": request["contract"]["package"]}
    for name, expected in packages.items():
        verify_files(folder / "runtime" / name, expected)
    need(sha(folder / "runtime/d4_two_node_resumefix/state_audit.py") == AUDIT_SHA,
         "Candidate lacks the approved empty-group audit source")
    cfg = request["config"]
    need(sha(Path(cfg["verl_src"]) / "verl/utils/checkpoint/megatron_checkpoint_manager.py") == MANAGER_SHA,
         "Installed checkpoint manager differs from the pinned, repaired version")
    t = cfg["training"]
    need((t["train_batch_size"], t["rollout_n"], t["ppo_mini_batch_size"], t["ppo_epochs"])
         == (256, 8, 32, 1), "Unexpected training batch contract")
    need(not cfg["resolved"]["judge_enabled"], "This continuation supports the four local-verifier domains")
    return dict(state=state, request=request, packages=packages, acceptance=acceptance)


def prepare(root, kind, operation):
    """Read/copy everything before asking the old owner to stop."""
    p = paths(root, kind)
    target = p["control"] / "operations" / operation
    planfile = target / "plan.json"
    if planfile.exists():
        plan = read(planfile)
        verify_files(target / "template", plan["template_files"])
        return plan
    inspected = inspect(root, kind)
    old = inspected["request"]
    source = Path(old["job_dir"])
    temp = target / "template"
    need(not temp.exists(), "Incomplete template exists; inspect before reusing operation")
    temp.mkdir(parents=True)
    for package, expected in inspected["packages"].items():
        for name in expected:
            src = source / "runtime" / package / name
            dst = temp / "runtime" / package / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    # Include the immutable isolated verifier archive, whose hash is also checked.
    for relative, expected in old["snapshot_files"].items():
        if relative.endswith(".zip"):
            dst = temp / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / relative, dst)
    reward = Path(old["reward"]["runtime_path"])
    need(reward.parent.parent == source and reward.name == "__init__.py", "Unknown reward wrapper layout")
    need(sha(reward) == old["reward"]["runtime_sha256"], "Reward source changed")
    shutil.copytree(reward.parent, temp / reward.parent.name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    if kind == "domain":
        need(old["reward"].get("repair_policy") == "math-verify-0.9-isolated-exact-product-timeout-zero-v2",
             "Domain verifier repair is missing")
        shutil.copy2(source / "math_repair.json", temp / "math_repair.json")
    # Keep audit sources separate from ordinary imported packages.
    audit = temp / "audit_source"
    audit.mkdir()
    for name in ("__init__.py", "core.py", "state_audit.py", "audit_offload.py"):
        shutil.copy2(source / "runtime/d4_two_node_resumefix" / name, audit / name)
    shutil.copytree(Path(__file__).parent, temp / "runtime/d4_training_continuation",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(p["root"] / "scripts/multidomain/node.sh", temp / "node.template.sh")
    # The existing dataset/agent/template/reward infrastructure is read-only.
    roots = {str(x.relative_to(p["root"])): sha(x) for sub in ("multidomain", "config/multidomain", "scripts/multidomain")
             for x in (p["root"] / sub).rglob("*")
             if x.is_file() and "__pycache__" not in x.parts and x.suffix != ".pyc"}
    plan = dict(version=VERSION, kind=kind, root=str(p["root"]), operation=operation,
                old_state=inspected["state"], old_request=old, acceptance=inspected["acceptance"],
                root_files=roots, template=str(temp), template_files=files(temp),
                single_lineage=uuid.uuid4().hex, candidate_lineage=uuid.uuid4().hex,
                training_policy=dict(POLICY))
    write(planfile, plan)
    return plan


def make_job(plan, role, nodes, authority, deadline):
    root, kind = Path(plan["root"]), plan["kind"]
    p = paths(root, kind)
    candidate = role == "candidate"
    need(role in ("candidate", "formal", "evaluation"), "Unsupported execution role")
    lineage = plan["candidate_lineage"] if candidate else authority["lineage"]
    need(lineage, "Formal new job requires an explicit lineage")
    folder = p["run"] / "jobs" / (role + "_" + uuid.uuid4().hex)
    folder.mkdir(parents=True)
    template = Path(plan["template"])
    verify_files(template, plan["template_files"])
    shutil.copytree(template, folder, dirs_exist_ok=True)
    old = plan["old_request"]
    reward_name = Path(old["reward"]["runtime_path"]).parent.name
    reward_file = folder / reward_name / "__init__.py"
    if kind == "domain":
        before = repr(str(Path(old["job_dir"]) / "math_diagnostics"))
        text = reward_file.read_text()
        need(text.count(before) == 1, "Unknown domain diagnostic-path binding")
        reward_file.write_text(text.replace(before, repr(str(folder / "math_diagnostics"))))
        math = read(folder / "math_repair.json")
        math["diagnostics"] = str(folder / "math_diagnostics")
        write(folder / "math_repair.json", math)
    entry = ("from pathlib import Path\nimport sys\nhere=Path(__file__).resolve().parent\n"
             f"sys.path.insert(0, {str(root)!r})\nsys.path.insert(0, str(here))\n"
             "sys.path.insert(0, str(here/'runtime'))\n"
             "from d4_training_continuation.runtime import main\nmain()\n")
    (folder / "entry.py").write_text(entry)
    node = (folder / "node.template.sh").read_text()
    original = '"$MD_PYTHON" -m multidomain.stage --request "$MD_REQUEST"'
    need(node.count(original) == 1, "Unknown node.sh entry point")
    node = node.replace(original, '"$MD_PYTHON" -I ' + shlex.quote(str(folder / "entry.py")) + ' execute --request "$MD_REQUEST"')
    (folder / "node.sh").write_text(node)
    job_name = "0390_d4db_cont" if kind == "domain" else "0390_d4r_cont"
    if role == "evaluation":
        job_name = "0390_d4_val"
    (folder / "stage.pbs").write_text(render_pbs(root, folder, nodes, job_name))
    # Legacy random requests said 25 while the launcher actually applied 5.
    # Preserve the established effective policy, not that stale request value.
    cfg = topology_config(training_config(old["config"]), nodes)
    # Remaining topology-derived config values are set per phase, never batch/sampler changes.
    contract = dict(version=VERSION, kind=kind, operation=plan["operation"], lineage=lineage,
                    role=role, config=cfg, source=authority, bound_check=plan["acceptance"]["identity"],
                    base_config=old["info"]["base_config"], data_dir=old["data_dir"], sampling=old.get("sampling"),
                    template_digest=digest(plan["template_files"]), root_files=plan["root_files"],
                    snapshot_files=files(folder), training_policy=dict(POLICY))
    if candidate and plan.get("opportunity_id"):
        contract["opportunity_id"] = plan["opportunity_id"]
    request = dict(version=VERSION, root=str(root), kind=kind, job_dir=str(folder),
                   run_dir=str(folder / "evaluation" if role == "evaluation" else p["run"] / "lineages" / lineage), control_dir=str(p["control"]),
                   stage="train", expected_nodes=nodes, job_name=job_name, lineage=lineage, role=role,
                   deadline_utc=deadline, contract=contract, identity=digest(contract), config=cfg,
                   data_dir=old["data_dir"], sampling=old.get("sampling"),
                   reward=dict(import_path="pkg://" + reward_name), source=authority,
                   training_policy=dict(POLICY))
    if contract.get("opportunity_id"):
        request["opportunity_id"] = contract["opportunity_id"]
    if plan.get("validation_policy"):
        from .validation_journal import POLICY as VALIDATION_POLICY
        from .evaluation import semantics
        need(plan["validation_policy"] == VALIDATION_POLICY, "Unknown validation plan")
        request["validation_policy"] = contract["validation_policy"] = dict(VALIDATION_POLICY)
        request["validation_semantics"] = contract["validation_semantics"] = semantics(request)
        request["identity"] = digest(contract)
    write(folder / "request.json", request)
    return dict(folder=str(folder), job_name=job_name)


def validate_request(path):
    path = Path(path).resolve()
    r = read(path)
    need(path == Path(r["job_dir"]) / "request.json", "Wrong request path")
    need(digest(r["contract"]) == r["identity"] and r["config"] == r["contract"]["config"], "Request contract changed")
    need(r.get("training_policy") == r["contract"].get("training_policy"), "Training policy contract changed")
    for key in ("validation_policy", "validation_semantics"):
        need(r.get(key) == r["contract"].get(key), "Validation contract changed: " + key)
    check_request(r)
    for key in ("kind", "role", "lineage", "source"):
        need(r[key] == r["contract"][key], "Request field changed: " + key)
    need(r.get("opportunity_id") == r["contract"].get("opportunity_id"), "Opportunity identity changed")
    for key in ("data_dir", "sampling"):
        need(r[key] == r["contract"][key], "Dataset/sampling contract changed: " + key)
    need(r["expected_nodes"] in (1, 2) and r["expected_nodes"] == r["config"]["training"]["train_nodes"], "Request topology changed")
    p = paths(r["root"], r["kind"])
    target = path.parent / "evaluation" if r["role"] == "evaluation" else p["run"] / "lineages" / r["lineage"]
    need(r["role"] in ("formal", "candidate", "evaluation"), "Unknown execution role")
    if r["role"] == "evaluation":
        need(r.get("validation_policy") and r["source"].get("record"), "Unbound evaluation-only request")
    need(path.parent.parent == p["run"] / "jobs" and Path(r["run_dir"]) == target
         and Path(r["control_dir"]) == p["control"], "Request run/control path changed")
    verify_files(path.parent, r["contract"]["snapshot_files"])
    verify_files(r["root"], r["contract"]["root_files"])
    if r.get("validation_policy"):
        from .validation_journal import POLICY as VALIDATION_POLICY
        from .evaluation import semantics
        need(r["validation_policy"] == VALIDATION_POLICY and r["validation_semantics"] == semantics(r),
             "Validation semantics changed")
    return r
