from pathlib import Path
import copy
import pytest
import torch

from d4_mopd import checkpoints, control
from d4_mopd.spec import PlannedSampler
from d4_teachers.common import read, write
from d4_training_continuation.scheduler import Unknown


@pytest.mark.parametrize("mode", ["missing_path", "working", "invalid_header"])
def test_te_header_probe_compiles_using_inherited_include_paths(tmp_path, mode):
    """A real compiler catches the difference between existing and usable headers."""
    import os
    import shutil
    import subprocess
    import sys
    assert shutil.which("g++"), "Compiler probe regression requires g++"
    root = tmp_path / "build paths with spaces"
    cuda, cudnn, nccl, inherited, audit = (root / p for p in ("cuda", "cudnn", "nccl", "inherited", "audit"))
    for path in (cuda, cudnn, nccl, inherited):
        (path / "include").mkdir(parents=True)
        (path / "lib").mkdir()
    audit.mkdir()
    (cuda / "include/cuda_runtime.h").write_text("#pragma once\n")
    (cudnn / "include/cudnn.h").write_text("#error broken_header\n" if mode == "invalid_header" else "#pragma once\n")
    (nccl / "include/nccl.h").write_text("#pragma once\n")
    env = {k: v for k, v in os.environ.items() if k not in ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH")}
    env.update(CUDA_HOME=str(cuda), CUDNN_PATH=str(cudnn), NCCL_HOME=str(nccl), CXX=shutil.which("g++"),
               CPATH=str(inherited / "include"), LIBRARY_PATH=str(inherited / "lib"),
               LD_LIBRARY_PATH=str(inherited / "lib"))
    script = Path(__file__).resolve().parents[2] / "scripts/mopd_bootstrap.sh"
    setup = "" if mode == "missing_path" else "mopd_te_include_environment; "
    result = subprocess.run(["bash", "-euo", "pipefail", "-c",
        'source "$1"; '+setup+'mopd_te_header_probe "$2" "$3"', "header-test",
        str(script), sys.executable, str(audit)], env=env, capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) == (mode == "working"), result.stdout + result.stderr
    proof = read(audit / "header_probe.json")
    assert proof["status"] == ("PASS" if mode == "working" else "FAIL")
    if mode == "working":
        assert (audit / "header_probe.o").is_file()
        assert proof["environment"]["CPATH"] == f"{cudnn}/include:{nccl}/include:{inherited}/include"
        for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            assert proof["environment"][key] == f"{cudnn}/lib:{nccl}/lib:{inherited}/lib"
    else:
        assert ("broken_header" if mode == "invalid_header" else "cudnn.h") in result.stderr
        assert "TE compiler header probe: PASS" not in result.stdout


@pytest.mark.parametrize("broken", [False, True])
def test_te_source_probe_checks_every_translation_unit_before_cuda_build(tmp_path, broken):
    import os
    import shutil
    import subprocess
    import sys
    source, audit, includes = (tmp_path / p for p in ("source", "audit", "include"))
    for path in (source / "build_tools", audit, includes): path.mkdir(parents=True)
    (source / "build_tools/__init__.py").write_text("")
    (source / "build_tools/pytorch.py").write_text('''
from pathlib import Path
from types import SimpleNamespace
def setup_pytorch_extension(*args):
    root = Path(__file__).parents[1]
    return SimpleNamespace(sources=[str(root / p) for p in ('a.cpp', 'b.cpp')],
        extra_compile_args={'cxx': ['-DREQUIRED_FLAG=7']}, include_dirs=[str(root.parent / 'include')],
        define_macros=[('REQUIRED_MACRO', '3')])
''')
    (includes / "required.h").write_text("#pragma once\n")
    (source / "a.cpp").write_text('#include <Python.h>\n#include <required.h>\n'
        '#if REQUIRED_FLAG != 7 || REQUIRED_MACRO != 3\n#error missing_flags\n#endif\n'
        '#ifndef TORCH_EXTENSION_NAME\n#error missing_torch_define\n#endif\nint a() { return 0; }\n')
    (source / "b.cpp").write_text("#error later_source_broken\n" if broken else "int b() { return 0; }\n")
    script = Path(__file__).resolve().parents[2] / "scripts/mopd_bootstrap.sh"
    result = subprocess.run(["bash", "-euo", "pipefail", "-c",
        'source "$1"; mopd_te_torch_source_probe "$2" "$3" "$4"', "source-test",
        str(script), sys.executable, str(audit), str(source)], env=dict(os.environ, CXX=shutil.which("g++")),
        capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) != broken, result.stdout + result.stderr
    proof = read(audit / "torch_source_probe.json")
    assert proof["status"] == ("FAIL" if broken else "PASS")
    assert len(proof["checked_sources"]) == proof["expected_sources"] == 2
    assert not list(source.glob("*.o"))
    if broken: assert "later_source_broken" in result.stderr


@pytest.mark.parametrize("location", ["path", "local", "cargo", "venv", "custom", "explicit", "missing", "bad_explicit", "old_cli"])
def test_uv_resolution_does_not_require_login_shell_path(tmp_path, location):
    import os
    import shutil
    import subprocess
    import sys
    root = tmp_path / "paths with spaces"
    bins, user, upstream, audit = (root / p for p in ("bin", "user", "upstream", "audit"))
    for p in (bins, user, upstream, audit): p.mkdir(parents=True)
    (bins / "readlink").symlink_to(shutil.which("readlink"))
    locations = dict(path=bins / "uv", local=user / ".local/bin/uv", cargo=user / ".cargo/bin/uv",
                     venv=upstream / ".venv/bin/uv", custom=root / "custom/uv", explicit=root / "chosen/uv",
                     old_cli=user / ".local/bin/uv")
    uv = locations.get(location)
    if uv:
        uv.parent.mkdir(parents=True, exist_ok=True)
        uv.write_text('#!/bin/sh\nif [ "$1" = "--version" ]; then printf "uv 0.13.6\\n"; exit 0; fi\n'
                      + ('exit 2\n' if location == "old_cli" else 'exit 0\n'))
        uv.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("MOPD_UV", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL")}
    env["PATH"] = str(bins)
    if location == "explicit": env["MOPD_UV"] = str(uv)
    if location == "bad_explicit": env["MOPD_UV"] = str(root / "absent/uv")
    if location == "custom": env["UV_INSTALL_DIR"] = str(uv.parent)
    script = Path(__file__).resolve().parents[2] / "scripts/mopd_bootstrap.sh"
    # The missing case is the initial bootstrap, which has no Python yet.
    python = "" if location == "missing" else sys.executable
    audit_arg = "" if location == "missing" else str(audit)
    result = subprocess.run([shutil.which("bash"), "-euo", "pipefail", "-c",
        'source "$1"; mopd_resolve_uv "$2" "$3" "$4" "$5"', "uv-test", str(script),
        str(upstream), python, audit_arg, str(user)], env=env, capture_output=True, text=True, timeout=20)
    success = location not in ("missing", "bad_explicit", "old_cli")
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    assert not (audit / "uv-tool").exists()
    if success:
        import hashlib
        record = read(audit / "uv.json")
        assert record["path"] == str(uv)
        assert record["sha256"] == hashlib.sha256(uv.read_bytes()).hexdigest()
        assert record["version"] == "uv 0.13.6"
    else:
        assert not (audit / "uv.json").exists()
    if location == "missing": assert "uv was not found" in result.stderr
    if location == "bad_explicit": assert "MOPD_UV must be" in result.stderr


def test_uv_fallback_installs_cli_outside_training_environment(tmp_path):
    import json
    import os
    import shutil
    import subprocess
    import sys
    bins, user, upstream, audit = (tmp_path / p for p in ("bin", "user", "upstream", "audit"))
    for p in (bins, user, upstream, audit): p.mkdir()
    (bins / "readlink").symlink_to(shutil.which("readlink"))
    calls = tmp_path / "calls.jsonl"
    wrapper = upstream / "python"
    wrapper.write_text(f'#!{sys.executable}\n'+r'''
import json, os, shutil, sys
from pathlib import Path
args = sys.argv[1:]
if args[:3] == ['-I', '-m', 'venv']:
    with open(os.environ['MOCK_CALLS'], 'a') as f: f.write(json.dumps(['venv', args[3]])+'\n')
    folder = Path(args[3]) / 'bin'; folder.mkdir(parents=True)
    shutil.copy2(__file__, folder / 'python')
elif args[:4] == ['-I', '-m', 'pip', '--isolated']:
    with open(os.environ['MOCK_CALLS'], 'a') as f: f.write(json.dumps(['pip', __file__, args])+'\n')
    uv = Path(__file__).with_name('uv')
    uv.write_text('#!/bin/sh\nprintf "uv 0.13.6\\n"\n')
    uv.chmod(0o755)
else:
    os.execv(sys.executable, [sys.executable] + args)
''')
    wrapper.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("MOPD_UV", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL")}
    env.update(PATH=str(bins), MOCK_CALLS=str(calls))
    script = Path(__file__).resolve().parents[2] / "scripts/mopd_bootstrap.sh"
    result = subprocess.run([shutil.which("bash"), "-euo", "pipefail", "-c",
        'source "$1"; mopd_resolve_uv "$2" "$3" "$4" "$5"', "uv-test", str(script),
        str(upstream), str(wrapper), str(audit), str(user)], env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    events = [json.loads(line) for line in calls.read_text().splitlines()]
    assert events[0] == ["venv", str(audit / "uv-tool")]
    assert events[1][1] == str(audit / "uv-tool/bin/python")
    assert events[1][2][-2:] == ["--no-deps", "uv==0.13.6"]
    assert read(audit / "uv.json")["path"] == str(audit / "uv-tool/bin/uv")
    assert list(upstream.iterdir()) == [wrapper]


@pytest.mark.parametrize("available,release,load_fails,selected,success", [
    ("nested", "13.0", False, "cuda/13.0/13.0.1", True),
    ("flat", "13.0", False, "cuda/13.0.1", True),
    ("both", "13.0", False, "cuda/13.0/13.0.1", True),
    ("neither", "13.0", False, None, False),
    ("nested", "12.8", False, "cuda/13.0/13.0.1", False),
    ("nested", "13.1", False, "cuda/13.0/13.0.1", False),
    ("nested", "13.0", True, "cuda/13.0/13.0.1", False),
])
def test_te_toolchain_resolves_abci_module_and_refuses_wrong_cuda(
        tmp_path, available, release, load_fails, selected, success):
    """Run the production shell check against a synthetic Modules installation."""
    import os
    import subprocess
    import sys
    cuda = tmp_path / "cuda" / "bin"
    cuda.mkdir(parents=True)
    nvcc = cuda / "nvcc"
    nvcc.write_text(f'#!/bin/sh\nprintf "Cuda compilation tools, release {release}, V{release}.88\\n"\n')
    nvcc.chmod(0o755)
    audit = tmp_path / "audit"
    audit.mkdir()
    script = Path(__file__).resolve().parents[2] / "scripts/mopd_bootstrap.sh"
    env = dict(os.environ, MOCK_AVAILABLE=available, MOCK_LOAD_FAIL=str(int(load_fails)),
               MOCK_CUDA=str(cuda), MOCK_AUDIT=str(audit))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", r'''
module() {
  case "$1" in
    -t) printf '/apps/modules:\ncuda/12.8/12.8.1\ncuda/13.0/13.0.1\n' >&2 ;;
    show)
      if [[ $2 == cuda/13.0/13.0.1 && ( $MOCK_AVAILABLE == nested || $MOCK_AVAILABLE == both ) ]] ||
         [[ $2 == cuda/13.0.1 && ( $MOCK_AVAILABLE == flat || $MOCK_AVAILABLE == both ) ]]; then
        printf 'modulefile %s\n' "$2" >&2
      else
        printf 'ERROR: Unable to locate a modulefile for %s\n' "$2" >&2
        return 1
      fi ;;
    load)
      printf '%s\n' "$*" > "$MOCK_AUDIT/load_called.txt"
      [[ $MOCK_LOAD_FAIL == 0 ]] || return 1
      export PATH="$MOCK_CUDA:$PATH" ;;
    list) printf 'loaded modules\n' >&2 ;;
    *) return 2 ;;
  esac
}
source "$1"
mopd_te_toolchain "$2" "$3"
touch "$3/build_permitted"
''', "toolchain-test", str(script), sys.executable, str(audit)],
        env=env, capture_output=True, text=True, timeout=20)
    assert (result.returncode == 0) == success, result.stdout + result.stderr
    assert (audit / "build_permitted").exists() == success
    assert "cuda/12.8/12.8.1" in (audit / "cuda_modules.txt").read_text()
    if selected:
        assert (audit / "load_called.txt").read_text().strip() == f"load gcc/13.2.0 {selected} cmake/4.1.1"
    else:
        assert not (audit / "load_called.txt").exists()
        assert "CUDA 13.0.1 module not found" in result.stderr
    if selected and not load_fails:
        proof = read(audit / "toolchain.json")
        assert proof["status"] == ("PASS" if success else "FAIL")
        assert proof["cuda_module"] == selected
        assert proof["cuda_home"] == str(cuda.parent)
        assert proof["nvcc_path"] == str(nvcc)
        if not success:
            assert "requires nvcc release 13.0" in result.stderr


def blocked_qualifications(manifest, tmp_path, monkeypatch):
    from d4_mopd import preflight, ARMS
    from d4_mopd import megatron_repair
    states = {}
    for i, arm in enumerate(ARMS):
        folder = tmp_path / "qualification" / arm / "attempts" / "failed"
        folder.mkdir(parents=True)
        (folder / "pbs.log").write_text("OSError: GLIBC_2.38 not found")
        states["qualification:"+arm] = dict(phase="qualification", arm=arm, status="BLOCKED", pending=None,
            error="Unclassified terminal job; inspect logs",
            job=dict(folder=str(folder), job_name="0390_mp_q"+arm, job_id=f"{10+i}.pbs1"))
    abi = dict(status="PASS", files=[dict(sha256="repaired")], incompatible=[])
    write(tmp_path / "states.json", states)
    guard = dict(status="PASS", sha256="guard")
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], versions={},
                                              native_abi=abi, megatron_import_guard=guard))
    write(tmp_path / "pipeline.json", dict(schema="d4-mopd-pipeline-v1", identity=manifest["identity"], arms=list(ARMS)))
    monkeypatch.setattr(control, "verify", lambda _:manifest)
    monkeypatch.setattr(preflight, "environment", lambda **_: {})
    monkeypatch.setattr(preflight, "native_abi", lambda:abi)
    monkeypatch.setattr(megatron_repair, "inspect", lambda:guard)
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=1)
    monkeypatch.setattr(control, "PBS", PBS)
    return states, abi


def test_explicit_retry_preserves_failures_and_parallel_pipeline_barrier(manifest, tmp_path, monkeypatch):
    from d4_mopd import ARMS
    states, _ = blocked_qualifications(manifest, tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(control, "start", lambda *args:calls.append(args) or dict(status="REGISTERED"))
    control.retry_qualification(manifest["manifest_path"], ARMS)
    current = read(tmp_path / "states.json")
    assert calls == [(manifest["manifest_path"], "pipeline", list(ARMS))]
    assert {s["status"] for s in current.values()} == {"READY"}
    for key, old in states.items():
        assert current[key]["previous_job"] == old["job"]
        assert Path(old["job"]["folder"], "pbs.log").read_text().startswith("OSError")
        assert read(current[key]["explicit_retry"])["before"] == states
    control.advance_pipeline(manifest, current)
    assert not any(s["phase"] == "train" for s in current.values())
    # A repeated command cannot erase a fresh pending/running submission.
    with pytest.raises(RuntimeError, match="BLOCKED"):
        control.retry_qualification(manifest["manifest_path"], ARMS)
    assert len(calls) == 1


@pytest.mark.parametrize("condition", ["active", "unknown", "pending", "checkpoint", "old_proof", "binary_change", "guardian",
                                       "guard_change", "missing_guard"])
def test_explicit_retry_refuses_ambiguous_or_changed_state(manifest, tmp_path, monkeypatch, condition):
    from contextlib import ExitStack
    from d4_mopd import ARMS, preflight
    from d4_training_continuation.common import locked
    states, abi = blocked_qualifications(manifest, tmp_path, monkeypatch)
    key = "qualification:domain_mix"
    if condition in ("active", "unknown"):
        class PBS:
            def observe(self, job):
                if condition == "unknown": raise Unknown("qstat timeout")
                return dict(job_state="R")
        monkeypatch.setattr(control, "PBS", PBS)
    if condition == "pending":
        states[key]["pending"] = {"folder":"ambiguous"}
        write(tmp_path / "states.json", states)
    if condition == "checkpoint":
        (tmp_path / "qualification/domain_mix/checkpoints/global_step_4").mkdir(parents=True)
    if condition == "old_proof":
        write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], versions={}))
    if condition == "binary_change":
        monkeypatch.setattr(preflight, "native_abi", lambda:dict(abi, files=[dict(sha256="changed")]))
    if condition == "guard_change":
        from d4_mopd import megatron_repair
        monkeypatch.setattr(megatron_repair, "inspect", lambda:dict(status="PASS", sha256="changed"))
    if condition == "missing_guard":
        proof = read(tmp_path / "cpu_preflight.json")
        proof.pop("megatron_import_guard")
        write(tmp_path / "cpu_preflight.json", proof)
    monkeypatch.setattr(control, "start", lambda *a:pytest.fail("must not launch"))
    with ExitStack() as stack:
        if condition == "guardian": stack.enter_context(locked(tmp_path / "guardian.lock"))
        with pytest.raises((RuntimeError, BlockingIOError, Unknown)):
            control.retry_qualification(manifest["manifest_path"], ARMS)
    assert read(tmp_path / "states.json") == states
    assert not list((tmp_path / "qualification_retries").glob("*.json"))


def test_structured_node_error_blocks_instead_of_external_retry(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    write(folder / "gpu_node_1.json", dict(status="FAIL", error="GLIBC_2.38 not found"))
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=271, comment="terminated by root@login3")
    control.tick(manifest, state, PBS())
    s = state["train:domain_mix"]
    assert s["status"] == "BLOCKED" and "GLIBC_2.38" in s["error"]


@pytest.mark.parametrize("build_fails", [False, True])
def test_environment_repair_keeps_states_frozen_and_requires_new_preflight(manifest, tmp_path, monkeypatch, build_fails):
    import os
    import subprocess
    import importlib.metadata
    from types import SimpleNamespace
    states, _ = blocked_qualifications(manifest, tmp_path, monkeypatch)
    before = read(tmp_path / "cpu_preflight.json")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda:[SimpleNamespace(metadata={"Name":"torch"}, version="2.13.0")])
    def build(args, **kwargs):
        assert args[3] == "--rebuild-te"
        from d4_mopd.repair_pipeline import lock_path
        assert len(kwargs["pass_fds"]) == 1
        assert os.fstat(kwargs["pass_fds"][0]).st_ino == lock_path(manifest).stat().st_ino
        assert control.active(lock_path(manifest))
        assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
        assert control.active(tmp_path / "guardian.lock")
        if build_fails: raise subprocess.CalledProcessError(1, args)
        write(Path(args[4]) / "build_receipt.json", dict(source_revision="pinned", sha256="wheel"))
    monkeypatch.setattr(control.subprocess, "run", build)
    if build_fails:
        with pytest.raises(subprocess.CalledProcessError): control.repair_te(manifest["manifest_path"])
    else:
        assert control.repair_te(manifest["manifest_path"])["status"] == "PASS"
    assert read(tmp_path / "states.json") == states
    assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
    audits = list((tmp_path / "environment_repairs").iterdir())
    assert len(audits) == 1
    assert read(audits[0] / "previous_cpu_preflight.json") == before
    assert read(audits[0] / "result.json")["status"] == ("FAIL" if build_fails else "PASS")
    assert not control.active(tmp_path / "guardian.lock")


@pytest.mark.parametrize("change", ["build_tools", "runtime_addition", "runtime_upgrade"])
def test_environment_repair_limits_package_changes(manifest, tmp_path, monkeypatch, change):
    import importlib.metadata
    from types import SimpleNamespace
    states, _ = blocked_qualifications(manifest, tmp_path, monkeypatch)
    packages = {"torch": "2.13.0"}
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": name}, version=version) for name, version in packages.items()])
    def build(args, **kwargs):
        packages.update(setuptools="80.10.2", wheel="0.48.0", pybind11="3.0.4", ninja="1.13.0")
        if change == "runtime_addition": packages["unexpected-runtime"] = "1.0"
        if change == "runtime_upgrade": packages["torch"] = "2.14.0"
        write(Path(args[4]) / "build_receipt.json", dict(source_revision="pinned", sha256="wheel"))
    monkeypatch.setattr(control.subprocess, "run", build)
    if change == "build_tools":
        assert control.repair_te(manifest["manifest_path"])["status"] == "PASS"
    else:
        with pytest.raises(RuntimeError, match="dependency"):
            control.repair_te(manifest["manifest_path"])
    assert read(tmp_path / "states.json") == states
    assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
    assert not control.active(tmp_path / "guardian.lock")


def checkpoint(root, step, identity="id", arm="domain_mix"):
    folder = root / f"global_step_{step}"
    actor = folder / "actor"
    contents = {}
    for key in ("model", "optimizer", "lr_scheduler", "rng_state"):
        path = actor / key / "shard.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-native-shard")
        contents[key] = {"path": key}
    write(actor / "ckpt_contents.json", dict(framework="megatron",global_step=step,world_size=8,contents=contents))
    (folder / "data.pt").write_bytes(b"loader")
    (folder / "driver.pt").write_bytes(b"driver")
    checkpoints.commit(folder, identity, arm, step, "sampler")
    return folder


def test_partial_newer_checkpoint_is_never_resumed(tmp_path):
    complete = checkpoint(tmp_path, 4)
    (tmp_path / "global_step_8/actor").mkdir(parents=True)
    assert checkpoints.latest(tmp_path, "id", "domain_mix", "sampler") == complete
    (complete / "data.pt").write_bytes(b"changed-loader")
    with pytest.raises(RuntimeError, match="changed"):
        checkpoints.latest(tmp_path, "id", "domain_mix", "sampler")


def test_weights_only_and_wrong_arm_are_rejected(tmp_path):
    p = checkpoint(tmp_path, 4)
    with pytest.raises(RuntimeError, match="another experiment"):
        checkpoints.latest(tmp_path, "id", "domain_only")
    (p / "actor/optimizer/shard.pt").unlink()
    with pytest.raises(RuntimeError, match="Missing checkpoint"):
        checkpoints.latest(tmp_path, "id", "domain_mix")


def test_adam_scheduler_rng_resume_matches_uninterrupted(tmp_path):
    torch.manual_seed(123)
    net = torch.nn.Linear(3, 2)
    optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)
    def step(model, opt, sched):
        x = torch.randn(5, 3)
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step(); sched.step()
    step(net, optim, scheduler)
    state = dict(model=net.state_dict(), optimizer=optim.state_dict(), scheduler=scheduler.state_dict(), rng=torch.get_rng_state())
    path = tmp_path / "toy.pt"
    torch.save(state, path)
    step(net, optim, scheduler)
    expected = copy.deepcopy(net.state_dict())
    resumed = torch.nn.Linear(3, 2)
    opt2 = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.LambdaLR(opt2, lambda _: 1.0)
    state = torch.load(path, weights_only=False)
    resumed.load_state_dict(state["model"]); opt2.load_state_dict(state["optimizer"])
    sched2.load_state_dict(state["scheduler"]); torch.set_rng_state(state["rng"])
    step(resumed, opt2, sched2)
    for key in expected:
        torch.testing.assert_close(resumed.state_dict()[key], expected[key], rtol=0, atol=0)


def running(manifest, tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    job = dict(folder=str(folder), job_name="0390_mp_tmix", job_id="12.pbs1")
    state = {"train:domain_mix": dict(phase="train", arm="domain_mix", status="R", job=job, pending=None)}
    return folder, job, state


@pytest.mark.parametrize("code,comment,retry", [(271,"terminated by root@login3",True),
    (265,"terminated by root@login1",True), (265,"application failure",False), (1,"error",False)])
def test_pbs_external_interruptions_only(manifest, tmp_path, code, comment, retry):
    folder, job, state = running(manifest, tmp_path)
    class PBS:
        def observe(self, j): return {"job_state":"F","Exit_status":code,"comment":comment}
    result = control.tick(manifest, state, PBS())["train:domain_mix"]
    assert result["status"] == ("READY" if retry else "BLOCKED")
    assert bool(result["job"]) != retry


def test_timeout_keeps_registered_job_without_resubmission(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    class PBS:
        def observe(self, j): raise Unknown("qstat timeout")
        def submit(self, *a): raise AssertionError("must not submit")
    result = control.tick(manifest, state, PBS())["train:domain_mix"]
    assert result["job"] == job
    assert "timeout" in result["observation_error"]


def test_ambiguous_qsub_reuses_same_pending_directory(manifest, tmp_path, monkeypatch):
    calls = []
    folder = tmp_path / "pending"
    folder.mkdir()
    pending = dict(folder=str(folder), job_name="0390_mp_tmix")
    monkeypatch.setattr(control, "new_attempt", lambda *args: pending)
    class PBS:
        def submit(self, folder, name):
            calls.append(folder)
            raise Unknown("ambiguous qsub")
    state = {"train:domain_mix": dict(phase="train", arm="domain_mix", status="READY", job=None)}
    control.tick(manifest, state, PBS()); control.tick(manifest, state, PBS())
    assert calls == [str(folder), str(folder)]
    assert state["train:domain_mix"]["pending"] == pending


def test_application_failure_does_not_become_infinite_271_retry(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    write(folder / "result.json", {"status":"FAIL", "error":"CUDA out of memory"})
    class PBS:
        def observe(self, j): return {"job_state":"F","Exit_status":271,"comment":"terminated"}
    assert control.tick(manifest, state, PBS())["train:domain_mix"]["status"] == "BLOCKED"


def test_qualification_requires_second_process_and_bound_success(manifest, tmp_path, monkeypatch):
    folder, job, _ = running(manifest, tmp_path)
    state = {"qualification:domain_mix":dict(phase="qualification", arm="domain_mix",status="R",job=job)}
    write(folder / "result.json", dict(status="RESUME_REQUIRED",step=4,identity=manifest["identity"],arm="domain_mix"))
    class PBS:
        def observe(self, job): return {"job_state":"F", "Exit_status":0}
        def submit(self, folder, name): return dict(folder=folder,job_name=name,job_id="13.pbs1")
    pbs = PBS()
    control.tick(manifest,state,pbs)
    assert state["qualification:domain_mix"]["status"] == "READY"
    assert not (tmp_path / "qualification/domain_mix/receipt.json").exists()
    second = tmp_path / "second"
    second.mkdir()
    monkeypatch.setattr(control,"new_attempt",lambda *a:dict(folder=str(second),job_name="0390_mp_qmix"))
    control.tick(manifest,state,pbs)
    write(second / "result.json",dict(status="PASS",step=8,resumed_from=4,identity=manifest["identity"],arm="domain_mix"))
    control.tick(manifest,state,pbs)
    receipt = read(tmp_path / "qualification/domain_mix/receipt.json")
    assert receipt["resumed_from"] == 4 and receipt["pbs_job"]["job_id"] == "13.pbs1"
    assert state["qualification:domain_mix"]["status"] == "DONE"


def test_stopped_training_is_rearmed_after_control_resumes(manifest, tmp_path):
    folder, job, state = running(manifest, tmp_path)
    write(folder / "result.json",dict(status="CONTROL_PAUSED",identity=manifest["identity"]))
    class PBS:
        def observe(self, job): return {"job_state":"F", "Exit_status":0}
    control.tick(manifest,state,PBS())
    assert state["train:domain_mix"]["status"] == "READY"


def test_completed_resume_provenance_survives_lost_result(tmp_path):
    p = checkpoint(tmp_path,8)
    checkpoints.commit(p,"id","domain_mix",8,"sampler",resumed_from=4)
    loaded = checkpoints.latest(tmp_path,"id","domain_mix","sampler")
    assert read(loaded / "mopd_complete.json")["resumed_from"] == 4


def validation(run, step, score, identity="id", arm="domain_mix"):
    from d4_mopd.metrics import SELECTION_KEY
    ref = checkpoints.evaluation_reference(run / "checkpoints", identity, arm, step, "sampler")
    row = dict(status="PASS", split="validation", identity=identity, arm=arm, step=step,
               metrics={SELECTION_KEY:score}, **ref)
    write(run / "validation" / f"step_{step}.json", row)
    return row


def test_best_checkpoint_not_latest_and_never_selected_by_test(tmp_path):
    from d4_mopd.metrics import select_best
    for step, score in [(0,.4),(20,.7),(40,.7),(60,.6)]:
        if step:
            checkpoint(tmp_path / "checkpoints", step)
        validation(tmp_path, step, score)
    write(tmp_path / "test/step_60.json", dict(score=.99))
    best = select_best(tmp_path, "id", "domain_mix")
    assert best["step"] == 20 and best["score"] == .7
    assert checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler").name == "global_step_20"
    (tmp_path / "checkpoints/global_step_20/actor/model/shard.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler")


def test_base_can_win_selection_and_receipt_changes_are_rejected(tmp_path):
    from d4_mopd.metrics import select_best
    validation(tmp_path, 0, .8)
    best = select_best(tmp_path, "id", "domain_mix")
    assert checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler") is None
    validation(tmp_path, 0, .9)
    with pytest.raises(RuntimeError, match="receipt changed"):
        checkpoints.restore_selected(best, tmp_path / "checkpoints", "id", "domain_mix", "sampler")


def test_training_success_registers_test_and_test_271_retries_same_selection(manifest, tmp_path):
    from d4_mopd.metrics import select_best
    folder, job, state = running(manifest, tmp_path)
    train = tmp_path / "runs/domain_mix"
    validation(train, 0, .8, identity=manifest["identity"])
    best = select_best(train, manifest["identity"], "domain_mix")
    write(folder / "result.json", dict(status="PASS", step=200, identity=manifest["identity"], selection=best))
    class PBS:
        code = 0
        submissions = []
        def observe(self, job):
            return dict(job_state="F", Exit_status=self.code, comment="terminated by root@login3")
        def submit(self, folder, name):
            self.submissions.append(folder)
            return dict(folder=folder, job_name=name, job_id="13.pbs1")
    pbs = PBS()
    control.tick(manifest, state, pbs)
    assert state["train:domain_mix"]["status"] == "DONE"
    assert state["test:domain_mix"]["status"] == "READY"
    control.tick(manifest, state, pbs)
    assert state["test:domain_mix"]["status"] == "SUBMITTED", state["test:domain_mix"]
    attempt = Path(state["test:domain_mix"]["job"]["folder"])
    assert read(attempt / "request.json")["selection"] == best
    pbs.code = 271
    control.tick(manifest, state, pbs)
    assert state["test:domain_mix"]["status"] == "READY"
    control.tick(manifest, state, pbs)
    second = Path(state["test:domain_mix"]["job"]["folder"])
    assert second != attempt and read(second / "request.json")["selection"] == best
    receipt = tmp_path / "final_tests/domain_mix/test/step_0.json"
    write(receipt, dict(status="PASS", split="test", identity=manifest["identity"], arm="domain_mix", step=0,
                        checkpoint_identity=best["checkpoint_identity"]))
    write(second / "result.json", dict(status="PASS", identity=manifest["identity"], step=0, selection=best,
                                       test_receipt=str(receipt)))
    pbs.code = 0
    control.tick(manifest, state, pbs)
    assert state["test:domain_mix"]["status"] == "DONE"
    assert read(tmp_path / "final_tests/domain_mix/receipt.json")["selection"] == best
    count = len(pbs.submissions)
    control.tick(manifest, state, pbs)
    assert len(pbs.submissions) == count


def test_full_validation_cache_survives_lost_best_pointer(manifest, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.metrics import SELECTION_KEY
    from d4_mopd.config import compose
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path)
    trainer.global_steps = 0
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(identity="sampler"))
    calls = []
    def validate(_):
        calls.append(1)
        return {SELECTION_KEY:.6}
    monkeypatch.setattr(RayPPOTrainer, "_validate", validate)
    trainer._validate()
    (tmp_path / "best_validation.json").unlink()
    assert trainer._validate()[SELECTION_KEY] == .6
    assert len(calls) == 1 and read(tmp_path / "best_validation.json")["step"] == 0


def test_test_evaluation_does_not_write_validation_selection(manifest, tmp_path, monkeypatch):
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.config import compose
    from d4_mopd.metrics import SELECTION_KEY
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path,
                             test_checkpoint=dict(step=20,checkpoint="chosen",checkpoint_identity="bound"))
    trainer.global_steps = 20
    monkeypatch.setattr(RayPPOTrainer, "_validate", lambda _: {SELECTION_KEY:.99})
    trainer._validate()
    assert read(tmp_path / "test/step_20.json")["split"] == "test"
    assert not (tmp_path / "best_validation.json").exists()
    assert not (tmp_path / "validation").exists()


@pytest.mark.parametrize("evaluation", [False, True])
def test_native_load_adapter_uses_selected_step_and_skips_off_grid_validation(manifest, tmp_path, monkeypatch, evaluation):
    import random
    import numpy as np
    from types import SimpleNamespace
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.config import compose
    from d4_mopd.metrics import select_best
    run = tmp_path / "runs/domain_mix"
    for step in (20,56):
        folder = checkpoint(run / "checkpoints", step, identity=manifest["identity"])
        torch.save(dict(random=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state()), folder / "driver.pt")
        checkpoints.commit(folder, manifest["identity"], "domain_mix", step, "sampler")
    validation(run, 20, .8, identity=manifest["identity"])
    selected = select_best(run, manifest["identity"], "domain_mix")
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path / "test" if evaluation else run,
                             test_checkpoint=selected if evaluation else None)
    trainer.global_steps = 0
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(identity="sampler"))
    def native_load(self):
        assert self.config.trainer.resume_mode == "resume_path"
        self.global_steps = int(self.config.trainer.resume_from_path.rsplit("_",1)[-1])
    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", native_load)
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    trainer._load_checkpoint()
    assert trainer.global_steps == (20 if evaluation else 56)
    assert trainer.config.trainer.val_before_train == evaluation


def pipeline(manifest):
    from d4_mopd import ARMS
    root = Path(manifest["manifest_path"]).parent
    write(root / "pipeline.json", dict(schema="d4-mopd-pipeline-v1", identity=manifest["identity"], arms=list(ARMS)))


def qualification_receipt(manifest, target_arm, **overrides):
    proof = dict(status="PASS", identity=manifest["identity"], scope="qualification", arm=target_arm, step=8, resumed_from=4)
    proof.update(overrides)
    write(control.task_root(manifest, "qualification", target_arm) / "receipt.json", proof)


@pytest.mark.parametrize("finish_order", [("domain_mix", "domain_only"), ("domain_only", "domain_mix")])
def test_parallel_qualifications_gate_both_fresh_training_arms(manifest, tmp_path, finish_order):
    pipeline(manifest)
    class PBS:
        submissions = []
        def submit(self, folder, name):
            self.submissions.append((folder, name))
            return dict(folder=folder, job_name=name, job_id=f"{len(self.submissions)}.pbs1")
        def observe(self, job):
            path = Path(job["folder"]) / "result.json"
            return dict(job_state="F", Exit_status=0) if path.exists() else dict(job_state="R")
    state, pbs = {}, PBS()
    control.tick(manifest, state, pbs)
    first_jobs = {arm: copy.deepcopy(state["qualification:" + arm]["job"]) for arm in finish_order}
    assert [n for _, n in pbs.submissions] == ["0390_mp_qmix", "0390_mp_qonly"]
    for arm in finish_order:
        key = "qualification:" + arm
        job = state[key]["job"]
        write(Path(job["folder"]) / "result.json", dict(status="RESUME_REQUIRED", step=4, identity=manifest["identity"], arm=arm))
        control.tick(manifest, state, pbs)
        assert state[key]["status"] == "READY"
        assert not any(k.startswith("train:") for k in state)
        control.tick(manifest, state, pbs)
        second = state[key]["job"]
        assert second != job
        write(Path(second["folder"]) / "result.json", dict(status="PASS", step=8, resumed_from=4, identity=manifest["identity"], arm=arm))
        control.tick(manifest, state, pbs)
        assert state[key]["status"] == "DONE"
        if arm == finish_order[0]:
            peer = state["qualification:" + finish_order[1]]
            assert peer["status"] == "R" and peer["job"] == first_jobs[finish_order[1]]
            assert not any(k.startswith("train:") for k in state)
    assert len(pbs.submissions) == 4
    assert state["train:domain_mix"]["status"] == state["train:domain_only"]["status"] == "READY"
    control.tick(manifest, state, pbs)
    resumed_names = ["0390_mp_qmix" if a == "domain_mix" else "0390_mp_qonly" for a in finish_order]
    assert [n for _, n in pbs.submissions] == ["0390_mp_qmix", "0390_mp_qonly"] + resumed_names + ["0390_mp_tmix", "0390_mp_tonly"]
    # Formal attempts use separate empty checkpoint roots, never the qualification weights.
    for arm in ("domain_mix", "domain_only"):
        request = read(Path(state["train:"+arm]["job"]["folder"]) / "request.json")
        assert request["run_dir"] == str(tmp_path / "runs" / arm)
        assert not (Path(request["run_dir"]) / "checkpoints").exists()
    control.tick(manifest, state, pbs)
    assert len(pbs.submissions) == 6
    # Finish both formal runs and exercise the automatic selected-model tests.
    from d4_mopd.metrics import select_best
    for arm in ("domain_mix", "domain_only"):
        train = tmp_path / "runs" / arm
        validation(train, 0, .8, identity=manifest["identity"], arm=arm)
        best = select_best(train, manifest["identity"], arm)
        folder = Path(state["train:"+arm]["job"]["folder"])
        write(folder / "result.json", dict(status="PASS", step=200, identity=manifest["identity"], arm=arm, selection=best))
    control.tick(manifest, state, pbs)
    control.tick(manifest, state, pbs)
    assert len(pbs.submissions) == 8
    for arm in ("domain_mix", "domain_only"):
        folder = Path(state["test:"+arm]["job"]["folder"])
        best = read(folder / "request.json")["selection"]
        receipt = tmp_path / "final_tests" / arm / "test/step_0.json"
        write(receipt, dict(status="PASS", split="test", identity=manifest["identity"], arm=arm, step=0,
                            checkpoint_identity=best["checkpoint_identity"]))
        write(folder / "result.json", dict(status="PASS", step=0, identity=manifest["identity"], arm=arm,
                                            selection=best, test_receipt=str(receipt)))
    control.tick(manifest, state, pbs)
    control.tick(manifest, state, pbs)
    assert all(s["status"] == "DONE" for s in state.values()) and len(pbs.submissions) == 8


def test_pipeline_restart_repairs_gap_and_blocked_qualification_never_promotes(manifest):
    pipeline(manifest)
    state = {"qualification:domain_mix": dict(phase="qualification", arm="domain_mix", status="BLOCKED")}
    control.advance_pipeline(manifest, state)
    assert len(state) == 2 and state["qualification:domain_only"]["status"] == "READY"
    state["qualification:domain_mix"]["status"] = "DONE"
    qualification_receipt(manifest, "domain_mix")
    control.advance_pipeline(manifest, state)
    assert state["qualification:domain_only"]["status"] == "READY"
    qualification_receipt(manifest, "domain_only")
    state["qualification:domain_only"]["status"] = "DONE"
    control.advance_pipeline(manifest, state)
    state["train:domain_mix"].update(status="R", job={"job_id":"keep-this-job"})
    control.advance_pipeline(manifest, state)
    assert state["train:domain_mix"]["job"]["job_id"] == "keep-this-job"


@pytest.mark.parametrize("bad", [{"identity":"other"}, {"resumed_from":0}, {"step":4}, {"arm":"domain_only"}])
def test_pipeline_requires_bound_full_qualification_receipt(manifest, bad):
    pipeline(manifest)
    state = {"qualification:domain_mix": dict(phase="qualification", arm="domain_mix", status="DONE")}
    qualification_receipt(manifest, "domain_mix", **bad)
    with pytest.raises(RuntimeError, match="qualification/reload"):
        control.advance_pipeline(manifest, state)
    assert set(state) == {"qualification:domain_mix", "qualification:domain_only"}


def test_pipeline_stop_prevents_submissions_and_automatic_transitions(manifest, tmp_path):
    pipeline(manifest)
    write(tmp_path / "STOP", {})
    class PBS:
        def submit(self, *args): raise AssertionError("stopped")
    state = {}
    control.tick(manifest, state, PBS())
    assert state == {}


def test_wrong_arm_job_success_is_not_a_qualification_pass(manifest, tmp_path):
    folder, job, _ = running(manifest, tmp_path)
    state = {"qualification:domain_mix": dict(phase="qualification", arm="domain_mix", status="R", job=job)}
    write(folder / "result.json", dict(status="PASS", identity=manifest["identity"], arm="domain_only", step=8, resumed_from=4))
    class PBS:
        def observe(self, job): return dict(job_state="F", Exit_status=0)
    control.tick(manifest, state, PBS())
    assert state["qualification:domain_mix"]["status"] == "BLOCKED"
    assert not (tmp_path / "qualification/domain_mix/receipt.json").exists()


def test_pipeline_start_persists_intent_is_idempotent_and_respects_stop(manifest, tmp_path, monkeypatch):
    monkeypatch.setattr(control, "verify", lambda _: manifest)
    monkeypatch.setattr(control, "active", lambda _: True)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], native_abi={"status":"PASS"}))
    for _ in range(2):
        result = control.start(manifest["manifest_path"], "pipeline", ("domain_mix", "domain_only"))
        assert result["automatic_pipeline"]
    assert list(read(tmp_path / "states.json")) == ["qualification:domain_mix", "qualification:domain_only"]
    assert read(tmp_path / "pipeline.json")["arms"] == ["domain_mix", "domain_only"]
    with pytest.raises(RuntimeError, match="another identity/arms"):
        control.start(manifest["manifest_path"], "pipeline", ("domain_mix",))
    write(tmp_path / "STOP", {})
    with pytest.raises(RuntimeError, match="stopped"):
        control.start(manifest["manifest_path"], "pipeline", ("domain_mix", "domain_only"))


def legacy_advance_pipeline_v1(m, state):
    # Frozen 63d59e2 dispatcher. tick/guardian are unchanged by the parallel
    # registration fix; exercise an existing guardian without replacing runtime.
    from d4_mopd import ARMS
    path = Path(m["manifest_path"]).parent / "pipeline.json"
    if not path.exists():
        return
    p = read(path)
    control.need(p["schema"] == "d4-mopd-pipeline-v1" and p["identity"] == m["identity"]
                 and p["arms"] == [a for a in ARMS if a in p["arms"]] and p["arms"],
                 "Pipeline identity or arm order changed")
    for arm in p["arms"]:
        s = state.setdefault("qualification:" + arm,
                             dict(phase="qualification", arm=arm, status="READY", job=None, pending=None))
        control.need(s["phase"] == "qualification" and s["arm"] == arm, "Pipeline qualification task changed")
        if s["status"] != "DONE":
            return
        control.qualification_passed(m, arm)
    for arm in p["arms"]:
        state.setdefault("train:" + arm, dict(phase="train", arm=arm, status="READY", job=None, pending=None))


@pytest.mark.parametrize("pending", [False, True])
@pytest.mark.parametrize("registration", ["pipeline", "qualification"])
def test_existing_frozen_guardian_adds_only_without_restarting_mix(manifest, tmp_path, monkeypatch, pending, registration):
    pipeline(manifest)
    write(tmp_path / "manifest.json", manifest)
    write(tmp_path / "cpu_preflight.json", dict(status="PASS", identity=manifest["identity"], native_abi={"status":"PASS"}))
    frozen = tmp_path / "runtime/d4_mopd/control.py"
    frozen.parent.mkdir(parents=True)
    frozen.write_text("frozen runtime must remain unchanged\n")
    protected = [tmp_path / n for n in ("manifest.json", "pipeline.json", "cpu_preflight.json")] + [frozen]
    original = {p: p.read_bytes() for p in protected}
    attempt = control.new_attempt(manifest, "qualification", "domain_mix")
    job = dict(attempt, job_id="2893794.pbs1")
    Path(attempt["folder"], "job_id.txt").write_text(job["job_id"])
    mix = dict(phase="qualification", arm="domain_mix", status="READY" if pending else "Q",
               job=None if pending else job, pending=attempt if pending else None)
    write(tmp_path / "states.json", {"qualification:domain_mix": mix})
    monkeypatch.setattr(control, "verify", lambda _: manifest)
    monkeypatch.setattr(control, "active", lambda _: True)
    def no_restart(*args, **kwargs):
        raise AssertionError("must reuse the existing guardian")
    monkeypatch.setattr(control.subprocess, "Popen", no_restart)
    for _ in range(2):
        result = control.start(manifest["manifest_path"], registration, ("domain_mix", "domain_only"))
        assert result["automatic_pipeline"]
        assert read(tmp_path / "states.json")["qualification:domain_mix"] == mix
    class PBS:
        submissions = []
        def submit(self, folder, name):
            # An accepted-but-pending qsub is recovered by its original path.
            identity = Path(folder) / "job_id.txt"
            if identity.exists():
                return dict(folder=folder, job_name=name, job_id=identity.read_text())
            self.submissions.append(name)
            return dict(folder=folder, job_name=name, job_id=f"new-{len(self.submissions)}.pbs1")
        def observe(self, job):
            return dict(job_state="Q")
    # The updated checkout registers tasks; the already-running guardian keeps
    # its frozen dispatcher and consumes the same shared state table.
    monkeypatch.setattr(control, "advance_pipeline", legacy_advance_pipeline_v1)
    state, pbs = read(tmp_path / "states.json"), PBS()
    control.tick(manifest, state, pbs)
    assert state["qualification:domain_mix"]["job"] == job
    only_job = copy.deepcopy(state["qualification:domain_only"]["job"])
    control.tick(manifest, state, pbs)
    assert state["qualification:domain_only"]["job"] == only_job
    assert pbs.submissions == ["0390_mp_qonly"]
    # Only may finish first under the old guardian; formal training still waits.
    qualification_receipt(manifest, "domain_only")
    state["qualification:domain_only"]["status"] = "DONE"
    control.tick(manifest, state, pbs)
    assert not any(k.startswith("train:") for k in state)
    qualification_receipt(manifest, "domain_mix")
    state["qualification:domain_mix"]["status"] = "DONE"
    control.tick(manifest, state, pbs)
    assert pbs.submissions == ["0390_mp_qonly", "0390_mp_tmix", "0390_mp_tonly"]
    assert all(p.read_bytes() == value for p, value in original.items())


@pytest.mark.parametrize("outcome", ["unknown", "271", "265", "application_fail"])
def test_parallel_peer_continues_during_interruption_or_observation_error(manifest, tmp_path, outcome):
    pipeline(manifest)
    class PBS:
        submissions = []
        interrupted = False
        def submit(self, folder, name):
            self.submissions.append(name)
            return dict(folder=folder, job_name=name, job_id=f"{len(self.submissions)}.pbs1")
        def observe(self, job):
            if job["job_id"] == "1.pbs1":
                if outcome == "unknown":
                    raise Unknown("qstat timeout")
                if not self.interrupted:
                    self.interrupted = True
                    return dict(job_state="F", Exit_status=271 if outcome == "application_fail" else int(outcome),
                                comment="terminated by root@login3")
            return dict(job_state="R")
    state, pbs = {}, PBS()
    control.tick(manifest, state, pbs)
    mix, only = state["qualification:domain_mix"], state["qualification:domain_only"]
    old_mix, old_only = copy.deepcopy(mix["job"]), copy.deepcopy(only["job"])
    if outcome == "application_fail":
        write(Path(old_mix["folder"]) / "result.json", dict(status="FAIL", error="explicit failure"))
    control.tick(manifest, state, pbs)
    assert only["status"] == "R" and only["job"] == old_only
    if outcome in ("271", "265"):
        assert mix["status"] == "READY" and mix["job"] is None
        assert mix["external_interruptions"] == 1
    else:
        assert mix["job"] == old_mix
        assert mix["status"] == ("BLOCKED" if outcome == "application_fail" else "SUBMITTED")
    control.tick(manifest, state, pbs)
    assert pbs.submissions == ["0390_mp_qmix", "0390_mp_qonly"] + (["0390_mp_qmix"] if outcome in ("271", "265") else [])
    assert only["job"] == old_only
    assert not any(k.startswith("train:") for k in state)


def test_retention_keeps_two_latest_and_best_with_complete_history(tmp_path):
    from d4_mopd.metrics import select_best
    root = tmp_path / "checkpoints"
    validation(tmp_path, 0, .4)
    for step in range(5, 36, 5):
        checkpoint(root, step)
        if step % 10 == 0:
            validation(tmp_path, step, .8 if step == 10 else .6)
    # Simulate interruption before the best pointer was published.
    assert not (tmp_path / "best_validation.json").exists()
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert result["retained_steps"] == [10, 30, 35]
    assert result["deleted_steps"] == [5, 15, 20, 25]
    assert len(list((tmp_path / "validation").glob("*.json"))) == 4
    assert checkpoints.latest(root, "id", "domain_mix", "sampler").name == "global_step_35"
    best = select_best(tmp_path, "id", "domain_mix")
    assert checkpoints.restore_selected(best, root, "id", "domain_mix", "sampler").name == "global_step_10"
    checkpoint(root, 40)
    validation(tmp_path, 40, .9)
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert result["retained_steps"] == [35, 40]
    assert 10 in result["deleted_steps"]


def test_retention_base_best_and_protected_final_test_selection(tmp_path):
    from d4_mopd.metrics import select_best
    root = tmp_path / "checkpoints"
    for step in (5, 10, 15, 20):
        checkpoint(root, step)
    validation(tmp_path, 10, .8)
    selected = select_best(tmp_path, "id", "domain_mix")
    write(tmp_path / "receipt.json", dict(status="PASS", identity="id", arm="domain_mix", selection=selected))
    # A frozen final-test request must remain loadable even if a later receipt
    # changes the current best; this is defensive and not the normal train order.
    validation(tmp_path, 0, .9)
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 20)
    assert result["retained_steps"] == [10, 15, 20] and result["best_step"] == 0
    assert checkpoints.restore_selected(selected, root, "id", "domain_mix", "sampler").name == "global_step_10"


def test_retention_interrupted_deletion_cannot_corrupt_resume_and_is_finished_on_restart(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    for step in (5, 10, 15):
        checkpoint(root, step)
    validation(tmp_path, 0, .8)
    original = checkpoints.shutil.rmtree
    def interrupted(path):
        (path / "actor/model/shard.pt").unlink()
        raise InterruptedError("simulated external kill during deletion")
    monkeypatch.setattr(checkpoints.shutil, "rmtree", interrupted)
    with pytest.raises(InterruptedError):
        checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert not (root / "global_step_5").exists()
    assert checkpoints.latest(root, "id", "domain_mix", "sampler").name == "global_step_15"
    monkeypatch.setattr(checkpoints.shutil, "rmtree", original)
    result = checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert result["retained_steps"] == [10, 15] and result["deleted_steps"] == [5]
    assert not list((root / ".retention_trash").iterdir())


def test_native_restart_retries_interrupted_first_validation(manifest, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from d4_mopd.integration import MOPDTrainer, RayPPOTrainer
    from d4_mopd.config import compose
    trainer = object.__new__(MOPDTrainer)
    trainer.config = compose(manifest, "domain_mix", tmp_path)
    trainer.global_steps = 0
    trainer.train_dataloader = SimpleNamespace(sampler=SimpleNamespace(identity="sampler"))
    (tmp_path / "validation").mkdir()
    monkeypatch.setenv("D4_MOPD_ATTEMPT", "synthetic")
    monkeypatch.setattr(RayPPOTrainer, "_load_checkpoint", lambda _: None)
    trainer._load_checkpoint()
    assert trainer.config.trainer.resume_mode == "disable" and trainer.config.trainer.val_before_train
    assert not (tmp_path / "retention.json").exists()


@pytest.mark.parametrize("corruption", ["wrong_arm", "latest_state", "missing_best", "symlink"])
def test_retention_refuses_deletion_when_identity_or_protected_state_is_invalid(tmp_path, corruption):
    root = tmp_path / "checkpoints"
    for step in (5, 10, 15, 20):
        checkpoint(root, step)
    validation(tmp_path, 5, .8)
    if corruption == "wrong_arm":
        path = root / "global_step_10/mopd_complete.json"
        write(path, dict(read(path), arm="domain_only"))
    elif corruption == "latest_state":
        (root / "global_step_20/data.pt").write_bytes(b"corrupted")
    elif corruption == "missing_best":
        (root / "global_step_5/mopd_complete.json").unlink()
    else:
        (root / "global_step_10").rename(tmp_path / "outside")
        (root / "global_step_10").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises((RuntimeError, FileNotFoundError)):
        checkpoints.prune(tmp_path, "id", "domain_mix", "sampler", 2, 200)
    assert all((root / f"global_step_{s}/data.pt").exists() for s in (5,10,15,20))


@pytest.mark.parametrize('failure', [None, 'patch', 'package_change', 'native_change', 'active_job', 'new_campaign'])
def test_megatron_import_repair_is_quiescent_and_preserves_frozen_experiment(
        manifest, tmp_path, monkeypatch, failure):
    import importlib.metadata
    from types import SimpleNamespace
    from d4_mopd import megatron_repair, preflight
    from d4_mopd.repair_pipeline import lock_path
    states, abi = blocked_qualifications(manifest, tmp_path, monkeypatch)
    old_proof = read(tmp_path / 'cpu_preflight.json')
    if failure == 'new_campaign':
        (tmp_path / 'states.json').unlink()
        (tmp_path / 'cpu_preflight.json').unlink()
    versions = {'torch': '2.13.0', 'megatron-core': '0.19.2'}
    monkeypatch.setattr(importlib.metadata, 'distributions', lambda: [
        SimpleNamespace(metadata={'Name': n}, version=v) for n, v in versions.items()])
    calls = []
    abi_state = {'changed': False}
    monkeypatch.setattr(preflight, 'native_abi', lambda: dict(abi,
        files=[dict(sha256='new')] if abi_state['changed'] else abi['files']))
    if failure == 'active_job':
        class PBS:
            def observe(self, job): return dict(job_state='R')
        monkeypatch.setattr(control, 'PBS', PBS)

    def apply(audit):
        calls.append('patch')
        assert control.active(lock_path(manifest))
        assert control.active(tmp_path / 'guardian.lock')
        assert read(tmp_path / 'cpu_preflight.json')['status'] == 'INVALIDATED'
        if failure == 'patch':
            raise RuntimeError('Unrecognized source fingerprint')
        if failure == 'package_change':
            versions['megatron-core'] = '0.20.0'
        if failure == 'native_change':
            abi_state['changed'] = True
        return dict(status='PASS', changed=True)

    monkeypatch.setattr(megatron_repair, 'apply', apply)
    monkeypatch.setattr(control, 'start', lambda *a: pytest.fail('Repair must not submit jobs'))
    monkeypatch.setattr(control, 'repair_te', lambda *a, **k: pytest.fail('Must preserve built TE'))
    protected = tmp_path / 'runtime'
    protected.mkdir()
    (protected / 'original.py').write_text('frozen runtime')
    if failure not in (None, 'new_campaign'):
        with pytest.raises(RuntimeError):
            control.repair_megatron_import(manifest['manifest_path'])
    else:
        proof = control.repair_megatron_import(manifest['manifest_path'])
        assert proof['status'] == 'PASS' and proof['kind'] == 'megatron-import'
    if failure == 'new_campaign':
        assert not (tmp_path / 'states.json').exists()
    else:
        assert read(tmp_path / 'states.json') == states
    assert (protected / 'original.py').read_text() == 'frozen runtime'
    assert not control.active(lock_path(manifest))
    if failure == 'active_job':
        assert calls == [] and read(tmp_path / 'cpu_preflight.json') == old_proof
    else:
        assert calls == ['patch']
        assert read(tmp_path / 'cpu_preflight.json')['status'] == 'INVALIDATED'
        audit = next((tmp_path / 'environment_repairs').iterdir())
        assert read(audit / 'result.json')['status'] == ('PASS' if failure in (None, 'new_campaign') else 'FAIL')


@pytest.mark.parametrize("outcome", ["recover", "stop", "active_job", "identity_mismatch", "progress",
                                      "stop_after_query", "progress_after_query"])
def test_repair_pipeline_waits_for_fresh_pbs_evidence_without_replaying_mutations(
        manifest, tmp_path, monkeypatch, outcome):
    import getpass
    import importlib.metadata
    import json
    import os
    import subprocess
    from types import SimpleNamespace
    from d4_mopd import megatron_repair, preflight, repair_pipeline as repair
    from d4_training_continuation import scheduler

    states, _ = blocked_qualifications(manifest, tmp_path, monkeypatch)
    old_proof = read(tmp_path / "cpu_preflight.json")
    write(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(repair, "verify", lambda _: manifest)
    monkeypatch.setattr(repair, "lock_path", lambda _: tmp_path / "environment.lock")
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [])
    monkeypatch.setattr(control, "PBS", scheduler.PBS)
    monkeypatch.setattr(control, "repair_te", lambda *a, **k: pytest.fail("TE must not be rebuilt"))
    jobs = {s["job"]["job_id"]: s["job"] for s in states.values()}
    events, queries, delays = [], [], []
    counts = {"before_patch": 0, "after_patch": 0}
    frozen = tmp_path / "runtime" / "original.py"
    frozen.parent.mkdir()
    frozen.write_text("frozen runtime")

    def command(args, **kwargs):
        # Exercise the real PBS timeout, JSON parsing and job-identity checks.
        assert args[:4] == ["qstat", "-xf", "-F", "json"]
        assert kwargs["timeout"] == 20
        stage = "after_patch" if "patch" in events else "before_patch"
        counts[stage] += 1
        n, jid = counts[stage], args[4]
        queries.append((stage, jid))
        if (stage, n) == ("before_patch", 1):
            raise subprocess.TimeoutExpired(args, 20)
        if (stage, n) == ("before_patch", 3):
            return SimpleNamespace(returncode=153, stdout="", stderr="qstat: Unknown Job Id " + jid)
        if (stage, n) == ("after_patch", 1):
            return SimpleNamespace(returncode=0, stdout=json.dumps({"pbs_server": "pbs1", "Jobs": {}}), stderr="")
        job = jobs[jid]
        row = dict(job_state="F", Exit_status=1, Job_Owner=getpass.getuser() + "@login2",
                   queue=scheduler.QUEUE, Job_Name=job["job_name"],
                   Submit_arguments=str(Path(job["folder"]) / "stage.pbs"))
        if stage == "before_patch" and n == 4:
            if outcome == "active_job":
                row["job_state"] = "R"
            elif outcome == "identity_mismatch":
                row["Job_Owner"] = "different-user@login2"
        if stage == "before_patch" and n == 5:
            if outcome == "stop_after_query":
                (tmp_path / "STOP").touch()
            elif outcome == "progress_after_query":
                (tmp_path / "qualification/domain_mix/checkpoints/global_step_4").mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout=json.dumps({"Jobs": {jid: row}}), stderr="")

    monkeypatch.setattr(scheduler.subprocess, "run", command)
    real_wait = repair.wait_for_pbs_retry

    def wait(root, delay):
        current = repair.status(tmp_path / "manifest.json")
        assert current["status"] == "WAITING_PBS"
        assert current["environment_lock_held"] and not current["heartbeat_stale"]
        assert current["pbs_wait"]["retry_after_seconds"] == delay
        assert read(tmp_path / "states.json") == states
        assert not list((tmp_path / "qualification_retries").glob("*.json"))
        if current["phase"] == "repair_megatron_import":
            assert events == []
        else:
            assert current["phase"] == "retry_qualification" and events == ["patch", "preflight"]
        assert read(tmp_path / "cpu_preflight.json") == old_proof
        delays.append(delay)
        if outcome == "stop":
            (tmp_path / "STOP").touch()
        elif outcome == "progress":
            (tmp_path / "qualification/domain_mix/checkpoints/global_step_4").mkdir(parents=True)
        real_wait(root, 0)

    monkeypatch.setattr(repair, "wait_for_pbs_retry", wait)

    def patch(audit):
        assert events == [] and counts["before_patch"] == 5
        assert read(tmp_path / "cpu_preflight.json")["status"] == "INVALIDATED"
        events.append("patch")
        return dict(status="PASS")

    def cpu(path):
        assert events == ["patch"]
        events.append("preflight")
        write(tmp_path / "cpu_preflight.json", old_proof)
        return old_proof

    def register(path, phase, arms):
        assert events == ["patch", "preflight"] and counts["after_patch"] == 3
        assert phase == "pipeline" and arms == list(control.ARMS)
        assert {s["status"] for s in read(tmp_path / "states.json").values()} == {"READY"}
        events.append("register")
        return dict(status="REGISTERED")

    monkeypatch.setattr(megatron_repair, "apply", patch)
    monkeypatch.setattr(preflight, "cpu", cpu)
    monkeypatch.setattr(control, "start", register)
    folder = tmp_path / "repair_pipelines" / "observations"
    folder.mkdir(parents=True)
    code = repair.snapshot(folder)
    write(folder / "request.json", dict(manifest=manifest["manifest_path"], identity=manifest["identity"],
                                       repair_kind="megatron-import", code_files=code))
    write(tmp_path / "repair_pipeline.json", dict(folder=str(folder), identity=manifest["identity"]))
    with repair.environment_lease(manifest) as fd:
        if outcome == "recover":
            repair.worker(tmp_path / "manifest.json", folder, os.dup(fd))
        else:
            with pytest.raises(RuntimeError):
                repair.worker(tmp_path / "manifest.json", folder, os.dup(fd))

    final = repair.status(tmp_path / "manifest.json")
    assert not final["environment_lock_held"]
    assert frozen.read_text() == "frozen runtime"
    assert final["pbs_observation_retries"] == len(delays)
    if outcome == "recover":
        assert final["status"] == "COMPLETE" and final["pbs_wait"] is None
        assert events == ["patch", "preflight", "register"] and delays == [30, 60, 60]
        assert queries == [("before_patch", j) for j in ("10.pbs1", "10.pbs1", "11.pbs1", "10.pbs1", "11.pbs1")] + [
            ("after_patch", j) for j in ("10.pbs1", "10.pbs1", "11.pbs1")]
        assert len(list((tmp_path / "environment_repairs").iterdir())) == 1
        assert len(list((tmp_path / "qualification_retries").glob("*.json"))) == 1
    else:
        assert final["status"] == "FAIL" and events == []
        assert read(tmp_path / "states.json") == states
        assert read(tmp_path / "cpu_preflight.json") == old_proof
        assert not (tmp_path / "environment_repairs").exists()
        assert not list((tmp_path / "qualification_retries").glob("*.json"))
