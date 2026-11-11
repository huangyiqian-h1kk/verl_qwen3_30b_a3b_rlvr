"""Reproduce the optional-import crash without importing any GPU packages."""
import builtins
import importlib.metadata as md
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from d4_mopd import megatron_repair as repair
from d4_teachers.common import read


def guard_runner(monkeypatch, fa4_version, broken=True):
    original_import = builtins.__import__
    calls = []
    kernel = object()

    def version(name):
        assert name == "flash-attn-4"
        calls.append("metadata")
        if fa4_version is None:
            raise md.PackageNotFoundError(name)
        return fa4_version

    def importer(name, *args, **kwargs):
        if name == "flash_attn.cute":
            calls.append("cute")
            if broken:
                raise AttributeError("module 'cutlass.cute.core' has no attribute 'ThrMma'")
            return SimpleNamespace(flash_attn_varlen_func=kernel)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(md, "version", version)
    monkeypatch.setattr(builtins, "__import__", importer)

    def run(block):
        namespace = {}
        exec(compile(block, "upstream_fa4_guard", "exec", dont_inherit=True), namespace)
        return namespace

    return run, calls, kernel


def test_original_guard_crashes_before_checking_absent_fa4(monkeypatch):
    run, calls, _ = guard_runner(monkeypatch, None)
    with pytest.raises(AttributeError, match="ThrMma"):
        run(repair.BEFORE)
    assert calls == ["cute"]


@pytest.mark.parametrize("version", [None, "4.0.0b19"])
def test_fixed_guard_skips_cute_when_fa4_absent_or_too_old(monkeypatch, version):
    run, calls, _ = guard_runner(monkeypatch, version)
    result = run(repair.AFTER)
    assert calls == ["metadata"]
    assert result["HAVE_FA4"] is False and result["flash_attn4_varlen_func"] is None


def test_fixed_guard_preserves_available_fa4_and_does_not_hide_broken_install(monkeypatch):
    with monkeypatch.context() as patch:
        run, calls, kernel = guard_runner(patch, "4.0.0b20", broken=False)
        result = run(repair.AFTER)
        assert calls == ["metadata", "cute"]
        assert result["HAVE_FA4"] is True and result["flash_attn4_varlen_func"] is kernel
    run, calls, _ = guard_runner(monkeypatch, "4.0.0b20")
    with pytest.raises(AttributeError, match="ThrMma"):
        run(repair.AFTER)
    assert calls == ["metadata", "cute"]


@pytest.fixture
def installation(tmp_path, monkeypatch):
    source = ("# fixture for pinned source\n" + repair.BEFORE +
              "\ntry:\n    from flash_mla import unused\nexcept ImportError:\n    pass\n").encode()
    updated = source.decode().replace(repair.BEFORE, repair.AFTER).encode()
    monkeypatch.setattr(repair, "ORIGINAL_SHA256", repair.sha(source))
    monkeypatch.setattr(repair, "PATCHED_SHA256", repair.sha(updated))
    target = tmp_path / "attention.py"
    target.write_bytes(source)
    target.chmod(0o640)
    monkeypatch.setattr(repair, "installed_path", lambda: target)
    versions = {"megatron-core": "0.19.2", "flash-attn": "2.8.3", "nvidia-cutlass-dsl": "4.8.0"}

    def version(name):
        if name not in versions:
            raise md.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(md, "version", version)
    return target, source, updated, versions


def test_apply_is_atomic_audited_idempotent_and_preflight_rejects_old_source(installation, tmp_path):
    target, source, updated, _ = installation
    with pytest.raises(RuntimeError, match="Missing metadata-first"):
        repair.inspect()
    native = tmp_path / "libtransformer_engine.so"
    native.write_bytes(b"unchanged native binary")
    before_modules = set(sys.modules)
    proof = repair.apply(tmp_path / "first")
    assert proof["status"] == "PASS" and proof["changed"] is True
    assert target.read_bytes() == updated and target.stat().st_mode & 0o777 == 0o640
    assert (tmp_path / "first/attention.before.py").read_bytes() == source
    assert read(tmp_path / "first/patch_plan.json")["after_sha256"] == repair.sha(updated)
    assert repair.inspect()["status"] == "PASS"
    assert not repair.apply(tmp_path / "second")["changed"]
    assert target.read_bytes() == updated and native.read_bytes() == b"unchanged native binary"
    assert not any(m.startswith(("megatron", "flash_attn", "cutlass")) for m in set(sys.modules) - before_modules)


@pytest.mark.parametrize("problem", ["source_changed", "fa4_installed", "wrong_version"])
def test_targeted_repair_refuses_unknown_environment_without_overwriting(installation, tmp_path, problem):
    target, _, _, versions = installation
    if problem == "source_changed":
        target.write_bytes(target.read_bytes() + b"# unrelated local change\n")
    elif problem == "fa4_installed":
        versions["flash-attn-4"] = "4.0.0b20"
    else:
        versions["megatron-core"] = "0.20.0"
    original = target.read_bytes()
    with pytest.raises(RuntimeError):
        repair.apply(tmp_path / "audit")
    assert target.read_bytes() == original


def test_package_lookup_does_not_allow_patching_outside_manifest_venv(monkeypatch, tmp_path):
    relative = "megatron/core/transformer/attention.py"
    fake = SimpleNamespace(version="0.19.2", files=[relative], locate_file=lambda _: tmp_path / "attention.py")
    monkeypatch.setattr(md, "distribution", lambda _: fake)
    with pytest.raises(RuntimeError, match="outside"):
        repair.installed_path()
