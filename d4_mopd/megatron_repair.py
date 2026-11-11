"""Backport upstream's metadata-first optional FA4 import to pinned MCore 0.19.2.

No kernels, package versions, or learner settings are changed. The whole-file
hash binds the backport to the exact source in verl's pinned uv.lock.
"""
import hashlib
import importlib.metadata as md
from pathlib import Path
import stat
import sys

from d4_teachers.common import need, write
from d4_training_continuation.common import atomic_text

SOURCE_REVISION = "4b4acac9a1d28ea6829c8d4f566d75698a21249d"
UPSTREAM_FIX = "https://github.com/NVIDIA/Megatron-LM/blob/5bce4b3f7ca0824600761ed2d53619a7c1e44e62/megatron/core/transformer/attention.py"
ORIGINAL_SHA256 = "a019cd00a708a22237383456a7ba85bf522201c14a5b7cb0c1e4f0f12eac9837"
PATCHED_SHA256 = "a733e2a6b5eb24e99fe4ca55a68f88bddf56ab0e5fd7ffd56a4b231c7262f5c8"
BEFORE = '''_MIN_FA4_VERSION = "4.0.0b20"
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _get_dist_version

    from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_func
    from packaging.version import Version as _Version

    try:
        HAVE_FA4 = _Version(_get_dist_version("flash-attn-4")) >= _Version(_MIN_FA4_VERSION)
    except PackageNotFoundError:
        HAVE_FA4 = False
except ImportError:
    HAVE_FA4 = False
'''
AFTER = '''_MIN_FA4_VERSION = "4.0.0b20"
flash_attn4_varlen_func = None
try:
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _get_dist_version

    from packaging.version import Version as _Version

    HAVE_FA4 = _Version(_get_dist_version("flash-attn-4")) >= _Version(_MIN_FA4_VERSION)
    if HAVE_FA4:
        from flash_attn.cute import flash_attn_varlen_func as flash_attn4_varlen_func
except (ImportError, PackageNotFoundError):
    HAVE_FA4 = False
'''


def sha(data):
    return hashlib.sha256(data).hexdigest()


def installed_path():
    dist = md.distribution("megatron-core")
    need(dist.version == "0.19.2", "FA4 import backport requires megatron-core==0.19.2")
    relative = Path("megatron/core/transformer/attention.py")
    need(any(str(f) == str(relative) for f in dist.files or ()), "MCore source missing from package record")
    path = Path(dist.locate_file(relative)).resolve()
    need(path.is_relative_to(Path(sys.prefix).resolve()), "MCore is outside the manifest virtual environment")
    return path


def package_evidence():
    versions = {name: md.version(name) for name in
                ("megatron-core", "flash-attn", "nvidia-cutlass-dsl")}
    need(versions["megatron-core"] == "0.19.2" and versions["flash-attn"] == "2.8.3",
         "Unexpected MCore/FlashAttention version; refusing this targeted repair")
    try:
        fa4 = md.version("flash-attn-4")
    except md.PackageNotFoundError:
        fa4 = None
    need(fa4 is None, "This recovery expects the pinned FA2-only stack, but flash-attn-4 is installed")
    return dict(versions=versions, flash_attn_4=fa4)


def patched_bytes(source):
    current = sha(source)
    if current == PATCHED_SHA256:
        return source
    need(current == ORIGINAL_SHA256, "MCore attention.py differs from the pinned source; refusing to overwrite")
    text = source.decode("utf-8")
    need(text.count(BEFORE) == 1, "Expected one original FA4 import block")
    updated = text.replace(BEFORE, AFTER, 1).encode("utf-8")
    need(sha(updated) == PATCHED_SHA256, "Unexpected FA4 backport result")
    compile(updated, "megatron/core/transformer/attention.py", "exec", dont_inherit=True)
    return updated


def inspect():
    """CPU-only evidence; never import Torch, Megatron, FlashAttention or CuTe."""
    packages = package_evidence()
    path = installed_path()
    source = path.read_bytes()
    need(sha(source) == PATCHED_SHA256,
         "Missing metadata-first FA4 import backport. Run repair-megatron-import; "
         "for the blocked two-arm pipeline use repair-and-retry --repair-kind megatron-import")
    # Exercise the installed guard itself. FA4 absence is verified above, so
    # only stdlib metadata and packaging.version are imported.
    text = source.decode("utf-8")
    start = text.index('_MIN_FA4_VERSION =')
    end = text.index("\ntry:\n    from flash_mla", start)
    namespace = {}
    exec(compile(text[start:end], str(path), "exec", dont_inherit=True), namespace)
    need(namespace["HAVE_FA4"] is False and namespace["flash_attn4_varlen_func"] is None,
         "Absent FA4 was not disabled by the installed guard")
    return dict(status="PASS", path=str(path), sha256=sha(source), **packages,
                upstream_fix=UPSTREAM_FIX, source_revision=SOURCE_REVISION,
                scope="installed FA4 metadata guard only; no model imports or GPU kernel execution")


def apply(audit):
    packages = package_evidence()
    target = installed_path()
    original = target.read_bytes()
    updated = patched_bytes(original)
    audit = Path(audit)
    audit.mkdir(parents=True, exist_ok=True)
    (audit / "attention.before.py").write_bytes(original)
    (audit / "attention.after.py").write_bytes(updated)
    plan = dict(path=str(target), before_sha256=sha(original), after_sha256=sha(updated),
                source_revision=SOURCE_REVISION, upstream_fix=UPSTREAM_FIX, **packages,
                changed=original != updated)
    write(audit / "patch_plan.json", plan)
    if original != updated:
        mode = stat.S_IMODE(target.stat().st_mode)
        atomic_text(target, updated.decode("utf-8"))
        target.chmod(mode)
    proof = inspect()
    write(audit / "import_guard.json", proof)
    return dict(plan, status="PASS", import_guard=proof)
