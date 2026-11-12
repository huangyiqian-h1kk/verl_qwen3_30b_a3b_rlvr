"""Exercise loader version errors in real isolated processes, without a GPU."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest

from d4_mopd import preflight, repair_pipeline
from d4_teachers.common import files


@pytest.fixture
def native_case(tmp_path, manifest, cpu_modules, monkeypatch):
    # Tiny synthetic ELF dependencies reproduce the real SONAME/version lookup
    # failure. They are test payloads, never used with the production trainer.
    old, new = (tmp_path / name for name in ("old-system-library", "module library"))
    source = tmp_path / "library.c"
    source.write_text("int fixture_value(void) { return 7; }\n")
    for folder, version in ((old, "3.4.31"), (new, "3.4.32")):
        folder.mkdir()
        script = folder / "symbols.map"
        script.write_text(f"GLIBCXX_{version} {{ global: fixture_value; local: *; }};\n")
        subprocess.run(["gcc", "-shared", "-fPIC", str(source), "-Wl,-soname,libstdc++.so.6",
                        "-Wl,--version-script=" + str(script), "-o", str(folder / "libstdc++.so.6")], check=True)
    consumer = tmp_path / "consumer.c"
    consumer.write_text("extern int fixture_value(void); int call_value(void) { return fixture_value(); }\n")
    extension = tmp_path / "extension.so"
    subprocess.run(["gcc", "-shared", "-fPIC", str(consumer), "-L" + str(new),
                    "-l:libstdc++.so.6", "-o", str(extension)], check=True)
    runtime = tmp_path / "selected runtime"
    package = runtime / "d4_mopd"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    marker = tmp_path / "native_import_started"
    (package / "integration.py").write_text("from pathlib import Path\nPath(" + repr(str(marker)) + ").touch()\n")
    (package / "batch_contract.py").write_text(
        "import ctypes\ndef probe():\n    lib = ctypes.CDLL(" + repr(str(extension)) + ")\n"
        "    assert lib.call_value() == 7\n    return {'status': 'PASS', 'gpu_execution_tested': False}\n")
    monkeypatch.setenv("MOPD_TEST_CXX_LIBRARY", str(new / "libstdc++.so.6"))
    monkeypatch.setenv("LD_LIBRARY_PATH", str(old))
    return manifest, runtime, old, new, extension, marker


def test_fresh_child_loads_module_library_before_native_import(native_case):
    m, runtime, old, new, extension, marker = native_case
    # The previous launch reproduces exactly the missing version category.
    failed = subprocess.run([sys.executable, "-I", "-c", "import ctypes,sys; ctypes.CDLL(sys.argv[1])", str(extension)],
                            capture_output=True, text=True, timeout=15)
    assert failed.returncode != 0 and "GLIBCXX_3.4.32" in failed.stderr and "not found" in failed.stderr
    before = files(runtime)
    proof = preflight.batch_metadata_probe(m, runtime)
    assert proof["status"] == "PASS" and proof["gpu_execution_tested"] is False
    assert marker.exists() and proof["compiler_runtime"]["library"] == str(new / "libstdc++.so.6")
    assert proof["compiler_runtime"]["sha256"] == hashlib.sha256((new / "libstdc++.so.6").read_bytes()).hexdigest()
    assert os.environ["LD_LIBRARY_PATH"] == str(old)  # Only the replay child is changed.
    assert files(runtime) == before


@pytest.mark.parametrize("failure", ["missing-module", "wrong-compiler", "missing-library", "preloaded-old-library"])
def test_bad_module_or_shadowed_library_cannot_pass_replay(native_case, monkeypatch, failure):
    m, runtime, old, new, extension, marker = native_case
    if failure == "missing-module": monkeypatch.setenv("MOPD_TEST_MODULE_AVAILABLE", "no")
    if failure == "wrong-compiler": monkeypatch.setenv("MOPD_TEST_GCC_VERSION", "11.2.0")
    if failure == "missing-library": monkeypatch.setenv("MOPD_TEST_CXX_LIBRARY", "libstdc++.so.6")
    if failure == "preloaded-old-library": monkeypatch.setenv("LD_PRELOAD", str(old / "libstdc++.so.6"))
    with pytest.raises(RuntimeError, match="failed CPU replay"):
        preflight.batch_metadata_probe(m, runtime)
    assert not marker.exists()


def test_native_application_error_remains_a_failure(native_case):
    m, runtime, old, new, extension, marker = native_case
    (runtime / "d4_mopd/batch_contract.py").write_text("def probe():\n    raise KeyError('broken_sample_id')\n")
    with pytest.raises(RuntimeError, match="broken_sample_id"):
        preflight.batch_metadata_probe(m, runtime)
    assert marker.exists()


def test_detached_repair_snapshot_includes_replay_bootstrap(tmp_path):
    inventory = repair_pipeline.snapshot(tmp_path)
    for name in ("cpu_replay.sh", "cpu_replay.py"):
        key = "d4_mopd/" + name
        assert inventory[key] == hashlib.sha256((Path(preflight.__file__).parent / name).read_bytes()).hexdigest()
