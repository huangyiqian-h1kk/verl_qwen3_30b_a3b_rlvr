import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
VERL = Path(os.environ.get("D4_MOPD_VERL_SRC", ROOT.parent / "upstream_verl")).resolve()
if not (VERL / "verl/trainer/distillation/losses.py").is_file():
    raise RuntimeError("Set D4_MOPD_VERL_SRC to the pinned upstream checkout; native tests must not be skipped")
sys.path.insert(0, str(VERL))

import pytest
from d4_mopd import DOMAINS
from d4_mopd.spec import load_spec


@pytest.fixture
def spec():
    return load_spec()


@pytest.fixture
def rows():
    return [{"domain": d, "sample_id": f"{d}:{i}"} for d in DOMAINS for i in range(31)]


@pytest.fixture
def manifest(spec, tmp_path):
    return dict(spec=spec, verl_src=str(VERL), max_prompt_tokens=4096,
                python=sys.executable,
                data={s: [str(tmp_path / (s+".parquet"))] for s in ("train", "validation", "test")},
                student={"path": str(tmp_path / "base")},
                teachers={d: {"path": str(tmp_path / d)} for d in DOMAINS},
                manifest_path=str(tmp_path / "manifest.json"), identity="synthetic-test-identity")


@pytest.fixture
def cpu_modules(tmp_path, monkeypatch):
    """Simulate the site's module command; replay still executes real native code."""
    import subprocess
    from types import SimpleNamespace
    library = Path(subprocess.check_output(["g++", "-print-file-name=libstdc++.so.6"], text=True).strip()).resolve()
    bins = tmp_path / "GCC module with spaces" / "bin"
    bins.mkdir(parents=True)
    compiler = bins / "g++"
    compiler.write_text('''#!/bin/sh
case "$1" in
  -dumpfullversion) printf '%s\\n' "$MOPD_TEST_GCC_VERSION" ;;
  -print-file-name=libstdc++.so.6) printf '%s\\n' "$MOPD_TEST_CXX_LIBRARY" ;;
  *) exit 1 ;;
esac
''')
    compiler.chmod(0o755)
    init = tmp_path / "modules.sh"
    init.write_text('''module() {
  [[ "$*" == 'load gcc/13.2.0' && "$MOPD_TEST_MODULE_AVAILABLE" == yes ]] || return 1
  export PATH="$MOPD_TEST_GCC_BIN:$PATH"
}
''')
    for key, value in dict(BASH_ENV=init, MOPD_TEST_GCC_BIN=bins, MOPD_TEST_GCC_VERSION="13.2.0",
                           MOPD_TEST_CXX_LIBRARY=library, MOPD_TEST_MODULE_AVAILABLE="yes").items():
        monkeypatch.setenv(key, str(value))
    return SimpleNamespace(library=library, compiler=compiler)
