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
