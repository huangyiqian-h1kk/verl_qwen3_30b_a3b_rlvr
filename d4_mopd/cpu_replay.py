"""Fresh-process native CPU replay; no stubs or optional-backend bypasses."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


def compiler_runtime():
    expected = Path(os.environ["D4_MOPD_CPU_CXX_LIBRARY"]).resolve(strict=True)
    # Load by SONAME, as a native dependency would, rather than force-loading the
    # expected absolute path and masking an already-loaded incompatible library.
    library = ctypes.CDLL("libstdc++.so.6", mode=ctypes.RTLD_GLOBAL)
    loaded = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and Path(fields[5]).name.startswith("libstdc++.so."):
            loaded.add(Path(fields[5]).resolve())
    if loaded != {expected}:
        raise RuntimeError(f"CPU replay loaded unexpected libstdc++: {sorted(map(str, loaded))}; expected {expected}")
    return library, dict(module="gcc/13.2.0", compiler=os.environ["D4_MOPD_CPU_CXX_COMPILER"],
                         version=os.environ["D4_MOPD_CPU_CXX_VERSION"], library=str(expected),
                         sha256=hashlib.sha256(expected.read_bytes()).hexdigest())


def main():
    runtime, verl = (Path(v).resolve() for v in sys.argv[1:])
    library, evidence = compiler_runtime()  # Retain the handle throughout replay.
    print("CPU_REPLAY_RUNTIME_JSON=" + json.dumps(evidence), flush=True)
    sys.path[:0] = [str(runtime), str(verl)]
    import d4_mopd.integration as integration
    import d4_mopd.batch_contract as contract
    for module in (integration, contract):
        if Path(module.__file__).resolve().parent.parent != runtime:
            raise RuntimeError("CPU replay imported an adapter outside the selected frozen runtime")
    result = contract.probe()
    # Also verify that the native imports did not introduce another C++ runtime.
    _, after = compiler_runtime()
    if after != evidence:
        raise RuntimeError("CPU replay compiler runtime changed during native imports")
    result["compiler_runtime"] = evidence
    print("BATCH_CONTRACT_JSON=" + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
