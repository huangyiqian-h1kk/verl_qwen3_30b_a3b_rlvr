"""Per-node CUDA JIT prerequisites; imports no GPU framework and uses no network."""
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import traceback


def write(path, result):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(path)


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def command(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, timeout=120, **kwargs)
    need(result.returncode == 0, f"{args[0]} failed ({result.returncode}): " + (result.stdout + result.stderr)[-6000:])
    return result.stdout.strip()


def probe(folder, rank):
    nvcc, ninja, cxx = (shutil.which(name) for name in ("nvcc", "ninja", "g++"))
    need(nvcc and ninja and cxx, "CUDA JIT needs nvcc, venv ninja and g++ on PATH")
    cuda = Path(os.environ.get("CUDA_HOME", "/nonexistent"))
    need((cuda / "include/cuda_runtime.h").is_file(), "CUDA_HOME has no cuda_runtime.h")
    need(Path(nvcc).resolve() == (cuda / "bin/nvcc").resolve()
         == Path(os.environ.get("CUDACXX", "/nonexistent")).resolve(), "CUDA compiler paths disagree")
    need(os.environ.get("CUDA_PATH") == str(cuda), "CUDA_PATH must match CUDA_HOME")
    need(Path(ninja).absolute() == Path(sys.executable).absolute().parent / "ninja",
         "ninja must come from the manifest venv/bin")
    need(not any("stubs" in Path(p).parts for p in os.environ.get("LD_LIBRARY_PATH", "").split(":")),
         "CUDA stub libraries must not be on runtime LD_LIBRARY_PATH")
    nvcc_version = command([nvcc, "--version"])
    release = re.search(r"\brelease\s+(\d+\.\d+),", nvcc_version)
    need(release and release[1] == "13.0", "Runtime nvcc must match the pinned torch CUDA 13.0 build")
    torch_version = md.version("torch")
    need(torch_version == "2.13.0+cu130", "Unexpected torch build for the CUDA runtime repair")
    ninja_version = command([ninja, "--version"])
    ninja_release = re.match(r"\d+\.\d+\.\d+", ninja_version)
    need(ninja_release and ninja_release[0] == md.version("ninja"),
         "ninja executable and installed package differ")
    cxx_version = command([cxx, "-dumpfullversion"])
    need(cxx_version == "13.2.0", "Expected the pinned GCC 13.2.0 host compiler")
    # Exercise the compiler through ninja before any expensive model loading.
    # This is a compile-only sm90 probe, not an end-to-end vLLM/GPU success claim.
    local = Path(os.environ["PBS_LOCALDIR"]) / ("cuda_probe_" + str(rank))
    local.mkdir(parents=True, exist_ok=True)
    (local / "probe.cu").write_text("#include <cuda_runtime.h>\n__global__ void probe(float *x) { x[threadIdx.x] += 1; }\n")
    build = shlex.join([nvcc, "-ccbin", cxx, "-arch=sm_90", "-c", "probe.cu", "-o", "probe.o"])
    (local / "build.ninja").write_text("rule cuda\n  command = " + build.replace("$", "$$")
                                       + "\nbuild probe.o: cuda probe.cu\n")
    output = command([ninja, "-v", "-j", "1"], cwd=local)
    need((local / "probe.o").is_file() and (local / "probe.o").stat().st_size > 0,
         "CUDA compiler did not produce the probe object")
    return dict(cuda_home=str(cuda), nvcc=nvcc, nvcc_version=nvcc_version, ninja=ninja,
                ninja_version=ninja_version, cxx=cxx, cxx_version=cxx_version, torch=torch_version,
                compile_probe=dict(status="PASS", directory=str(local), output=output),
                gpu_execution_tested=False)


def main():
    action, folder, rank = sys.argv[1:]
    folder = Path(folder)
    path = folder / f"cuda_runtime_{rank}.json"
    request = json.loads((folder / "request.json").read_text())
    result = dict(status="RUNNING", stage="cuda-runtime", rank=int(rank),
                  host=socket.gethostname(), identity=request["identity"])
    if action == "begin":
        write(path, result)
        return
    if action == "fail":
        previous = json.loads(path.read_text()) if path.exists() else {}
        if previous.get("status") != "FAIL":
            write(path, dict(result, status="FAIL", error="CUDA module/tool setup failed; inspect pbs.log and cuda_modules log"))
        return
    need(action == "probe", "Unknown CUDA runtime action")
    try:
        result.update(probe(folder, int(rank)), status="PASS")
    except Exception as exc:
        write(path, dict(result, status="FAIL", error=str(exc), traceback=traceback.format_exc()))
        raise
    write(path, result)
    print(f"[0390] CUDA JIT prerequisites PASS rank={rank}: nvcc 13.0, venv ninja, sm90 compile", flush=True)


if __name__ == "__main__":
    main()
