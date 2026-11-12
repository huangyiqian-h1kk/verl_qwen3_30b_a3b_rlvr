#!/usr/bin/env python3
"""Read-only MOPD failure evidence; stdlib only, also usable via git show | python -I -."""
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import re
import sys


ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR = re.compile(
    r"Traceback \(most recent call last\)|\bERROR\b|\bFATAL\b|"
    r"\b\w*(?:Error|Exception)\b|\bSystemExit\b|out of memory|"
    r"undefined symbol|\bSIG(?:TERM|KILL|SEGV|ABRT)\b|\bKilled\b", re.I
)
CAUSE = re.compile(r"\b(?:\w*(?:Error|Exception)|SystemExit):[^\n]*")
CONFIG = re.compile(
    r"^(distillation|actor_rollout_ref|mopd):|^\s*(?:teacher_models|model_path|"
    r"key|tensor_model_parallel_size|max_model_len|max_num_seqs|"
    r"max_num_batched_tokens|gpu_memory_utilization|dtype|enforce_eager|"
    r"engine_kwargs|attention_backend|moe_backend|enable_chunked_prefill|"
    r"enable_prefix_caching|load_format|nnodes|n_gpus_per_node):"
)
MIB = 1024 * 1024


def clip(text, limit):
    data = ANSI.sub("", str(text)).encode("utf-8")
    if len(data) <= limit:
        return data.decode("utf-8")
    half = limit // 2
    return (data[:half].decode("utf-8", errors="replace")
            + f"\n[省略 {len(data) - 2 * half} bytes；保留开头与结尾]\n"
            + data[-half:].decode("utf-8", errors="replace"))


def read_object(path):
    with Path(path).open("rb") as stream:
        data = stream.read(2 * MIB + 1)
    if len(data) > 2 * MIB:
        raise ValueError(f"JSON exceeds 2 MiB: {path}")
    value = json.loads(data)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def optional_object(path):
    try:
        return read_object(path)
    except (OSError, ValueError) as exc:
        return {"inspection_error": str(exc)}


def contained(root, value):
    path = Path(value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Refusing a path outside the campaign: {path}")
    return path


def chunks(path, budget):
    """Bound file reads, preserving both early failures and final shutdown messages."""
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(0)
        if size <= budget:
            return size, [(0, stream.read(size))]
        head = budget * 3 // 4
        prefix = stream.read(head)
        offset = size - (budget - head)
        stream.seek(offset)
        return size, [(0, prefix), (offset, stream.read(budget - head))]


def log_excerpt(path, *, scan_bytes=16 * MIB, tail_lines=60):
    try:
        size, samples = chunks(path, scan_bytes)
    except OSError as exc:
        return f"不可读: {exc}\n"
    report = [f"原文件 bytes={size}; 本次读取 bytes={sum(len(b) for _, b in samples)}"]
    if len(samples) > 1:
        report.append("[扫描未覆盖中段；可提高 --scan-mib，或直接查阅原日志。]")
    for offset, data in samples:
        lines = ANSI.sub("", data.decode("utf-8", errors="replace")).splitlines()
        report.append(f"[日志片段 byte offset={offset}；以下行号相对本片段]")
        first, last, causes = [], deque(maxlen=2), {}
        for index, line in enumerate(lines):
            if ERROR.search(line):
                if len(first) < 4:
                    first.append(index)
                last.append(index)
            cause = CAUSE.search(line)
            if cause and len(causes) < 32:
                causes.setdefault(clip(cause.group(), 1500), index + 1)
        report.append("候选异常行（去重，至多 32 条；需要结合上下文判断根因）:")
        report.extend(f"{index:>7}: {cause}" for cause, index in causes.items())
        selected = set(range(max(0, len(lines) - tail_lines), len(lines)))
        # Unique inner exceptions may lie between repeated optional-import
        # warnings and the final Ray wrapper. Retain their nearby stack frames.
        for index in set(first) | set(last) | {i - 1 for i in causes.values()}:
            selected.update(range(max(0, index - 6), min(len(lines), index + 49)))
        if not first:
            report.append("本片段无匹配错误行；仍保留末尾。")
        previous = -1
        for index in sorted(selected):
            if index != previous + 1:
                report.append(f"[省略行 {previous + 2}–{index}]")
            report.append(f"{index + 1:>7}: {clip(lines[index], 1800)}")
            previous = index
    return clip("\n".join(report), 96 * 1024) + "\n"


def show_json(out, path, limit=32 * 1024):
    print(f"\n文件: {path}", file=out)
    print(clip(json.dumps(optional_object(path), ensure_ascii=False, indent=2), limit), file=out)


def show_attempt(out, root, key, job, state, scan_bytes, *, previous=False):
    label = "上次作业" if previous else "当前登记作业"
    print(f"\n========== {key} | {label} | {job.get('job_id')} ==========", file=out)
    try:
        folder = contained(root, job["folder"])
    except (KeyError, OSError, ValueError) as exc:
        print(f"跳过: {exc}", file=out)
        return
    print(f"目录: {folder}", file=out)
    result = optional_object(folder / "result.json")
    if not previous:
        summary = {k: state.get(k) for k in ("status", "phase", "observed_utc", "pbs",
                   "retry_reason", "external_interruptions", "observation_error")}
        summary["result_status"] = result.get("status")
        print(json.dumps(summary, ensure_ascii=False, indent=2), file=out)
        if str((state.get("pbs") or {}).get("Exit_status")) == "271" and result.get("status") == "FAIL":
            print("判定提示: PBS=271，同时 result.json=FAIL；不能仅凭退出数字覆盖应用失败。"
                  "需结合 termination_evidence、完整 traceback、PBS comment 与日志时间。", file=out)
    show_json(out, folder / "result.json", 64 * 1024)
    for name in ("finished", "job_id.txt"):
        try:
            _, data = chunks(folder / name, 4096)
            print(f"{name}: {clip(data[0][1].decode(errors='replace'), 2048)}", file=out)
        except OSError as exc:
            print(f"{name}: {exc}", file=out)
    if not previous:
        for name in ("request.json", "resume.json", "cuda_runtime_0.json", "cuda_runtime_1.json",
                     "gpu_node_0.json", "gpu_node_1.json"):
            show_json(out, folder / name, 12 * 1024)
        for name in ("ray_start_0.log", "ray_start_1.log"):
            print(f"\n文件: {folder / name}", file=out)
            print(clip(log_excerpt(folder / name, scan_bytes=min(scan_bytes, MIB), tail_lines=20), 6000), file=out)
        cfg = folder / "resolved.yaml"
        print(f"\n配置摘录: {cfg}", file=out)
        try:
            size, samples = chunks(cfg, 2 * MIB)
            selected = [line for _, data in samples for line in data.decode(errors="replace").splitlines()
                        if CONFIG.search(line)]
            print(clip("\n".join(selected), 24 * 1024), file=out)
            if size > 2 * MIB:
                print("[配置超过 2 MiB，未覆盖中段。]", file=out)
        except OSError as exc:
            print(str(exc), file=out)
    print(f"\n错误上下文及末尾: {folder / 'pbs.log'}", file=out)
    excerpt = log_excerpt(folder / "pbs.log", scan_bytes=scan_bytes)
    print(clip(excerpt, 24 * 1024) if previous else excerpt, file=out)


def build_report(manifest_path, *, scan_bytes=16 * MIB):
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    manifest, states = read_object(manifest_path), read_object(root / "states.json")
    out = io.StringIO()
    print("MOPD failure inspection v1", file=out)
    print("检查时间（日本时间）:", datetime.now(timezone(timedelta(hours=9))).isoformat(), file=out)
    print("Campaign:", root, "\nIdentity:", manifest.get("identity"), file=out)
    print("文件与已保存 PBS 观察的快照；本命令不查询 PBS，不证明当前进程存活。", file=out)
    print("仅采集状态、配置和有限日志；不读取模型权重、数据集或 rollout 生成文件。", file=out)
    for name in ("heartbeat.json", "launcher_repair.json", "repair_pipeline.json"):
        show_json(out, root / name, 8000)
    pointer = optional_object(root / "repair_pipeline.json")
    if pointer.get("folder"):
        try:
            repair = contained(root, pointer["folder"])
            show_json(out, repair / "status.json", 16 * 1024)
        except (OSError, ValueError) as exc:
            print(str(exc), file=out)
    pending = []
    summaries = []
    for key, state in states.items():
        if not isinstance(state, dict):
            continue
        job = state.get("job")
        summaries.append(f"{key}: {state.get('status')} | job={(job or {}).get('job_id')}"
                         f" | PBS exit={(state.get('pbs') or {}).get('Exit_status')}")
        if job:
            show_attempt(out, root, key, job, state, scan_bytes)
        else:
            print(f"\n{key}: 没有登记 job；pending={state.get('pending')}", file=out)
        old = state.get("previous_job")
        if old and old.get("folder") != (job or {}).get("folder"):
            pending.append((key, old, state))
    # Current jobs come first so prior failures cannot crowd out current causes.
    for key, job, state in pending:
        show_attempt(out, root, key, job, state, scan_bytes, previous=True)
    print("\n只读采集完成：未调用 qsub/qdel，未修改环境、manifest、监护或作业状态。", file=out)
    return out.getvalue(), summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", help="New report outside the campaign; never overwrite an existing file")
    parser.add_argument("--scan-mib", type=int, default=16, help="Max bytes read per PBS log, in MiB (1–128)")
    args = parser.parse_args()
    if not 1 <= args.scan_mib <= 128:
        parser.error("--scan-mib must be between 1 and 128")
    root = Path(args.manifest).resolve().parent
    target = Path(args.output).resolve() if args.output else None
    if target and target.is_relative_to(root):
        parser.error("--output must be outside the campaign")
    report, summaries = build_report(args.manifest, scan_bytes=args.scan_mib * MIB)
    if target:
        with target.open("x", encoding="utf-8") as out:
            out.write(report)
        print("\n".join(summaries))
        print(f"RESULT_REPORT={target}\nREPORT_BYTES={len(report.encode('utf-8'))}")
    else:
        print(report, end="")


if __name__ == "__main__":
    main()
