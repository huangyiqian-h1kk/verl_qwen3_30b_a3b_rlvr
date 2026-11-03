#!/usr/bin/env python3
"""Read-only campaign report; standard library only, also runnable via git show | python -I -."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess


TZ = timezone(timedelta(hours=8))
DOMAINS = ("math", "if", "conversational_pivot", "logic_algorithmic")
EXTRA = ("math_step50", "cp_step80")
DEEP = ("deepmath_base", "deepmath_math20", "deepmath_math50")
DEEP_TEST = ("deepmath_if70", "deepmath_cp80")
FOLLOWUP = ("if_step70", "math_continued")


def short(value, limit=400):
    return str(value if value is not None else "—").replace("\n", " ")[:limit]


def when(value):
    try:
        return (
            datetime.fromisoformat(value).astimezone(TZ).isoformat(timespec="seconds")
        )
    except (ValueError, TypeError):
        return short(value)


def number(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else "—"


def score(value):
    return f"{number(value.get('mean'))} (n={value.get('n', '?')})" if value else "—"


def table(headers, rows):
    print(" | ".join(headers))
    for row in rows:
        print(" | ".join(short(x, 250) for x in row))


class Reader:
    def __init__(self):
        self.errors = []

    def read(self, path):
        path = Path(path)
        try:
            if path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError("summary JSON exceeds 8 MiB; skipped")
            value = json.loads(path.read_text())
            if not isinstance(value, dict):
                raise ValueError("expected JSON object")
            return value
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            self.errors.append(f"{path}: {exc}")
            return {}

    def receipt(self, path):
        value = self.read(path)
        if not value:
            return value, "未生成/不可读"
        payload = {k: v for k, v in value.items() if k != "content_sha256"}
        checksum = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest()
        if value.get("content_sha256") != checksum:
            self.errors.append(f"{path}: receipt checksum missing/mismatch")
            return {}, "收据校验失败"
        return value, value.get("status", "未知")


def lock_held(path):
    try:
        with path.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            return False
    except FileNotFoundError:
        return False
    except OSError as exc:
        return "未知: " + str(exc)


def newest(paths, limit):
    ordered = []
    for path in paths:
        try:
            ordered.append((path.stat().st_mtime_ns, str(path), path))
        except FileNotFoundError:
            pass
    return [p for _, _, p in sorted(ordered, reverse=True)[:limit]]


def tail(path, lines):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 12000))
            return (
                "\n".join(stream.read().decode(errors="replace").splitlines()[-lines:])
                or "（空）"
            )
    except OSError as exc:
        return "不可读: " + str(exc)


def observe(job):
    jid = job.get("job_id", "")
    if not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9._-]+)?", jid):
        return {}, "无合法 job ID"
    try:
        proc = subprocess.run(
            ["qstat", "-xf", "-F", "json", jid],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=20,
            check=False,
        )
        if proc.returncode:
            raise RuntimeError(
                f"qstat exit={proc.returncode}: {proc.stderr or proc.stdout}"
            )
        row = json.loads(proc.stdout).get("Jobs", {}).get(jid)
        if not row:
            raise RuntimeError("PBS 未返回该 job；不能据此判断已结束")
        if row.get("Job_Name") != job.get("job_name"):
            raise RuntimeError("PBS job 名称与记录不一致")
        return row, None
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {}, short(exc, 500)


def report(root, logs=0, history=5, no_pbs=False):
    if not (root / "campaign.json").is_file():
        raise SystemExit("Campaign 不存在: " + str(root))
    reader = Reader()
    c = reader.read(root / "campaign.json")
    if not c:
        raise SystemExit("无法读取 campaign.json")
    policy = c.get("policy", {})
    cp_plan = reader.read(root / "checkpoint_evaluation.json")
    dm_plan = reader.read(root / "deepmath_evaluation.json")
    dm_test_plan = reader.read(root / "deepmath_if70_cp80_test.json")
    fu_plan = reader.read(root / "teacher_followup.json")
    registered = (
        set(DOMAINS)
        | {"base"}
        | set(cp_plan.get("targets", {}))
        | set(dm_plan.get("targets", {}))
        | set(fu_plan.get("targets", {}))
        | set(dm_test_plan.get("targets", {}))
    )
    discovered = {p.parent.name for p in (root / "runs").glob("*/state.json")}
    tasks = [*DOMAINS, "base", *EXTRA, *DEEP]
    if fu_plan:
        tasks += list(FOLLOWUP)
    if dm_test_plan:
        tasks += list(DEEP_TEST)
    tasks += sorted(discovered - set(tasks))
    states = {t: reader.read(root / "runs" / t / "state.json") for t in tasks}
    beat = reader.read(root / "heartbeat.json")
    try:
        age = (
            datetime.now(timezone.utc) - datetime.fromisoformat(beat["at_utc"])
        ).total_seconds()
        stale = age > max(120, policy.get("poll_seconds", 30) * 4)
    except (KeyError, ValueError, TypeError):
        age, stale = None, True
    print("检查时间（UTC+8）:", datetime.now(TZ).isoformat(timespec="seconds"))
    print("Campaign:", root)
    print(
        "报告格式: teacher-inspection-v1 | Campaign identity:", c.get("identity", "—")
    )
    print(
        "监护锁持有:",
        lock_held(root / "guardian.lock"),
        "| 心跳过期:",
        stale,
        "| 心跳年龄(s):",
        number(age),
        "| host/PID:",
        beat.get("host"),
        beat.get("pid"),
    )
    print(
        "最近心跳:",
        when(beat.get("at_utc")),
        "| Campaign STOP:",
        (root / "STOP.json").exists(),
    )
    print("监护策略:", beat.get("retry_policy"), beat.get("observation_policy"))
    print(
        "以下是持续变化的文件/PBS快照；PASS 为已写出收据的状态，未重读模型或大批量生成文件。"
    )

    print("\n[teacher 训练；math_continued 的 best 分数为两个来源 validation 等权均值]")
    training = []
    training_tasks = (*DOMAINS, *(("math_continued",) if fu_plan else ()))
    for task in training_tasks:
        run = root / "runs" / task
        state, progress = states[task], reader.read(run / "progress.json")
        best, done = (
            reader.read(run / "best.json"),
            reader.read(run / "training_complete.json"),
        )
        export = reader.read(run / "export.json")
        training.append(
            [
                task,
                state.get("status"),
                state.get("phase"),
                progress.get("step"),
                progress.get("latest_full_step"),
                best.get("step"),
                number(best.get("score")),
                done.get("status", "未完成"),
                done.get("early_stop"),
                export.get("status", "未生成"),
                when(progress.get("at_utc")),
            ]
        )
    table(
        [
            "teacher",
            "控制器状态",
            "阶段",
            "最近记录步",
            "完整保存步(记录)",
            "best步",
            "best本域val",
            "训练完成",
            "early_stop",
            "HF导出",
            "训练进度时间",
        ],
        training,
    )
    print(
        "最近记录步包含中断前未保存的工作；恢复点应看完整保存步及新 job 的 resume 记录。"
    )

    print("\n[四域 test：Math 列仅为 Nemotron；DeepMath 另列]")
    rows, receipts = [], {}
    for task in (*DOMAINS, "base", *EXTRA, *(FOLLOWUP if fu_plan else ())):
        receipt, status = reader.receipt(root / "runs" / task / "test.json")
        receipts[task] = receipt
        scores = dict(receipt.get("scores", {}))
        if receipt.get("math_sources"):
            scores["math"] = receipt["math_sources"]["nemotron"]
        rows.append(
            [
                task,
                status,
                receipt.get("step"),
                *[score(scores.get(d)) for d in DOMAINS],
                (receipt.get("binding", {}).get("verifier") or "—")[:12],
            ]
        )
    table(
        [
            "模型",
            "test收据",
            "被评估步",
            "Math/Nemotron",
            "IF",
            "CP",
            "Logic",
            "verifier前12位",
        ],
        rows,
    )
    revisions = {
        r.get("binding", {}).get("verifier")
        for r in receipts.values()
        if r.get("status") == "PASS"
    }
    if len(revisions) > 1:
        print("注意：已完成 test 使用了不同 verifier 版本，比较时需保留此区别。")
    if fu_plan:
        print("\n[采用版本与 Math 续训]")
        print(
            "CP: 用户指定 cp_step80；复用已完成四域 test；原 CP50 validation 选择记录保留。"
        )
        print("IF: 当前采用 IF40，IF70 是补充 test，尚未自动替换。")
        print(
            "Math续训策略:", json.dumps(fu_plan.get("math_policy"), ensure_ascii=False)
        )
        chosen = reader.read(root / "selected_teachers.json")
        for domain, item in chosen.get("teachers", {}).items():
            print(
                domain,
                "采用任务=",
                item.get("source_task"),
                "step=",
                item.get("selected_step"),
                "选择依据=",
                item.get("selection_source", "validation"),
                "待完成替换=",
                item.get("pending_replacement", "—"),
            )
        receipt = receipts.get("math_continued", {})
        print(
            "Math续训最终test:",
            json.dumps(receipt.get("math_sources", {}), ensure_ascii=False),
        )

    print("\n[DeepMath 补充 validation/test]")
    data = reader.read(root / "deepmath_evaluation/data/manifest.json")
    print("数据已准备:", bool(data), "| 三模型计划已注册:", bool(dm_plan))
    print("DeepMath计划 verifier:", dm_plan.get("verifier_identity", "—"))
    if dm_test_plan:
        print(
            "IF70/CP80 test补测已注册 | verifier:",
            dm_test_plan.get("verifier_identity", "—"),
        )
    if data:
        print(
            "数据分层:",
            json.dumps(
                {
                    s: {k: v for k, v in info.items() if k in ("rows", "sources")}
                    for s, info in data.get("splits", {}).items()
                },
                ensure_ascii=False,
            ),
        )
    rows = []
    for task in (*DEEP, *(DEEP_TEST if dm_test_plan else ())):
        receipt, status = reader.receipt(root / "runs" / task / "heldout.json")
        for split in ("test",) if task in DEEP_TEST else ("validation", "test"):
            scores = receipt.get("split_scores", {}).get(split, {})
            rows.append(
                [
                    task,
                    split,
                    status,
                    receipt.get("step"),
                    score(scores.get("math")),
                    score(scores.get("math/source/deepmath_l8")),
                    score(scores.get("math/source/deepmath_l9")),
                ]
            )
    table(["模型", "split", "收据", "步数", "整体", "L8", "L9"], rows)

    print("\n[全部任务的作业、评估进度和近期错误]")
    jobs = {}
    for task in tasks:
        run, state = root / "runs" / task, states[task]
        label = (
            state.get("status", "未注册/无状态")
            if task in registered
            else "未注册（现有目录不等于已投递）"
        )
        print(
            f"\n{task}: {label} / {state.get('phase', '—')} | STOP={(run / 'STOP.json').exists()}"
            f" | 无新增可恢复进度的中断计数={state.get('no_progress_retries', 0)}"
            f" | 状态时间={when(state.get('observed_utc'))}"
        )
        for key in ("waiting_for", "error", "previous_error"):
            if state.get(key):
                print(f"  {key}:", short(state[key], 650))
        candidates = newest((run / "evaluations").glob("*/progress.json"), 1)
        if candidates:
            p = candidates[0]
            value = reader.read(p)
            print(
                "  最新持久化评估（可能来自前次作业）:",
                p.parent.name,
                f"| rows={value.get('completed_rows')}/{value.get('expected_rows')}"
                f" | batches={value.get('completed_batches')} | 本进程复用={value.get('cached_batches_this_process')}"
                f" | {when(value.get('at_utc'))} | receipt存在={(p.parent / 'receipt.json').is_file()}",
            )
        kind = (
            "当前job"
            if state.get("job")
            else "待确认投递"
            if state.get("pending")
            else "上次job"
        )
        job = (
            state.get("job") or state.get("pending") or state.get("previous_job") or {}
        )
        if not job:
            print("  无作业记录")
            continue
        folder = Path(job["folder"]) if job.get("folder") else None
        print(" ", kind, job.get("job_id", "尚无ID"), "|", folder)
        if job.get("job_id"):
            jobs[task] = job
        if folder:
            result = reader.read(folder / "result.json")
            resume = reader.read(folder / "train/resume.json") or reader.read(
                folder / "train/resume_source.json"
            )
            print(
                "  result:",
                result.get("status", "未生成"),
                "| resume step:",
                resume.get("step", "—"),
                "| error:",
                short(result.get("error"), 650),
            )
            if logs:
                print("  pbs.log 末尾:", folder / "pbs.log")
                print(tail(folder / "pbs.log", logs))
    print(
        "\n计数仅是诊断信息，不能据此断言重试被停止；旧 error 字段也可能保留在 DONE 任务中。"
    )

    if not no_pbs:
        print(
            "\n[PBS 实时查询：每个任务的当前/上次 job，单次超时20秒，最多3个并行查询]",
            flush=True,
        )
        items = list(jobs.items())
        with ThreadPoolExecutor(max_workers=3) as pool:
            observations = pool.map(observe, [job for _, job in items])
            for (task, job), (row, error) in zip(items, observations):
                if error:
                    print(task, job["job_id"], "| PBS UNKNOWN:", error, flush=True)
                    continue
                print(
                    task,
                    job["job_id"],
                    "| state=",
                    row.get("job_state"),
                    "| exit=",
                    row.get("Exit_status"),
                    "| used=",
                    row.get("resources_used", {}).get("walltime"),
                    "| limit=",
                    row.get("Resource_List", {}).get("walltime"),
                    "|",
                    short(row.get("comment"), 500),
                    flush=True,
                )

    print("\n[本域 validation 历史及最近训练记录]")
    for task in training_tasks:
        run = root / "runs" / task
        print("\n", task)
        history_rows = []
        for path in (run / "evaluations").glob("validation_*/receipt.json"):
            receipt, status = reader.receipt(path)
            if receipt:
                history_rows.append(
                    [
                        receipt.get("step", -1),
                        status,
                        score(receipt.get("scores", {}).get(task)),
                        number(receipt.get("elapsed_seconds")),
                        when(receipt.get("at_utc")),
                        score(receipt.get("math_sources", {}).get("nemotron")),
                        score(receipt.get("math_sources", {}).get("deepmath")),
                        number(receipt.get("selection_score")),
                    ]
                )
        table(
            [
                "val步",
                "收据",
                "本域reward",
                "本次完成尝试耗时(s)",
                "完成时间",
                "Nemotron",
                "DeepMath",
                "来源等权均值",
            ],
            sorted(history_rows, key=lambda r: r[0]),
        )
        recent = newest((run / "jobs").glob("*/train/step_*.json"), history)
        table(
            [
                "最近写出步",
                "reward",
                "0-adv组比例",
                "平均长度",
                "预算耗尽比例",
                "train+save秒",
                "时间",
                "job目录",
            ],
            [
                [
                    v.get("step"),
                    number(v.get("reward_mean")),
                    number(v.get("zero_advantage_group_fraction")),
                    number(v.get("response_length_mean")),
                    number(v.get("response_at_budget_fraction")),
                    number(v.get("train_and_save_seconds")),
                    when(v.get("at_utc")),
                    p.parent.parent.name,
                ]
                for p in reversed(recent)
                for v in [reader.read(p)]
            ],
        )
    print(
        "\n近期训练行按文件写出时间排序，保留中断重跑的重复步；不将其解释为新增训练步。"
    )
    if reader.errors:
        print("\n[读取/摘要校验问题]")
        for error in reader.errors:
            print(error)
    print("\n只读检查完成：未调用 qsub/qdel，未修改 campaign、监护或任务状态。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--logs",
        type=int,
        default=0,
        help="Tail up to N lines of each latest job log (default: 0)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=5,
        help="Recent training records per teacher (default: 5)",
    )
    parser.add_argument(
        "--no-pbs", action="store_true", help="Only inspect persisted files"
    )
    args = parser.parse_args()
    if not 0 <= args.logs <= 100 or not 0 <= args.history <= 50:
        parser.error("--logs must be 0..100; --history must be 0..50")
    report(args.campaign.resolve(), args.logs, args.history, args.no_pbs)


if __name__ == "__main__":
    main()
