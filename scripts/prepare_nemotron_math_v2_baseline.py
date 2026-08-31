#!/usr/bin/env python3
"""Prepare a filtered, reproducible Nemotron-Math-v2 baseline sample.

For repeated sampling, pass ``--download-dir`` to download/reuse one split
parquet, or pass an existing file with ``--local-file``. Local parquet input is
scanned in record batches while excluding the very large ``messages``
trajectory column. Eligible problems are UUID-deduplicated and selected with
reservoir sampling, so the full parquet is never loaded into memory. The
Dataset Viewer rows API remains available when neither local option is used.

The defaults implement the requested problem-level filter exactly:

* read the ``medium`` trajectory split;
* keep rows with ``1 <= metadata.reason_medium_no_tool.pass <= 7``;
* do not restrict ``data_source``.

Nemotron-Math-v2 stores one row per retained correct trajectory, so the API
sampler still applies inverse trajectory-multiplicity acceptance and UUID
deduplication to approximate uniform sampling over problems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import socket
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DEFAULT_DATASET = "nvidia/Nemotron-Math-v2"
DEFAULT_SPLIT = "medium"
DEFAULT_SEED = 20260728
DEFAULT_METADATA_KEY = "reason_medium_no_tool"
DEFAULT_MIN_METADATA_PASS = 1
DEFAULT_MAX_METADATA_PASS = 7
API_ROOT = "https://datasets-server.huggingface.co"
LOCAL_PARQUET_COLUMNS = [
    "uuid",
    "problem",
    "expected_answer",
    "original_expected_answer",
    "changed_answer_to_majority",
    "data_source",
    "metadata",
    "used_in",
    "url",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_json(url: str, retries: int = 12, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "nemotron-math-v2-baseline-preparer/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code == 429 and attempt + 1 < retries:
                retry_after = exc.headers.get("Retry-After", "").strip()
                try:
                    wait_seconds = max(1.0, float(retry_after))
                except ValueError:
                    wait_seconds = min(30.0 * (2**attempt), 300.0)
                print(
                    f"[rate-limit] HTTP 429; waiting {wait_seconds:.0f}s "
                    f"before retry {attempt + 2}/{retries}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            if attempt + 1 < retries and 500 <= exc.code < 600:
                wait_seconds = min(2.0**attempt, 30.0)
                print(
                    f"[server-error] HTTP {exc.code}; waiting "
                    f"{wait_seconds:.0f}s before retry {attempt + 2}/{retries}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait_seconds)
                continue
            break
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"request failed after {retries} attempts: {url}") from last_error


def dataset_splits(dataset: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"dataset": dataset})
    payload = get_json(f"{API_ROOT}/splits?{query}")
    splits = payload.get("splits")
    if not isinstance(splits, list) or not splits:
        raise RuntimeError(f"no splits returned for {dataset}")
    return splits


def dataset_split_sizes(dataset: str) -> list[dict[str, Any]]:
    """Return per-split row counts from the Dataset Viewer `/size` endpoint."""
    query = urllib.parse.urlencode({"dataset": dataset})
    payload = get_json(f"{API_ROOT}/size?{query}")
    size = payload.get("size")
    splits = size.get("splits") if isinstance(size, dict) else None
    if not isinstance(splits, list) or not splits:
        raise RuntimeError(f"no split sizes returned for {dataset}")
    return splits


def resolve_splits(
    dataset: str, config: str, split: str
) -> tuple[str, list[dict[str, Any]]]:
    candidates = dataset_splits(dataset)
    size_lookup = {
        (str(item.get("config")), str(item.get("split"))): int(
            item.get("num_rows") or 0
        )
        for item in dataset_split_sizes(dataset)
    }
    enriched: list[dict[str, Any]] = []
    for item in candidates:
        normalized = dict(item)
        key = (str(item.get("config")), str(item.get("split")))
        normalized["num_rows"] = size_lookup.get(key, 0)
        enriched.append(normalized)
    candidates = enriched

    if config == "auto":
        by_config: dict[str, list[dict[str, Any]]] = {}
        for item in candidates:
            by_config.setdefault(str(item.get("config")), []).append(item)
        if not by_config:
            raise RuntimeError(f"no configurations returned for {dataset}")
        config = max(
            by_config,
            key=lambda name: sum(
                int(item.get("num_rows") or 0) for item in by_config[name]
            ),
        )

    config_candidates = [
        item for item in candidates if str(item.get("config")) == config
    ]
    if split == "all":
        matching = [
            item
            for item in config_candidates
            if str(item.get("split", "")).startswith(
                ("low", "medium", "high")
            )
        ]
    elif split in {"low", "medium", "high"}:
        matching = [
            item
            for item in config_candidates
            if str(item.get("split", "")).startswith(split)
        ]
    else:
        matching = [
            item
            for item in config_candidates
            if item.get("split") == split
        ]
    if not matching:
        available = sorted(
            {
                f"{item.get('config')}/{item.get('split')}"
                for item in candidates
            }
        )
        raise RuntimeError(
            f"cannot find config={config!r}, split={split!r}; "
            f"available={available}"
        )
    for item in matching:
        if int(item.get("num_rows") or 0) <= 0:
            raise RuntimeError(f"invalid num_rows for split: {item}")
    return config, matching


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    offset: int,
    length: int,
) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        }
    )
    payload = get_json(f"{API_ROOT}/rows?{query}")
    result: list[dict[str, Any]] = []
    for wrapped in payload.get("rows", []):
        if isinstance(wrapped, dict) and isinstance(wrapped.get("row"), dict):
            result.append(wrapped["row"])
    return result


def iter_local_rows(
    path: Path, parquet_batch_size: int
) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    if isinstance(value, dict):
                        yield value
        return
    if path.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "local parquet streaming requires pyarrow"
            ) from exc
        parquet_file = pq.ParquetFile(path)
        available = set(parquet_file.schema_arrow.names)
        required = {"problem", "expected_answer", "metadata"}
        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"local parquet is missing required columns: {missing}"
            )
        columns = [
            column for column in LOCAL_PARQUET_COLUMNS if column in available
        ]
        for batch in parquet_file.iter_batches(
            batch_size=parquet_batch_size,
            columns=columns,
            use_threads=True,
        ):
            yield from batch.to_pylist()
        return
    raise ValueError(f"unsupported local file: {path}")


def stable_uid(row: dict[str, Any], problem: str) -> str:
    value = str(row.get("uuid") or "").strip()
    if value:
        return value
    return "sha256:" + hashlib.sha256(problem.encode("utf-8")).hexdigest()


def trajectory_multiplicity(
    row: dict[str, Any], reasoning_regime: str = "all"
) -> int:
    """Return the number of retained correct trajectories for this problem.

    Nemotron-Math-v2 stores one row per retained correct solution. Sampling
    rows directly therefore over-samples problems with more correct solutions.
    The metadata contains pass counts for the six generation configurations.
    With a named reasoning regime, only that regime's tool/no-tool pass counts
    are summed; with ``all``, all six configurations are summed.
    """
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return 1
    total = 0
    prefix = (
        f"reason_{reasoning_regime}_"
        if reasoning_regime in {"low", "medium", "high"}
        else ""
    )
    for key, value in metadata.items():
        if prefix and not str(key).startswith(prefix):
            continue
        if isinstance(value, dict):
            try:
                total += max(0, int(value.get("pass", 0)))
            except (TypeError, ValueError):
                continue
    return max(1, total)


def metadata_pass_values(
    row: dict[str, Any], metadata_key: str
) -> tuple[int, int, float] | None:
    """Return ``(pass, count, accuracy)`` for one metadata regime."""
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        return None
    block = metadata.get(metadata_key)
    if not isinstance(block, dict):
        return None
    try:
        pass_count = int(block["pass"])
        sample_count = int(block["count"])
        accuracy = float(block.get("accuracy", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if pass_count < 0 or sample_count < 0 or pass_count > sample_count:
        return None
    return pass_count, sample_count, accuracy


def apply_metadata_pass_filter(
    row: dict[str, Any],
    metadata_key: str,
    min_pass: int,
    max_pass: int,
) -> tuple[tuple[int, int, float] | None, str]:
    """Apply an inclusive pass-count filter to one metadata regime."""
    values = metadata_pass_values(row, metadata_key)
    if values is None:
        return None, "missing_or_invalid_metadata_filter"
    pass_count, _, _ = values
    if pass_count < min_pass:
        return None, "metadata_pass_below_min"
    if pass_count > max_pass:
        return None, "metadata_pass_above_max"
    return values, "accepted"


def load_excluded_uids(paths: list[Path]) -> set[str]:
    """Load UID columns from prior problem/group parquet or JSONL files."""
    excluded: set[str] = set()
    for path in paths:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path, columns=["uid"])
            values = frame["uid"].tolist()
        elif path.suffix == ".jsonl":
            values = []
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict) or not value.get("uid"):
                        raise ValueError(
                            f"{path}:{line_number} has no non-empty uid"
                        )
                    values.append(value["uid"])
        else:
            raise ValueError(
                f"unsupported exclusion file (use parquet or JSONL): {path}"
            )
        excluded.update(str(value).strip() for value in values if str(value).strip())
    return excluded


def resolve_or_download_local_file(args: argparse.Namespace) -> None:
    """Resolve ``--local-file`` or download one repository parquet once."""
    args.downloaded_repo_file = None
    if args.download_dir is None:
        return
    if args.local_file is not None:
        raise ValueError("--local-file and --download-dir are mutually exclusive")
    repo_file = args.download_repo_file or f"data/{args.split}.parquet"
    args.download_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "--download-dir requires the huggingface_hub package"
        ) from exc
    print(
        f"[download] ensuring {args.dataset}/{repo_file} under "
        f"{args.download_dir}",
        file=sys.stderr,
        flush=True,
    )
    downloaded = hf_hub_download(
        repo_id=args.dataset,
        repo_type="dataset",
        filename=repo_file,
        revision=args.revision,
        local_dir=args.download_dir,
    )
    args.local_file = Path(downloaded)
    args.downloaded_repo_file = repo_file


def normalize_row(
    row: dict[str, Any],
    allow_majority_answer: bool,
    max_problem_chars: int,
) -> tuple[dict[str, Any] | None, str]:
    problem = str(row.get("problem") or "").strip()
    answer = str(row.get("expected_answer") or "").strip()
    if not problem:
        return None, "empty_problem"
    if not answer:
        return None, "empty_answer"
    if len(problem) > max_problem_chars:
        return None, "problem_too_long_chars"
    changed = bool(row.get("changed_answer_to_majority", False))
    if changed and not allow_majority_answer:
        return None, "majority_answer_replaced"

    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    result = {
        "uid": stable_uid(row, problem),
        "problem": problem,
        "expected_answer": answer,
        "original_expected_answer": str(
            row.get("original_expected_answer") or ""
        ),
        "changed_answer_to_majority": changed,
        "data_source": str(row.get("data_source") or "unknown"),
        "metadata_json": json.dumps(
            metadata, ensure_ascii=False, sort_keys=True, default=str
        ),
        "used_in_json": json.dumps(
            row.get("used_in") or [], ensure_ascii=False, default=str
        ),
        "source_url": str(row.get("url") or ""),
        "total_trajectory_multiplicity": trajectory_multiplicity(row),
    }
    return result, "accepted"


def render_prompt(tokenizer: Any, system_prompt: str, problem: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def filter_prompt_length(
    rows: list[dict[str, Any]],
    tokenizer_path: str,
    system_prompt: str,
    max_prompt_tokens: int,
    rejected: Counter[str],
) -> list[dict[str, Any]]:
    if not tokenizer_path:
        return rows
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, trust_remote_code=True, local_files_only=True
    )
    kept: list[dict[str, Any]] = []
    for row in rows:
        rendered = render_prompt(tokenizer, system_prompt, row["problem"])
        token_count = len(tokenizer.encode(rendered, add_special_tokens=False))
        if token_count > max_prompt_tokens:
            rejected["prompt_too_long_tokens"] += 1
            continue
        row["prompt_tokens"] = token_count
        kept.append(row)
    return kept


def collect_from_api(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config, split_specs = resolve_splits(
        args.dataset, args.config, args.split
    )
    rng = random.Random(args.seed)
    accepted: dict[str, dict[str, Any]] = {}
    rejected: Counter[str] = Counter()
    rows_seen = 0
    requests = 0
    inverse_weight_rejections = 0
    reasoning_regime = (
        args.split if args.split in {"low", "medium", "high"} else "all"
    )

    # Collect extra rows before tokenizer filtering so that a few long prompts
    # do not reduce the final requested sample size.
    target_before_token_filter = max(
        args.sample_size + args.sample_reserve,
        args.sample_size,
    )

    pages: list[tuple[str, int, int]] = []
    for spec in split_specs:
        split_name = str(spec["split"])
        num_rows = int(spec["num_rows"])
        for offset in range(0, num_rows, args.page_size):
            pages.append(
                (split_name, offset, min(args.page_size, num_rows - offset))
            )
    rng.shuffle(pages)

    for split_name, offset, length in pages[: args.max_requests]:
        if len(accepted) >= target_before_token_filter:
            break
        if requests:
            time.sleep(args.request_interval)
        page = fetch_rows(
            args.dataset, config, split_name, offset, length
        )
        requests += 1
        rows_seen += len(page)
        for raw in page:
            metadata_values, reason = apply_metadata_pass_filter(
                raw,
                metadata_key=args.metadata_key,
                min_pass=args.min_metadata_pass,
                max_pass=args.max_metadata_pass,
            )
            if metadata_values is None:
                rejected[reason] += 1
                continue
            multiplicity = trajectory_multiplicity(raw, reasoning_regime)
            if rng.random() >= 1.0 / multiplicity:
                inverse_weight_rejections += 1
                continue
            normalized, reason = normalize_row(
                raw,
                allow_majority_answer=args.allow_majority_answer,
                max_problem_chars=args.max_problem_chars,
            )
            if normalized is None:
                rejected[reason] += 1
                continue
            uid = normalized["uid"]
            if uid in args.excluded_uids:
                rejected["excluded_prior_uid"] += 1
                continue
            if uid in accepted:
                rejected["duplicate_problem"] += 1
                continue
            pass_count, sample_count, accuracy = metadata_values
            normalized["metadata_filter_key"] = args.metadata_key
            normalized["metadata_filter_pass"] = pass_count
            normalized["metadata_filter_count"] = sample_count
            normalized["metadata_filter_accuracy"] = accuracy
            normalized["sampling_trajectory_multiplicity"] = multiplicity
            accepted[uid] = normalized
        print(
            f"[sampling] requests={requests} rows_seen={rows_seen} "
            f"unique_accepted={len(accepted)}/{target_before_token_filter}",
            file=sys.stderr,
            flush=True,
        )

    if len(accepted) < args.sample_size:
        raise RuntimeError(
            f"only collected {len(accepted)} unique usable problems after "
            f"{requests} requests; increase --max-requests"
        )
    audit = {
        "backend": "rows_api",
        "resolved_config": config,
        "requested_split": args.split,
        "resolved_splits": {
            str(item["split"]): int(item["num_rows"])
            for item in split_specs
        },
        "total_trajectory_rows": sum(
            int(item["num_rows"]) for item in split_specs
        ),
        "requests": requests,
        "rows_seen": rows_seen,
        "problem_sampling": (
            "random trajectory pages; inverse retained-trajectory "
            "multiplicity acceptance; UUID deduplication"
        ),
        "sampling_reasoning_regime": reasoning_regime,
        "request_interval_seconds": args.request_interval,
        "inverse_multiplicity_rejections": inverse_weight_rejections,
        "metadata_filter": {
            "key": args.metadata_key,
            "min_pass_inclusive": args.min_metadata_pass,
            "max_pass_inclusive": args.max_metadata_pass,
        },
        "excluded_uid_count": len(args.excluded_uids),
        "rejected": dict(rejected),
    }
    return list(accepted.values()), audit


def collect_from_local(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    assert args.local_file is not None
    rng = random.Random(args.seed)
    selected_reservoir: list[dict[str, Any]] = []
    seen_uids: set[str] = set()
    rejected: Counter[str] = Counter()
    rows_seen = 0
    eligible_unique_problems = 0
    target_before_token_filter = max(
        args.sample_size + args.sample_reserve,
        args.sample_size,
    )

    parquet_rows = None
    parquet_row_groups = None
    if args.local_file.suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "local parquet streaming requires pyarrow"
            ) from exc
        parquet_file = pq.ParquetFile(args.local_file)
        parquet_rows = int(parquet_file.metadata.num_rows)
        parquet_row_groups = int(parquet_file.metadata.num_row_groups)

    for raw in iter_local_rows(args.local_file, args.local_batch_size):
        rows_seen += 1
        normalized, reason = normalize_row(
            raw,
            allow_majority_answer=args.allow_majority_answer,
            max_problem_chars=args.max_problem_chars,
        )
        if normalized is None:
            rejected[reason] += 1
            continue
        uid = normalized["uid"]
        if uid in seen_uids:
            rejected["duplicate_problem_trajectory"] += 1
            continue
        seen_uids.add(uid)
        if uid in args.excluded_uids:
            rejected["excluded_prior_uid"] += 1
            continue
        metadata_values, reason = apply_metadata_pass_filter(
            raw,
            metadata_key=args.metadata_key,
            min_pass=args.min_metadata_pass,
            max_pass=args.max_metadata_pass,
        )
        if metadata_values is None:
            rejected[reason] += 1
            continue
        pass_count, sample_count, accuracy = metadata_values
        normalized["metadata_filter_key"] = args.metadata_key
        normalized["metadata_filter_pass"] = pass_count
        normalized["metadata_filter_count"] = sample_count
        normalized["metadata_filter_accuracy"] = accuracy
        normalized["sampling_trajectory_multiplicity"] = (
            trajectory_multiplicity(raw, args.split)
        )

        eligible_unique_problems += 1
        if len(selected_reservoir) < target_before_token_filter:
            selected_reservoir.append(normalized)
        else:
            replacement = rng.randrange(eligible_unique_problems)
            if replacement < target_before_token_filter:
                selected_reservoir[replacement] = normalized

        if rows_seen % args.local_progress_every == 0:
            print(
                f"[local-scan] rows={rows_seen} "
                f"unique={len(seen_uids)} "
                f"eligible={eligible_unique_problems} "
                f"reservoir={len(selected_reservoir)}/"
                f"{target_before_token_filter}",
                file=sys.stderr,
                flush=True,
            )

    if len(selected_reservoir) < args.sample_size:
        raise RuntimeError(
            f"only found {eligible_unique_problems} unique eligible problems "
            f"in {args.local_file}; need at least {args.sample_size}"
        )
    audit = {
        "backend": "local_file_streaming_reservoir",
        "local_file": str(args.local_file),
        "download_dir": (
            str(args.download_dir) if args.download_dir is not None else None
        ),
        "downloaded_repo_file": args.downloaded_repo_file,
        "dataset_revision": args.revision,
        "local_file_size_bytes": args.local_file.stat().st_size,
        "local_parquet_rows": parquet_rows,
        "local_parquet_row_groups": parquet_row_groups,
        "local_batch_size": args.local_batch_size,
        "local_progress_every": args.local_progress_every,
        "rows_seen": rows_seen,
        "unique_problems_seen": len(seen_uids),
        "eligible_unique_problems": eligible_unique_problems,
        "reservoir_size": len(selected_reservoir),
        "problem_sampling": (
            "full sequential local scan; UUID deduplication; uniform "
            "reservoir sampling over eligible unique problems"
        ),
        "metadata_filter": {
            "key": args.metadata_key,
            "min_pass_inclusive": args.min_metadata_pass,
            "max_pass_inclusive": args.max_metadata_pass,
        },
        "excluded_uid_count": len(args.excluded_uids),
        "rejected": dict(rejected),
    }
    return selected_reservoir, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default="auto")
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--revision",
        default="main",
        help="Hugging Face dataset revision used with --download-dir",
    )
    parser.add_argument("--metadata-key", default=DEFAULT_METADATA_KEY)
    parser.add_argument(
        "--min-metadata-pass",
        type=int,
        default=DEFAULT_MIN_METADATA_PASS,
    )
    parser.add_argument(
        "--max-metadata-pass",
        type=int,
        default=DEFAULT_MAX_METADATA_PASS,
    )
    parser.add_argument(
        "--exclude-uids-from",
        type=Path,
        action="append",
        default=[],
        metavar="PARQUET_OR_JSONL",
        help=(
            "exclude UIDs found in a prior problem/group parquet or JSONL; "
            "may be repeated"
        ),
    )
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--sample-reserve", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=200,
        help="Dataset Viewer API only; accepted but ignored in local mode",
    )
    parser.add_argument(
        "--request-interval",
        type=float,
        default=5.0,
        help="Dataset Viewer API only; accepted but ignored in local mode",
    )
    parser.add_argument("--max-problem-chars", type=int, default=12000)
    parser.add_argument("--allow-majority-answer", action="store_true")
    parser.add_argument(
        "--local-file",
        type=Path,
        help=(
            "local split parquet/JSONL; parquet is streamed without reading "
            "the messages column or loading the full file into memory"
        ),
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        help=(
            "download/reuse the split parquet under this directory, then "
            "stream it locally; mutually exclusive with --local-file"
        ),
    )
    parser.add_argument(
        "--download-repo-file",
        help=(
            "repository-relative parquet path; defaults to "
            "data/<split>.parquet"
        ),
    )
    parser.add_argument(
        "--local-batch-size",
        type=int,
        default=16384,
        help="number of local parquet rows decoded per record batch",
    )
    parser.add_argument(
        "--local-progress-every",
        type=int,
        default=250000,
        help="print local scan progress after this many trajectory rows",
    )
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--system-prompt-file", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise SystemExit("--sample-size must be positive")
    if not args.metadata_key:
        raise SystemExit("--metadata-key must be non-empty")
    if args.min_metadata_pass < 0:
        raise SystemExit("--min-metadata-pass must be non-negative")
    if args.max_metadata_pass < args.min_metadata_pass:
        raise SystemExit(
            "--max-metadata-pass must be greater than or equal to "
            "--min-metadata-pass"
        )
    if not 1 <= args.page_size <= 100:
        raise SystemExit("--page-size must be between 1 and 100")
    if args.request_interval < 0:
        raise SystemExit("--request-interval must be non-negative")
    if args.local_batch_size <= 0:
        raise SystemExit("--local-batch-size must be positive")
    if args.local_progress_every <= 0:
        raise SystemExit("--local-progress-every must be positive")
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing file: {args.output}")
    try:
        resolve_or_download_local_file(args)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"failed to resolve local split parquet: {exc}") from exc
    if args.local_file is not None and not args.local_file.is_file():
        raise SystemExit(f"local file not found: {args.local_file}")
    for path in args.exclude_uids_from:
        if not path.is_file():
            raise SystemExit(f"UID exclusion file not found: {path}")
    try:
        args.excluded_uids = load_excluded_uids(args.exclude_uids_from)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"failed to load UID exclusion file: {exc}") from exc

    system_prompt = ""
    if args.system_prompt_file:
        system_prompt = args.system_prompt_file.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise SystemExit("system prompt file is empty")
    if args.tokenizer_path and not system_prompt:
        raise SystemExit("--system-prompt-file is required with --tokenizer-path")

    if args.local_file is not None:
        print(
            "[local] --max-requests and --request-interval are ignored",
            file=sys.stderr,
            flush=True,
        )
        rows, audit = collect_from_local(args)
    else:
        rows, audit = collect_from_api(args)

    rejected = Counter(audit.get("rejected", {}))
    rows = filter_prompt_length(
        rows,
        args.tokenizer_path,
        system_prompt,
        args.max_prompt_tokens,
        rejected,
    )
    if len(rows) < args.sample_size:
        raise SystemExit(
            f"only {len(rows)} rows remain after prompt-length filtering; "
            "increase --sample-reserve and rerun"
        )

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    selected = rows[: args.sample_size]
    for index, row in enumerate(selected):
        row["baseline_index"] = index
        row["sample_seed"] = args.seed
        row["source_dataset"] = args.dataset
        row["source_split"] = args.split

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_parquet(args.output, index=False)
    audit.update(
        {
            "dataset": args.dataset,
            "requested_sample_size": args.sample_size,
            "written_sample_size": len(selected),
            "seed": args.seed,
            "metadata_filter": {
                "key": args.metadata_key,
                "min_pass_inclusive": args.min_metadata_pass,
                "max_pass_inclusive": args.max_metadata_pass,
            },
            "exclude_uids_from": [
                str(path) for path in args.exclude_uids_from
            ],
            "excluded_uid_count": len(args.excluded_uids),
            "allow_majority_answer": args.allow_majority_answer,
            "max_problem_chars": args.max_problem_chars,
            "max_prompt_tokens": (
                args.max_prompt_tokens if args.tokenizer_path else None
            ),
            "tokenizer_path": args.tokenizer_path or None,
            "rejected": dict(rejected),
            "source_counts": dict(
                Counter(row["data_source"] for row in selected)
            ),
            "changed_answer_to_majority_count": sum(
                bool(row["changed_answer_to_majority"]) for row in selected
            ),
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
        }
    )
    audit_path = args.output.with_suffix(".audit.json")
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
