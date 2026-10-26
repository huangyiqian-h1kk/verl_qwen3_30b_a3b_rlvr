"""Filter frozen Arrow tables; never rebuild, rebalance, or change split membership."""

from collections import Counter
from pathlib import Path
import shutil
import uuid
from . import DOMAINS, VALIDATION_COUNTS
from .common import digest, file_sha, need, read, write

SPLITS = ("train", "validation", "test")


def validate_rows(tables, expected_train=10000, validation_counts=VALIDATION_COUNTS):
    seen_ids, seen_groups = {}, {}
    for split, rows in tables.items():
        counts = Counter(r["domain"] for r in rows)
        need(set(counts) == set(DOMAINS), "Missing or unexpected parent domains")
        if split == "train":
            need(
                counts == Counter({d: expected_train for d in DOMAINS}),
                "Parent train counts changed",
            )
        if split == "validation":
            need(
                counts == Counter(validation_counts), "Parent validation counts changed"
            )
        for row in rows:
            sid, group = row["sample_id"], row["split_group_id"]
            need(
                isinstance(sid, str) and sid and sid not in seen_ids,
                "Duplicate/missing sample ID",
            )
            need(isinstance(group, str) and group, "Missing split group")
            need(
                group not in seen_groups or seen_groups[group] == split,
                "Split-group leakage",
            )
            seen_ids[sid], seen_groups[group] = split, split
            need(
                0 < row["prompt_tokens"] < 65536 and row["generation_max_tokens"] > 0,
                "Invalid frozen token budget",
            )


def describe(rows):
    return dict(
        rows=len(rows),
        ids=[r["sample_id"] for r in rows],
        content_sha256=digest(rows),
        sources=dict(Counter(r["source"] for r in rows)),
        categories=dict(Counter(str(r.get("category", "")) for r in rows)),
        tasks=dict(Counter(str(r.get("task", "")) for r in rows)),
    )


def prepare(
    parent, destination, expected_train=10000, validation_counts=VALIDATION_COUNTS
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parent, destination = Path(parent).resolve(), Path(destination).resolve()
    parent_hashes = {s: file_sha(parent / (s + ".parquet")) for s in SPLITS}
    if (destination / "manifest.json").exists():
        result = verify(destination)
        need(
            result["parent_sha256"] == parent_hashes,
            "Prepared teacher data has another parent",
        )
        return result
    need(
        not destination.exists(),
        "Incomplete data directory; inspect it before retrying",
    )
    tables = {s: pq.read_table(parent / (s + ".parquet")) for s in SPLITS}
    rows = {s: t.to_pylist() for s, t in tables.items()}
    validate_rows(rows, expected_train, validation_counts)
    staging = destination.with_name("." + destination.name + "." + uuid.uuid4().hex)
    staging.mkdir(parents=True)
    manifest = dict(
        version="frozen-teacher-slices-v1",
        parent=str(parent),
        parent_sha256=parent_hashes,
        domains={},
    )
    try:
        for domain in DOMAINS:
            manifest["domains"][domain] = {}
            folder = staging / domain
            folder.mkdir()
            for split, table in tables.items():
                selected = table.filter(
                    pa.array([r["domain"] == domain for r in rows[split]])
                )
                target = folder / (split + ".parquet")
                pq.write_table(selected, target)
                info = describe(selected.to_pylist())
                need(
                    pq.read_table(target).to_pylist() == selected.to_pylist(),
                    "Arrow round-trip changed rows",
                )
                manifest["domains"][domain][split] = dict(
                    info, file_sha256=file_sha(target)
                )
        need(
            parent_hashes == {s: file_sha(parent / (s + ".parquet")) for s in SPLITS},
            "Parent data changed during slicing",
        )
        write(staging / "manifest.json", manifest)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def verify(directory):
    directory = Path(directory)
    m = read(directory / "manifest.json")
    need(set(m["domains"]) == set(DOMAINS), "Incomplete teacher manifest")
    for domain, splits in m["domains"].items():
        need(set(splits) == set(SPLITS), "Incomplete teacher splits")
        for split, info in splits.items():
            need(
                file_sha(directory / domain / (split + ".parquet"))
                == info["file_sha256"],
                "Frozen teacher data changed",
            )
    return m
