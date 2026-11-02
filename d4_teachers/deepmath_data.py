"""Freeze supplementary DeepMath held-outs against the actual teacher inputs.

This is CPU preparation, not training. No upstream dataset is downloaded. The
original train/validation/test tables and their membership are never edited.
"""

from collections import Counter, defaultdict
import copy
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import sys
import unicodedata
import uuid

from .common import digest, file_sha, lock, need, read, write

VERSION = "deepmath-heldouts-v1"
SPLITS = ("validation", "test")
LEVELS = (8, 9)


def canonical(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    for token in (r"\left", r"\right", r"\!", r"\,", "$", r"\(", r"\)", r"\[", r"\]"):
        text = text.replace(token, "")
    return re.sub(r"\s+", "", text)


def question_key(text):
    return digest(canonical(text))


def messages(row):
    value = row.get("messages_json", row.get("prompt", []))
    value = json.loads(value) if isinstance(value, str) else value
    need(isinstance(value, list), "Unsupported prompt representation")
    return value


def last_user(row):
    users = [m["content"] for m in messages(row) if m.get("role") == "user"]
    need(users and isinstance(users[-1], str), "Missing text user question")
    return users[-1]


def records(path):
    """Read only question metadata, never the large released reasoning solutions."""
    path = Path(path)
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        names = pq.read_schema(path).names
        keep = {
            "question",
            "problem",
            "final_answer",
            "answer",
            "expected_answer",
            "difficulty",
            "level",
            "difficulty_level",
            "topic",
            "original_index",
            "source_index",
            "index",
            "prompt",
            "reward_model",
            "extra_info",
        }
        yield from pq.read_table(
            path, columns=[n for n in names if n in keep]
        ).to_pylist()
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    need(isinstance(row, dict), "Expected one DeepMath object per line")
                    yield row


def unpack(row):
    extra = row.get("extra_info") or {}
    extra = json.loads(extra) if isinstance(extra, str) else extra
    question = row.get("question", row.get("problem"))
    if question is None and row.get("prompt") is not None:
        question = last_user(row)
    answer = row.get("final_answer", row.get("expected_answer", row.get("answer")))
    if answer is None:
        reward = row.get("reward_model") or {}
        reward = json.loads(reward) if isinstance(reward, str) else reward
        answer = reward.get("ground_truth")
    difficulty = row.get("difficulty", row.get("level", row.get("difficulty_level")))
    if difficulty is None:
        difficulty = extra.get("difficulty", extra.get("difficulty_level"))
    need(isinstance(question, str) and question.strip(), "DeepMath question missing")
    need(
        isinstance(answer, (str, int, float)) and str(answer).strip(),
        "DeepMath answer missing",
    )
    try:
        level = int(float(difficulty))
    except (TypeError, ValueError, OverflowError) as e:
        raise RuntimeError("DeepMath difficulty missing/invalid") from e
    return dict(
        question=question.strip(),
        answer=str(answer).strip(),
        level=level,
        topic=str(row.get("topic", extra.get("topic", ""))),
        original_index=row.get(
            "original_index", row.get("source_index", row.get("index"))
        ),
    )


def discover(c):
    parent = Path(c["policy"]["parent_root"])
    roots = [
        parent / "data",
        parent / "assets",
        parent / "supplemental_data",
        parent.parent / "verl_qwen3_30b_a3b_rlvr" / "data",
    ]
    found = {"raw": [], "holdout": [], "exclude_probe": []}
    for root in roots:
        if not root.is_dir():
            continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = sorted(
                d
                for d in dirs
                if not d.startswith(".")
                and d
                not in {
                    "outputs",
                    "checkpoints",
                    "model_snapshots",
                    "artifacts",
                    "hf_home",
                }
            )
            for name in sorted(names):
                path = Path(directory) / name
                label = str(path.relative_to(root)).lower()
                if "deepmath" not in label or not (
                    name.endswith((".parquet", ".jsonl", ".jsonl.gz"))
                ):
                    continue
                if "probe" in label:
                    kind = "exclude_probe"
                elif any(
                    x in name.lower() for x in ("holdout", "validation", "_val", "test")
                ):
                    kind = "holdout"
                elif name.startswith("deepmath_l89_") and name.endswith(".jsonl.gz"):
                    kind = "raw"
                elif "raw_deepmath" in label and name.endswith(".parquet"):
                    kind = "raw"
                else:
                    continue
                found[kind].append(str(path.resolve()))
    # Prefer the frozen light input used by the 10K build over the 2GB upstream files.
    compact = [p for p in found["raw"] if p.endswith(".jsonl.gz")]
    if compact:
        found["raw"] = compact
    return {k: sorted(set(v)) for k, v in found.items()}


def input_records(inputs):
    candidates, origins, rejected = [], [], Counter()
    for kind in ("raw", "holdout", "exclude_probe"):
        for name in inputs[kind]:
            path = Path(name).resolve()
            before = file_sha(path)
            origin = dict(path=str(path), sha256=before, kind=kind, rows=0)
            for index, raw in enumerate(records(path)):
                origin["rows"] += 1
                if kind == "exclude_probe":
                    question = raw.get("question", raw.get("problem"))
                    if question is None:
                        question = last_user(raw)
                    need(
                        isinstance(question, str) and question.strip(),
                        "Cannot identify prior probe question",
                    )
                    row = dict(
                        question=question.strip(),
                        answer="",
                        level=None,
                        topic="",
                        original_index=None,
                    )
                else:
                    try:
                        row = unpack(raw)
                    except RuntimeError as e:
                        rejected[str(e)] += 1
                        continue
                    if row["level"] not in LEVELS:
                        rejected["outside_levels_8_9"] += 1
                        continue
                row.update(
                    origin=dict(path=str(path), sha256=before, row=index), kind=kind
                )
                # Honor an existing named validation/test partition when available.
                label = path.name.lower()
                row["preferred_split"] = (
                    (
                        "validation"
                        if "validation" in label or "_val" in label
                        else "test"
                        if "test" in label
                        else None
                    )
                    if kind == "holdout"
                    else None
                )
                candidates.append(row)
            need(file_sha(path) == before, "DeepMath input changed while reading")
            origins.append(origin)
    need(
        any(r["kind"] == "raw" for r in candidates),
        "No usable raw DeepMath L8/L9 input. Pass --input /path/deepmath_l89_*.jsonl.gz (or raw parquet).",
    )
    return candidates, origins, dict(rejected)


def question_index(candidates, normalized=False):
    index = defaultdict(list)
    for row in candidates:
        q = canonical(row["question"]) if normalized else row["question"]
        if len(q) >= 32:
            index[q[:32]].append((q, row))
    return index


def matches(text, index):
    for i in range(max(0, len(text) - 31)):
        for q, row in index.get(text[i : i + 32], ()):
            if text.startswith(q, i):
                yield i, q, row


def infer_template(train, raw):
    """Infer the exact frozen scaffold from raw-question anchors, not a new prompt."""
    deep = [r for r in train if r["source"] in ("deepmath_l8", "deepmath_l9")]
    need(deep, "Frozen Math training has no recognized DeepMath rows")
    index = question_index(raw)
    templates = {}
    witnesses = []
    for row in deep[:128]:
        msgs = messages(row)
        position = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        user = msgs[position]["content"]
        hits = list(matches(user, index))
        if not hits:
            continue
        offset, q, original = max(hits, key=lambda h: len(h[1]))
        prefix, suffix = user[:offset], user[offset + len(q) :]
        body = copy.deepcopy(msgs)
        body[position]["content"] = ""
        spec = dict(
            messages=body,
            user_index=position,
            prefix=prefix,
            suffix=suffix,
            tools_json=row["tools_json"],
            request_max_output_tokens=row.get("request_max_output_tokens"),
        )
        templates[digest(spec)] = spec
        witnesses.append(dict(sample_id=row["sample_id"], origin=original["origin"]))
        if len(witnesses) >= 8:
            break
    need(
        len(witnesses) >= 2 and len(templates) == 1,
        "Cannot identify a unique frozen DeepMath prompt scaffold from raw training anchors",
    )
    spec = next(iter(templates.values()))
    need(
        json.loads(spec["tools_json"]) == [],
        "DeepMath scaffold unexpectedly contains tools",
    )
    for row in deep:
        body = messages(row)
        user = body[spec["user_index"]]["content"]
        body[spec["user_index"]]["content"] = ""
        need(
            body == spec["messages"]
            and user.startswith(spec["prefix"])
            and user.endswith(spec["suffix"])
            and row["tools_json"] == spec["tools_json"]
            and row.get("request_max_output_tokens")
            == spec["request_max_output_tokens"],
            "Frozen DeepMath training uses multiple prompt scaffolds",
        )
    return spec, witnesses, deep[0]


def remove_training_and_existing_eval(candidates, frozen_rows, template):
    blocked, counts = set(), Counter()
    index = question_index(candidates, normalized=True)
    for split, rows in frozen_rows.items():
        for row in rows:
            values = [last_user(row)]
            payload = json.loads(row["verifier_json"])
            values.extend(
                v
                for v in (row.get("question"), payload.get("question"))
                if isinstance(v, str)
            )
            for text in values:
                blocked.add(question_key(text))
                for _, _, candidate in matches(canonical(text), index):
                    blocked.add(question_key(candidate["question"]))
            if row["domain"] == "math":
                text = last_user(row)
                pre, post = template["prefix"], template["suffix"]
                if text.startswith(pre) and text.endswith(post):
                    q = text[len(pre) : len(text) - len(post) if post else None]
                    blocked.add(question_key(q))
            counts[split] += 1
    probes = {
        question_key(r["question"]) for r in candidates if r["kind"] == "exclude_probe"
    }
    by_key = defaultdict(list)
    for r in candidates:
        by_key[question_key(r["question"])].append(r)
    eligible, excluded = [], Counter()
    for key, versions in by_key.items():
        if key in blocked:
            excluded["frozen_train_or_existing_eval"] += 1
        elif key in probes:
            excluded["prior_probe"] += 1
        elif len({canonical(v["answer"]) for v in versions}) != 1:
            excluded["conflicting_reference_answers"] += 1
        elif len({v["level"] for v in versions}) != 1:
            excluded["conflicting_difficulty"] += 1
        elif len({v["preferred_split"] for v in versions if v["preferred_split"]}) > 1:
            excluded["overlapping_old_validation_test"] += 1
        else:
            chosen = min(
                versions,
                key=lambda v: (
                    v["kind"] != "holdout",
                    not bool(v["preferred_split"]),
                    v["origin"]["path"],
                    v["origin"]["row"],
                ),
            )
            eligible.append(dict(chosen, key=key))
    return eligible, dict(
        rows_checked=dict(counts),
        exclusions=dict(excluded),
        normalization="NFKC, casefold, whitespace/LaTeX delimiter normalization; exact and wrapped-text overlap, not semantic decontamination",
        blocked_keys_sha256=digest(sorted(blocked | probes)),
    )


def strata_targets(n, training_counts):
    total = sum(training_counts.values())
    need(
        total > 0 and n >= 2,
        "Need DeepMath training counts and at least two held-out rows",
    )
    n9 = max(1, min(n - 1, round(n * training_counts[9] / total)))
    return {8: n - n9, 9: n9}


def choose(eligible, n, seed, training_counts, convert):
    """Stable, disjoint, difficulty-stratified selection with explicit shortages."""
    targets = strata_targets(n, training_counts)
    used, result, failures = set(), {s: [] for s in SPLITS}, Counter()
    converted = {}

    def candidate(row):
        if row["key"] not in converted:
            converted[row["key"]] = convert(row)
            if converted[row["key"]] is None:
                failures["reference_or_input_ineligible"] += 1
        return converted[row["key"]]

    for split in SPLITS:
        pool = sorted(
            (r for r in eligible if r["preferred_split"] in (None, split)),
            key=lambda r: (
                r["kind"] != "holdout",
                r["preferred_split"] != split,
                digest([seed, split, r["key"]]),
            ),
        )
        for level in LEVELS:
            for row in pool:
                if (
                    sum(x["difficulty_level"] == level for x in result[split])
                    >= targets[level]
                ):
                    break
                if row["level"] == level and row["key"] not in used:
                    value = candidate(row)
                    if value is not None:
                        result[split].append(dict(value, evaluation_split=split))
                        used.add(row["key"])
        # Do not silently fill an L9 deficit with easier L8 questions.
        need(
            len(result[split]) == n,
            f"Insufficient clean DeepMath {split}: requested strata {targets}, got "
            f"{dict(Counter(r['difficulty_level'] for r in result[split]))}. Inspect data audit or supply more held-out/raw inputs.",
        )
        result[split].sort(key=lambda r: digest([seed, split, r["sample_id"]]))
    return result, dict(
        requested_per_split=n, target_level_counts=targets, rejected=dict(failures)
    )


def make_converter(c, template, prototype, tokenizer, score_reference):
    from multidomain.template import render

    cfg = c["contract"]["base_config"]["data"]

    def convert(row):
        question = row["question"]
        if re.search(
            r"!\[|<image|\[asy\]|\\begin\{tikz|(?:shown|given) in (?:the|this) (?:figure|diagram)",
            question,
            re.I,
        ):
            return None
        msgs = copy.deepcopy(template["messages"])
        msgs[template["user_index"]]["content"] = (
            template["prefix"] + question + template["suffix"]
        )
        ids = render(tokenizer, msgs, [])
        if not 0 < len(ids) < cfg["max_context_tokens"]:
            return None
        proof = score_reference(row["answer"])
        need(
            proof.get("status") in ("ok", "timeout", "error"),
            "Unknown reference preflight result",
        )
        if proof.get("status") == "error":
            if proof.get("phase") == "gold_parse" and str(
                proof.get("error", "")
            ).startswith("ValueError: No gold target"):
                return None
            # Infrastructure faults must never be silently converted into data filtering.
            raise RuntimeError("Reference verifier failed: " + str(proof))
        if proof.get("status") != "ok" or proof.get("score") != 1.0:
            return None
        payload = json.loads(prototype["verifier_json"])
        need("expected_answer" in payload, "Unknown frozen Math verifier payload")
        payload["expected_answer"] = row["answer"]
        if "question" in payload:
            payload["question"] = question
        source = "deepmath_l" + str(row["level"])
        value = dict(prototype)
        value.update(
            sample_id="deepmath_heldout/" + row["key"],
            domain="math",
            source=source,
            category=source,
            task="deepmath",
            difficulty_level=row["level"],
            messages_json=json.dumps(
                msgs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            tools_json="[]",
            verifier_json=json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
            split_group_id=row["key"],
            dedup_key=row["key"],
            prompt_tokens=len(ids),
            prompt_token_hash=digest(ids),
            generation_max_tokens=min(
                cfg["max_context_tokens"] - len(ids),
                template["request_max_output_tokens"] or cfg["max_response_tokens"],
            ),
            provenance_json=json.dumps(
                dict(
                    source="zwhe99/DeepMath-103K",
                    origin=row["origin"],
                    prior_partition=row["kind"],
                    topic=row["topic"],
                    question_sha256=row["key"],
                    reference_selfcheck=proof,
                ),
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        if "leakage_keys" in value:
            value["leakage_keys"] = ["question:" + row["key"]]
        return value

    return convert


def directory(c):
    return Path(c["path"]) / "deepmath_evaluation" / "data"


def verify(c):
    root = directory(c)
    m = read(root / "manifest.json")
    need(
        m["identity"] == digest({k: v for k, v in m.items() if k != "identity"})
        and m["campaign_identity"] == c["identity"]
        and m["version"] == VERSION,
        "DeepMath data manifest changed",
    )
    for name, sha in m["files"].items():
        need(file_sha(root / name) == sha, "DeepMath held-out file changed: " + name)
    return m


def prepare(c, inputs=None, size=256, seed=42, tokenizer=None, score_reference=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .checkpoint_evaluation import verifier
    from multidomain.template import render, tokenizer_fingerprint

    root = directory(c)
    with lock(root.parent / "prepare.lock"):
        if (root / "manifest.json").exists():
            m = verify(c)
            need(
                m["size_per_split"] == size
                and m["seed"] == seed
                and m["verifier_identity"] == verifier(c),
                "A different DeepMath preparation already exists",
            )
            if inputs is not None:
                supplied = {str(Path(p).resolve()) for v in inputs.values() for p in v}
                need(
                    supplied == {r["path"] for r in m["inputs"]},
                    "DeepMath input selection changed",
                )
            return m
        inputs = discover(c) if inputs is None else inputs
        need(
            inputs.get("raw"),
            "No DeepMath raw input found; run discover or pass --input PATH",
        )
        candidates, origins, rejected = input_records(inputs)
        parent_manifest = read(Path(c["path"]) / "data/manifest.json")
        frozen = {
            s: [
                r
                for d in parent_manifest["domains"]
                for r in pq.read_table(
                    Path(c["path"]) / "data" / d / (s + ".parquet")
                ).to_pylist()
            ]
            for s in ("train", "validation", "test")
        }
        train = [r for r in frozen["train"] if r["domain"] == "math"]
        template, anchors, prototype = infer_template(train, candidates)
        eligible, audit = remove_training_and_existing_eval(
            candidates, frozen, template
        )
        counts = {
            level: sum(r["source"] == "deepmath_l" + str(level) for r in train)
            for level in LEVELS
        }
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                c["contract"]["base_model"], local_files_only=True
            )
        for row in [r for r in train if r["source"].startswith("deepmath_")][:16]:
            ids = render(tokenizer, messages(row), json.loads(row["tools_json"]))
            need(
                len(ids) == row["prompt_tokens"]
                and digest(ids) == row["prompt_token_hash"],
                "Preparation renderer/tokenizer differs from the frozen training prompts",
            )
        if score_reference is None:
            from .math_test_repair import (
                active_revision,
                revision_directory,
                worker_score,
            )

            revision = active_revision(c)
            package = (
                revision_directory(c, revision["identity"]) / "runtime/d4_db_mathfix"
                if revision
                else Path(c["path"]) / "template/runtime/d4_db_mathfix"
            )

            def score_reference(answer):
                key = digest([verifier(c), answer])
                cache = root.parent / "reference_checks" / (key + ".json")
                if cache.exists():
                    proof = read(cache)
                    need(
                        proof["identity"] == key
                        and proof["checksum"] == digest(proof["result"]),
                        "Reference preflight cache changed",
                    )
                    return proof["result"]
                result = worker_score(
                    Path(c["contract"]["overlay"]) / "bin/python",
                    package,
                    answer,
                    r"\boxed{" + answer + "}",
                )
                write(cache, dict(identity=key, result=result, checksum=digest(result)))
                return result

        convert = make_converter(c, template, prototype, tokenizer, score_reference)
        completed = 0

        def checked(row):
            nonlocal completed
            value = convert(row)
            completed += 1
            if completed % 32 == 0:
                print(
                    f"DeepMath CPU preflight: checked {completed} candidate references",
                    file=sys.stderr,
                    flush=True,
                )
            return value

        selected, sampling = choose(eligible, size, seed, counts, checked)
        stage = root.with_name(".staging_" + uuid.uuid4().hex)
        stage.mkdir(parents=True)
        try:
            outputs = {}
            for split in SPLITS:
                rows = selected[split]
                pq.write_table(pa.Table.from_pylist(rows), stage / (split + ".parquet"))
                need(
                    pq.read_table(stage / (split + ".parquet")).to_pylist() == rows,
                    "DeepMath Arrow round trip changed",
                )
                outputs[split] = dict(
                    rows=len(rows),
                    ids=[r["sample_id"] for r in rows],
                    sources=dict(Counter(r["source"] for r in rows)),
                    content_sha256=digest(rows),
                )
            need(
                not set(outputs["validation"]["ids"]) & set(outputs["test"]["ids"]),
                "Held-out split overlap",
            )
            m = dict(
                version=VERSION,
                campaign_identity=c["identity"],
                parent_data_digest=c["contract"]["data_digest"],
                size_per_split=size,
                seed=seed,
                inputs=origins,
                input_rejections=rejected,
                audit=audit,
                sampling=sampling,
                training_source_counts={str(k): v for k, v in counts.items()},
                template=template,
                template_anchors=anchors,
                tokenizer=tokenizer_fingerprint(tokenizer),
                verifier_identity=verifier(c),
                splits=outputs,
                model_weights_loaded=False,
                gpu_jobs_submitted=0,
                files={
                    s + ".parquet": file_sha(stage / (s + ".parquet")) for s in SPLITS
                },
            )
            m["identity"] = digest(m)
            write(stage / "manifest.json", m)
            need(not root.exists(), "Unowned DeepMath data directory exists")
            stage.rename(root)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        return verify(c)
