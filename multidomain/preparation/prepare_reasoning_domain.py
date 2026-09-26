#!/usr/bin/env python3
"""Extract published RG candidates and audit task/category counts, without generating new questions.

This does not create an eligible training index: tokenization, verifier acceptance,
deduplication and train/validation/test splitting are still separate stages.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml


def main():
    kit = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    ap.add_argument('--config', type=Path, default=kit / 'config/multidomain/initial.yaml')
    a = ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        ap.error('Choose an empty output directory')
    config = yaml.safe_load(a.config.read_text())['domains']['logic_algorithmic']
    mapping = json.loads((kit / 'config/multidomain/reasoning_gym_task_categories.json').read_text())
    lock = json.loads((kit / 'config/multidomain/sources.lock.json').read_text())['sources']['reasoning_gym']
    source = a.source.read_bytes()
    if hashlib.sha256(source).hexdigest() != lock['raw_sha256']:
        raise ValueError('Source differs from the pinned published dataset')
    raw_counts, selected_counts, tasks = Counter(), Counter(), Counter()
    selected, index = [], []
    questions = set()
    for number, line in enumerate(source.splitlines(keepends=True), 1):
        row = json.loads(line)
        task = row['metadata']['source_dataset']
        category = mapping[task]
        if category not in config['categories']:
            continue
        raw_counts[category] += 1
        if task in config['exclude_tasks']:
            continue
        selected.append(line)
        selected_counts[category] += 1
        tasks[task] += 1
        question_hash = hashlib.sha256(row['question'].encode()).hexdigest()
        questions.add(question_hash)
        index.append({'uuid': row['uuid'], 'raw_row_1based': number, 'task': task,
                      'category': category, 'question_sha256': question_hash,
                      'domain': 'logic_algorithmic', 'source': 'reasoning_gym'})
    a.output_dir.mkdir(parents=True, exist_ok=True)
    data_path = a.output_dir / 'logic_algorithmic_candidates.jsonl'
    data_path.write_bytes(b''.join(selected))
    (a.output_dir / 'candidate_provenance.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in index))
    report = {
        'status': 'candidate_pool_not_yet_training_eligible',
        'dataset': lock['repo_id'], 'revision': lock['revision'], 'source_sha256': lock['raw_sha256'],
        'categories': config['categories'], 'excluded_tasks': config['exclude_tasks'],
        'raw_four_category_rows': sum(raw_counts.values()), 'raw_category_counts': dict(raw_counts),
        'selected_rows': len(selected), 'selected_tasks': len(tasks),
        'selected_category_counts': dict(selected_counts), 'selected_task_counts': dict(sorted(tasks.items())),
        'unique_exact_questions': len(questions), 'deduplicated': False,
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'source_lines_preserved_byte_for_byte': True,
        'license': 'CC BY 4.0; NVIDIA attribution and per-row license fields retained',
        'not_performed': ['Qwen tokenization', 'complete verifier acceptance', 'deduplication', 'train/validation/test split'],
    }
    (a.output_dir / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k:report[k] for k in ['raw_four_category_rows','selected_rows','selected_tasks','unique_exact_questions','selected_category_counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
