#!/usr/bin/env python3
"""Run on a machine with Internet. Download pinned data and restore Math placeholders.

Does not download model weights, prepare train parquet, or submit jobs.
Dependencies for this separate preparation environment: huggingface_hub, datasets, PyYAML.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import yaml


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser(description=__doc__)
    kit = Path(__file__).resolve().parents[2]
    ap.add_argument('--lock', type=Path, default=kit / 'config/multidomain/sources.lock.json')
    ap.add_argument('--config', type=Path, default=kit / 'config/multidomain/initial.yaml')
    ap.add_argument('--output-dir', type=Path, required=True)
    a = ap.parse_args()
    lock = json.loads(a.lock.read_text())
    config = yaml.safe_load(a.config.read_text())
    from multidomain.preparation.check_config import resolve
    config = resolve(config)
    enabled_sources = list(lock['sources'])  # Freeze one complete corpus before optional-domain ablations.
    manifest = {'sources': {}, 'restoration_dependencies': {k: lock['dependencies'][k] for k in ('dapo', 'skywork')}}
    for source in enabled_sources:
        info = lock['sources'][source]
        dest = a.output_dir / source / info['revision']
        patterns = info['files'] + ['README.md']
        if info.get('restore_script'):
            patterns.append(info['restore_script'])
        snapshot_download(repo_id=info['repo_id'], repo_type='dataset', revision=info['revision'],
                          allow_patterns=patterns, local_dir=dest)
        manifest['sources'][source] = {
            **info,
            'files_on_disk': [{'path': str((dest / p).resolve()),
                               'relative_path': str((dest / p).relative_to(a.output_dir)),
                               'bytes': (dest / p).stat().st_size,
                               'sha256': file_hash(dest / p)} for p in info['files']],
        }
        if source == 'math':
            from datasets import load_dataset
            spec = importlib.util.spec_from_file_location('official_math_restore', dest / info['restore_script'])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            hf = {}
            for key, split in (('dapo', 'train'), ('skywork', 'math')):
                dep = lock['dependencies'][key]
                hf[(dep['repo_id'], split)] = load_dataset(dep['repo_id'], revision=dep['revision'], split=split)
            restored_path = dest / 'restored/train.jsonl'
            restored_path.parent.mkdir(parents=True, exist_ok=True)
            provenance_path = dest / 'restored/provenance.jsonl'
            total, restored = 0, 0
            with (dest / 'data/train.jsonl').open() as fin, restored_path.open('w') as fout, provenance_path.open('w') as pf:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    ph = row.get('_hf_question_placeholder')
                    if ph:
                        pf.write(json.dumps({'uuid': row.get('uuid'), 'placeholder': ph}, ensure_ascii=False) + '\n')
                        row = module.restore_row(row, hf)
                        restored += 1
                    if not str(row.get('question') or '').strip() or not str(row.get('expected_answer') or '').strip():
                        raise ValueError(f'Math restoration left empty question/answer at row {total}')
                    total += 1
                    fout.write(json.dumps(row, ensure_ascii=False) + '\n')
            manifest['sources']['math']['restored'] = {
                'path': str(restored_path.resolve()),
                'relative_path': str(restored_path.relative_to(a.output_dir)),
                'sha256': file_hash(restored_path),
                'rows': total, 'restored_rows': restored, 'provenance_path': str(provenance_path.resolve()),
                'provenance_relative_path': str(provenance_path.relative_to(a.output_dir)),
                'provenance_sha256': file_hash(provenance_path),
                'restore_script_sha256': file_hash(dest / info['restore_script']),
            }
        a.output_dir.mkdir(parents=True, exist_ok=True)
        (a.output_dir / 'download_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(f'Fetched and hashed: {source}', flush=True)
    (a.output_dir / 'sources.lock.json').write_text(json.dumps(lock, ensure_ascii=False, indent=2) + '\n')
    print('Raw data prepared. Tokenization, deduplication, verifier checks and final balance still follow.')


if __name__ == '__main__':
    main()
