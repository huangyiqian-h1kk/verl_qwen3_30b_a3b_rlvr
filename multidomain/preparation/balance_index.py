#!/usr/bin/env python3
"""Split eligible sample indexes by leakage group, then balance enabled domains.

Input JSONL fields: sample_id, domain, source, split_group_id, prompt_tokens,
eligible=true, plus optional provenance. No prompt truncation or raw-data parsing.
The full data adapter must produce this index after recovery, validation and dedup.
This utility selects IDs; it does NOT claim to export training-ready parquet.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from multidomain.preparation.check_config import response_budget


def digest(seed, purpose, key):
    value = json.dumps([seed, purpose, key], ensure_ascii=False, separators=(',', ':'))
    return hashlib.sha256(value.encode()).hexdigest()


def group_split(seed, key, ratios):
    x = int(digest(seed, 'group-split-v1', key), 16) / (1 << 256)
    return 'train' if x < ratios[0] else 'validation' if x < sum(ratios[:2]) else 'test'


def select(rows, n, seed, purpose, balance_sources=False):
    if n > len(rows):
        raise ValueError('Cannot select without replacement beyond capacity')
    if not balance_sources:
        return sorted(rows, key=lambda r: digest(seed, purpose, r['sample_id']))[:n]
    by_source = defaultdict(list)
    for row in rows:
        by_source[row['source']].append(row)
    quotas = {s: 0 for s in by_source}
    for _ in range(n):
        available = [s for s in quotas if quotas[s] < len(by_source[s])]
        s = min(available, key=lambda s: (quotas[s], digest(seed, purpose, s)))
        quotas[s] += 1
    result = []
    for source, items in by_source.items():
        result.extend(select(items, quotas[source], seed, purpose + '/' + source))
    return result


def make_selection(rows, config):
    seed, dc = config['seed'], config['data']
    active = [k for k, v in config['domains'].items() if v['enabled']]
    pools = {split: {d: [] for d in active} for split in ('train', 'validation', 'test')}
    seen = set()
    assignments = []
    excluded = Counter()
    for row in rows:
        for field in ('sample_id', 'domain', 'source', 'split_group_id', 'prompt_tokens', 'eligible'):
            if field not in row:
                raise ValueError(f'Index row is missing {field}')
        if row['domain'] not in active:
            excluded['disabled_domain'] += 1
            continue
        if row['source'] not in config['domains'][row['domain']]['sources']:
            raise ValueError(f'Invalid domain/source mapping for {row["sample_id"]}')
        if row['eligible'] is not True:
            excluded['ineligible'] += 1
            continue
        if not isinstance(row['prompt_tokens'], int) or row['prompt_tokens'] <= 0:
            raise ValueError('prompt_tokens must be a positive integer from the actual model template')
        budget = response_budget(row['prompt_tokens'], dc, row.get('request_max_output_tokens'))
        if budget == 0:
            excluded['overlength'] += 1
            continue
        row = {**row, 'generation_max_tokens': budget}
        if not isinstance(row['sample_id'], str) or not row['sample_id']:
            raise ValueError('sample_id must be a nonempty string')
        if not isinstance(row['split_group_id'], str) or not row['split_group_id']:
            raise ValueError('split_group_id must be a nonempty globally scoped string')
        if row['sample_id'] in seen:
            raise ValueError(f'Duplicate sample_id: {row["sample_id"]}; deduplicate upstream')
        seen.add(row['sample_id'])
        split = group_split(seed, row['split_group_id'], dc['split_ratios'])
        pools[split][row['domain']].append(row)
        assignments.append({'sample_id': row['sample_id'], 'split_group_id': row['split_group_id'], 'split': split})
    counts = {s: {d: len(items) for d, items in p.items()} for s, p in pools.items()}
    for split, by_domain in counts.items():
        if min(by_domain.values()) == 0:
            raise ValueError(f'An enabled domain has an empty {split} pool: {by_domain}; inspect group coverage')
    if_sources = {r['source'] for r in pools['train']['if']}
    if if_sources != set(config['domains']['if']['sources']):
        raise ValueError(f'IF training pool does not cover all three sources: {if_sources}')
    targets = {
        'train': min(counts['train'].values()),
        'validation': min(dc['validation_max_per_domain'], min(counts['validation'].values())),
        'test': min(dc['test_max_per_domain'], min(counts['test'].values())),
    }
    chosen = {}
    for split, p in pools.items():
        selected = []
        for domain, items in p.items():
            selected.extend(select(items, targets[split], seed, split + '/select/' + domain,
                                   balance_sources=(domain == 'if')))
        chosen[split] = sorted(selected, key=lambda r: digest(seed, split + '/shuffle', r['sample_id']))
    selected_train = {r['sample_id'] for r in chosen['train']}
    chosen['reserve_train'] = [r for items in pools['train'].values() for r in items if r['sample_id'] not in selected_train]
    selected_groups = {s: {r['split_group_id'] for r in chosen[s]} for s in ('train', 'validation', 'test')}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if selected_groups[a] & selected_groups[b]:
            raise AssertionError(f'Group leakage: {a}/{b}')
    n = targets['train']
    manifest = {
        'seed': seed, 'enabled_domains': active, 'eligible_pool_counts': counts,
        'target_rows_per_domain': targets, 'bottleneck_domains': [d for d in active if counts['train'][d] == n],
        'selected_rows': {s: len(v) for s, v in chosen.items()},
        'selected_source_counts': {s: dict(Counter(r['source'] for r in chosen[s])) for s in ('train', 'validation', 'test')},
        'excluded_index_rows': dict(excluded), 'group_intersections': 0,
        'train_batches_per_epoch_drop_last': len(chosen['train']) // config['training']['train_batch_size'],
        'train_tail_rows_per_epoch_drop_last': len(chosen['train']) % config['training']['train_batch_size'],
        'notes': ['Equal sample counts, not equal tokens or compute.',
                  'Unselected held-out rows remain held out; never refill training from them.',
                  'Dataset is balanced; shuffled minibatches and dropped epoch tails need not be exactly balanced.'],
    }
    return chosen, assignments, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, type=Path, help='JSON emitted by check_config.py')
    ap.add_argument('--index', required=True, type=Path)
    ap.add_argument('--output-dir', required=True, type=Path)
    a = ap.parse_args()
    if a.output_dir.exists() and any(a.output_dir.iterdir()):
        ap.error('Use a new output directory; existing experiment selections are not overwritten')
    c = json.loads(a.config.read_text())
    from multidomain.preparation.check_config import resolve
    c = resolve(c)
    with a.index.open() as f:
        selected, assignments, manifest = make_selection((json.loads(line) for line in f if line.strip()), c)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    for name, items in {**selected, 'split_assignments': assignments}.items():
        with (a.output_dir / (name + '.index.jsonl')).open('w') as f:
            for r in items:
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n')
    manifest['config_sha256'] = hashlib.sha256(a.config.read_bytes()).hexdigest()
    manifest['index_sha256'] = hashlib.sha256(a.index.read_bytes()).hexdigest()
    (a.output_dir / 'balance_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
