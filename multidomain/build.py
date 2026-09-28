"""Stream pinned raw data into an auditable SQLite store and fixed parquet splits."""
import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from multidomain.adapters import adapt
from multidomain.common import ROOT, DataError, digest, dumps, file_sha, read_config, write_json
from multidomain.preparation.check_config import response_budget
from multidomain.preparation.balance_index import group_split, select
from multidomain.template import render, tokenizer_fingerprint


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        parent = self.parent[key]
        if parent != key:
            self.parent[key] = self.find(parent)
        return self.parent[key]

    def join(self, keys):
        roots = [self.find(k) for k in keys]
        root = min(roots)
        for key in roots:
            self.parent[key] = root


def rows_from(path):
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            yield from batch.to_pylist()
    else:
        with path.open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def category_select(rows, n, seed, purpose):
    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get('category', '')].append(row)
    quotas = {k: int(n * len(v) / len(rows)) for k, v in buckets.items()}
    remainder = n - sum(quotas.values())
    order = sorted(buckets, key=lambda k: (-(n * len(buckets[k]) / len(rows) - quotas[k]), k))
    for k in order[:remainder]:
        quotas[k] += 1
    return [r for k in sorted(buckets) for r in select(buckets[k], quotas[k], seed, purpose + '/' + k)]


def choose(index, config):
    pools = {s: defaultdict(list) for s in ('train', 'validation', 'test')}
    seed = config['seed']
    active = [d for d, c in config['domains'].items() if c['enabled']]
    for row in index:
        if row['eligible'] and row['domain'] in active:
            split = group_split(seed, row['split_group_id'], config['data']['split_ratios'])
            pools[split][row['domain']].append(row)
    counts = {s: {d: len(p[d]) for d in active} for s, p in pools.items()}
    if any(n == 0 for p in counts.values() for n in p.values()):
        raise DataError('Empty domain split: ' + dumps(counts))
    if {r['source'] for r in pools['train']['if']} != set(config['domains']['if']['sources']):
        raise DataError('IF train pool does not cover all configured sources')
    n = min(counts['train'].values())
    selected = {}
    for split, domains in pools.items():
        picked = []
        for domain in active:
            rows = domains[domain]
            # Evaluation selection depends only on this domain, so optional domains cannot change its IDs.
            count = n if split == 'train' else min(len(rows), config['data'][split + '_max_per_domain'])
            purpose = split + '/select/' + domain
            fn = category_select if domain == 'logic_algorithmic' else select
            kwargs = {'balance_sources': domain == 'if'} if fn is select else {}
            picked.extend(fn(rows, count, seed, purpose, **kwargs))
        selected[split] = sorted(picked, key=lambda r: digest([seed, split, r['sample_id']]))
    all_selected = {r['sample_id'] for rows in selected.values() for r in rows}
    selected['reserve'] = [r for r in index if r['sample_id'] not in all_selected]
    for split in ('train', 'validation', 'test'):
        selected[split + '_ids'] = [r['sample_id'] for r in selected[split]]
    return selected, {'eligible_pool_counts': counts, 'train_rows_per_domain': n,
                      'selected_counts': {s: dict(Counter(r['domain'] for r in selected[s])) for s in ('train', 'validation', 'test')},
                      'selected_source_counts': {s: dict(Counter(r['source'] for r in selected[s])) for s in ('train', 'validation', 'test')},
                      'train_batches_per_epoch_drop_last': len(selected['train']) // config['training']['train_batch_size'],
                      'train_tail_rows_per_epoch_drop_last': len(selected['train']) % config['training']['train_batch_size']}


def export_parquet(db, selected, path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    try:
        for start in range(0, len(selected), 128):
            batch = []
            for idx in selected[start:start + 128]:
                row = json.loads(db.execute('select record from samples where id=?', (idx['sample_id'],)).fetchone()[0])
                row.update({k: idx[k] for k in ('split_group_id', 'prompt_tokens', 'prompt_token_hash', 'generation_max_tokens')})
                batch.append({k: v for k, v in row.items() if k not in ('leakage_keys', 'dedup_key', 'eligible', 'request_max_output_tokens')})
            table = pa.Table.from_pylist(batch)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression='zstd')
            writer.write_table(table)
    finally:
        if writer:
            writer.close()


def build(config, raw_root, output, tokenizer_path, judge_tokenizer_path=None):
    from transformers import AutoTokenizer
    raw_root, output = Path(raw_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise DataError('Output is not empty; choose a new data ID or inspect the failed build')
    output.mkdir(parents=True, exist_ok=True)
    lock = json.loads((ROOT / 'config/multidomain/sources.lock.json').read_text())
    downloads = json.loads((raw_root / 'download_manifest.json').read_text())
    if set(downloads['sources']) != set(lock['sources']):
        raise DataError('Freeze all eight raw sources before choosing optional-domain mixtures')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    token_hash = tokenizer_fingerprint(tokenizer)
    judge_tok = None
    if config['domains']['science']['enabled']:
        if not judge_tokenizer_path:
            raise DataError('Science preparation requires the judge tokenizer for context budgeting')
        judge_tok = AutoTokenizer.from_pretrained(judge_tokenizer_path, local_files_only=True)
    db = sqlite3.connect(output / 'canonical.sqlite')
    db.execute('create table samples (id text primary key, dedup text unique, record text)')
    groups, index, exclusions = Groups(), [], Counter()
    inputs = []
    with (output / 'excluded.jsonl').open('w') as rejected:
        # Build leakage groups from all available configured sources, even if a domain is disabled later.
        for source, info in downloads['sources'].items():
            expected = lock['sources'][source]
            if info['revision'] != expected['revision'] or info['repo_id'] != expected['repo_id']:
                raise DataError('Raw source revision mismatch: ' + source)
            files = [info['restored']] if source == 'math' else info['files_on_disk']
            for meta in files:
                path = raw_root / meta['relative_path']
                if file_sha(path) != meta['sha256']:
                    raise DataError('Raw checksum mismatch: ' + str(path))
                inputs.append({'source': source, 'revision': info['revision'], 'file': meta['relative_path'], 'sha256': meta['sha256']})
                for number, raw in enumerate(rows_from(path), 1):
                    origin = {'repo_id': info['repo_id'], 'revision': info['revision'], 'file': meta['relative_path'],
                              'row_1based': number, 'raw_sha256': meta['sha256']}
                    try:
                        row, reason = adapt(raw, source, origin, config)
                    except Exception as exc:
                        raise DataError(f'{source}:{meta["relative_path"]}:{number}: {exc}') from exc
                    if row is None:
                        exclusions[reason] += 1
                        continue
                    groups.join(row['leakage_keys'])
                    existing = db.execute('select id from samples where dedup=?', (row['dedup_key'],)).fetchone()
                    if existing is not None:
                        exclusions['exact_duplicate'] += 1
                        keep, discard = sorted((existing[0], row['sample_id']))
                        rejected.write(dumps({'origin': origin, 'reason': 'exact_duplicate', 'sample_id': discard, 'kept_id': keep}) + '\n')
                        if keep == existing[0]:
                            continue
                        db.execute('delete from samples where id=?', (existing[0],))
                    db.execute('insert into samples values(?,?,?)', (row['sample_id'], row['dedup_key'], dumps(row)))
                    ids = render(tokenizer, json.loads(row['messages_json']), json.loads(row['tools_json']))
                    budget = response_budget(len(ids), config['data'], row['request_max_output_tokens'])
                    eligible, reason = budget > 0, None if budget > 0 else 'policy_context_overflow'
                    if source == 'science' and eligible and judge_tok is not None:
                        from multidomain.reward import science_prompt
                        prompt = science_prompt(json.loads(row['verifier_json']), '')
                        judge_base = len(render(judge_tok, [{'role': 'user', 'content': prompt}], []))
                        # Validate fixed judge input now; validate actual generated candidates at runtime.
                        if judge_base + 8192 >= config['judge']['max_model_len']:
                            eligible, reason = False, 'judge_question_reference_context_overflow'
                    idx = {k: row[k] for k in ('sample_id', 'domain', 'source', 'category', 'task')}
                    idx.update(eligible=eligible, prompt_tokens=len(ids), prompt_token_hash=digest(ids),
                               generation_max_tokens=budget, first_leakage_key=row['leakage_keys'][0])
                    index.append(idx)
                    if reason:
                        exclusions[reason] += 1
                        rejected.write(dumps({'sample_id': row['sample_id'], 'reason': reason, 'prompt_tokens': len(ids)}) + '\n')
                    if number % 1000 == 0:
                        db.commit()
                        print(f'{source}: {number} rows', flush=True)
    db.commit()
    kept_ids = {r[0] for r in db.execute('select id from samples')}
    index = [r for r in index if r['sample_id'] in kept_ids]
    for idx in index:
        idx['split_group_id'] = digest(groups.find(idx.pop('first_leakage_key')))
    selected, balance = choose(index, config)
    with (output / 'index.jsonl').open('w') as f:
        for idx in index:
            f.write(dumps(idx) + '\n')
    for split in ('train', 'validation', 'test'):
        export_parquet(db, selected[split], output / (split + '.parquet'))
        write_json(output / (split + '_ids.json'), selected[split + '_ids'])
    # Small deterministic acceptance/evaluation sets come only from their assigned split.
    smoke, baseline = [], []
    for domain in balance['selected_counts']['train']:
        pool = [r for r in selected['train'] if r['domain'] == domain]
        random_rows = select(pool, min(8, len(pool)), config['seed'], 'smoke/' + domain)
        longest = sorted(pool, key=lambda r: (-r['prompt_tokens'], r['sample_id']))[:4]
        smoke.extend({r['sample_id']: r for r in random_rows + longest}.values())
        baseline.extend(select(pool, min(64, len(pool)), config['seed'], 'baseline/' + domain))
    export_parquet(db, smoke, output / 'smoke.parquet')
    export_parquet(db, baseline, output / 'baseline_train.parquet')
    hashes = {p.name: file_sha(p) for p in output.glob('*.parquet')}
    write_json(output / 'manifest.json', {'status': 'data_prepared_verifier_acceptance_pending', 'config': config,
        'config_sha256': digest(config), 'tokenizer_fingerprint': token_hash,
        'judge_tokenizer_fingerprint': tokenizer_fingerprint(judge_tok) if judge_tok else None,
        'input_files': inputs, 'files': hashes, **balance, 'exclusions': dict(exclusions),
        'headroom_lt': {str(n): sum(0 < r['generation_max_tokens'] < n for r in index) for n in (1024, 4096, 8192)},
        'eval_policy': 'per-domain fixed IDs; disabled domains never change remaining eval selections',
        'group_policy': 'global normalized question (non-Pivot), exact prompt (Pivot), and published trajectory/instance connected components',
        'unselected_heldout_policy': 'remain held out in canonical.sqlite; never refill training'})
    db.close()
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default=str(ROOT / 'config/multidomain/initial.yaml'))
    ap.add_argument('--raw-dir', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--tokenizer')
    ap.add_argument('--judge-tokenizer')
    a = ap.parse_args()
    c = read_config(a.config)
    print(build(c, a.raw_dir, a.output_dir, a.tokenizer or c['training']['model_path'],
                a.judge_tokenizer or c['judge']['model_path']))

if __name__ == '__main__':
    main()
