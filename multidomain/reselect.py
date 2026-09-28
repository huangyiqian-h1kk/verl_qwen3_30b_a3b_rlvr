"""Derive an optional-domain subset from a prepared full canonical pool."""
import copy
import json
import sqlite3
from pathlib import Path

from multidomain.build import choose, export_parquet
from multidomain.common import DataError, digest, file_sha, write_json
from multidomain.preparation.balance_index import select


def comparable(config):
    value = copy.deepcopy(config)
    value.pop('resolved', None)
    value.pop('project_root', None)
    value['acceptance'].pop('infrastructure_nodes', None)
    for domain in value['domains'].values():
        domain.pop('enabled', None)
    return value


def reselect(config, source, output, expected_manifest_sha256):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise DataError('Derived data must have a separate directory')
    if output.exists() and any(output.iterdir()):
        raise DataError('Derived output is not empty; keep it for diagnosis and choose a new data ID')
    manifest_path = source / 'manifest.json'
    if not expected_manifest_sha256 or file_sha(manifest_path) != expected_manifest_sha256:
        raise DataError('Parent manifest differs from the explicitly selected accepted dataset')
    parent = json.loads(manifest_path.read_text())
    if digest(parent['config']) != parent['config_sha256']:
        raise DataError('Parent config checksum mismatch')
    if comparable(parent['config']) != comparable(config):
        raise DataError('Reuse permits only domain disabling, project path and infrastructure-node changes')
    active = set(config['resolved']['enabled_domains'])
    previous = set(parent['config']['resolved']['enabled_domains'])
    if not active < previous or config['resolved']['judge_enabled']:
        raise DataError('This derivation requires a strict domain subset with science disabled')
    for name, checksum in parent['files'].items():
        if Path(name).name != name or file_sha(source / name) != checksum:
            raise DataError('Parent parquet checksum mismatch: ' + name)
    index_path, canonical = source / 'index.jsonl', source / 'canonical.sqlite'
    initial_hashes = {p.name: file_sha(p) for p in (index_path, canonical)}
    index = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
    if len({row['sample_id'] for row in index}) != len(index):
        raise DataError('Duplicate sample IDs in the parent index')
    selected, balance = choose(index, config)
    # Anchor reuse to the accepted six-domain selection, including full pool counts.
    old_selected, old_balance = choose(index, parent['config'])
    for key in ('eligible_pool_counts', 'selected_counts', 'selected_source_counts'):
        if old_balance[key] != parent[key]:
            raise DataError('Parent index no longer reproduces its manifest: ' + key)
    import pyarrow.parquet as pq
    for split in ('train', 'validation', 'test'):
        old_rows = pq.read_table(source / (split + '.parquet'), columns=['sample_id', 'domain']).to_pylist()
        if [row['sample_id'] for row in old_rows] != old_selected[split + '_ids']:
            raise DataError('Parent index no longer reproduces accepted IDs: ' + split)
        if split != 'train':
            expected = [row['sample_id'] for row in old_rows if row['domain'] in active]
            if selected[split + '_ids'] != expected:
                raise DataError('Held-out IDs changed in a retained domain: ' + split)
    with sqlite3.connect(canonical.as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('pragma quick_check').fetchone()[0] != 'ok':
            raise DataError('Parent SQLite integrity check failed')
        if {row[0] for row in db.execute('select id from samples')} != {row['sample_id'] for row in index}:
            raise DataError('Canonical store and index IDs differ')
        output.mkdir(parents=True, exist_ok=True)
        for split in ('train', 'validation', 'test'):
            export_parquet(db, selected[split], output / (split + '.parquet'))
            write_json(output / (split + '_ids.json'), selected[split + '_ids'])
        smoke, baseline = [], []
        for domain in config['resolved']['enabled_domains']:
            pool = [r for r in selected['train'] if r['domain'] == domain]
            random_rows = select(pool, min(8, len(pool)), config['seed'], 'smoke/' + domain)
            longest = sorted(pool, key=lambda r: (-r['prompt_tokens'], r['sample_id']))[:4]
            smoke.extend({r['sample_id']: r for r in random_rows + longest}.values())
            baseline.extend(select(pool, min(64, len(pool)), config['seed'], 'baseline/' + domain))
        export_parquet(db, smoke, output / 'smoke.parquet')
        export_parquet(db, baseline, output / 'baseline_train.parquet')
    # No writes to, copies of, or symlinks into the source canonical database.
    # The training/evaluation parquets are self-contained; provenance retains its path.
    for path in (index_path, canonical):
        if file_sha(path) != initial_hashes[path.name]:
            raise DataError('Parent pool changed during derivation')
    if file_sha(manifest_path) != expected_manifest_sha256:
        raise DataError('Parent manifest changed during derivation')
    write_json(output / 'manifest.json', {
        'status': 'data_prepared_verifier_acceptance_pending', 'config': config,
        'config_sha256': digest(config), 'tokenizer_fingerprint': parent['tokenizer_fingerprint'],
        'judge_tokenizer_fingerprint': None, 'input_files': parent['input_files'],
        'files': {p.name: file_sha(p) for p in output.glob('*.parquet')}, **balance,
        'derived_from': {'directory': str(source), 'manifest_sha256': expected_manifest_sha256,
                         'pool_hashes_at_derivation': initial_hashes,
                         'pool_hash_scope': 'recorded now; parent manifest did not hash the canonical pool',
                         'selection_reproduced': True, 'retained_domain_eval_ids_unchanged': True},
        'exclusions_inherited_from_parent_all_domains': parent.get('exclusions', {}),
        'headroom_lt': {str(n): sum(0 < r['generation_max_tokens'] < n for r in index
                                  if r['domain'] in active) for n in (1024, 4096, 8192)},
        'eval_policy': parent['eval_policy'], 'group_policy': parent['group_policy'],
        'unselected_heldout_policy': 'remain in parent canonical pool; never refill training'})
    print(json.dumps({'derived_data': str(output), **balance}, ensure_ascii=False, indent=2), flush=True)
    return output
