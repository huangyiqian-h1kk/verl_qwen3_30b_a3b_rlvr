"""Fail before GPU allocation or model loading when inputs differ from the frozen run."""
import importlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from multidomain.common import ROOT, DataError, digest, file_sha


def check(config, data_dir, models=False):
    actual = subprocess.check_output(['git', '-C', config['verl_src'], 'rev-parse', 'HEAD'], text=True).strip()
    if actual != config['verl_commit']:
        raise DataError(f'verl commit mismatch: {actual}')
    for name, version in [('reasoning-gym', '0.1.25'), ('math-verify', '0.8.0')]:
        if importlib.metadata.version(name) != version:
            raise DataError(f'{name} must be {version}')
    for name in ('yaml', 'pyarrow', 'transformers', 'openapi_schema_validator', 'xmltodict', 'aiohttp'):
        importlib.import_module(name)
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    if manifest['config_sha256'] != digest(config):
        raise DataError('Experiment config differs from the data manifest; use a new data ID')
    for name, expected in manifest['files'].items():
        if file_sha(data_dir / name) != expected:
            raise DataError('Prepared dataset checksum mismatch: ' + name)
    if models:
        from transformers import AutoTokenizer
        from multidomain.template import tokenizer_fingerprint
        for key, expected in [('training', manifest['tokenizer_fingerprint']), ('judge', manifest['judge_tokenizer_fingerprint'])]:
            if key == 'judge' and not config['resolved']['judge_enabled']:
                continue
            folder = Path(config[key]['model_path'])
            tok = AutoTokenizer.from_pretrained(folder, local_files_only=True)
            if tokenizer_fingerprint(tok) != expected:
                raise DataError(f'{key} tokenizer differs from the data build')
            asset_file = folder / 'multidomain_asset_manifest.json'
            if not asset_file.is_file():
                raise DataError('Verify pinned model assets first: ' + str(folder))
            assets = json.loads(asset_file.read_text())
            lock = json.loads((ROOT / 'config/multidomain/sources.lock.json').read_text())['dependencies']['policy' if key == 'training' else 'judge']
            if not assets.get('weights_revision_verified') or assets['revision'] != lock['revision']:
                raise DataError('Run multidomain.assets with --weights or --verify-existing-weights: ' + str(folder))
            for name, meta in assets['files'].items():
                file = folder / name
                if not file.is_file() or file.stat().st_size != meta['bytes'] or file.stat().st_mtime_ns != meta['mtime_ns']:
                    raise DataError('Model asset changed since verification: ' + str(file))
            index = folder / 'model.safetensors.index.json'
            if not index.is_file():
                raise DataError('Missing sharded model index: ' + str(index))
            weights = set(json.loads(index.read_text())['weight_map'].values())
            missing = [name for name in weights if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
            if missing:
                raise DataError(f'{key} missing weight shards: {missing[:5]}')
            model_config = json.loads((folder / 'config.json').read_text())
            if key == 'judge' and model_config.get('quantization_config', {}).get('quant_method') not in ('fp8', 'compressed-tensors'):
                raise DataError('Judge directory is not the configured native FP8 model')
    return manifest
