"""Download pinned tokenizer files or weights on an Internet-connected machine."""
import argparse
import json
from pathlib import Path
from multidomain.common import ROOT, file_sha, write_json


def main():
    from huggingface_hub import snapshot_download, HfApi
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=('policy', 'judge'), required=True)
    ap.add_argument('--destination', required=True)
    ap.add_argument('--weights', action='store_true', help='Include all safetensor weight shards; can be hundreds of GB')
    ap.add_argument('--verify-existing-weights', action='store_true', help='Verify existing shards against pinned HF SHA256 metadata without downloading them')
    a = ap.parse_args()
    lock = json.loads((ROOT / 'config/multidomain/sources.lock.json').read_text())
    info = lock['dependencies'][a.model]
    patterns = ['config.json', 'generation_config.json', '*token*', '*vocab*', 'merges.txt', 'chat_template*', '*.jinja']
    if a.weights:
        patterns += ['*.safetensors', '*.safetensors.index.json']
    destination = Path(a.destination)
    snapshot_download(repo_id=info['repo_id'], revision=info['revision'], allow_patterns=patterns, local_dir=destination)
    files = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': file_sha(p)}
             for p in destination.iterdir() if p.is_file() and p.name != 'multidomain_asset_manifest.json'
             and (a.weights or a.verify_existing_weights or p.suffix != '.safetensors')}
    verified = False
    if a.weights or a.verify_existing_weights:
        remote = HfApi().model_info(info['repo_id'], revision=info['revision'], files_metadata=True)
        expected = {f.rfilename: f.lfs.sha256 for f in remote.siblings if f.rfilename.endswith('.safetensors') and f.lfs}
        if not expected:
            raise RuntimeError('Pinned model has no safetensor checksum metadata')
        for name, sha in expected.items():
            if name not in files or files[name]['sha256'] != sha:
                raise RuntimeError('Missing or wrong model shard: ' + name)
        verified = True
    write_json(destination / 'multidomain_asset_manifest.json', {'repo_id': info['repo_id'], 'revision': info['revision'],
        'weights_revision_verified': verified, 'files': files})
    print(destination)

if __name__ == '__main__':
    main()
