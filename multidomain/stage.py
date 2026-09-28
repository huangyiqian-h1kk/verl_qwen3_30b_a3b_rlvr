"""One stage per PBS allocation. Reports are bound to code, config, and prepared data."""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from multidomain.common import ROOT, digest, file_sha, write_json


def code_fingerprint():
    paths = list((ROOT / 'multidomain').rglob('*.py')) + list((ROOT / 'config/multidomain').glob('*')) + list((ROOT / 'multidomain/_vendor').glob('*.txt')) + list((ROOT / 'multidomain/_vendor').glob('*.json'))
    return digest({str(p.relative_to(ROOT)): file_sha(p) for p in sorted(paths) if p.is_file()})


def require(reports, stages, fingerprint):
    for stage in stages:
        path = reports / (stage + '.json')
        if not path.is_file():
            raise RuntimeError('Missing acceptance report; run --stage ' + stage)
        report = json.loads(path.read_text())
        if report.get('status') != 'PASS' or report.get('fingerprint') != fingerprint:
            raise RuntimeError('Acceptance is failed or stale; rerun --stage ' + stage)


def wait_nodes(count):
    import ray
    ray.init(address='auto', ignore_reinit_error=True)
    deadline = time.monotonic() + 300
    while True:
        nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0)]
        if len(nodes) == count and all(n['Resources']['GPU'] == 8 for n in nodes):
            return
        if time.monotonic() > deadline:
            raise RuntimeError(f'Expected {count} Ray nodes with 8 GPUs each')
        time.sleep(2)


def execute(req):
    from multidomain.preflight import check
    stage, c = req['stage'], req['config']
    data_dir, run_dir = Path(req['data_dir']), Path(req['run_dir'])
    reports = run_dir / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    # Single writer per run ID, including baseline/test/checkpoint operations.
    lock = (run_dir / 'stage.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    base = {'config': digest(c), 'code': code_fingerprint(), 'data_id': data_dir.name}
    fingerprint = digest(base)
    report = {'status': 'FAIL', 'stage': stage, 'fingerprint': fingerprint, 'inputs': base,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'job_id': os.getenv('PBS_JOBID'),
              'optimizer_steps': 'training' if stage == 'train' else 0}
    try:
        if stage == 'prepare-data':
            if req.get('source_data_dir'):
                from multidomain.reselect import reselect
                reselect(c, req['source_data_dir'], data_dir, req['source_manifest_sha256'])
            else:
                from multidomain.build import build
                build(c, req['raw_dir'], data_dir, c['training']['model_path'], c['judge']['model_path'])
            if c['resolved']['judge_enabled']:
                from multidomain.calibration import make_template
                make_template(data_dir, run_dir / 'review/science_128.jsonl')
        elif stage == 'check-infra':
            subprocess.run([sys.executable, '-m', 'multidomain.check_infra', '--expected-nodes', str(req['expected_nodes']),
                            '--output', str(reports / 'infrastructure_detail.json')], check=True)
        else:
            require(reports, ['prepare-data'], fingerprint)
            manifest = check(c, data_dir, models=stage not in ('check-verifiers',))
            report['manifest_sha256'] = file_sha(data_dir / 'manifest.json')
            for previous in ('check-verifiers', 'judge-calibration', 'smoke', 'backward', 'baseline', 'train'):
                path = reports / (previous + '.json')
                if path.exists() and previous != stage:
                    old = json.loads(path.read_text())
                    if old.get('status') == 'PASS' and old.get('manifest_sha256') != report['manifest_sha256']:
                        raise RuntimeError('Prepared data changed since acceptance; use a new run ID')
            if stage == 'check-verifiers':
                from multidomain.verify import check as verify
                verify(data_dir, reports / 'verifiers_detail.json')
                import asyncio
                from multidomain.check_framework import check as check_framework
                asyncio.run(check_framework(c, data_dir))
                subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests/multidomain'), '-v'], check=True)
            else:
                require(reports, ['check-infra', 'check-verifiers'], fingerprint)
                wait_nodes(req['expected_nodes'])
                from multidomain.runtime_config import compose
                from omegaconf import OmegaConf
                if stage == 'judge-calibration':
                    if not req.get('calibration_file'):
                        raise ValueError('Complete the human review in review/science_128.jsonl; pass --calibration-file PATH')
                    from multidomain.calibration import run
                    cfg = compose(c, data_dir, run_dir, 'smoke')
                    run(cfg, req['calibration_file'], reports / 'judge_calibration_detail.json')
                else:
                    if c['resolved']['judge_enabled']:
                        require(reports, ['judge-calibration'], fingerprint)
                    if stage in ('backward', 'baseline', 'train', 'evaluate'):
                        require(reports, ['smoke'], fingerprint)
                    if stage in ('train', 'evaluate'):
                        require(reports, ['backward', 'baseline'], fingerprint)
                    if stage == 'evaluate':
                        require(reports, ['train'], fingerprint)
                    substages = ['baseline-train', 'baseline-val'] if stage == 'baseline' else [stage]
                    for substage in substages:
                        model_path = None
                        if stage == 'evaluate':
                            root = run_dir / 'checkpoints'
                            step = int((root / 'latest_checkpointed_iteration.txt').read_text().strip())
                            checkpoint = root / f'global_step_{step}' / 'actor'
                            spec = json.loads((checkpoint / 'ckpt_contents.json').read_text())
                            # Canonical location at the pinned verl commit.
                            model_path = str(checkpoint / 'huggingface')
                            if not (Path(model_path) / 'config.json').is_file():
                                candidates = list(checkpoint.glob('**/huggingface/config.json'))
                                if len(candidates) != 1:
                                    raise RuntimeError('Final HF export missing or ambiguous')
                                model_path = str(candidates[0].parent)
                        cfg = compose(c, data_dir, run_dir, substage, model_path)
                        path = Path(req['job_dir']) / (substage + '.resolved.yaml')
                        OmegaConf.save(cfg, path, resolve=True)
                        # Separate Ray job per substage ensures GPU actors and pools are released.
                        subprocess.run([sys.executable, '-m', 'multidomain.launch', '--config', str(path),
                            '--stage', substage, '--run-dir', str(run_dir), '--data-dir', str(data_dir)], check=True)
        report['status'] = 'PASS'
    except BaseException as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(reports / (stage + '.json'), report)
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        lock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', required=True)
    req = json.loads(Path(ap.parse_args().request).read_text())
    if req['stage'] != 'accept-four':
        execute(req)
        return
    c = req['config']
    if set(c['resolved']['enabled_domains']) != {'math', 'if', 'conversational_pivot', 'logic_algorithmic'} or req['expected_nodes'] != 1:
        raise ValueError('accept-four requires the four-domain single-node configuration')
    # A restart uses only matching, successful reports from this NEW experiment.
    from multidomain.preflight import check
    reports = Path(req['run_dir']) / 'reports'
    fp = digest({'config': digest(c), 'code': code_fingerprint(), 'data_id': Path(req['data_dir']).name})
    for stage in ('prepare-data', 'check-verifiers', 'check-infra', 'smoke'):
        report_file = reports / (stage + '.json')
        old = json.loads(report_file.read_text()) if report_file.exists() else {}
        matches = old.get('status') == 'PASS' and old.get('fingerprint') == fp
        if matches:
            if stage != 'check-infra':
                check(c, Path(req['data_dir']), models=False)
                if stage != 'prepare-data' and old.get('manifest_sha256') != file_sha(Path(req['data_dir']) / 'manifest.json'):
                    raise RuntimeError('Prepared data changed after acceptance')
            print('[0390] Reuse matching four-domain PASS: ' + stage, flush=True)
            continue
        print('[0390] Running four-domain stage: ' + stage, flush=True)
        execute(dict(req, stage=stage))
    print('[0390] FOUR-DOMAIN ACCEPTANCE PASS; optimizer_steps=0', flush=True)

if __name__ == '__main__':
    main()
