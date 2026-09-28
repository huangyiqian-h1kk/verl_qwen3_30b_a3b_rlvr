#!/usr/bin/env python3
"""Check real Ray actors on every allocated node before loading model weights."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time


def inspect_process(project_root, expected_versions):
    root = Path(project_root)
    expected_prefix = root / '.venv-multidomain'
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise RuntimeError(f'Wrong Python environment on {socket.gethostname()}: '
                           f'executable={sys.executable}, prefix={sys.prefix}, '
                           f'expected={expected_prefix}')
    modules = ('xmltodict', 'math_verify', 'latex2sympy2_extended',
               'reasoning_gym', 'openapi_schema_validator', 'yaml', 'pyarrow', 'aiohttp')
    locations = {name: str(importlib.import_module(name).__file__) for name in modules}
    versions = {name: importlib.metadata.version(name) for name in expected_versions}
    if versions != expected_versions:
        raise RuntimeError(f'Scoring dependency mismatch on {socket.gethostname()}: '
                           f'actual={versions}, expected={expected_versions}')
    from multidomain.reward import local_score
    from multidomain.template import parse_completion
    schema = json.dumps({'type': 'object', 'properties': {'value': {'type': 'integer'}}})
    for fmt, answer in [('json', '{"value":3}'), ('xml', '<value>3</value>')]:
        payload = {'source': 'if_structured', 'schema_type': fmt,
                   'schema_str': schema, 'response_mode': 'text'}
        outputs, error = parse_completion(answer)
        if error or local_score(payload, outputs) != 1:
            raise RuntimeError(f'Structured {fmt} scoring failed on {socket.gethostname()}')
    sys.path.insert(0, str(root / 'scripts/multidomain'))
    from check_reward_entry import check_entry
    math_entry = check_entry(root)
    return {'host': socket.gethostname(), 'python': sys.executable, 'prefix': sys.prefix,
            'versions': versions, 'module_files': locations, 'structured_json_xml': 'PASS',
            'math_reward_entry': math_entry}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', required=True, type=Path)
    a = ap.parse_args()
    request = json.loads(a.request.read_text())
    root = Path(request['config']['project_root'])
    sys.path.insert(0, str(root))
    from multidomain.dependencies import EXPECTED
    from multidomain.common import write_json
    destination = Path(request['job_dir']) / 'ray_environment.json'
    report = {'status': 'FAIL', 'scope': 'Python environment, structured scoring and real spawn math entry on every Ray node',
              'job_id': os.environ.get('PBS_JOBID'), 'nodes': []}
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    try:
        report['driver'] = inspect_process(str(root), EXPECTED)
        ray.init(address='auto', log_to_driver=True)
        deadline = time.monotonic() + 300
        while True:
            nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0)]
            if len(nodes) == request['expected_nodes'] and all(n['Resources']['GPU'] == 8 for n in nodes):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for all allocated Ray nodes with 8 GPUs each')
            time.sleep(2)
        # Actors with a runtime_env exercise the same worker startup category as
        # Runner/RewardLoopWorker. Hard affinity verifies each actual cluster node.
        @ray.remote(num_cpus=1, num_gpus=0, max_restarts=0)
        class EnvironmentProbe:
            def inspect(self, project_root, expected):
                return inspect_process(project_root, expected)

        actors = [EnvironmentProbe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(n['NodeID'], soft=False),
            runtime_env={'env_vars': {'MD_ENVIRONMENT_PROBE': '1'}}).remote() for n in nodes]
        try:
            reports = ray.get([actor.inspect.remote(str(root), EXPECTED) for actor in actors], timeout=180)
            report['nodes'] = [dict(item, node_id=node['NodeID']) for node, item in zip(nodes, reports)]
        finally:
            for actor in actors:
                ray.kill(actor, no_restart=True)
        report['status'] = 'PASS'
        print('[0390] Ray worker environment PASS: ' + json.dumps(report, ensure_ascii=False), flush=True)
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(destination, report)
        ray.shutdown()


if __name__ == '__main__':
    main()
