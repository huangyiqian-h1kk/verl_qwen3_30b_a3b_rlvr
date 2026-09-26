"""Audit project dependencies against the unchanged ABCI base environment.

Only the three exact, user-observed baseline diagnostics below can remain.
This is CPU dependency acceptance, never proof of GPU/runtime compatibility.
Uses only the standard library so it also runs before installing dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys


SCORING = frozenset({
    'math-verify', 'latex2sympy2-extended', 'antlr4-python3-runtime',
    'reasoning-gym', 'arckit', 'bfi', 'cellpylib', 'magiccube',
    'pycosat', 'pyfiglet', 'tabulate', 'zss',
    'openapi-schema-validator', 'xmltodict',
})
EXPECTED = {
    'math-verify': '0.8.0', 'latex2sympy2-extended': '1.10.2',
    'reasoning-gym': '0.1.25', 'openapi-schema-validator': '0.6.3',
    'xmltodict': '1.0.2',
}
KNOWN_BASELINE = {
    'outlines 0.1.11 has requirement outlines_core==0.1.26, but you have outlines-core 0.2.11.':
        {'outlines': '0.1.11', 'outlines-core': '0.2.11', 'vllm': '0.11.0'},
    'megatron-core 0.13.1 has requirement numpy<2.0.0, but you have numpy 2.2.6.':
        {'megatron-core': '0.13.1', 'numpy': '2.2.6'},
    'decord 0.6.0 is not supported on this platform': {'decord': '0.6.0'},
}
BOOTSTRAP = {'pip', 'setuptools', 'wheel'}
PROBE = r'''
import importlib.metadata as m
import json, re, sys
from pathlib import Path
packages = {}
for d in m.distributions():
    name = re.sub(r"[-_.]+", "-", d.metadata["Name"]).lower()
    if name not in packages:
        packages[name] = {"version": d.version, "location": str(Path(d.locate_file("")).resolve())}
print(json.dumps({"python": sys.executable, "prefix": sys.prefix, "packages": packages}, sort_keys=True))
'''


def snapshot(python):
    env = {**os.environ, 'PIP_DISABLE_PIP_VERSION_CHECK': '1', 'NO_COLOR': '1'}
    probe = subprocess.run([str(python), '-I', '-c', PROBE], env=env,
                           text=True, capture_output=True, check=True)
    data = json.loads(probe.stdout)
    check = subprocess.run([str(python), '-I', '-m', 'pip', 'check'], env=env,
                           text=True, capture_output=True)
    data['pip_check'] = {
        'returncode': check.returncode, 'stdout': check.stdout, 'stderr': check.stderr,
        'issues': [line.strip() for line in check.stdout.splitlines()
                   if line.strip() and line.strip() != 'No broken requirements found.'],
    }
    return data


def analyze_snapshots(before, after, project):
    errors = []
    if before['packages'] != after['packages']:
        errors.append('Base package versions or locations changed during installation.')
    for label, data in [('base before', before), ('base after', after), ('project', project)]:
        check = data['pip_check']
        if (check['returncode'] not in (0, 1)
                or bool(check['issues']) != bool(check['returncode'])):
            errors.append(f'{label}: pip check did not produce a valid dependency result.')
        for issue in check['issues']:
            expected = KNOWN_BASELINE.get(issue)
            if expected is None:
                errors.append(f'{label}: unrecognized dependency issue: {issue}')
            else:
                for name, version in expected.items():
                    if data['packages'].get(name, {}).get('version') != version:
                        errors.append(f'{label}: baseline issue package changed: {name}')
    baseline = set(before['pip_check']['issues'])
    if set(after['pip_check']['issues']) != baseline:
        errors.append('Base pip check diagnostics changed during installation.')
    for issue in sorted(set(project['pip_check']['issues']) - baseline):
        errors.append(f'Project introduced a dependency issue: {issue}')
    for name, base_package in before['packages'].items():
        if name not in SCORING | BOOTSTRAP:
            if project['packages'].get(name) != base_package:
                errors.append(f'Protected base package was replaced, shadowed, or lost: {name}')
    for name, version in EXPECTED.items():
        if project['packages'].get(name, {}).get('version') != version:
            errors.append(f'Project requires {name}=={version}.')
    status = ('FAIL' if errors else 'CPU_DEPENDENCIES_READY_WITH_INHERITED_CONFLICTS'
              if baseline else 'CPU_DEPENDENCIES_READY')
    return {
        'status': status, 'scope': 'CPU dependency preparation; GPU runtime not accepted here',
        'errors': errors, 'inherited_conflicts': sorted(baseline),
        'project_pip_check_issues': project['pip_check']['issues'],
        'base_only_diagnostics': sorted(baseline - set(project['pip_check']['issues'])),
        'note': 'A baseline-only platform diagnostic is retained, not declared resolved.',
        'scoring_versions': {name: project['packages'].get(name) for name in EXPECTED},
        'base_before': before, 'base_after': after, 'project': project,
    }


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='command', required=True)
    capture = sub.add_parser('capture')
    capture.add_argument('--python', required=True)
    capture.add_argument('--output', required=True)
    capture.add_argument('--constraints', required=True)
    check = sub.add_parser('check')
    check.add_argument('--base-python', required=True)
    check.add_argument('--project-python', required=True)
    check.add_argument('--before', required=True)
    check.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.command == 'capture':
        data = snapshot(args.python)
        write_json(args.output, data)
        lines = [f'{name}=={p["version"]}' for name, p in sorted(data['packages'].items())
                 if name not in SCORING | BOOTSTRAP]
        Path(args.constraints).write_text('\n'.join(lines) + '\n')
        print('Base dependency snapshot captured; training versions constrained.')
        return
    before = json.loads(Path(args.before).read_text())
    after, project = snapshot(args.base_python), snapshot(args.project_python)
    result = analyze_snapshots(before, after, project)
    expected_prefix = str(Path(args.project_python).parent.parent.resolve())
    if str(Path(project['prefix']).resolve()) != expected_prefix:
        result['errors'].append('Project interpreter is not using the expected virtual environment.')
        result['status'] = 'FAIL'
    write_json(args.output, result)
    print(result['status'])
    for issue in result['inherited_conflicts']:
        print('INHERITED (unresolved):', issue)
    for name, info in result['scoring_versions'].items():
        print(f'{name}: {info["version"] if info else "MISSING"}')
    print('Report:', args.output)
    if result['errors']:
        for error in result['errors']:
            print('ERROR:', error, file=sys.stderr)
        raise SystemExit(1)
    freeze = subprocess.run([args.project_python, '-I', '-m', 'pip', 'list', '--format=freeze',
                             '--disable-pip-version-check'], text=True, capture_output=True, check=True)
    Path(args.output).with_name('resolved.requirements.txt').write_text(freeze.stdout)
    print('CPU dependency check completed; GPU/model acceptance remains pending.')


if __name__ == '__main__':
    main()
