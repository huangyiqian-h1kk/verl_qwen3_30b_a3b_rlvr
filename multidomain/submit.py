"""Generate an exact PBS script and submit it; --dry-run prints it without qsub."""
import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from multidomain.common import ROOT, read_config, write_json

STAGES = ('accept-four', 'prepare-data', 'check-verifiers', 'check-infra', 'judge-calibration', 'smoke', 'backward', 'baseline', 'train', 'evaluate')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=STAGES)
    ap.add_argument('--config', default='config/multidomain/initial.yaml')
    ap.add_argument('--data-id', default='all_six_v1')
    ap.add_argument('--run-id', default='qwen30b_d6_uniform_ctx64k_seed42_v1')
    ap.add_argument('--raw-dir', default=str(ROOT / 'data/raw'))
    ap.add_argument('--calibration-file')
    ap.add_argument('--source-data-dir')
    ap.add_argument('--source-manifest-sha256')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    for value in (a.data_id, a.run_id):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
            ap.error('IDs may contain only letters, digits, underscores, dots and hyphens')
    c = read_config(a.config)
    if Path(c['project_root']).resolve() != ROOT:
        ap.error(f'config.project_root must be this checkout: {ROOT}')
    if bool(a.source_data_dir) != bool(a.source_manifest_sha256):
        ap.error('--source-data-dir and --source-manifest-sha256 must be supplied together')
    if a.stage == 'judge-calibration' and not c['resolved']['judge_enabled']:
        ap.error('Science is disabled; no judge calibration is needed')
    if a.stage == 'accept-four' and (c['resolved']['domain_count'] != 4 or c['resolved']['judge_enabled'] or c['domains']['swe_pivot']['enabled'] or not a.source_data_dir):
        ap.error('accept-four requires four_domain.yaml and the accepted parent pool')
    cpu = a.stage in ('prepare-data', 'check-verifiers')
    nodes = 1 if cpu or a.stage == 'judge-calibration' else c['resolved']['pbs_nodes_for_training_job']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    job = ROOT / 'outputs/multidomain' / a.run_id / 'jobs' / (stamp + '_' + a.stage)
    job.mkdir(parents=True, exist_ok=False)
    request = {'stage': a.stage, 'config': c, 'data_dir': str(ROOT / 'data/multidomain' / a.data_id),
               'raw_dir': str(Path(a.raw_dir).resolve()), 'run_dir': str(job.parents[1]), 'job_dir': str(job),
               'calibration_file': str(Path(a.calibration_file).resolve()) if a.calibration_file else None,
               'expected_nodes': nodes,
               'source_data_dir': str(Path(a.source_data_dir).resolve()) if a.source_data_dir else None,
               'source_manifest_sha256': a.source_manifest_sha256}
    write_json(job / 'request.json', request)
    q = shlex.quote
    sched = c['scheduler']
    label = {'accept-four': 'accept', 'prepare-data': 'data', 'check-verifiers': 'verify', 'check-infra': 'infra', 'judge-calibration': 'judge'}.get(a.stage, a.stage)
    job_name = f'0390_d{c["resolved"]["domain_count"]}_{label}'
    lines = ['#!/bin/bash', f'#PBS -P {sched["project"]}', f'#PBS -q {sched["queue"]}',
             f'#PBS -v RTYPE={sched["rtype"]}', f'#PBS -l select={nodes}',
             f'#PBS -l walltime={"01:00:00" if a.stage == "check-infra" else sched["walltime"]}',
             f'#PBS -N {job_name}', '#PBS -j oe', '#PBS -k oe', 'set -euo pipefail',
             f'exec > >(tee -a {q(str(job / "pbs.log"))}) 2>&1', 'export PYTHONUNBUFFERED=1',
             f'cd {q(str(ROOT))}', 'if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi',
             'module load gcc/13.2.0 cuda/12.8/12.8.1 cudnn/9.10/9.10.2 nccl/2.29/2.29.7-1', 'source /home/aci18769hm/opt/miniforge3/etc/profile.d/conda.sh',
             f'conda activate {q(c["environment_prefix"])}']
    if cpu:
        lines += [f'source {q(str(ROOT / "scripts/multidomain/env.sh"))}',
                  'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',
                  f'python -m multidomain.stage --request {q(str(job / "request.json"))}']
    else:
        lines += ['module load hpcx/2.20', 'module load nccl/2.29/2.29.7-1',
                  f'awk \'!seen[$0]++\' "$PBS_NODEFILE" > {q(str(job / "hosts"))}',
                  f'[[ $(wc -l < {q(str(job / "hosts"))}) -eq {nodes} ]]',
                  f'mpirun -np {nodes} --map-by ppr:1:node --bind-to none --hostfile {q(str(job / "hosts"))} '
                  f'bash {q(str(ROOT / "scripts/multidomain/node.sh"))} {q(str(ROOT))} {q(str(job / "request.json"))} {q(str(job))}']
    script = job / 'stage.pbs'
    script.write_text('\n'.join(lines) + '\n')
    print('PBS script:', script)
    if a.dry_run:
        print(script.read_text())
    else:
        result = subprocess.run(['qsub', str(script)], text=True, capture_output=True, check=True,
                                env={k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')})
        (job / 'job_id.txt').write_text(result.stdout)
        print('Submitted:', result.stdout.strip())
        print('Log:', job / 'pbs.log')

if __name__ == '__main__':
    main()
