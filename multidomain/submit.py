"""Generate an exact PBS script and submit it; --dry-run prints it without qsub."""
import argparse
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path
from multidomain.common import ROOT, read_config, write_json

STAGES = ('prepare-data', 'check-verifiers', 'check-infra', 'judge-calibration', 'smoke', 'backward', 'baseline', 'train', 'evaluate')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--stage', required=True, choices=STAGES)
    ap.add_argument('--config', default='config/multidomain/initial.yaml')
    ap.add_argument('--data-id', default='all_six_v1')
    ap.add_argument('--run-id', default='qwen30b_d6_uniform_ctx64k_seed42_v1')
    ap.add_argument('--raw-dir', default=str(ROOT / 'data/raw'))
    ap.add_argument('--calibration-file')
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    for value in (a.data_id, a.run_id):
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', value) or value in ('.', '..'):
            ap.error('IDs may contain only letters, digits, underscores, dots and hyphens')
    c = read_config(a.config)
    if Path(c['project_root']).resolve() != ROOT:
        ap.error(f'config.project_root must be this checkout: {ROOT}')
    cpu = a.stage in ('prepare-data', 'check-verifiers')
    nodes = 1 if cpu or a.stage == 'judge-calibration' else 2 if a.stage == 'check-infra' else c['resolved']['pbs_nodes_for_training_job']
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    job = ROOT / 'outputs/multidomain' / a.run_id / 'jobs' / (stamp + '_' + a.stage)
    job.mkdir(parents=True, exist_ok=False)
    request = {'stage': a.stage, 'config': c, 'data_dir': str(ROOT / 'data/multidomain' / a.data_id),
               'raw_dir': str(Path(a.raw_dir).resolve()), 'run_dir': str(job.parents[1]), 'job_dir': str(job),
               'calibration_file': str(Path(a.calibration_file).resolve()) if a.calibration_file else None,
               'expected_nodes': nodes}
    write_json(job / 'request.json', request)
    q = shlex.quote
    sched = c['scheduler']
    lines = ['#!/bin/bash', f'#PBS -P {sched["project"]}', f'#PBS -q {sched["queue"]}',
             f'#PBS -v RTYPE={sched["rtype"]}', f'#PBS -l select={nodes}:mpiprocs=1',
             f'#PBS -l walltime={"01:00:00" if a.stage == "check-infra" else sched["walltime"]}',
             f'#PBS -N md_{a.stage}', '#PBS -j oe', f'#PBS -o {job / "pbs.log"}', 'set -euo pipefail',
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
        result = subprocess.run(['qsub', str(script)], text=True, capture_output=True, check=True)
        (job / 'job_id.txt').write_text(result.stdout)
        print('Submitted:', result.stdout.strip())
        print('Log:', job / 'pbs.log')

if __name__ == '__main__':
    main()
