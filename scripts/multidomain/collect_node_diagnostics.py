#!/usr/bin/env python3
"""Preserve current PBS allocation's Ray logs and resource snapshots on shared storage."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading

FILE_LIMIT = 4 << 20
ROUND_LIMIT = 256 << 20


def run_command(args):
    try:
        p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, errors='replace', timeout=8)
        return {'command': args, 'returncode': p.returncode, 'output': p.stdout[-(2 << 20):]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': args, 'error': str(exc)}


def read_small(path):
    try:
        with path.open() as f:
            return f.read(65536)
    except OSError as exc:
        return 'UNAVAILABLE: ' + str(exc)


def cgroup_memory():
    result = {}
    lines = read_small(Path('/proc/self/cgroup')).splitlines()
    for line in lines:
        fields = line.split(':', 2)
        if len(fields) != 3:
            continue
        _, controllers, relative = fields
        if controllers == '':
            mount = Path('/sys/fs/cgroup')
            names = ('memory.current', 'memory.max', 'memory.peak', 'memory.events', 'memory.events.local')
        elif 'memory' in controllers.split(','):
            mount = Path('/sys/fs/cgroup/memory')
            names = ('memory.usage_in_bytes', 'memory.limit_in_bytes', 'memory.max_usage_in_bytes',
                     'memory.failcnt', 'memory.oom_control')
        else:
            continue
        location = mount / relative.lstrip('/')
        # A PBS memory limit can be set on a parent cgroup.
        for _ in range(5):
            if not location.is_relative_to(mount):
                break
            for name in names:
                path = location / name
                if path.is_file():
                    result[str(path)] = read_small(path)
            if location == mount:
                break
            location = location.parent
    return result


def collect_logs(local_root, ray_root, destination, known):
    entries, candidates, seen = [], [], set()
    # The head uses --temp-dir=$RAY_TMPDIR; Ray's default on a worker
    # may append another /ray. Inspect both, only inside this PBS allocation.
    for base in (ray_root, ray_root / 'ray', local_root / 'multidomain/ray',
                 local_root / 'multidomain/ray/ray'):
        if not base.resolve().is_relative_to(local_root):
            continue
        for session in base.glob('session_*'):
            logs = (session / 'logs').resolve()
            if not logs.is_relative_to(local_root) or logs in seen or not logs.is_dir():
                continue
            seen.add(logs)
            for path in logs.rglob('*'):
                if path.is_symlink() or not path.is_file():
                    continue
                if path.suffix not in ('.out', '.err', '.log', '.json', '.txt'):
                    continue
                candidates.append(path)
    candidates.sort(key=lambda p: (0 if p.suffix == '.err' else
                                   1 if p.name.startswith('worker-') else
                                   2 if p.name.startswith('raylet') else 3, str(p)))
    total = 0
    for path in candidates:
        try:
            stat = path.stat()
            item = {'source': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
            total += min(stat.st_size, FILE_LIMIT)
            if total > ROUND_LIMIT:
                item['skipped'] = '256 MiB per-snapshot retained-log budget'
                entries.append(item)
                continue
            relative = path.relative_to(local_root)
            target = destination / 'ray_logs' / relative
            key = (stat.st_size, stat.st_mtime_ns)
            item.update({'saved': str(target.relative_to(destination)),
                         'truncated': stat.st_size > FILE_LIMIT})
            if known.get(str(path)) != key or not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                with path.open('rb') as f:
                    if stat.st_size <= FILE_LIMIT:
                        data = f.read(FILE_LIMIT)
                    else:
                        head = f.read(65536)
                        f.seek(-int(FILE_LIMIT - 65536), os.SEEK_END)
                        data = head + b'\n[DIAGNOSTICS: MIDDLE OMITTED; SEE INDEX FOR ORIGINAL SIZE]\n' + f.read(FILE_LIMIT - 65536)
                temporary = target.with_name(target.name + '.copying')
                temporary.write_bytes(data)
                temporary.replace(target)
                known[str(path)] = key
            entries.append(item)
        except OSError as exc:
            entries.append({'source': str(path), 'error': str(exc)})
    return entries


def snapshot(args, known, phase):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'phase': phase, 'host': socket.gethostname(), 'rank': args.rank,
              'pbs_job_id': os.environ.get('PBS_JOBID'),
              'diagnostics_only': True,
              'environment': {key: os.environ.get(key) for key in (
                  'VLLM_LOGGING_LEVEL', 'PYTHONFAULTHANDLER', 'CUDA_VISIBLE_DEVICES',
                  'PBS_LOCALDIR', 'RAY_TMPDIR', 'TMPDIR')},
              'memory': read_small(Path('/proc/meminfo')),
              'cgroup_memory': cgroup_memory(),
              'gpu': run_command(['nvidia-smi', '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu', '--format=csv']),
              'processes': run_command(['ps', '-u', str(os.getuid()), '-o', 'pid,ppid,stat,rss,etimes,comm']),
              'filesystems': run_command(['df', '-h', str(args.local_root), '/dev/shm'])}
    if phase != 'periodic':
        report['gpu_detail'] = run_command(['nvidia-smi', '-q'])
    entries = collect_logs(args.local_root, args.ray_root, output, known)
    report['ray_files_seen'] = len(entries)
    with (output / 'resource_snapshots.jsonl').open('a') as f:
        f.write(json.dumps(report, ensure_ascii=False) + '\n')
    temporary = output / 'ray_log_index.json.tmp'
    temporary.write_text(json.dumps({'utc': report['utc'], 'files': entries}, indent=2) + '\n')
    temporary.replace(output / 'ray_log_index.json')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--local-root', type=Path, required=True)
    ap.add_argument('--ray-root', type=Path, required=True)
    ap.add_argument('--rank', required=True)
    ap.add_argument('--watch', action='store_true')
    ap.add_argument('--parent-pid', type=int)
    ap.add_argument('--phase', default='manual')
    args = ap.parse_args()
    args.local_root = args.local_root.resolve()
    args.ray_root = args.ray_root.resolve()
    if not args.ray_root.is_relative_to(args.local_root):
        raise ValueError('Ray directory must belong to the current PBS local directory')
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    known = {}
    def capture(phase):
        try:
            snapshot(args, known, phase)
        except Exception as exc:
            print('Diagnostic collection failed:', repr(exc), flush=True)
    if not args.watch:
        capture(args.phase)
        return
    capture('startup')
    while not stop.wait(15):
        if args.parent_pid and os.getppid() != args.parent_pid:
            break
        capture('periodic')
    capture('watcher_exit')


if __name__ == '__main__':
    main()
