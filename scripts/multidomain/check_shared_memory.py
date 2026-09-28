#!/usr/bin/env python3
"""Check every allocated node's /dev/shm before starting Ray or model workers.

16 GiB is this project's startup headroom floor, not a vLLM capacity estimate.
A 64 MiB mmap write checks actual backing pages, not just apparent file size.
Only the unique probe file created by this script is ever removed.
"""
import argparse
import datetime
import hashlib
import json
import mmap
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tempfile
import time
import uuid

GIB = 1024 ** 3
MIN_FREE = 16 * GIB
PROBE_BYTES = 64 * 1024 ** 2


def write_probe(path, size):
    # A SIGBUS must not generate a large core file or kill the checking parent.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    with open(path, 'r+b', buffering=0) as stream:
        stream.truncate(size)
        with mmap.mmap(stream.fileno(), size) as buffer:
            for offset in range(0, size, mmap.PAGESIZE):
                buffer[offset] = 123
            for offset in range(0, size, mmap.PAGESIZE):
                if buffer[offset] != 123:
                    raise RuntimeError('Shared-memory readback failed')


def inspect_node(shm=Path('/dev/shm'), minimum=MIN_FREE, probe_size=PROBE_BYTES):
    report = {'host': socket.gethostname(), 'status': 'FAIL', 'path': str(shm),
              'minimum_free_bytes': minimum, 'probe_bytes': probe_size,
              'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    probe = None
    try:
        stat = os.statvfs(shm)
        report.update(total_bytes=stat.f_blocks * stat.f_frsize,
                      free_bytes=stat.f_bavail * stat.f_frsize,
                      free_inodes=stat.f_favail)
        if report['free_bytes'] < minimum:
            raise RuntimeError(f"/dev/shm free {report['free_bytes'] / GIB:.3f} GiB < startup floor {minimum / GIB:g} GiB")
        if stat.f_favail == 0:
            raise RuntimeError('/dev/shm has no free inodes')
        fd, name = tempfile.mkstemp(prefix='0390_shm_probe_', dir=shm)
        probe = Path(name)
        os.close(fd)
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                 '--probe', name, '--probe-bytes', str(probe_size)],
                                capture_output=True, text=True, timeout=30)
        report['probe_returncode'] = result.returncode
        if result.returncode != 0:
            raise RuntimeError(f'Shared-memory write failed (returncode={result.returncode}): {result.stderr[-2000:]}')
        report['status'] = 'PASS'
    except Exception as exc:
        report['reason'] = f'{type(exc).__name__}: {exc}'
    finally:
        if probe is not None:
            probe.unlink(missing_ok=True)
    return report


def publish(directory, report):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"rank{report['rank']}.json"
    temp = path.with_suffix(f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(report, indent=2) + '\n')
    temp.replace(path)


def wait_all(directory, expected, session, rank, nonce, timeout=180, poll=0.25):
    deadline = time.monotonic() + timeout
    while True:
        reports = []
        for peer in range(expected):
            path = directory / f'rank{peer}.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            if row.get('session') == session and row.get('rank') == peer:
                reports.append(row)
        if len(reports) == expected:
            if reports[rank]['nonce'] != nonce:
                raise RuntimeError('Concurrent preflight for the same rank/job directory')
            # Every rank acknowledges the same fresh set of random nonces. Old
            # reports/acks from a PBS requeue cannot satisfy this barrier.
            token = hashlib.sha256(json.dumps(reports, sort_keys=True).encode()).hexdigest()
            ack = directory / f'ack{rank}.json'
            temp = ack.with_suffix(f'.{os.getpid()}.tmp')
            temp.write_text(json.dumps({'token': token}) + '\n')
            temp.replace(ack)
            if all((directory / f'ack{peer}.json').exists() and
                   json.loads((directory / f'ack{peer}.json').read_text()).get('token') == token
                   for peer in range(expected)):
                return reports
        if time.monotonic() >= deadline:
            received = [row['rank'] for row in reports]
            raise RuntimeError(f'Shared-memory preflight timed out; received ranks {received}, expected {expected}. Ray was not started on this node.')
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--request', type=Path)
    ap.add_argument('--rank', type=int)
    ap.add_argument('--probe', type=Path)
    ap.add_argument('--probe-bytes', type=int, default=PROBE_BYTES)
    args = ap.parse_args()
    if args.probe is not None:
        write_probe(args.probe, args.probe_bytes)
        return 0
    if args.request is None or args.rank is None:
        ap.error('--request and --rank are required')
    request = json.loads(args.request.read_text())
    expected = int(request['expected_nodes'])
    if not 0 <= args.rank < expected:
        raise RuntimeError('Invalid allocated-node rank')
    job_id = os.environ.get('PBS_JOBID')
    if not job_id:
        raise RuntimeError('Expected PBS_JOBID from the PBS launcher')
    session = hashlib.sha256(job_id.encode()).hexdigest()
    directory = Path(request['job_dir']) / 'preflight' / ('shared_memory_' + session[:16])
    report = inspect_node()
    nonce = uuid.uuid4().hex
    report.update(rank=args.rank, job_id=job_id, session=session, nonce=nonce)
    publish(directory, report)
    free = report.get('free_bytes')
    available = 'unknown' if free is None else f'{free / GIB:.3f} GiB'
    print(f"[0390] SHM {report['status']} host={report['host']} rank={args.rank} available={available}; {report.get('reason', '64 MiB mmap write/read passed')}; report={directory / ('rank' + str(args.rank) + '.json')}", flush=True)
    reports = wait_all(directory, expected, session, args.rank, nonce)
    failed = [row for row in reports if row['status'] != 'PASS']
    if failed:
        print('[0390] SHM PREFLIGHT FAIL: no Ray/model startup. ' + '; '.join(
            f"{row['host']}: {row.get('reason', 'failed')}" for row in failed), flush=True)
        return 2
    if args.rank == 0:
        print(f'[0390] SHM PREFLIGHT PASS: all {expected} nodes passed; Ray startup permitted.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[0390] SHM PREFLIGHT ERROR: {error}', file=sys.stderr, flush=True)
        sys.exit(2)
