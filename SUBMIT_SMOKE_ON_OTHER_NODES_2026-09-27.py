#!/usr/bin/env python3
"""Submit one 0390 smoke job with automatic or explicit reserved-node allocation.

Use --auto-free-nodes to select two currently idle nii hosts, excluding hnode552.
Use --auto-nodes for unrestricted PBS selection, or --hosts for a fixed pair.
Uses ABCI's ordinary qsub wrapper. If it rejects the request, stop.
Reservation-listing failures are recorded without blocking normal PBS submission.
Checks the actual allocation and each node's /dev/shm before launching models.
Does not modify project source, acceptance files, dependencies or old outputs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

QUEUE = 'R9920261000'
RESERVATION = QUEUE + '.pbs1'
TOOL_VERSION = '2026-09-28.4'
SHM_GUARD = '#!/usr/bin/env python3\n"""Check every allocated node\'s /dev/shm before starting Ray or model workers.\n\n16 GiB is this project\'s startup headroom floor, not a vLLM capacity estimate.\nA 64 MiB mmap write checks actual backing pages, not just apparent file size.\nOnly the unique probe file created by this script is ever removed.\n"""\nimport argparse\nimport datetime\nimport hashlib\nimport json\nimport mmap\nimport os\nfrom pathlib import Path\nimport resource\nimport socket\nimport subprocess\nimport sys\nimport tempfile\nimport time\nimport uuid\n\nGIB = 1024 ** 3\nMIN_FREE = 16 * GIB\nPROBE_BYTES = 64 * 1024 ** 2\n\n\ndef write_probe(path, size):\n    # A SIGBUS must not generate a large core file or kill the checking parent.\n    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n    with open(path, \'r+b\', buffering=0) as stream:\n        stream.truncate(size)\n        with mmap.mmap(stream.fileno(), size) as buffer:\n            for offset in range(0, size, mmap.PAGESIZE):\n                buffer[offset] = 123\n            for offset in range(0, size, mmap.PAGESIZE):\n                if buffer[offset] != 123:\n                    raise RuntimeError(\'Shared-memory readback failed\')\n\n\ndef inspect_node(shm=Path(\'/dev/shm\'), minimum=MIN_FREE, probe_size=PROBE_BYTES):\n    report = {\'host\': socket.gethostname(), \'status\': \'FAIL\', \'path\': str(shm),\n              \'minimum_free_bytes\': minimum, \'probe_bytes\': probe_size,\n              \'created_utc\': datetime.datetime.now(datetime.timezone.utc).isoformat()}\n    probe = None\n    try:\n        stat = os.statvfs(shm)\n        report.update(total_bytes=stat.f_blocks * stat.f_frsize,\n                      free_bytes=stat.f_bavail * stat.f_frsize,\n                      free_inodes=stat.f_favail)\n        if report[\'free_bytes\'] < minimum:\n            raise RuntimeError(f"/dev/shm free {report[\'free_bytes\'] / GIB:.3f} GiB < startup floor {minimum / GIB:g} GiB")\n        if stat.f_favail == 0:\n            raise RuntimeError(\'/dev/shm has no free inodes\')\n        fd, name = tempfile.mkstemp(prefix=\'0390_shm_probe_\', dir=shm)\n        probe = Path(name)\n        os.close(fd)\n        result = subprocess.run([sys.executable, str(Path(__file__).resolve()),\n                                 \'--probe\', name, \'--probe-bytes\', str(probe_size)],\n                                capture_output=True, text=True, timeout=30)\n        report[\'probe_returncode\'] = result.returncode\n        if result.returncode != 0:\n            raise RuntimeError(f\'Shared-memory write failed (returncode={result.returncode}): {result.stderr[-2000:]}\')\n        report[\'status\'] = \'PASS\'\n    except Exception as exc:\n        report[\'reason\'] = f\'{type(exc).__name__}: {exc}\'\n    finally:\n        if probe is not None:\n            probe.unlink(missing_ok=True)\n    return report\n\n\ndef publish(directory, report):\n    directory.mkdir(parents=True, exist_ok=True)\n    path = directory / f"rank{report[\'rank\']}.json"\n    temp = path.with_suffix(f\'.{os.getpid()}.tmp\')\n    temp.write_text(json.dumps(report, indent=2) + \'\\n\')\n    temp.replace(path)\n\n\ndef wait_all(directory, expected, session, rank, nonce, timeout=180, poll=0.25):\n    deadline = time.monotonic() + timeout\n    while True:\n        reports = []\n        for peer in range(expected):\n            path = directory / f\'rank{peer}.json\'\n            if not path.exists():\n                continue\n            row = json.loads(path.read_text())\n            if row.get(\'session\') == session and row.get(\'rank\') == peer:\n                reports.append(row)\n        if len(reports) == expected:\n            if reports[rank][\'nonce\'] != nonce:\n                raise RuntimeError(\'Concurrent preflight for the same rank/job directory\')\n            # Every rank acknowledges the same fresh set of random nonces. Old\n            # reports/acks from a PBS requeue cannot satisfy this barrier.\n            token = hashlib.sha256(json.dumps(reports, sort_keys=True).encode()).hexdigest()\n            ack = directory / f\'ack{rank}.json\'\n            temp = ack.with_suffix(f\'.{os.getpid()}.tmp\')\n            temp.write_text(json.dumps({\'token\': token}) + \'\\n\')\n            temp.replace(ack)\n            if all((directory / f\'ack{peer}.json\').exists() and\n                   json.loads((directory / f\'ack{peer}.json\').read_text()).get(\'token\') == token\n                   for peer in range(expected)):\n                return reports\n        if time.monotonic() >= deadline:\n            received = [row[\'rank\'] for row in reports]\n            raise RuntimeError(f\'Shared-memory preflight timed out; received ranks {received}, expected {expected}. Ray was not started on this node.\')\n        time.sleep(poll)\n\n\ndef main():\n    ap = argparse.ArgumentParser(description=__doc__)\n    ap.add_argument(\'--request\', type=Path)\n    ap.add_argument(\'--rank\', type=int)\n    ap.add_argument(\'--probe\', type=Path)\n    ap.add_argument(\'--probe-bytes\', type=int, default=PROBE_BYTES)\n    args = ap.parse_args()\n    if args.probe is not None:\n        write_probe(args.probe, args.probe_bytes)\n        return 0\n    if args.request is None or args.rank is None:\n        ap.error(\'--request and --rank are required\')\n    request = json.loads(args.request.read_text())\n    expected = int(request[\'expected_nodes\'])\n    if not 0 <= args.rank < expected:\n        raise RuntimeError(\'Invalid allocated-node rank\')\n    job_id = os.environ.get(\'PBS_JOBID\')\n    if not job_id:\n        raise RuntimeError(\'Expected PBS_JOBID from the PBS launcher\')\n    session = hashlib.sha256(job_id.encode()).hexdigest()\n    directory = Path(request[\'job_dir\']) / \'preflight\' / (\'shared_memory_\' + session[:16])\n    report = inspect_node()\n    nonce = uuid.uuid4().hex\n    report.update(rank=args.rank, job_id=job_id, session=session, nonce=nonce)\n    publish(directory, report)\n    free = report.get(\'free_bytes\')\n    available = \'unknown\' if free is None else f\'{free / GIB:.3f} GiB\'\n    print(f"[0390] SHM {report[\'status\']} host={report[\'host\']} rank={args.rank} available={available}; {report.get(\'reason\', \'64 MiB mmap write/read passed\')}; report={directory / (\'rank\' + str(args.rank) + \'.json\')}", flush=True)\n    reports = wait_all(directory, expected, session, args.rank, nonce)\n    failed = [row for row in reports if row[\'status\'] != \'PASS\']\n    if failed:\n        print(\'[0390] SHM PREFLIGHT FAIL: no Ray/model startup. \' + \'; \'.join(\n            f"{row[\'host\']}: {row.get(\'reason\', \'failed\')}" for row in failed), flush=True)\n        return 2\n    if args.rank == 0:\n        print(f\'[0390] SHM PREFLIGHT PASS: all {expected} nodes passed; Ray startup permitted.\', flush=True)\n    return 0\n\n\nif __name__ == \'__main__\':\n    try:\n        sys.exit(main())\n    except Exception as error:\n        print(f\'[0390] SHM PREFLIGHT ERROR: {error}\', file=sys.stderr, flush=True)\n        sys.exit(2)\n'


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def scheduler_environment():
    return {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}


def scheduler_snapshot(command, job, stem):
    result = subprocess.run(command, text=True, capture_output=True,
                            env=scheduler_environment(), timeout=60)
    (job / (stem + '.stdout.txt')).write_text(result.stdout)
    (job / (stem + '.stderr.txt')).write_text(result.stderr)
    (job / (stem + '.query.json')).write_text(json.dumps({
        'command': command, 'returncode': result.returncode,
    }, indent=2) + '\n')
    need(result.returncode == 0,
         f'{command[0]} failed (rc={result.returncode}); no job submitted. '
         f'stdout={result.stdout[:1000]!r}, stderr={result.stderr[:1000]!r}; details: {job}')
    return result.stdout


def pbs_attributes(text):
    attributes = {}
    for line in text.splitlines():
        match = re.match(r'^\s+([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(.*?)\s*$', line)
        if match:
            attributes[match[1]] = match[2]
    return attributes


def resolve_resource(nodes, name, section, key, seen=None):
    seen = set() if seen is None else seen
    marker = (name, section, key)
    if marker in seen or name not in nodes:
        raise ValueError(f'Unresolvable PBS resource: {marker}')
    seen.add(marker)
    value = nodes[name].get(section, {}).get(key)
    if isinstance(value, str) and value.startswith('@'):
        return resolve_resource(nodes, value[1:], section, key, seen)
    return value


def resource_number(value):
    if value is None:
        return 0
    result = float(value)
    if result < 0 or result != result or result == float('inf'):
        raise ValueError(f'Invalid PBS numeric resource: {value!r}')
    return result


def resource_words(value):
    if value is None:
        return set()
    if isinstance(value, list):
        return {str(x).strip() for x in value}
    return {x.strip() for x in str(value).split(',') if x.strip()}


def idle_nii_hosts(payload, group='nii', excluded=('hnode552',)):
    """Use actual allocatable vnodes, not the host-level -aS aggregate.

    A 'free' vnode may still contain jobs, so inspect jobs and assigned resources.
    Every positive-CPU vnode must explicitly resolve to the queue's node_group.
    Scheduler eligibility and the actual /dev/shm health are checked separately.
    """
    nodes = payload.get('nodes')
    need(isinstance(nodes, dict) and bool(nodes), 'pbsnodes JSON has no nonempty nodes mapping')
    by_host = {}
    unparsed = []
    for name, row in nodes.items():
        try:
            host = resolve_resource(nodes, name, 'resources_available', 'host')
            if host is None:
                host = name.split('[')[0]
            host = str(host).split('.')[0]
            if not re.fullmatch(r'hnode\d+', host):
                unparsed.append(name)
                continue
            by_host.setdefault(host, []).append(name)
        except (ValueError, TypeError, AttributeError):
            unparsed.append(name)
    rows = []
    for host, names in sorted(by_host.items()):
        reasons = []
        ncpus = ngpus = 0
        leaves = []
        for name in names:
            row = nodes[name]
            try:
                cpus = resource_number(resolve_resource(nodes, name, 'resources_available', 'ncpus'))
                if cpus == 0:
                    continue
                leaves.append(name)
                ncpus += cpus
                ngpus += resource_number(resolve_resource(nodes, name, 'resources_available', 'ngpus'))
                groups = resource_words(resolve_resource(nodes, name, 'resources_available', 'node_group'))
                if group not in groups:
                    reasons.append('node_group_not_' + group)
                if resource_words(row.get('state')) != {'free'}:
                    reasons.append('vnode_not_free')
                if row.get('jobs'):
                    reasons.append('has_jobs')
                # Even if a sibling has no CPUs, jobs there still preclude full-host use.
                for key in ('ncpus', 'ngpus', 'njobs_rt_HF', 'njobs_rt_HG', 'njobs_rt_HC'):
                    if resource_number(resolve_resource(nodes, name, 'resources_assigned', key)) != 0:
                        reasons.append('resources_in_use')
                if row.get('resv'):
                    reasons.append('has_reservation')
                if row.get('queue') not in (None, '', QUEUE):
                    reasons.append('other_queue')
            except (ValueError, TypeError, AttributeError):
                reasons.append('unresolved_resource')
        # Distinguish a natural vnode with ncpus=0 from actual NUMA vnodes.
        # If both parent and children report CPU capacity, do not double count.
        if any('[' in name for name in leaves) and any('[' not in name for name in leaves):
            reasons.append('ambiguous_parent_and_child_capacity')
        if any(nodes[name].get('jobs') for name in names):
            reasons.append('has_jobs')
        if ncpus != 192 or ngpus != 8:
            reasons.append('not_one_192cpu_8gpu_host')
        if host in excluded:
            reasons.append('excluded_host')
        rows.append({'host': host, 'vnodes': leaves, 'ncpus': ncpus, 'ngpus': ngpus,
                     'eligible_candidate': not reasons, 'reasons': sorted(set(reasons))})
    candidates = [row['host'] for row in rows if row['eligible_candidate']]
    # Prefer previously observed healthy hosts when they are currently idle.
    # This is only a preference; startup preflight is always required.
    preferred = {'hnode539': 0, 'hnode624': 1, 'hnode589': 2}
    candidates.sort(key=lambda host: (preferred.get(host, 3), host))
    return candidates, rows, unparsed


def select_current_idle_hosts(job):
    queue_text = scheduler_snapshot(['qstat', '-Qf', QUEUE], job, 'queue_snapshot')
    attributes = pbs_attributes(queue_text)
    need(re.search(r'^Queue:\s*' + re.escape(QUEUE) + r'\s*$', queue_text, re.MULTILINE),
         'qstat did not identify the expected queue; no job submitted')
    need(attributes.get('default_chunk.node_group') == 'nii',
         'Queue node_group is missing or changed; no job submitted; see ' + str(job))
    need(attributes.get('enabled', '').lower() == 'true' and
         attributes.get('started', '').lower() == 'true', 'Queue is disabled or stopped; no job submitted')
    node_text = scheduler_snapshot(['pbsnodes', '-av', '-F', 'json'], job, 'nodes_snapshot')
    try:
        payload = json.loads(node_text)
    except ValueError as error:
        raise RuntimeError('pbsnodes did not return valid JSON; no job submitted; see ' + str(job)) from error
    candidates, rows, unparsed = idle_nii_hosts(payload)
    available = attributes.get('resources_available.ncpus')
    assigned = attributes.get('resources_assigned.ncpus')
    limit = attributes.get('resources_max.ncpus')
    cpu_headroom = None
    if available is not None and assigned is not None:
        ceiling = min(float(available), float(limit)) if limit is not None else float(available)
        cpu_headroom = int(ceiling - float(assigned))
    selected = candidates[:2] if len(candidates) >= 2 else []
    reasons = {}
    for row in rows:
        for reason in row['reasons']:
            reasons[reason] = reasons.get(reason, 0) + 1
    summary = {'mode': 'auto-free-nodes', 'queue': QUEUE, 'node_group': 'nii',
               'excluded_hosts': ['hnode552'], 'candidate_hosts': candidates,
               'selected_hosts': selected, 'queue_ncpus_headroom_snapshot': cpu_headroom,
               'required_ncpus': 384, 'rejected_host_reason_counts': reasons,
               'unparsed_vnodes': unparsed}
    (job / 'node_selection.json').write_text(json.dumps({**summary, 'hosts': rows}, indent=2) + '\n')
    (job / 'node_selection_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('[0390] Live node-selection summary:', json.dumps(summary), flush=True)
    print('[0390] Node-selection report:', job / 'node_selection_summary.json', flush=True)
    need(bool(selected), 'Fewer than two fully idle nii hosts after excluding hnode552; no job submitted. '
         'Run the same command later to take a fresh snapshot; no automatic retries.')
    need(cpu_headroom is None or cpu_headroom >= 384,
         f'Queue CPU headroom is {cpu_headroom}, below the 384 CPUs needed; no job submitted. '
         'Run the same command later to take a fresh snapshot.')
    print('[0390] Dynamically selected hosts:', ', '.join(selected), flush=True)
    print('[0390] This is a scheduler snapshot, not a reservation or a SHM health guarantee.', flush=True)
    return selected


def reservation_hosts(text):
    match = re.search(r'^\s*resv_nodes\s*=\s*(.*?)(?=\n\s*[A-Za-z_][A-Za-z0-9_.]*\s*=|\Z)',
                      text, re.MULTILINE | re.DOTALL)
    need(match is not None, 'qrstat did not return resv_nodes; cannot verify target nodes')
    unfolded = re.sub(r'\s+', '', match.group(1))
    hosts = set(re.findall(r'\b(hnode\d+)(?:\[|[.:+)]|$)', unfolded))
    need(bool(hosts), 'Could not parse reserved host names from resv_nodes')
    return hosts


def allocation_checker(hosts):
    if hosts is None:
        return '''import os
from pathlib import Path
actual = {line.strip().split('.')[0] for line in Path(os.environ['PBS_NODEFILE']).read_text().splitlines() if line.strip()}
print('[0390] PBS automatic allocation; allocated hosts:', sorted(actual), flush=True)
if len(actual) != 2:
    raise SystemExit('[0390] STOP: expected two distinct allocated nodes; no model startup.')
'''
    return '''import os
from pathlib import Path
expected = EXPECTED_HOSTS
actual = {line.strip().split('.')[0] for line in Path(os.environ['PBS_NODEFILE']).read_text().splitlines() if line.strip()}
print('[0390] requested hosts:', sorted(expected), 'allocated hosts:', sorted(actual), flush=True)
if actual != expected:
    raise SystemExit('[0390] STOP: allocated hosts differ from the request; no model startup. The qsub wrapper may not support host constraints.')
'''.replace('EXPECTED_HOSTS', repr(set(hosts)))


def prepare_job(root, job, hosts):
    request = json.loads((job / 'request.json').read_text())
    need(request['stage'] == 'smoke' and request['expected_nodes'] == 2,
         'Expected a two-node smoke request')
    sched = request['config']['scheduler']
    need((sched['project'], sched['queue'], sched['rtype']) == ('gcg51557', QUEUE, 'rt_HF'),
         'Unexpected scheduler configuration; preserved without submission')
    script = job / 'stage.pbs'
    original = script.read_text()
    required = ('#PBS -P gcg51557', '#PBS -q ' + QUEUE, '#PBS -v RTYPE=rt_HF',
                '#PBS -N 0390_md_smoke', '#PBS -j oe', '#PBS -k oe')
    need(all(line in original.splitlines() for line in required), 'Unexpected PBS header')
    need(original.splitlines().count('#PBS -l select=2') == 1, 'Unexpected select directive')
    launches = [line for line in original.splitlines() if line.startswith('mpirun ')]
    need(len(launches) == 1, 'Expected exactly one MPI launch')
    python = root / '.venv-multidomain/bin/python'
    need(python.is_file(), 'Missing project overlay Python: ' + str(python))
    prefix, separator, _ = launches[0].partition(' bash ')
    need(bool(separator), 'Unexpected node.sh launch format')
    q = shlex.quote
    (job / 'shared_memory_preflight.py').write_text(SHM_GUARD)
    (job / 'check_allocation.py').write_text(allocation_checker(hosts))
    wrapper = job / 'preflight_node.sh'
    wrapper.write_text('#!/bin/bash\nset -euo pipefail\nexec ' + q(str(python)) + ' ' +
                       q(str(job / 'shared_memory_preflight.py')) + ' --request ' +
                       q(str(job / 'request.json')) + ' --rank "${OMPI_COMM_WORLD_RANK:?}"\n')
    checks = q(str(python)) + ' ' + q(str(job / 'check_allocation.py')) + '\n'
    checks += prefix + ' bash ' + q(str(wrapper)) + '\n'
    modified = original
    if hosts is not None:
        modified = original.replace('#PBS -l select=2\n',
                                    '#PBS -l select=' + '+'.join('1:host=' + host for host in hosts) + '\n', 1)
    modified = modified.replace(launches[0], checks + launches[0], 1)
    (job / 'stage.automatic_nodes.pbs').write_text(original)
    script.write_text(modified)
    subprocess.run(['bash', '-n', str(script)], check=True)
    return script


def main():
    print('[0390] Alternate-node submitter version:', TOOL_VERSION, flush=True)
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument('--hosts', nargs=2, default=['hnode589', 'hnode624'])
    mode.add_argument('--auto-nodes', action='store_true',
                      help='Let PBS allocate any two eligible reserved nodes; check SHM before startup')
    mode.add_argument('--auto-free-nodes', action='store_true',
                      help='Select two currently idle nii hosts, excluding hnode552; one qsub only')
    ap.add_argument('--config', default='config/multidomain/initial.yaml')
    ap.add_argument('--data-id', default='all_six_v2_ep8')
    ap.add_argument('--run-id', default='qwen30b_d6_uniform_ctx64k_seed42_v2_ep8_pkg')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    root = Path.cwd().resolve()
    need((root / 'multidomain/submit.py').is_file(), 'Run from the data_mixture_rl project root')
    if args.auto_free_nodes:
        args.hosts = None
        print('[0390] Node selection: fresh PBS snapshot, nii hosts, excluding hnode552.', flush=True)
    elif args.auto_nodes:
        args.hosts = None
        print('[0390] Node selection: automatic (select=2); no hostname constraints.', flush=True)
    else:
        need(len(set(args.hosts)) == 2 and all(re.fullmatch(r'hnode\d+', h) for h in args.hosts),
             'Specify two different hnode names')
        need('hnode552' not in args.hosts, 'This workaround excludes hnode552')
        # Scheduler wrappers must not import the project's NumPy compatibility hook.
        reservation = subprocess.run(['qrstat', '-f', RESERVATION], text=True, capture_output=True, env={k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')})
        # Listing a reservation is informative; qsub/PBS remains the authority for
        # queue access and resource eligibility. A failed listing is not a PASS.
        available = None
        if reservation.returncode == 0:
            try:
                available = reservation_hosts(reservation.stdout)
            except RuntimeError as error:
                print('[0390] Reservation listing could not be parsed:', error, flush=True)
        if available is not None:
            need(set(args.hosts) <= available,
                 'Requested hosts are not both in the current reservation. Reserved hosts: ' + ', '.join(sorted(available)))
            print('Reservation membership verified:', ', '.join(args.hosts), flush=True)
        else:
            print(f'[0390] Reservation membership not verified by qrstat: rc={reservation.returncode}, '
                  f'stdout={reservation.stdout!r}, stderr={reservation.stderr!r}. '
                  'Continuing with the normal qsub command and reserved queue; allocation and SHM checks remain enabled.', flush=True)
    generate = subprocess.run([sys.executable, '-m', 'multidomain.submit', '--stage', 'smoke',
                               '--config', args.config, '--data-id', args.data_id,
                               '--run-id', args.run_id, '--dry-run'], text=True, capture_output=True)
    need(generate.returncode == 0, generate.stdout + generate.stderr)
    match = re.search(r'^PBS script: (.+)$', generate.stdout, re.MULTILINE)
    need(match is not None, 'Could not identify generated PBS script:\n' + generate.stdout)
    script = Path(match.group(1)).resolve()
    need(script.is_relative_to(root / 'outputs/multidomain') and script.name == 'stage.pbs',
         'Unexpected generated PBS path')
    job = script.parent
    if args.auto_free_nodes:
        args.hosts = select_current_idle_hosts(job)
    elif not args.auto_nodes:
        (job / 'reservation_at_submission.txt').write_text(reservation.stdout)
        (job / 'reservation_query.json').write_text(json.dumps({
            'command': ['qrstat', '-f', RESERVATION], 'returncode': reservation.returncode,
            'stdout': reservation.stdout, 'stderr': reservation.stderr,
            'membership_verified': available is not None,
        }, indent=2) + '\n')
    script = prepare_job(root, job, args.hosts)
    print('PBS script:', script, flush=True)
    print('Requested hosts:', 'automatic (PBS chooses two)' if args.hosts is None else ', '.join(args.hosts), flush=True)
    print('Log:', job / 'pbs.log', flush=True)
    if args.dry_run:
        print(script.read_text())
        return
    submitted = subprocess.run(['qsub', str(script)], text=True, capture_output=True, env={k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')})
    (job / 'qsub.stdout.txt').write_text(submitted.stdout)
    (job / 'qsub.stderr.txt').write_text(submitted.stderr)
    need(submitted.returncode == 0,
         f'qsub rejected the request (exit code {submitted.returncode}); no fallback or automatic retry was attempted.\nstdout: {submitted.stdout!r}\nstderr: {submitted.stderr!r}')
    (job / 'job_id.txt').write_text(submitted.stdout)
    print('Submitted:', submitted.stdout.strip(), flush=True)
    if submitted.stderr:
        print(submitted.stderr, file=sys.stderr)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('[0390] STOP:', error, file=sys.stderr, flush=True)
        sys.exit(2)
