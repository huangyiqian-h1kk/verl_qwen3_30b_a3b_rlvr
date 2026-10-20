#!/usr/bin/env python3
"""Coordinate the two existing 0390 owners; preserve their handoff protocol.

Standard-library only. This module is also frozen beside each installed entry.
It never cancels a PBS job, edits a training snapshot, or deletes a checkpoint.
"""
import argparse
from contextlib import ExitStack, contextmanager, nullcontext
import datetime as dt
import fcntl
import getpass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

VERSION = '2026-10-20.capacity.2'
PREVIOUS_VERSION = '2026-10-20.capacity.1'
SUPPORTED_VERSIONS = {PREVIOUS_VERSION, VERSION}
# Exact previously published adapter, accepted for an in-place operational upgrade.
PREVIOUS_SHA256 = '7c490748fbf33183c89c632dd8be3d840586c8ef5aafed7f0f93774b77f71f22'
LIMITS_VERSION = '2026-10-19.monitor30-training24.1'
BASE = 'qwen30b_d4_10k_uniform_ctx64k_seed42_v1'
QUEUE = 'R9920261000'
RESERVATION = QUEUE + '.pbs1'
ROLES = {'domain': ('_db_n2_once_v1', 'd4_db_switch'),
         'random': ('_n2_once_v1', 'd4_managed_switch')}
PRE = {'WAIT_CHECK', 'WAIT_CHECKPOINT', 'WAIT_READY'}
UTC = dt.timezone.utc


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def now():
    return dt.datetime.now(UTC)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with tmp.open('w') as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def active(path):
    try:
        with locked(path):
            return False
    except BlockingIOError:
        return True


def controls(root):
    return {kind: Path(root) / 'outputs/multidomain' / (BASE + suffix) / 'handoff'
            for kind, (suffix, _) in ROLES.items()}


def area(root):
    return Path(root) / 'outputs/multidomain' / (BASE + '_capacity')


def command(args, root=None, timeout=20):
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME')}
    env['LC_ALL'] = 'C'
    result = subprocess.run(args, cwd=root, env=env, text=True, capture_output=True, timeout=timeout)
    need(result.returncode == 0, '{}: exit status {}; stdout={!r}; stderr={!r}'.format(
        ' '.join(map(str, args)), result.returncode, result.stdout[:600], result.stderr[:600]))
    return result.stdout


def attributes(text):
    """PBS long-format attributes, including indented resv_nodes continuations."""
    result, key = {}, None
    for line in text.splitlines():
        match = re.match(r'^\s+([\w.]+)\s*=\s*(.*)$', line)
        if match:
            key = match[1]
            result[key] = match[2].strip()
        elif key and line[:1].isspace():
            result[key] += line.strip()
        else:
            key = None
    return result


def reservation_records(text):
    """Split full reservation records before reading attributes; never merge them."""
    records, current = [], []
    identity = None
    for line in text.splitlines():
        match = re.match(r'^\s*(?:Resv\s+ID|Reservation\s+Id)\s*:\s*(\S+)\s*$', line, re.I)
        if match:
            if current:
                row = attributes('\n'.join(current))
                if row:
                    records.append(dict(row, reservation_id=identity))
            current, identity = [], match[1]
        else:
            current.append(line)
    row = attributes('\n'.join(current))
    if row:
        records.append(dict(row, reservation_id=identity))
    return records


def listed_reservations(text):
    """Use the actual Resv ID / Queue columns, including a different PBS server."""
    result = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1] == QUEUE and re.fullmatch(r'[RS]\d+(?:\.[\w.-]+)?', fields[0]):
            result.append(fields[0])
    return list(dict.fromkeys(result))


def reservation_for_queue(cached_id=None):
    """Discover queue binding; every accepted detailed record must name QUEUE."""
    attempts, ids = [], []
    deadline = time.monotonic() + 25

    def query(args):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError('Reservation lookup timed out. ' + ' | '.join(attempts))
        try:
            text = command(args, timeout=min(10, remaining))
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
            attempts.append(str(exc))
            return None
        records = [r for r in reservation_records(text) if r.get('queue') == QUEUE and r.get('resv_nodes')]
        need(len(records) <= 1, 'Multiple reservation records bind the same queue; refusing to select one')
        if records:
            return records[0]
        ids.extend(listed_reservations(text))
        attempts.append(' '.join(args) + ': no detailed record for ' + QUEUE)
        return None

    if cached_id:
        found = query(['qrstat', '-f', cached_id])
        if found:
            return dict(found, reservation_id=found.get('reservation_id') or cached_id)
    found = query(['qrstat'])
    if found:
        return found
    # Query the exact IDs observed in the scheduler listing first.
    for rid in dict.fromkeys(ids):
        found = query(['qrstat', '-f', rid])
        if found:
            return dict(found, reservation_id=found.get('reservation_id') or rid)
    for args in (['qrstat', '-f'], ['qrstat', '-F']):
        found = query(args)
        if found:
            return found
    # Some site wrappers accept the short identifier. These are query probes,
    # not assumed bindings: a successful result still must name QUEUE explicitly.
    for rid in dict.fromkeys([*ids, QUEUE, RESERVATION]):
        for flag in ('-f', '-F'):
            args = ['qrstat', flag, rid]
            if any(a.startswith(' '.join(args) + ':') for a in attempts):
                continue
            found = query(args)
            if found:
                return dict(found, reservation_id=found.get('reservation_id') or rid)
    raise RuntimeError('Reservation lookup unavailable. ' + ' | '.join(attempts))


def words(value):
    if isinstance(value, list):
        return {str(x).strip() for x in value if str(x).strip()}
    return {x.strip() for x in str(value or '').split(',') if x.strip()}


def hold_types(value):
    return set(str(value or 'n').replace(',', '').replace(' ', '')) - {'n'}


def number(value):
    value = float(value or 0)
    need(math.isfinite(value) and value >= 0, 'Invalid PBS resource value')
    return value


def resource(nodes, name, section, key, seen=None):
    seen = set() if seen is None else seen
    marker = (name, section, key)
    need(marker not in seen and name in nodes, 'Unresolvable indirect vnode resource')
    seen.add(marker)
    value = nodes[name].get(section, {}).get(key)
    if isinstance(value, str) and value.startswith('@'):
        return resource(nodes, value[1:], section, key, seen)
    return value


def hosts_from_exec(value):
    return set(re.findall(r'(?:^|[+(])\s*(hnode\d+)(?:\[\d+\])?:', str(value or '')))


def free_hosts(nodes_payload, reservation, queue):
    """Count physical reserved 192-CPU/8-GPU hosts, never NUMA vnodes."""
    nodes = nodes_payload.get('nodes')
    need(isinstance(nodes, dict) and nodes, 'pbsnodes returned no vnode mapping')
    need(queue.get('enabled', '').lower() == 'true'
         and queue.get('started', '').lower() == 'true', 'Reserved queue is not enabled/started')
    need(reservation.get('queue') == QUEUE, 'Reservation does not bind the expected queue')
    need(reservation.get('reserve_state') in ('RESV_RUNNING', '5', 'RN'),
         'Reservation is not running')
    reserved = set(re.findall(r'(hnode\d+(?:\[\d+\])?):', reservation.get('resv_nodes', '')))
    need(reserved, 'Reservation has no parseable resv_nodes; do not infer capacity from cluster totals')
    group = queue.get('default_chunk.node_group')
    need(group, 'Queue node_group is missing')
    groups = {}
    for name in nodes:
        host = str(resource(nodes, name, 'resources_available', 'host') or name.split('[')[0]).split('.')[0]
        if re.fullmatch(r'hnode\d+', host):
            groups.setdefault(host, []).append(name)
    free, details = [], {}
    for host, names in groups.items():
        if not any(n in reserved or host in reserved for n in names):
            continue
        reasons, leaves, cpus, gpus = set(), [], 0, 0
        for name in names:
            row = nodes[name]
            ncpu = number(resource(nodes, name, 'resources_available', 'ncpus'))
            if ncpu:
                leaves.append(name)
                cpus += ncpu
                gpus += number(resource(nodes, name, 'resources_available', 'ngpus'))
                if name not in reserved and host not in reserved:
                    reasons.add('partially_reserved_host')
                if group not in words(resource(nodes, name, 'resources_available', 'node_group')):
                    reasons.add('wrong_node_group')
            if words(row.get('state')) != {'free'}:
                reasons.add('not_free')
            if row.get('jobs'):
                reasons.add('has_jobs')
            if any(number(resource(nodes, name, 'resources_assigned', key)) for key in
                   ('ncpus', 'ngpus', 'njobs_rt_HF', 'njobs_rt_HG', 'njobs_rt_HC')):
                reasons.add('assigned_resources')
            if row.get('queue') not in (None, '', QUEUE):
                reasons.add('other_queue')
            # Belonging to THIS reservation is expected, not a rejection reason.
            own_reservations = {QUEUE, str(reservation.get('reservation_id') or '').split('.')[0]}
            if any(r.split('.')[0] not in own_reservations for r in words(row.get('resv'))):
                reasons.add('other_reservation')
        if cpus != 192 or gpus != 8:
            reasons.add('not_one_whole_gpu_host')
        if any('[' in n for n in leaves) and any('[' not in n for n in leaves):
            reasons.add('double_counted_parent')
        details[host] = sorted(reasons)
        if not reasons:
            free.append(host)
    free.sort()
    headroom = None
    if 'resources_available.ncpus' in queue:
        headroom = max(0, number(queue['resources_available.ncpus'])
                       - number(queue.get('resources_assigned.ncpus')))
    slots = len(free) if headroom is None else min(len(free), int(headroom // 192))
    return dict(free_hosts=free, free_slots=slots, hosts=details,
                queue_ncpus_headroom=headroom, reserved_hosts=sorted(details))


def validate_job(job, row, root, kind, role):
    suffix = ROLES[kind][0] if role == 'dual' else ('_db_v1' if kind == 'domain' else '')
    folder = Path(job['folder']).resolve()
    need(folder.parent == Path(root) / 'outputs/multidomain' / (BASE + suffix) / 'jobs',
         'Job directory is outside the controlled run')
    expected = ('0390_d4dbn2_train' if kind == 'domain' else '0390_d4n2_once') if role == 'dual' else (
        '0390_d4db_train' if kind == 'domain' else '0390_d4_train')
    need(row.get('Job_Owner', '').split('@')[0] == getpass.getuser()
         and row.get('queue') == QUEUE and row.get('Job_Name') == expected,
         'PBS job owner/queue/name does not match the registered controller')
    need(str(row.get('Submit_arguments', '')).replace('\n', '').replace('\t', '') == str(folder / 'stage.pbs'),
         'PBS job script does not match the registered snapshot')
    resources = row.get('Resource_List', {})
    need(int(resources.get('nodect', 0)) == (2 if role == 'dual' else 1), 'Unexpected job node count')


def decision(kind, states, rows, capacity):
    """Pure policy. Capacity includes our live allocations; queued demand is reserved."""
    state = states[kind]
    if state['phase'] not in PRE:
        return False, 'handoff_already_started'
    if state.get('handoff_blocked') or state.get('check_blocked'):
        return False, 'existing_controller_block'
    if any(s.get('pending_job') or s.get('pending_check') for s in states.values()):
        return False, 'submission_recovery_in_progress'
    for other, s in states.items():
        if other != kind and s['phase'] in PRE:
            if rows.get((other, 'dual'), {}).get('job_state') in ('R', 'E', 'S', 'B'):
                return False, 'other_candidate_already_started'
    if kind == 'random':
        domain = states['domain']
        if domain['phase'] != 'DUAL_ACTIVE' or not domain.get('handoff_completed_utc'):
            return False, 'domain_handoff_has_priority'
        if rows.get(('domain', 'dual'), {}).get('job_state') != 'R':
            return False, 'domain_primary_not_running'
    shortfall = 0
    for owner, s in states.items():
        if s['phase'] == 'DONE':
            continue
        primary = 'single' if s['phase'] in PRE else 'dual'
        row = rows.get((owner, primary), {})
        if row.get('job_state') != 'R':
            shortfall += 1 if primary == 'single' else 2
    if capacity['available_to_experiment'] < 4:
        return False, 'fewer_than_four_available_nodes'
    if capacity['free_slots'] < 2 + shortfall:
        return False, 'keep_capacity_for_primary_jobs_and_two_node_candidate'
    return True, 'domain_first' if kind == 'domain' else 'domain_active_and_spare_capacity'


class Policy:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.area = area(self.root)
        self.controls = controls(self.root)
        self.reservation_id = None

    def jobs(self):
        payload = json.loads(command(['qstat', '-f', '-F', 'json']))
        need(isinstance(payload, dict) and ('Jobs' in payload or 'pbs_server' in payload), 'Invalid qstat JSON')
        return payload.get('Jobs', {})

    def observe(self, job, kind, role):
        result = json.loads(command(['qstat', '-xf', '-F', 'json', job['job_id']]))
        row = result['Jobs'][job['job_id']]
        validate_job(job, row, self.root, kind, role)
        return row

    def capacity(self, rows):
        resv = reservation_for_queue(self.reservation_id)
        self.reservation_id = resv.get('reservation_id')
        queue_text = command(['qstat', '-Qf', QUEUE])
        need(re.search(r'^Queue:\s*' + QUEUE + r'\s*$', queue_text, re.M), 'Unexpected queue response')
        nodes = json.loads(command(['pbsnodes', '-av', '-F', 'json']))
        result = free_hosts(nodes, resv, attributes(queue_text))
        occupied = set()
        for row in rows.values():
            if row.get('job_state') not in ('R', 'E', 'S', 'B'):
                continue
            hosts = hosts_from_exec(row.get('exec_vnode'))
            need(len(hosts) == int(row['Resource_List']['nodect']), 'Cannot identify current physical allocation')
            occupied.update(hosts)
        occupied &= set(result['reserved_hosts'])
        need(not occupied.intersection(result['free_hosts']), 'Inconsistent qstat/pbsnodes snapshot; retry later')
        result.update(owned_hosts=sorted(occupied), available_to_experiment=result['free_slots'] + len(occupied),
                      reservation_id=resv.get('reservation_id'))
        return result

    def set_hold(self, kind, job, allow, journal):
        """Only our recorded user hold is released; running jobs are left alone."""
        row = self.observe(job, kind, 'dual')
        jid, folder = job['job_id'], Path(job['folder'])
        held = hold_types(row.get('Hold_Types'))
        receipt = folder / 'capacity_hold.json'
        if receipt.exists():
            marker = read(receipt)
            need(marker['version'] in SUPPORTED_VERSIONS and marker['kind'] == kind
                 and marker['script_sha256'] == sha(folder / 'stage.pbs'), 'Candidate hold receipt changed')
            # #PBS -h acquired this hold at initial submission, before any crash.
            if jid not in journal and 'u' in held:
                journal[jid] = dict(kind=kind, status='HELD', initial_submission=True)
        if row.get('job_state') not in ('Q', 'H', 'W'):
            return
        owned = journal.get(jid, {}).get('status') in ('INTENT', 'HELD', 'RELEASE_INTENT')
        if allow:
            if not owned or 'u' not in held:
                return
            # Do not touch operator/system holds or another user's original hold.
            if held != {'u'}:
                return
            journal[jid]['status'] = 'RELEASE_INTENT'
            write(self.area / 'holds.json', journal)
            command(['qrls', '-h', 'u', jid])
            journal[jid]['status'] = 'RELEASED'
        else:
            if held:
                return
            journal[jid] = dict(kind=kind, status='INTENT', at_utc=now().isoformat())
            write(self.area / 'holds.json', journal)
            command(['qhold', '-h', 'u', jid])
            after = self.observe(job, kind, 'dual')
            # PBS may dispatch between query and hold. Remove our hold if it did.
            if after.get('job_state') == 'R':
                command(['qrls', '-h', 'u', jid])
                journal[jid]['status'] = 'RELEASED'
            else:
                journal[jid]['status'] = 'HELD'
        write(self.area / 'holds.json', journal)

    def reconcile(self, mutate=True):
        # Read-only status never acquires the mutation lock. Each state file is
        # atomically published; the report is observational, not a release permit.
        with (locked(self.area / 'policy.lock') if mutate else nullcontext()):
            states = {k: read(p / 'state.json') for k, p in self.controls.items()}
            jobs, rows = self.jobs(), {}
            for kind, state in states.items():
                for role in ('single', 'dual'):
                    job = state.get(role)
                    if job and job['job_id'] in jobs:
                        row = jobs[job['job_id']]
                        validate_job(job, row, self.root, kind, role)
                        rows[kind, role] = row
            error = None
            try:
                capacity = self.capacity(rows)
            except Exception as exc:
                capacity, error = None, str(exc)
            choices = {k: decision(k, states, rows, capacity) if capacity else (False, 'capacity_query_unknown')
                       for k in ROLES}
            report = dict(version=VERSION, at_utc=now().isoformat(), queue=QUEUE,
                          read_only=not mutate,
                          capacity=capacity, query_error=error,
                          jobs={k: {role: dict(job_id=states[k].get(role, {}).get('job_id'),
                                      state=rows.get((k, role), {}).get('job_state'),
                                      hold=rows.get((k, role), {}).get('Hold_Types'),
                                      comment=rows.get((k, role), {}).get('comment'))
                                    for role in ('single', 'dual') if states[k].get(role)} for k in ROLES},
                          choices={k: dict(allow=v[0], reason=v[1]) for k, v in choices.items()})
            if mutate:
                path = self.area / 'holds.json'
                journal = read(path) if path.exists() else {}
                # Denials first: never release domain while a queued random can start.
                for allow in (False, True):
                    for kind in ROLES:
                        state = states[kind]
                        if state['phase'] not in PRE or not state.get('dual') or choices[kind][0] != allow:
                            continue
                        self.set_hold(kind, state['dual'], allow, journal)
                write(self.area / 'status.json', report)
            return report


def hold_script(text, folder, kind):
    folder = Path(folder)
    need(not (folder / 'qsub_attempted.json').exists(), 'Refuse to change a submitted script')
    need('#PBS -q ' + QUEUE in text.splitlines(), 'Candidate is not in reserved queue')
    need(not re.search(r'^#PBS\s+-[hc](?:\s|$)', text, re.M), 'Candidate already has a hold/checkpoint directive')
    # This is PBS checkpointing only; application model/optimizer checkpoints are unchanged.
    name = '0390_d4dbn2_train' if kind == 'domain' else '0390_d4n2_once'
    # The existing create_job also performs this rename after rendering. Do it
    # here too so the hold receipt describes the final bytes it will hash.
    text = text.replace('#PBS -N 0390_d4n2_train\n', '#PBS -N ' + name + '\n')
    result = text.replace('\n', '\n#PBS -h\n#PBS -c n\n', 1)
    write(folder / 'capacity_hold.json', dict(version=VERSION, kind=kind,
          script_sha256=hashlib.sha256(result.encode()).hexdigest(), at_utc=now().isoformat()))
    return result


def install_hooks(root, kind):
    module = importlib.import_module(ROLES[kind][1] + '.controller')
    cls = module.Controller
    if getattr(cls, '_0390_capacity', False):
        return
    policy = Policy(root)
    original_submit, original_tick = cls.submit, cls.tick
    pbs = importlib.import_module('d4_two_node_resumefix.scheduler')
    original_render = pbs.render_pbs
    context = {'first': False}

    def render(project, folder, action, walltime, hosts=()):
        text = original_render(project, folder, action, walltime, hosts)
        if context['first']:
            need(action == 'train' and Path(folder).parent == controls(root)[kind].parent / 'jobs',
                 'Unexpected candidate renderer')
            return hold_script(text, folder, kind)
        return text

    def submit(self, first):
        context['first'] = bool(first)
        try:
            return original_submit(self, first)
        finally:
            context['first'] = False

    def maintenance(self):
        try:
            report = policy.reconcile()
            self.save(capacity_policy_version=VERSION, capacity_error=report['query_error'],
                      capacity_decision=report['choices'][kind], capacity_policy_busy=False)
        except BlockingIOError:
            # The other owner is performing the same shared policy update.
            # This is expected contention, not a resource query failure.
            self.save(capacity_policy_busy=True)
        except Exception as exc:
            # Resource queries must never disable the existing single-owner retry loop.
            self.save(capacity_error=str(exc))

    def tick(self):
        start, end = self.state.get('window_started_utc'), self.state.get('deadline_utc')
        if start and end:
            self.save(monitor_window_hours=(dt.datetime.fromisoformat(end) - dt.datetime.fromisoformat(start)).total_seconds() / 3600)
        maintenance(self)
        original_tick(self)  # Owns all retry, readiness, cancellation, and fallback decisions.
        maintenance(self)

    cls.submit, cls.tick, cls._0390_capacity = submit, tick, True
    pbs.render_pbs = render


def entry_text(module_path, expected):
    return ('from pathlib import Path\nimport hashlib, importlib.util, sys\n'
            'p=Path(' + repr(str(module_path)) + ')\n'
            'if hashlib.sha256(p.read_bytes()).hexdigest()!=' + repr(expected) + ': raise RuntimeError("Capacity policy changed")\n'
            's=importlib.util.spec_from_file_location("d4_capacity_policy",p)\n'
            'm=importlib.util.module_from_spec(s);s.loader.exec_module(m)\n'
            'm.boot(Path(__file__).resolve().parent,sys.argv[1:])\n')


def limits_entry(cfg):
    return ('from pathlib import Path\nimport hashlib, json, sys\n'
            'here=Path(__file__).resolve().parent\n'
            'policy=here/"runtime_limits_30h_24h"\n'
            'if hashlib.sha256((policy/"limits_policy.py").read_bytes()).hexdigest()!=' + repr(cfg['policy_sha256']) + ': raise RuntimeError("Limits policy changed")\n'
            'sys.path.insert(0,str(policy))\nfrom limits_policy import boot\n'
            'boot(here,sys.argv[1:])\n')


def validate_control(root, kind, control, allow_pending=False):
    plan, state = read(control / 'plan.json'), read(control / 'state.json')
    need(digest(plan['contract']) == plan['identity'] == state['identity'], 'Controller contract differs')
    need(allow_pending or (not state.get('pending_job') and not state.get('pending_check')),
         'Submission pending: let the current owner finish/adopt it before refreshing')
    cfg = read(control / 'runtime_limits_30h_24h/config.json')
    need(cfg['version'] == LIMITS_VERSION and cfg['kind'] == kind and Path(cfg['root']) == root
         and cfg['contract_identity'] == plan['identity'], 'Expected existing 30h/24h adapter is missing or changed')
    need(sha(control / 'runtime_limits_30h_24h/limits_policy.py') == cfg['policy_sha256'], 'Limits adapter changed')
    for name, expected in cfg['guarded_packages'].items():
        actual = {p.name: sha(p) for p in sorted((control / 'runtime' / name).glob('*.py'))}
        need(actual == expected, 'Frozen controller package changed: ' + name)
    text = (control / 'entry.py').read_text()
    if text != limits_entry(cfg):
        path = control / 'runtime_capacity/policy.py'
        need(path.exists() and text == entry_text(path, sha(path))
             and sha(path) in {sha(__file__), PREVIOUS_SHA256},
             'Unrecognized controller entry; no operational adapter overwritten')
    return state


def boot(control, argv):
    cfg = read(control / 'runtime_limits_30h_24h/config.json')
    root, kind = Path(cfg['root']), cfg['kind']
    # Same frozen source and contract guards as the installed operational entry.
    validate_control(root, kind, control, allow_pending=True)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(control / 'runtime'))
    sys.path.insert(0, str(control / 'runtime_limits_30h_24h'))
    limits = importlib.import_module('limits_policy')
    if kind == 'random':
        limits.install_random(root, control)
    install_hooks(root, kind)
    return limits.boot(control, argv)


def target_deadline(old, hours, at):
    end = dt.datetime.fromisoformat(old)
    need(end.tzinfo is not None and math.isfinite(hours) and hours > 0, 'Invalid extension')
    return (max(at, end) + dt.timedelta(hours=hours)).isoformat()


def stop_controls(owners):
    for path in owners.values():
        if active(path / 'controller.lock'):
            (path / 'stop.request').touch()
    end = time.monotonic() + 45
    while any(active(p / 'controller.lock') for p in owners.values()) and time.monotonic() < end:
        time.sleep(.25)
    need(not any(active(p / 'controller.lock') for p in owners.values()),
         'Owner is finishing a tick. Rerun the SAME operation after it stops; no process was force-killed')


def start_owner(root, path, target, python):
    state = read(path / 'state.json')
    if state['phase'] == 'DONE':
        return
    if active(path / 'controller.lock'):
        need(abs((dt.datetime.fromisoformat(state['deadline_utc']) - dt.datetime.fromisoformat(target)).total_seconds()) < 10,
             'Active owner has a different deadline; do not start another owner')
        return
    hours = (dt.datetime.fromisoformat(target) - now()).total_seconds() / 3600
    need(hours > 0, 'Recorded extension already expired; use a new operation ID')
    print(command([str(python), '-I', str(path / 'entry.py'), 'resume', '--root', str(root),
                   '--hours', str(hours)], root, timeout=40), end='')
    state = read(path / 'state.json')
    need(active(path / 'controller.lock') and state['controller_status'] == 'RUNNING'
         and abs((dt.datetime.fromisoformat(state['deadline_utc']) - dt.datetime.fromisoformat(target)).total_seconds()) < 10,
         'Actual running owner/deadline was not confirmed')


def apply(root, hours, operation, python, upgrade=False):
    need(re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', operation), 'Invalid operation ID')
    owners = controls(root)
    record = area(root) / 'operations' / (operation + '.json')
    with locked(area(root) / 'install.lock'):
        if record.exists():
            op = read(record)
            need(op['version'] in SUPPORTED_VERSIONS and (upgrade or op['extra_hours'] == hours),
                 'Operation ID reused with different options')
            need(upgrade or op['version'] == VERSION, 'Use upgrade to update the adapter while preserving its saved deadlines')
            if upgrade:
                states = {k: validate_control(root, k, p) for k, p in owners.items()}
                outdated = any(not (p / 'runtime_capacity/policy.py').exists()
                               or sha(p / 'runtime_capacity/policy.py') != sha(__file__) for p in owners.values())
                if outdated or op['version'] != VERSION:
                    for kind, state in states.items():
                        # Never shorten a separately extended live owner either.
                        old = dt.datetime.fromisoformat(op['targets'][kind])
                        live = dt.datetime.fromisoformat(state['deadline_utc'])
                        if (live - old).total_seconds() > 10:
                            op['targets'][kind] = live.isoformat()
                    op.update(previous_version=op['version'], version=VERSION, phase='PREPARED',
                              upgrade_utc=now().isoformat(), deadline_extension_in_this_upgrade=0)
                    write(record, op)
        else:
            need(not upgrade, 'No saved extension operation exists; upgrade never creates a new 24-hour extension')
            states = {k: validate_control(root, k, p) for k, p in owners.items()}
            at = now()
            op = dict(version=VERSION, extra_hours=hours, phase='PREPARED', created_utc=at.isoformat(),
                      targets={k: target_deadline(s['deadline_utc'], hours, at) for k, s in states.items()})
            write(record, op)  # Rerunning after interruption cannot accidentally add another 24h.
        if op['phase'] in ('PREPARED', 'STOPPING'):
            for kind, path in owners.items():
                validate_control(root, kind, path)
            op['phase'] = 'STOPPING'
            write(record, op)
            stop_controls(owners)
            with ExitStack() as stack:
                for path in owners.values():
                    stack.enter_context(locked(path / 'controller.lock'))
                for kind, path in owners.items():
                    validate_control(root, kind, path)
                for kind, path in owners.items():
                    dest = path / 'runtime_capacity/policy.py'
                    backup = area(root) / 'backups' / operation / VERSION / kind
                    backup.mkdir(parents=True, exist_ok=True)
                    for name in ('entry.py', 'state.json', 'plan.json'):
                        if not (backup / name).exists():
                            shutil.copy2(path / name, backup / name)
                    if dest.exists() and not (backup / 'policy.py').exists():
                        shutil.copy2(dest, backup / 'policy.py')
                    atomic_text(dest, Path(__file__).read_text())
                    atomic_text(path / 'entry.py', entry_text(dest, sha(dest)))
            op['phase'] = 'INSTALLED'
            write(record, op)
        errors, warnings = {}, {}
        capacity_ready = False
        # Apply holds before either refreshed owner can release/start a candidate.
        try:
            report = Policy(root).reconcile()
            capacity_ready = report.get('capacity') is not None and not report.get('query_error')
            if not capacity_ready:
                warnings['capacity'] = report.get('query_error') or 'Capacity has not been observed'
        except BlockingIOError:
            warnings['capacity'] = 'The other owner is updating shared policy; status will query without taking its lock'
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            errors['capacity'] = str(exc)
        # Always attempt BOTH owners even if capacity information is unavailable.
        for kind, path in owners.items():
            try:
                start_owner(root, path, op['targets'][kind], python)
            except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
                errors[kind] = str(exc)
        phase = ('RUNNING' if capacity_ready else 'RUNNING_CAPACITY_WAIT') if not errors else 'NEEDS_RETRY'
        op.update(phase=phase, errors=errors, warnings=warnings, capacity_ready=capacity_ready,
                  monitors_verified=not any(k in errors for k in owners), updated_utc=now().isoformat())
        write(record, op)
        print(json.dumps(op, ensure_ascii=False, indent=2))
        need(not errors, 'Incomplete refresh; rerun the SAME operation ID after inspecting errors above')


def status(root):
    result = {}
    for kind, path in controls(root).items():
        state = read(path / 'state.json')
        result[kind] = {key: state.get(key) for key in ('phase', 'controller_status', 'host', 'pid',
            'deadline_utc', 'single', 'dual', 'last_error', 'capacity_error', 'capacity_decision')}
        result[kind]['controller_active'] = active(path / 'controller.lock')
    try:
        result['capacity_now'] = Policy(root).reconcile(mutate=False)
    except Exception as exc:
        result['capacity_now'] = dict(read_only=True, capacity=None, query_error=str(exc))
    cap = result['capacity_now'].get('capacity')
    if cap:
        result['capacity_now']['capacity'] = {k: v for k, v in cap.items() if k not in ('hosts', 'reserved_hosts')}
    result['monitoring_active'] = all(result[k]['controller_active'] and result[k]['controller_status'] == 'RUNNING'
                                      for k in ROLES)
    result['capacity_ready'] = cap is not None and not result['capacity_now'].get('query_error')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'apply', 'upgrade', 'status'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--extra-hours', type=float, default=24)
    parser.add_argument('--operation', default='20261020-plus24')
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'preflight':
        for name in ('qstat', 'pbsnodes', 'qrstat', 'qhold', 'qrls'):
            need(shutil.which(name), 'Missing PBS command: ' + name)
        for kind, path in controls(root).items():
            validate_control(root, kind, path)
        status(root)
    elif args.action in ('apply', 'upgrade'):
        apply(root, args.extra_hours, args.operation, args.python, upgrade=args.action == 'upgrade')
    else:
        status(root)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        raise SystemExit('[0390] ' + str(exc)) from exc
