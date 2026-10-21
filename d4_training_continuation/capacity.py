"""Audited 0390 reserved-queue physical-host accounting (read only)."""
import math
import re
from . import QUEUE
from .common import need
NODE_GROUP = "nii"
PROJECT = "gcg51557"

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

def words(value):
    if isinstance(value, list):
        return {str(x).strip() for x in value if str(x).strip()}
    return {x.strip() for x in str(value or '').split(',') if x.strip()}

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

def queue_pool_profile(queue):
    """Validate the actual 0390 execution-queue profile, not its name as a Resv ID.

    Uploaded queue/node snapshots bind R9920261000 to nii and authorize gcg51557.
    This is an explicit site-specific profile, not a fallback to global free nodes.
    PBS still decides placement; future/foreign reservations are excluded below.
    """
    need(queue.get('queue_type') == 'Execution' and queue.get('default_chunk.node_group') == NODE_GROUP,
         'Expected reserved execution queue / nii node-group profile is unavailable')
    need(queue.get('acl_group_enable', '').lower() == 'true' and PROJECT in words(queue.get('acl_groups')),
         'Expected project authorization is missing from the queue profile')
    need(number(queue.get('resources_available.ncpus')) > 0,
         'Queue CPU capacity is missing; do not infer headroom from cluster totals')

def free_hosts(nodes_payload, reservation, queue):
    """Count eligible physical 192-CPU/8-GPU hosts, never NUMA vnodes.

    reservation=None selects the audited execution-queue/node-group profile.
    An explicit reservation record retains the stricter resv_nodes membership.
    """
    nodes = nodes_payload.get('nodes')
    need(isinstance(nodes, dict) and nodes, 'pbsnodes returned no vnode mapping')
    need(queue.get('enabled', '').lower() == 'true'
         and queue.get('started', '').lower() == 'true', 'Reserved queue is not enabled/started')
    reserved = None
    if reservation is None:
        queue_pool_profile(queue)
    else:
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
        if reserved is not None:
            if not any(n in reserved or host in reserved for n in names):
                continue
        elif not any(number(resource(nodes, n, 'resources_available', 'ncpus')) > 0
                     and group in words(resource(nodes, n, 'resources_available', 'node_group')) for n in names):
            continue
        reasons, leaves, cpus, gpus = set(), [], 0, 0
        for name in names:
            row = nodes[name]
            ncpu = number(resource(nodes, name, 'resources_available', 'ncpus'))
            if ncpu:
                leaves.append(name)
                cpus += ncpu
                gpus += number(resource(nodes, name, 'resources_available', 'ngpus'))
                if reserved is not None and name not in reserved and host not in reserved:
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
            own_reservations = ({QUEUE, str(reservation.get('reservation_id') or '').split('.')[0]}
                                if reservation is not None else set())
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
                queue_ncpus_headroom=headroom, pool_hosts=sorted(details), node_group=group,
                capacity_source='reservation_nodes' if reservation is not None else 'queue_node_group')
