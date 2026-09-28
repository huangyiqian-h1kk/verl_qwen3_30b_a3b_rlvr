#!/usr/bin/env python3
"""ABCI infrastructure acceptance: Ray and NCCL across the requested GPUs.

No policy, judge, optimizer or training data is loaded. This is NOT a model-
memory or verifier acceptance test. See README for the subsequent val_only stage.
"""
import argparse
import datetime
import json
import os
import platform
import socket
import time
from pathlib import Path


def main():
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--expected-nodes', type=int, default=2)
    ap.add_argument('--gpus-per-node', type=int, default=8)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    ray.init(address=os.environ.get('RAY_ADDRESS', 'auto'))
    deadline = time.monotonic() + 300
    while True:
        nodes = [n for n in ray.nodes() if n['Alive'] and n['Resources'].get('GPU', 0) > 0]
        if len(nodes) == a.expected_nodes and all(n['Resources']['GPU'] == a.gpus_per_node for n in nodes):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Expected {a.expected_nodes} nodes x {a.gpus_per_node} GPUs; got {nodes}')
        time.sleep(2)

    @ray.remote(num_gpus=1, num_cpus=1)
    class GPUProbe:
        def info(self):
            import torch
            torch.cuda.set_device(0)
            props = torch.cuda.get_device_properties(0)
            x = torch.ones((128, 128), device='cuda')
            value = (x @ x).sum().item()
            assert value == 2097152.0
            return {'host': socket.gethostname(), 'ip': ray.util.get_node_ip_address(),
                    'node_id': str(ray.get_runtime_context().get_node_id()),
                    'gpu_ids': ray.get_gpu_ids(), 'gpu_name': props.name,
                    'memory_bytes': props.total_memory, 'torch': torch.__version__,
                    'cuda': torch.version.cuda, 'nccl': list(torch.cuda.nccl.version()),
                    'python': platform.python_version()}

        def free_port(self):
            with socket.socket() as s:
                s.bind(('', 0))
                return s.getsockname()[1]

        def all_reduce(self, rank, world_size, address):
            import torch
            import torch.distributed as dist
            torch.cuda.set_device(0)
            dist.init_process_group(backend='nccl', init_method=address, rank=rank,
                                    world_size=world_size, timeout=datetime.timedelta(seconds=120))
            timings = []
            try:
                for size in (1, 1024, 1048576):
                    x = torch.full((size,), rank + 1.0, dtype=torch.float32, device='cuda')
                    torch.cuda.synchronize()
                    start = time.monotonic()
                    dist.all_reduce(x)
                    torch.cuda.synchronize()
                    expected = world_size * (world_size + 1) / 2
                    assert bool(torch.all(x == expected).item()), f'Rank {rank}: incorrect all-reduce'
                    timings.append({'elements': size, 'seconds': time.monotonic() - start})
                dist.barrier()
            finally:
                dist.destroy_process_group()
            return {'rank': rank, 'checks': timings}

    groups, actors = [], []
    report = {'kind': 'infrastructure_only', 'status': 'FAIL',
              'optimizer_steps': 0, 'policy_loaded': False, 'judge_loaded': False}
    try:
        for _ in range(a.expected_nodes):
            pg = placement_group([{'GPU': 1, 'CPU': 1}] * a.gpus_per_node, strategy='STRICT_PACK')
            groups.append(pg)
        ray.get([g.ready() for g in groups], timeout=180)
        for pg in groups:
            actors.extend(GPUProbe.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg, placement_group_bundle_index=i)).remote() for i in range(a.gpus_per_node))
        info = ray.get([actor.info.remote() for actor in actors], timeout=180)
        devices = {(x['node_id'], tuple(x['gpu_ids'])) for x in info}
        assert len(devices) == a.expected_nodes * a.gpus_per_node, 'GPU assignments overlap'
        assert len({x['node_id'] for x in info}) == a.expected_nodes
        for field in ('torch', 'cuda', 'nccl', 'python'):
            assert len({str(x[field]) for x in info}) == 1, f'Environment mismatch: {field}'
        port = ray.get(actors[0].free_port.remote())
        address = f'tcp://{info[0]["ip"]}:{port}'
        results = ray.get([actor.all_reduce.remote(rank, len(actors), address)
                           for rank, actor in enumerate(actors)], timeout=240)
        report.update(status='PASS', devices=info, nccl_results=results)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        for actor in actors:
            ray.kill(actor, no_restart=True)
        for group in groups:
            remove_placement_group(group)
        ray.shutdown()
    print(json.dumps({'status': report['status'], 'devices': len(report['devices']), 'optimizer_steps': 0}))


if __name__ == '__main__':
    main()
