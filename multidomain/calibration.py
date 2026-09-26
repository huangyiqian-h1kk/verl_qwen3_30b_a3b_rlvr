"""Human-labelled Science judge calibration; never manufacture review labels."""
import asyncio
import json
from pathlib import Path
from multidomain.common import write_json, file_sha


def make_template(data_dir, destination):
    import pyarrow.parquet as pq
    rows = [r for r in pq.read_table(data_dir / 'validation.parquet').to_pylist() if r['domain'] == 'science'][:64]
    if len(rows) < 64:
        raise ValueError('Need at least 64 held-out Science questions for calibration')
    records = []
    for row in rows:
        p = json.loads(row['verifier_json'])
        for candidate in (p['expected_answer'], ''):
            records.append({'sample_id': row['sample_id'], 'question': p['question'], 'expected_answer': p['expected_answer'],
                            'output_regex': p.get('output_regex'), 'candidate': candidate,
                            'human_label': None, 'reviewer': '', 'review_note': ''})
    destination = Path(destination)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))


def run(config, labelled_path, output):
    rows = [json.loads(line) for line in Path(labelled_path).read_text().splitlines() if line.strip()]
    if len(rows) != 128 or sum(r.get('human_label') == 1 for r in rows) != 64 or sum(r.get('human_label') == 0 for r in rows) != 64:
        raise ValueError('Review exactly 128 cases: 64 human-confirmed equivalent and 64 non-equivalent')
    if any(not r.get('reviewer', '').strip() or not r.get('candidate', '').strip() for r in rows):
        raise ValueError('Every calibration case needs a nonempty candidate and reviewer')
    import ray
    from transformers import AutoTokenizer
    from verl.single_controller.ray import RayResourcePool
    from verl.experimental.reward_loop.reward_model import RewardModelManager
    from multidomain.reward import science_score
    ray.init(address='auto', ignore_reinit_error=True)
    pool = RayResourcePool(process_on_nodes=[8], name_prefix='science_calibration', max_colocate_count=1)
    manager = RewardModelManager(config.reward.reward_model, pool)
    if len(manager.rollout_replicas) != 1:
        raise RuntimeError('Expected one frozen judge replica')
    tokenizer = AutoTokenizer.from_pretrained(config.reward.reward_model.model_path, local_files_only=True)
    async def evaluate():
        return await asyncio.gather(*(science_score(r, r['candidate'], manager.get_router_address(), tokenizer) for r in rows))
    scores = asyncio.run(evaluate())
    accuracy = sum(s == r['human_label'] for s, r in zip(scores, rows)) / 128
    false_positives = sum(s == 1 and r['human_label'] == 0 for s, r in zip(scores, rows)) / 64
    report = {'status': 'PASS' if accuracy >= .95 and false_positives <= .05 else 'FAIL',
              'accuracy': accuracy, 'false_positive_rate': false_positives, 'cases': 128,
              'labels_sha256': file_sha(labelled_path), 'results': [dict(r, judge_score=s) for r, s in zip(rows, scores)]}
    write_json(output, report)
    if report['status'] != 'PASS':
        raise RuntimeError('Science judge calibration did not meet the agreed thresholds')
