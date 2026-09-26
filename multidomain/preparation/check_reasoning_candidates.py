#!/usr/bin/env python3
"""Reference checks for the selected published RG rows; no model generation.

Boxnet witnesses are constructed using fixed movement rules solely for tests.
Rush Hour rows have no gold solution: check board parsing, and test the scorer
on a separate small solved/unsolved fixture. Report this coverage limitation.
"""
import argparse
import copy
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

import reasoning_gym
from reasoning_gym.games.rush_hour import Board


def boxnet_witness(row):
    state = copy.deepcopy(row['metadata']['initial_state'])
    moves = []
    while True:
        found = next(((pos, item) for pos, items in state.items() for item in items
                      if item.startswith('box_')), None)
        if found is None:
            break
        pos, box = found
        target = 'target_' + box[4:]
        dest = next(key for key, items in state.items() if target in items)
        x, y = map(float, pos.split('_'))
        tx, ty = map(float, dest.split('_'))
        while (x, y) != (tx, ty):
            nx, ny = x, y
            if x != tx:
                nx += 1 if tx > x else -1
            else:
                ny += 1 if ty > y else -1
            next_pos = f'{nx}_{ny}'
            assert next_pos in state
            moves.append({f'Agent[{x}, {y}]': f'move({box}, square[{nx}, {ny}])'})
            state[pos].remove(box)
            state[next_pos].append(box)
            pos, x, y = next_pos, nx, ny
        moves.append({f'Agent[{x}, {y}]': f'move({box}, {target})'})
        state[pos].remove(box)
        state[pos].remove(target)
    return json.dumps(moves)


def main():
    kit = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, default=kit / 'data/reasoning_gym/logic_algorithmic_candidates.jsonl')
    ap.add_argument('--output', type=Path, default=kit / 'reports/reasoning_candidates_checks.json')
    args = ap.parse_args()
    version = importlib.metadata.version('reasoning-gym')
    if version != '0.1.25':
        raise ValueError(f'Expected reasoning-gym==0.1.25, got {version}')
    functions, tasks, totals, flags = {}, {}, Counter(), []
    for line in args.data.read_text().splitlines():
        row = json.loads(line)
        task = row['metadata']['source_dataset']
        if task not in functions:
            functions[task] = reasoning_gym.get_score_answer_fn(task)
            tasks[task] = Counter()
        answer = row['answer']
        if task == 'boxnet':
            answer = boxnet_witness(row)
            tasks[task]['constructed_witness'] += 1
        elif task == 'rush_hour':
            Board(row['metadata']['board_config'])
            tasks[task]['board_parse_only_no_reference'] += 1
            totals['board_parse_only_no_reference'] += 1
            continue
        elif task == 'graph_color':
            answer = json.dumps(row['metadata']['possible_answer'])
        elif task == 'propositional_logic':
            answer = row['metadata']['example_answer']
        score = float(functions[task](answer=answer, entry=row))
        category = 'reference_score_1' if score == 1.0 else 'reference_flag'
        tasks[task][category] += 1
        totals[category] += 1
        if score != 1.0:
            flags.append({'uuid': row['uuid'], 'task': task, 'score': score})
    rush = functions['rush_hour']
    fixture = {'metadata': {'board_config': 'oooooo' * 2 + 'AAoooo' + 'oooooo' * 3}}
    assert rush(answer='A+4', entry=fixture) == 1.0
    assert rush(answer='A-1', entry=fixture) == 0.01
    assert rush(answer='', entry=fixture) == 0.0
    box = functions['boxnet']
    fixture = {'metadata': {'initial_state': {'0.5_0.5': ['box_red', 'target_red']}}}
    assert box(answer='[{"Agent[0.5, 0.5]":"move(box_red, target_red)"}]', entry=fixture) == 1.0
    assert box(answer='[]', entry=fixture) == 0.05
    assert box(answer='not JSON', entry=fixture) == 0.0
    report = {'version': version, 'totals': dict(totals), 'tasks': tasks,
              'reference_flags': flags, 'planning_boundary_fixtures_passed': 6,
              'scope': 'CPU references and fixed-rule simulation; no policy model, no tokenizer, not a full acceptance pass',
              'limitations': ['Rush Hour real-row solutions have not been tested',
                              'Propositional reference flags require review before training',
                              'Boxnet constructed solutions are test witnesses, not added training answers']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'totals':dict(totals),'planning_boundary_fixtures_passed':6}, ensure_ascii=False))


if __name__ == '__main__':
    main()
