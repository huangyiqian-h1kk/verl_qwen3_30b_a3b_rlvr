"""Test witnesses for fixed-rule games. These are never added as training answers."""
from collections import deque


def rush_hour_witness(board_config, max_states=200000):
    from reasoning_gym.games.rush_hour import Board, TARGET
    board = Board(board_config)
    pieces = board._pieces
    start = tuple(p.position for p in pieces)
    queue, parents = deque([start]), {start: None}
    while queue:
        state = queue.popleft()
        if state[0] == TARGET:
            moves = []
            while parents[state] is not None:
                prev, move = parents[state]
                moves.append(move)
                state = prev
            return ' '.join(reversed(moves))
        masks = [sum(1 << (pos + n * p.stride) for n in range(p.size)) for pos, p in zip(state, pieces)]
        occupied = sum(masks)
        for i, p in enumerate(pieces):
            if p.fixed:
                continue
            free_mask = occupied ^ masks[i]
            for sign in (-1, 1):
                for amount in range(1, 6):
                    pos = state[i] + sign * amount * p.stride
                    end = pos + (p.size - 1) * p.stride
                    if pos < 0 or end >= 36 or (p.stride == 1 and (pos // 6 != state[i] // 6 or end // 6 != state[i] // 6)):
                        break
                    new_mask = sum(1 << (pos + n * p.stride) for n in range(p.size))
                    if new_mask & free_mask:
                        break
                    new_state = state[:i] + (pos,) + state[i + 1:]
                    if new_state not in parents:
                        parents[new_state] = (state, f'{chr(65 + i)}{sign * amount:+d}')
                        queue.append(new_state)
                        if len(parents) > max_states:
                            raise RuntimeError('Rush Hour witness search exceeded its state limit')
    raise RuntimeError('Published Rush Hour board has no solution under the native rules')


def reasoning_witness(payload):
    task = payload['metadata']['source_dataset']
    if task == 'rush_hour':
        return rush_hour_witness(payload['metadata']['board_config'])
    if task == 'boxnet':
        from multidomain.preparation.check_reasoning_candidates import boxnet_witness
        return boxnet_witness(payload)
    if task == 'graph_color':
        import json
        return json.dumps(payload['metadata']['possible_answer'])
    if task == 'propositional_logic':
        return payload['metadata']['example_answer']
    return payload.get('answer')
