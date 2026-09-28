"""Check real selected rows against the pinned upstream verifier bodies."""
import asyncio
import json
from collections import Counter
from types import SimpleNamespace
from multidomain.common import write_json
from multidomain.reward import compute_score, local_score, math_score
from multidomain.template import parse_completion
from multidomain.upstream import method_namespace, format_scorer, structured_scorer, pivot_namespace, obj
from multidomain.witnesses import reasoning_witness


def official_score(payload, outputs):
    source = payload['source']
    if source in ('if_freeform', 'if_citation'):
        cls = method_namespace('format_verification.py', 'FormatVerificationResourcesServer', ['verify'],
                               {'FormatVerificationVerifyResponse': lambda **kw: SimpleNamespace(**kw)})
        scorer = cls()
        for name in ('_extract_assistant_text', '_verify_regex', '_verify_string_match'):
            setattr(scorer, name, getattr(format_scorer(), name))
        body = SimpleNamespace(response=obj({'output': outputs}), verifier=payload['verifier'], model_dump=lambda: {})
    elif source == 'if_structured':
        from enum import StrEnum
        class SchemaType(StrEnum):
            JSON='json'; YAML='yaml'; XML='xml'; TOML='toml'; CSV='csv'
        cls = method_namespace('structured.py', 'StructuredOutputsResourcesServer', ['verify'],
            {'SchemaType': SchemaType, 'StructuredOutputsVerifyResponse': lambda **kw: SimpleNamespace(**kw)})
        scorer = cls()
        for name in ('extract_tool_call_payload', 'evaluate_structured_object_response', 'evaluate_structured_output_response'):
            setattr(scorer, name, getattr(structured_scorer(), name))
        body = SimpleNamespace(**{k: payload.get(k) for k in ('schema_type','schema_str','response_mode','tool_name','tool_payload_key')},
            response=obj({'output': outputs, 'error': None}), model_dump=lambda: {})
    elif source.endswith('_pivot'):
        ns = pivot_namespace()
        cls = method_namespace('pivot_app.py', 'SingleStepToolUseArgumentComparisonResourcesServer', ['verify'],
            {**ns, 'SingleStepToolUseArgumentComparisonVerifyResponse': lambda **kw: SimpleNamespace(**kw)})
        scorer = cls()
        scorer.config = SimpleNamespace(tool_call_comparator_config=ns['ToolCallComparatorConfig'](
            word_count_similarity_threshold=0.0 if source == 'swe_pivot' else 0.1))
        body = SimpleNamespace(response=obj({'output': outputs}), expected_action=obj(payload['expected_action']), model_dump=lambda: {})
    else:
        raise ValueError(source)
    return asyncio.run(scorer.verify(body)).reward


def check(data_dir, output):
    import pyarrow.parquet as pq
    from multidomain.common import digest
    rows = pq.read_table(data_dir / 'train.parquet').to_pylist()
    rows.sort(key=lambda r: digest(r['sample_id']))
    count, tested, branches = Counter(), 0, Counter()
    rg_tasks = set()
    cases = []
    for row in rows:
        payload = json.loads(row['verifier_json'])
        source = payload['source']
        if source == 'science':
            continue  # Requires separately human-labelled judge calibration.
        # Cover every RG task and every structured response format, in addition to 32 rows/source.
        branch = (payload.get('schema_type'), payload.get('response_mode')) if source == 'if_structured' else row.get('task', '')
        key = source + '/' + str(branch)
        if count[source] >= 32 and branches[key] >= 1:
            continue
        completions = ['', 'invalid answer']
        if source.endswith('_pivot'):
            action = payload['expected_action']
            completions += [action['content']] if action['type'] == 'message' else [
                '<tool_call>' + json.dumps({'name': action['name'], 'arguments': json.loads(action['arguments'])}) + '</tool_call>']
        elif source == 'math':
            completions.append('\\boxed{' + payload['expected_answer'] + '}')
        elif source == 'reasoning_gym':
            witness = reasoning_witness(payload)
            if witness is not None:
                completions.append('<answer>' + str(witness) + '</answer>')
        else:
            completions += ['{}', '[]']
        for completion in completions:
            outputs, error = parse_completion(completion)
            if error:
                raise AssertionError('Invalid acceptance fixture')
            actual = asyncio.run(compute_score(row['domain'], completion, row['verifier_json'],
                extra_info={'raw_completion': completion}))['score']
            if source == 'math':
                expected = math_score(payload['expected_answer'], completion)
            elif source == 'reasoning_gym':
                import reasoning_gym
                from reasoning_gym.utils import extract_answer
                answer = extract_answer(completion, tag_name='answer')
                if answer is None:
                    answer = completion.strip()
                expected = reasoning_gym.get_score_answer_fn(payload['metadata']['source_dataset'])(answer=answer, entry=payload)
                if completion.startswith('<answer>') and actual != 1:
                    raise AssertionError(f'RG witness failed: {row["sample_id"]} {actual}')
            else:
                expected = official_score(payload, outputs)
            if actual != expected:
                raise AssertionError((row['sample_id'], actual, expected))
            tested += 1
        count[source] += 1
        branches[key] += 1
    required = {r['source'] for r in rows} - {'science'}
    if any(count[s] < 32 for s in required):
        raise AssertionError('Need >=32 real train rows per non-judge source: ' + str(count))
    write_json(output, {'status': 'PASS', 'scope': 'real-row adapter/upstream score parity; candidate fixture coverage',
                       'sources': dict(count), 'branches': dict(branches), 'candidate_comparisons': tested,
                       'science': 'separate human-labelled calibration required' if any(r['source'] == 'science' for r in rows) else 'disabled; no judge calibration required',
                       'limitations': ['Structured positive-output completeness is checked in model smoke, not inferred from empty-output parity.',
                                       'Math comparator shares math-verify==0.8.0; this checks adapter parity, not mathematical correctness of every gold label.']})
