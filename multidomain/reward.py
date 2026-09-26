"""verl async reward entry point. Model mistakes yield reward; infrastructure faults raise."""
import asyncio
import json
import math
import multiprocessing
import re
from functools import lru_cache
from types import SimpleNamespace
from multidomain.common import InfrastructureError
from multidomain.template import parse_completion, output_text
from multidomain.upstream import VENDOR, format_scorer, structured_scorer, pivot_namespace, obj


def local_score(payload, outputs):
    source = payload['source']
    text = output_text(outputs)
    if source in ('if_freeform', 'if_citation'):
        v = payload['verifier']
        fn = format_scorer()._verify_string_match if v['type'] == 'string_match' else format_scorer()._verify_regex
        return fn(text, v)[0]
    if source == 'if_structured':
        scorer = structured_scorer()
        if payload.get('response_mode') == 'tool_call':
            body = SimpleNamespace(response=obj({'output': outputs, 'error': None}),
                                   tool_name=payload.get('tool_name'), tool_payload_key=payload.get('tool_payload_key'))
            value, error, _ = scorer.extract_tool_call_payload(body)
            return 0.0 if error else scorer.evaluate_structured_object_response(payload['schema_str'], value)[0]
        return scorer.evaluate_structured_output_response(payload.get('schema_type') or 'json', payload['schema_str'], text)[0]
    if source.endswith('_pivot'):
        ns = pivot_namespace()
        actual = ns['extract_tool_call_or_text'](obj({'output': outputs}))
        expected = payload['expected_action']
        if actual is None:
            return 0.0
        if expected['type'] == 'message':
            return float(actual.type == 'output_text')
        if actual.type != 'function_call':
            return 0.0
        comp = ns['ToolCallComparator'](config=ns['ToolCallComparatorConfig'](
            word_count_similarity_threshold=0.0 if source == 'swe_pivot' else 0.1))
        return comp.compare_tool_call(ns['ExpectedFunctionCall'](**expected), actual)[0]
    if source == 'reasoning_gym':
        from reasoning_gym.utils import extract_answer
        answer = extract_answer(text, tag_name='answer')
        if answer is None:
            match = re.search(r'\\boxed\{([^}]+)\}', text)
            answer = match.group(1).strip() if match else text.strip()
        task = payload['metadata']['source_dataset']
        return float(reasoning_dataset(task)(answer=answer, entry=payload))
    raise ValueError('No local verifier for ' + source)


@lru_cache(maxsize=64)
def reasoning_dataset(task):
    import reasoning_gym
    return reasoning_gym.get_score_answer_fn(task)


def _math_child(expected, generated, pipe):
    try:
        from math_verify.metric import math_metric
        from math_verify import ExprExtractionConfig, LatexExtractionConfig
        expected = expected.strip()
        if expected.startswith('\\(') and expected.endswith('\\)'):
            expected = expected[2:-2].strip()
        if expected.startswith('$') and expected.endswith('$') and len(expected) > 1:
            expected = expected[1:-1].strip()
        metric = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                             pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()))
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            score, _ = metric(['\\boxed{' + expected + '}'], [generated])
        pipe.send(('ok', float(score)))
    except BaseException as exc:
        pipe.send(('error', type(exc).__name__ + ': ' + str(exc)))
    finally:
        pipe.close()


def math_score(expected, generated, timeout=10):
    # A dedicated spawn child permits terminating a stuck symbolic calculation.
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_math_child, args=(expected, generated, child))
    process.start()
    child.close()
    try:
        if not parent.poll(timeout):
            raise InfrastructureError(f'Math verifier exceeded {timeout}s')
        try:
            status, value = parent.recv()
        except EOFError as exc:
            raise InfrastructureError('Math verifier child exited without a result') from exc
        if status != 'ok':
            raise InfrastructureError('Math verifier: ' + value)
        return value
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()


def science_candidate(payload, text):
    regex = payload.get('output_regex')
    if regex:
        matches = list(re.finditer(regex, text, flags=re.MULTILINE | re.DOTALL))
        if matches:
            match = matches[-1]
            return next((x.strip() for x in match.groups() if isinstance(x, str) and x.strip()), match.group(0).strip())
    return text.strip() or '[NO VALID ANSWER EXTRACTED]'


def science_prompt(payload, text):
    return (VENDOR / 'science_prompt.txt').read_text().format(question=payload['question'],
        expected_answer=payload['expected_answer'], generated_answer=science_candidate(payload, text))


async def science_score(payload, text, address, tokenizer):
    import aiohttp
    if not address or tokenizer is None:
        raise InfrastructureError('Science needs its dedicated frozen judge server and tokenizer')
    prompt = science_prompt(payload, text)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
                                         add_generation_prompt=True, enable_thinking=False)
    if hasattr(ids, 'keys'):
        ids = ids['input_ids']
    if len(ids) + 8192 > 131072:
        raise InfrastructureError('Judge context overflow: data budget check must be revisited')
    last = None
    for attempt in range(3):
        try:
            # One reward worker + shared semaphore = a global 32-request limit.
            async with judge_semaphore():
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
                    async with session.post(f'http://{address}/v1/completions', json={
                        'model': str(tokenizer.name_or_path), 'prompt': ids, 'max_tokens': 8192,
                        'temperature': 0.0, 'top_p': 1.0}) as response:
                        response.raise_for_status()
                        result = await response.json()
            choice = result['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise InfrastructureError('Judge verdict truncated')
            labels = re.findall(r'\[\[A(?:!=|=)B\]\]', choice['text'])
            if not labels or len(set(labels)) != 1:
                raise InfrastructureError('Judge verdict missing or contradictory')
            return float(labels[-1] == '[[A=B]]')
        except (aiohttp.ClientError, asyncio.TimeoutError, KeyError, ValueError, InfrastructureError) as exc:
            last = exc
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)
    raise InfrastructureError(f'Science judge failed after 3 attempts: {last}') from last


@lru_cache(maxsize=1)
def judge_semaphore():
    return asyncio.Semaphore(32)


@lru_cache(maxsize=1)
def local_semaphore():
    return asyncio.Semaphore(8)


async def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                        reward_router_address=None, reward_model_tokenizer=None, **kwargs):
    payload = json.loads(ground_truth)
    extra_info = extra_info or {}
    if 'raw_completion' not in extra_info:
        raise InfrastructureError('Reward requires MultiDomainAgentLoop raw_completion, including tool markers')
    outputs, error = parse_completion(extra_info['raw_completion'])
    source = payload['source']
    if error:
        score = 0.0
    elif source == 'math':
        async with local_semaphore():
            score = await asyncio.to_thread(math_score, payload['expected_answer'], output_text(outputs))
    elif source == 'science':
        score = await science_score(payload, output_text(outputs), reward_router_address, reward_model_tokenizer)
    else:
        async with local_semaphore():
            score = await asyncio.to_thread(local_score, payload, outputs)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise InfrastructureError(f'Invalid reward from {source}: {score}')
    # Identical keys for every domain: required by verl's batched reward collation.
    return {'score': score, 'acc': score, 'parse_error': float(error is not None)}
