"""Versioned adapters from published NeMo Gym rows to a uniform text-only record."""
import json
import re
import unicodedata
from multidomain.common import DataError, ROOT, digest, dumps
from multidomain.template import convert_request

SOURCE_DOMAIN = {'math': 'math', 'if_freeform': 'if', 'if_citation': 'if', 'if_structured': 'if',
                 'science': 'science', 'conversational_pivot': 'conversational_pivot',
                 'swe_pivot': 'swe_pivot', 'reasoning_gym': 'logic_algorithmic'}
CATEGORIES = json.loads((ROOT / 'config/multidomain/reasoning_gym_task_categories.json').read_text())


def unpack(value):
    return json.loads(value) if isinstance(value, str) else value


def normalize_question(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def adapt(raw, source, origin, config):
    """Return (record, None), or (None, explicit native-subset exclusion reason)."""
    row = {k: v for k, v in raw.items() if v is not None}
    for field in ('responses_create_params', 'metadata', 'template_metadata', 'verifier', 'expected_action'):
        if field in row:
            row[field] = unpack(row[field])
    domain = SOURCE_DOMAIN[source]
    params = row['responses_create_params']
    md = row.get('metadata') or {}
    if source == 'science':
        if unpack(row.get('agent_ref', {})).get('name') != config['domains']['science']['agent_filter']:
            return None, 'science_agent_outside_native_no_tool_subset'
        if params.get('tools'):
            raise DataError('Science no-tool agent unexpectedly contains tools')
    if source == 'reasoning_gym':
        if row.get('uuid') in {'4fbf3afc-93cf-48d3-997d-d6df1d506dac', '6c50ea1e-0097-4da5-a16a-6b1f85539e6c'}:
            return None, 'published_propositional_reference_under_review'
        task = md['source_dataset']
        category = CATEGORIES[task]
        rg = config['domains']['logic_algorithmic']
        if category not in rg['categories'] or task in rg['exclude_tasks']:
            return None, 'reasoning_task_outside_selected_scope'
    messages, tools = convert_request(params)
    users = [m['content'] for m in messages if m['role'] == 'user']
    question = row.get('question') or (users[-1] if users else '')
    if not isinstance(question, str) or not question.strip():
        raise DataError('Missing task question')
    payload = {'source': source}
    if source == 'math':
        if row.get('_hf_question_placeholder'):
            raise DataError('Restore Math placeholders with fetch_raw before building data')
        payload['expected_answer'] = row['expected_answer']
    elif source in ('if_freeform', 'if_citation'):
        payload['verifier'] = row['verifier']
        v = payload['verifier']
        if v.get('type') not in ('regex', 'inline_prose', 'string_match'):
            raise DataError('Unknown IF verifier type')
        for pattern in v.get('verify_regex', []) + v.get('patterns', []):
            re.compile(pattern)
    elif source == 'if_structured':
        fields = ('schema_str', 'schema_type', 'response_mode', 'tool_name', 'tool_payload_key',
                  'source_record_id', 'problem_type', 'schema_repr')
        payload.update({k: row.get(k) for k in fields})
        json.loads(payload['schema_str'])
        payload['schema_type'] = payload['schema_type'] or 'json'
        payload['response_mode'] = payload['response_mode'] or 'text'
        if payload['schema_type'] not in ('json', 'yaml', 'xml', 'toml', 'csv'):
            raise DataError('Unsupported structured format')
        if payload['response_mode'] not in ('text', 'tool_call'):
            raise DataError('Unsupported structured response_mode')
        if payload['response_mode'] == 'tool_call' and not tools:
            raise DataError('Tool-call structured row has no native tool schema')
    elif source == 'science':
        expected = row.get('expected_answer', md.get('expected_answer'))
        if expected is None:
            raise DataError('Science expected_answer missing')
        regex = row.get('template_metadata', {}).get('output_regex')
        if regex:
            re.compile(regex)
        payload.update(question=question, expected_answer=str(expected), output_regex=regex)
    elif source.endswith('_pivot'):
        action = row['expected_action']
        if action['type'] == 'function_call':
            args = unpack(action['arguments'])
            if not isinstance(args, dict):
                raise DataError('Pivot expected arguments must be an object')
            action = {**action, 'arguments': dumps(args)}
        elif action['type'] != 'message':
            raise DataError('Unknown Pivot expected action')
        payload['expected_action'] = action
    elif source == 'reasoning_gym':
        payload.update(question=row['question'], answer=raw.get('answer'), metadata=md)
    native_cap = params.get('max_output_tokens')
    if native_cap is not None and (type(native_cap) is not int or native_cap < 1):
        raise DataError('Invalid native max_output_tokens')
    # Same question is a global leakage key even when it occurs in multiple domains.
    keys = ['prompt:' + digest([messages, tools])] if source.endswith('_pivot') else ['question:' + digest(normalize_question(question))]
    for key in ('trajectory_id', 'conversation_id', 'source_record_id', 'instance_id', 'problem_id'):
        value = row.get(key, md.get(key))
        if value is not None:
            keys.append(f'{source}/{key}:' + digest(value))
    if source == 'swe_pivot' and not md.get('instance_id'):
        raise DataError('SWE row needs metadata.instance_id for split isolation')
    if source.endswith('_pivot') and not any(k in row or k in md for k in ('trajectory_id', 'conversation_id', 'instance_id')):
        # First user request groups turns when no trajectory identifier is published.
        raise DataError('Pivot row needs a published trajectory/conversation/instance ID; do not group unrelated dialogues by generic greetings')
    sample_id = source + '/' + digest({'origin': origin, 'native_id': row.get('uuid', row.get('id'))})
    provenance = {**origin, 'native_id': row.get('uuid', row.get('id')), 'license': row.get('license'),
                  'adapter_version': 1, 'request_controls': {k: v for k, v in params.items() if k not in ('input', 'tools', 'instructions')},
                  'historical_reasoning': 'published_summary_text_preserved_as_assistant_content'}
    dedup_payload = {k: v for k, v in payload.items() if k != 'source_record_id'}
    dedup_key = digest([messages, tools, dedup_payload])
    if source == 'reasoning_gym':
        dedup_key = digest([source, normalize_question(question), md['source_dataset'], raw.get('answer')])
    return {'sample_id': sample_id, 'domain': domain, 'source': source,
            'messages_json': dumps(messages), 'tools_json': dumps(tools), 'verifier_json': dumps(payload),
            'provenance_json': dumps(provenance), 'leakage_keys': keys,
            'dedup_key': dedup_key,
            'request_max_output_tokens': native_cap,
            'category': CATEGORIES[md['source_dataset']] if source == 'reasoning_gym' else '',
            'task': md.get('source_dataset', ''), 'eligible': True}, None
