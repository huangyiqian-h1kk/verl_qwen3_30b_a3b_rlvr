"""Responses input to Qwen chat messages; one shared renderer for build and rollout."""
import json
import re
from multidomain.common import DataError, digest


def text_content(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise DataError('Expected text content or a list of text blocks')
    out = []
    for block in content:
        if block.get('type') not in ('input_text', 'output_text', 'text'):
            raise DataError(f'Unsupported non-text content: {block.get("type")}')
        out.append(block['text'])
    return ''.join(out)


def convert_request(params):
    raw = params.get('input')
    if isinstance(raw, str):
        raw = [{'role': 'user', 'content': raw}]
    if not isinstance(raw, list) or not raw:
        raise DataError('Missing responses_create_params.input')
    messages = []
    if params.get('instructions'):
        messages.append({'role': 'system', 'content': params['instructions']})
    calls = {}
    for item in raw:
        kind = item.get('type', 'message')
        if kind == 'message':
            role = item['role']
            if role == 'developer':
                role = 'system'
            if role not in ('system', 'user', 'assistant', 'tool'):
                raise DataError(f'Unsupported role: {role}')
            msg = {'role': role, 'content': text_content(item.get('content') or '')}
            if item.get('tool_calls'):
                msg['tool_calls'] = item['tool_calls']
            if item.get('tool_call_id'):
                msg['tool_call_id'] = item['tool_call_id']
            messages.append(msg)
        elif kind == 'function_call':
            arguments = item['arguments']
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise DataError('Historical tool arguments must be a JSON object')
            call_id = item.get('call_id', item.get('id', ''))
            calls[call_id] = item['name']
            call = {'type': 'function', 'id': call_id, 'function': {'name': item['name'], 'arguments': arguments}}
            if messages and messages[-1]['role'] == 'assistant':
                messages[-1].setdefault('tool_calls', []).append(call)
            else:
                messages.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
        elif kind == 'function_call_output':
            value = item['output']
            messages.append({'role': 'tool', 'content': text_content(value), 'tool_call_id': item.get('call_id', '')})
        elif kind == 'reasoning':
            summary = item.get('summary') or []
            if item.get('encrypted_content'):
                raise DataError('Encrypted reasoning history cannot be rendered')
            text = '\n'.join(b['text'] for b in summary if b.get('type') == 'summary_text')
            if text:
                messages.append({'role': 'assistant', 'content': text})
        else:
            raise DataError(f'Unsupported Responses input item: {kind}')
    tools = []
    for tool in params.get('tools') or []:
        if tool.get('type') != 'function':
            raise DataError(f'Only offline function schemas are supported: {tool.get("type")}')
        fn = dict(tool.get('function', tool))
        fn.pop('type', None)
        fn.pop('strict', None)  # No constrained decoding; schema is still included in the prompt.
        tools.append({'type': 'function', 'function': fn})
    return messages, tools


def render(tokenizer, messages, tools):
    kwargs = {'tools': tools} if tools else {}
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                        enable_thinking=False, **kwargs)
    from collections.abc import Mapping
    if isinstance(ids, Mapping):
        ids = ids['input_ids']
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    if isinstance(ids, list) and len(ids) == 1 and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not isinstance(ids[0], int):
        raise DataError('Tokenizer returned an unsupported prompt representation')
    return ids


def tokenizer_fingerprint(tokenizer):
    return digest({'chat_template': tokenizer.chat_template, 'vocab': tokenizer.get_vocab(),
                   'special_tokens': tokenizer.special_tokens_map})


def parse_completion(raw):
    """Parse Qwen tool markers without repairing malformed model JSON or executing tools."""
    text = raw
    for token in ('<|im_end|>', '<|endoftext|>'):
        if text.endswith(token):
            text = text[:-len(token)]
    # Instruct-2507 has no thinking mode; do not silently strip ordinary answer text.
    outputs, cursor = [], 0
    pattern = re.compile(r'<tool_call>\s*(.*?)\s*</tool_call>', re.DOTALL)
    for match in pattern.finditer(text):
        before = text[cursor:match.start()]
        if before.strip():
            outputs.append({'type': 'message', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': before}]})
        try:
            call = json.loads(match.group(1))
            if not isinstance(call, dict) or not isinstance(call.get('name'), str) or not isinstance(call.get('arguments'), dict):
                raise ValueError('Tool call needs name and object arguments')
        except (ValueError, TypeError):
            return [], 'malformed_tool_call'
        outputs.append({'type': 'function_call', 'name': call['name'], 'arguments': json.dumps(call['arguments'], ensure_ascii=False)})
        cursor = match.end()
    rest = text[cursor:]
    if '<tool_call>' in rest or '</tool_call>' in rest:
        return [], 'incomplete_tool_call'
    if rest.strip():
        outputs.append({'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': rest}]})
    return outputs, None


def output_text(outputs):
    return ''.join(c['text'] for o in outputs if o['type'] == 'message' for c in o['content'] if c['type'] == 'output_text')
