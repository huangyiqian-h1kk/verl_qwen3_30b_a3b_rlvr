"""Load only pinned, hash-checked scoring code, without NeMo web-server dependencies.

The bundled sources remain byte-for-byte upstream. AST selection omits service
imports and request classes; selected scoring function bodies are unchanged.
"""
import ast
import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

VENDOR = Path(__file__).with_name('_vendor')


def obj(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: obj(v) for k, v in value.items()})
    if isinstance(value, list):
        return [obj(v) for v in value]
    return value


def checked_tree(name):
    data = (VENDOR / name).read_bytes()
    manifest = json.loads((VENDOR / 'manifest.json').read_text())
    if hashlib.sha256(data).hexdigest() != manifest[name]['sha256']:
        raise RuntimeError('Modified upstream scoring source: ' + name)
    return ast.parse(data)


def compile_nodes(nodes, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<pinned scoring functions>', 'exec'), namespace)
    return namespace


def method_namespace(file, cls, methods, imports):
    tree = checked_tree(file)
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    selected = [copy.deepcopy(n) for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods]
    if {n.name for n in selected} != set(methods):
        raise RuntimeError('Upstream method contract changed')
    selected_cls = ast.ClassDef(name=cls, bases=[], keywords=[], body=selected, decorator_list=[])
    ns = {'__name__': 'multidomain.pinned', **imports}
    compile_nodes([selected_cls], ns)
    return ns[cls]


@lru_cache
def format_scorer():
    import re
    return method_namespace('format_verification.py', 'FormatVerificationResourcesServer',
        ['_extract_assistant_text', '_verify_regex', '_verify_string_match'], {'re': re})()


@lru_cache
def structured_scorer():
    import csv, io, tomllib, xmltodict, yaml
    from typing import Dict
    from openapi_schema_validator import validate
    from enum import StrEnum
    ns = {'StrEnum': StrEnum}
    enum = next(n for n in checked_tree('structured.py').body if isinstance(n, ast.ClassDef) and n.name == 'SchemaType')
    compile_nodes([enum], ns)
    cls = method_namespace('structured.py', 'StructuredOutputsResourcesServer',
        ['extract_tool_call_payload', 'parse_content', 'strictify_schema', 'coerce_xml_types',
         'coerce_csv_types', '_coerce_csv_scalar', 'evaluate_structured_object_response', 'evaluate_structured_output_response'],
        dict(csv=csv, io=io, json=json, tomllib=tomllib, xmltodict=xmltodict, yaml=yaml, Dict=Dict,
             SchemaType=ns['SchemaType'], validate_against_schema_openapi=validate))
    scorer = cls()
    scorer.config = SimpleNamespace(xml_coerce_types=True)
    return scorer


@lru_cache
def pivot_namespace():
    ns = {'__name__': 'multidomain.pinned', 'NeMoGymResponseFunctionToolCall': SimpleNamespace,
          'NeMoGymResponse': SimpleNamespace, 'NeMoGymResponseOutputText': SimpleNamespace}
    for file in ('pivot_comparator.py', 'pivot_response.py'):
        nodes = [n for n in checked_tree(file).body if not (isinstance(n, ast.ImportFrom) and (n.module or '').startswith('nemo_gym'))]
        compile_nodes(nodes, ns)
    for name in ('ExpectedMessage', 'ExpectedFunctionCall', 'ToolCallComparatorConfig', 'ToolCallComparator'):
        ns[name].model_rebuild(_types_namespace=ns)
    return ns
