import asyncio
import copy
import json
import unittest
from pathlib import Path
from multidomain.common import ROOT, read_config, DataError
from multidomain.template import convert_request, parse_completion, render
from multidomain.adapters import adapt
from multidomain.reward import local_score, compute_score, science_candidate
from multidomain.build import Groups, choose
from multidomain.verify import official_score


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.config = read_config(ROOT / 'config/multidomain/initial.yaml')

    def test_native_tool_history_and_summary(self):
        request = {'input': [{'role':'system','content':'native instruction'}, {'role':'user','content':'Q'},
            {'type':'reasoning','summary':[{'type':'summary_text','text':'Published history'}]},
            {'type':'function_call','name':'read','arguments':'{"file":"x"}','call_id':'c'},
            {'type':'function_call_output','call_id':'c','output':'file body'}],
            'tools':[{'type':'function','name':'read','parameters':{'type':'object'},'strict':True}]}
        messages, tools = convert_request(request)
        self.assertEqual(messages[-2]['content'], 'Published history')
        self.assertEqual(messages[-2]['tool_calls'][0]['function']['arguments'], {'file':'x'})
        self.assertEqual(messages[-1]['content'], 'file body')
        self.assertEqual(tools[0]['function']['name'], 'read')

    def test_no_image_or_unknown_input_silently_dropped(self):
        with self.assertRaises(DataError):
            convert_request({'input':[{'role':'user','content':[{'type':'input_image','image_url':'x'}]}]})

    def test_science_native_subset(self):
        row = {'responses_create_params': {'input':[{'role':'user','content':'Q'}]},
            'expected_answer':'A','agent_ref':{'name':'ns_tools_simple_agent'}}
        self.assertIsNone(adapt(row,'science',{},self.config)[0])
        row['agent_ref']['name']='equivalence_llm_judge_simple_agent'
        self.assertEqual(json.loads(adapt(row,'science',{},self.config)[0]['verifier_json'])['expected_answer'],'A')
        row['responses_create_params']['tools']=[{'type':'function','name':'x'}]
        with self.assertRaises(DataError): adapt(row,'science',{},self.config)

    def test_math_placeholder_cannot_reach_training(self):
        row={'responses_create_params':{'input':[{'role':'user','content':'Q'}]},'_hf_question_placeholder':{'repo':'x'},'expected_answer':'2'}
        with self.assertRaises(DataError): adapt(row,'math',{},self.config)

    def test_qwen_renderer_receives_row_tools_without_truncation(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs=kwargs
                return [1,2,3]
        tok=Tokenizer()
        tools=[{'type':'function','function':{'name':'f'}}]
        self.assertEqual(render(tok,[{'role':'user','content':'Q'}],tools),[1,2,3])
        self.assertEqual(tok.kwargs['tools'], tools)
        self.assertNotIn('truncation',tok.kwargs)

    def test_known_invalid_rg_references_are_reported(self):
        row={'uuid':'4fbf3afc-93cf-48d3-997d-d6df1d506dac','metadata':{'source_dataset':'propositional_logic'},'responses_create_params':{}}
        result, reason=adapt(row,'reasoning_gym',{},self.config)
        self.assertIsNone(result)
        self.assertIn('under_review',reason)


class RewardTests(unittest.TestCase):
    def test_tool_markers_preserved_and_first_call_wins(self):
        p={'source':'swe_pivot','expected_action':{'type':'function_call','name':'run','arguments':'{"command":"python -m pytest"}'}}
        for completion, expected in [
            ('<tool_call>{"name":"run","arguments":{"command":"echo different words"}}</tool_call>',1),
            ('<tool_call>{"name":"wrong","arguments":{}}</tool_call><tool_call>{"name":"run","arguments":{"command":"python -m pytest"}}</tool_call>',0)]:
            outputs,error=parse_completion(completion+'<|im_end|>')
            self.assertIsNone(error)
            self.assertEqual(local_score(p,outputs),expected)
            self.assertEqual(local_score(p,outputs),official_score(p,outputs))

    def test_malformed_tool_output_and_missing_transport(self):
        p={'source':'swe_pivot','expected_action':{'type':'message','content':'answer'}}
        out=asyncio.run(compute_score('swe_pivot','',''+json.dumps(p),{'raw_completion':'<tool_call>{bad}</tool_call>'}))
        self.assertEqual(out['score'],0)
        with self.assertRaises(RuntimeError):asyncio.run(compute_score('swe_pivot','answer',json.dumps(p),{}))

    def test_pivot_message_official_type_only(self):
        p={'source':'conversational_pivot','expected_action':{'type':'message','content':'reference'}}
        outputs,_=parse_completion('arbitrary message')
        self.assertEqual(local_score(p,outputs),1)
        self.assertEqual(local_score(p,outputs),official_score(p,outputs))

    def test_format_parity(self):
        for p,text in [({'source':'if_freeform','verifier':{'type':'regex','verify_regex':['^- .+','^- x.*'],'verify_min_matches':2}},'- x'),
                       ({'source':'if_citation','verifier':{'type':'string_match','expected_markers':['[1]'],'patterns':[r'\[\d+\]']}},'A[1] B[9]')]:
            outputs,_=parse_completion(text)
            self.assertEqual(local_score(p,outputs),0)
            self.assertEqual(local_score(p,outputs),official_score(p,outputs))

    def test_science_regex_last_capture_and_fallback(self):
        self.assertEqual(science_candidate({'output_regex':r'Answer: (.*?)\.'},'Answer: x. Answer: y.'),'y')
        self.assertEqual(science_candidate({'output_regex':r'no match'},'full response'),'full response')

    def test_structured_formats_and_tools(self):
        schema=json.dumps({'type':'object','properties':{'value':{'type':'integer'}}})
        cases=[('json','{"value":3}',1),('yaml','value: 3',1),('toml','value=3',1),('xml','<value>3</value>',1),('json','{"value":"bad"}',0),('json','{"value":3,"extra":4}',0)]
        for fmt,text,expected in cases:
            p={'source':'if_structured','schema_type':fmt,'schema_str':schema,'response_mode':'text'}
            outputs,_=parse_completion(text)
            self.assertEqual(local_score(p,outputs),expected)
            self.assertEqual(local_score(p,outputs),official_score(p,outputs))
        p={'source':'if_structured','schema_type':'csv','response_mode':'text',
           'schema_str':json.dumps({'type':'array','items':{'type':'object','properties':{'value':{'type':'integer'}}}})}
        outputs,_=parse_completion('value\n3')
        self.assertEqual(local_score(p,outputs),1)
        self.assertEqual(local_score(p,outputs),official_score(p,outputs))
        p={'source':'if_structured','schema_type':'json','schema_str':schema,'response_mode':'tool_call','tool_name':'extract','tool_payload_key':'payload'}
        call='<tool_call>{"name":"extract","arguments":{"payload":{"value":3}}}</tool_call>'
        for text,expected in [(call,1),(call+call,0),(call.replace('extract','wrong'),0)]:
            outputs,_=parse_completion(text)
            self.assertEqual(local_score(p,outputs),expected)
            self.assertEqual(local_score(p,outputs),official_score(p,outputs))

    def test_invalid_judge_verdict_is_infrastructure_error(self):
        from unittest.mock import patch, AsyncMock
        from multidomain.reward import science_score
        from multidomain.common import InfrastructureError
        class Tokenizer:
            name_or_path='frozen-judge'
            def apply_chat_template(self, *a, **kw):return [1,2,3]
        class Response:
            async def __aenter__(self):return self
            async def __aexit__(self,*a):return False
            def raise_for_status(self):pass
            async def json(self):return {'choices':[{'text':'not a verdict','finish_reason':'stop'}]}
        class Session(Response):
            def __init__(self, *a, **kw):pass
            def post(self, *a, **kw):return Response()
        with patch('aiohttp.ClientSession',Session), patch('asyncio.sleep',new_callable=AsyncMock):
            with self.assertRaises(InfrastructureError):
                asyncio.run(science_score({'question':'Q','expected_answer':'A'},'candidate','test.invalid',Tokenizer()))


class SelectionTests(unittest.TestCase):
    def test_connected_components_transitive_and_order_independent(self):
        a,b=Groups(),Groups()
        keys=[['q1','instanceA'],['q2','instanceA'],['q2','conversationB']]
        for k in keys:a.join(k)
        for k in reversed(keys):b.join(k)
        self.assertEqual(a.find('q1'),a.find('conversationB'))
        self.assertEqual(a.find('q1'),b.find('q1'))

    def test_fixed_evaluation_ids_across_domain_switches(self):
        c=read_config(ROOT/'config/multidomain/initial.yaml')
        rows=[]
        for d,dc in c['domains'].items():
            for source in dc['sources']:
                for i in range(700):
                    rows.append({'sample_id':f'{source}/{i}','domain':d,'source':source,'eligible':True,
                                 'split_group_id':f'{source}/{i}','category':str(i%4)})
        first,_=choose(rows,c)
        c=copy.deepcopy(c);c['domains']['swe_pivot']['enabled']=False
        second,_=choose(rows,c)
        for split in ('validation','test'):
            self.assertEqual([r['sample_id'] for r in first[split] if r['domain']!='swe_pivot'],[r['sample_id'] for r in second[split]])
        self.assertEqual(len({r['sample_id'] for r in first['train']}),len(first['train']))

if __name__=='__main__':unittest.main()
