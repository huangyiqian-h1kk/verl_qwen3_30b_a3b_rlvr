"""Single assistant turn, native per-row tools, shared 64K budget, no tool execution."""
import json
from uuid import uuid4
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from multidomain.common import DataError, digest
from multidomain.template import render


class MultiDomainAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, priority=0, **kwargs):
        messages = list(kwargs['raw_prompt'])
        tools = json.loads(kwargs['tools_json'])
        extra = kwargs['extra_info']
        ids = await self.loop.run_in_executor(None, lambda: render(self.tokenizer, messages, tools))
        if len(ids) != extra['prompt_tokens'] or digest(ids) != extra['prompt_token_hash']:
            raise DataError('Prompt tokenization differs from frozen preparation: ' + extra['sample_id'])
        budget = min(int(extra['generation_max_tokens']), 65536 - len(ids))
        if budget < 1:
            raise DataError('No generation headroom')
        params = dict(sampling_params)
        params['max_tokens'] = budget
        metrics = {}
        with simple_timer('generate_sequences', metrics):
            result = await self.server_manager.generate(request_id=uuid4().hex, prompt_ids=ids,
                sampling_params=params, priority=int(priority))
        if len(result.token_ids) > budget:
            raise DataError('Serving backend exceeded the per-record response budget')
        metrics['num_preempted'] = result.num_preempted if result.num_preempted is not None else -1
        fields = dict(result.extra_fields or {})
        fields.update(raw_completion=self.tokenizer.decode(result.token_ids, skip_special_tokens=False),
                      turn_scores=[], tool_rewards=[])
        import os
        if os.environ.get('MD_STAGE') in ('smoke', 'baseline-train', 'baseline-val', 'evaluate'):
            import socket
            from pathlib import Path
            directory = Path(os.environ['MD_RUN_DIR']) / 'raw_generations' / os.environ['MD_STAGE']
            directory.mkdir(parents=True, exist_ok=True)
            record = {'sample_id': extra['sample_id'], 'source': extra['source'],
                      'prompt_tokens': len(ids), 'response_tokens': len(result.token_ids),
                      'generation_max_tokens': budget, 'completion': fields['raw_completion']}
            with (directory / f'{socket.gethostname()}_{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        return AgentLoopOutput(prompt_ids=ids, response_ids=result.token_ids,
            response_mask=[1] * len(result.token_ids), response_logprobs=result.log_probs,
            routed_experts=result.routed_experts, multi_modal_data={}, num_turns=2,
            metrics=metrics, extra_fields=fields)
