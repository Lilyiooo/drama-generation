import json
import os


def install():
    import narrative_memory_feedback_v2_7_self_evolving.run_generation as generation
    from narrative_memory_feedback_v2_7_self_evolving.io_utils import append_jsonl, utc_now
    from narrative_memory_feedback_v2_7_self_evolving.schema import load_protocol

    original_prompt = generation.build_compact_extraction_prompt

    def compact(**kwargs):
        return original_prompt(**kwargs) + '\n\n字段校验补充：primary_move_id、secondary_move_ids 中的每个元素，以及非空 writeback_candidate.primary_move_id，都只能逐字选自允许的 move_id 列表。mechanism_family 不是 move_id，禁止把 RESOURCE_LOSS 等机制名称填入这些字段。无适用次要动作时 secondary_move_ids 输出 []，不能自行发明或改写 ID。仍须完整提取事实，不能用空状态回避任务。'

    generation.build_compact_extraction_prompt = compact
    original_call = generation.call_and_record
    taxonomy = load_protocol()[4]
    move_schema = {'type': 'string', 'enum': [item['move_id'] for item in taxonomy]}
    strings = {'type': 'array', 'items': {'type': 'string'}}
    obligation = {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'type': {'type': 'string', 'enum': sorted(generation.OBLIGATION_TYPES)},
            'description': {'type': 'string', 'minLength': 1},
            'required_payoff': {'type': 'string', 'minLength': 1},
            'participants': strings,
            'deadline': {'anyOf': [{'type': 'string'}, {'type': 'null'}]},
        },
        'required': ['type', 'description', 'required_payoff', 'participants', 'deadline'],
    }
    feedback = {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'memory_id': {'type': 'string'}, 'evidence': {'type': 'string'}, 'harm_flags': strings,
            **{field: {'type': 'boolean'} for field in ['adopted', 'expected_effect_achieved',
                                                        'state_supported', 'anchor_supported', 'causal_supported']},
        },
        'required': ['memory_id', 'evidence', 'harm_flags', 'adopted', 'expected_effect_achieved',
                     'state_supported', 'anchor_supported', 'causal_supported'],
    }
    schema = {
        'type': 'object', 'additionalProperties': False,
        'properties': {
            'state_delta': {'type': 'object', 'additionalProperties': False,
                            'properties': {field: strings for field in generation.STATE_FIELDS},
                            'required': list(generation.STATE_FIELDS)},
            'resolved_goals': strings, 'resolved_unknown': strings, 'retracted_facts': strings,
            'new_obligations': {'type': 'array', 'items': obligation},
            'resolved_obligations': strings,
            'primary_move_id': move_schema,
            'secondary_move_ids': {'type': 'array', 'items': move_schema},
            'move_evidence': {'type': 'string'},
            'state_constraint_count': {'type': 'integer', 'minimum': 0},
            'state_consistency_issues': strings,
            'card_feedback': {'type': 'array', 'items': feedback},
            'writeback_candidate': {'anyOf': [
                {'type': 'null'},
                {'type': 'object', 'properties': {
                    'primary_move_id': move_schema, 'move_id': move_schema,
                    'quality_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
                    'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
                }, 'required': ['primary_move_id', 'quality_score', 'confidence']},
            ]},
        },
        'required': ['primary_move_id', 'secondary_move_ids', 'state_delta', 'resolved_goals',
                     'resolved_unknown', 'retracted_facts', 'new_obligations', 'resolved_obligations',
                     'move_evidence', 'state_constraint_count', 'state_consistency_issues',
                     'card_feedback', 'writeback_candidate'],
    }

    class ConstrainedClient:
        def __init__(self, client):
            self.client = client

        def complete(self, messages, settings, *, seed, purpose):
            from openai import OpenAI
            request = {
                'model': self.client.model, 'messages': messages,
                'temperature': settings['temperature'], 'max_tokens': settings['max_output_tokens'],
                'seed': seed, 'response_format': settings['response_format'],
            }
            for key in ['top_p', 'presence_penalty', 'frequency_penalty']:
                if key in settings:
                    request[key] = settings[key]
            if self.client.disable_thinking:
                request['extra_body'] = {'chat_template_kwargs': {'enable_thinking': False}}
            with OpenAI(api_key=self.client.api_key, base_url=self.client.base_url,
                        timeout=float(os.environ.get('NMF_API_TIMEOUT_SECONDS', '600')), max_retries=2) as client:
                response = client.chat.completions.create(**request)
            choice = response.choices[0]
            if choice.finish_reason == 'length' or not choice.message.content:
                raise RuntimeError(f'{purpose}: constrained extraction truncated or empty')
            return choice.message.content

    def call(client, **kwargs):
        if kwargs['purpose'].startswith('extraction_retry_'):
            kwargs['settings'] = {**kwargs['settings'], 'response_format': {
                'type': 'json_schema',
                'json_schema': {'name': 'extraction_move_ids', 'schema': schema},
            }}
            client = ConstrainedClient(client)
        response = original_call(client, **kwargs)
        if kwargs['purpose'] == 'extraction' or kwargs['purpose'].startswith('extraction_retry_'):
            append_jsonl(kwargs['output_dir'] / 'recovery_extraction_responses.jsonl', {
                'job_id': kwargs['job']['job_id'], 'purpose': kwargs['purpose'],
                'created_at': utc_now(), 'response': response,
                'response_format': kwargs['settings'].get('response_format'),
            })
        return response

    generation.call_and_record = call
