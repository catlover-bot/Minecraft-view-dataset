"""Pure contracts and fake HTTP; never query real credentials or providers."""
import copy
from decimal import Decimal
import io
import json
import socket
import urllib.error

import pytest

from offline_evaluation.experiments.goal015 import adapter as a
from offline_evaluation.resmoke_v2 import adapter as old_adapter
from offline_evaluation.resmoke_v2.schema import action_schema
from offline_evaluation.server_v2 import providers
from offline_evaluation.server_v2.cohort_v4 import _serialize_exact, strict_saved_json, typed_tree
from offline_evaluation.server_v2.core import ValidationError


S = '{"summary":"  A stone building. 日本語  "}'
PRICING = {'openai': {'input_usd_per_million': .25, 'output_usd_per_million': 2},
           'anthropic': {'input_usd_per_million': 1, 'output_usd_per_million': 5}}


def envelope(provider, *, model=None, refusal=False, stop=None, text='{"operations":[]}', usage=True):
    if provider == 'openai':
        block = {'type': 'refusal', 'refusal': 'fixture'} if refusal else {'type': 'output_text', 'text': text}
        data = {'id': 'fixture', 'model': model or a.MODELS[provider],
                'status': 'incomplete' if stop else 'completed',
                'incomplete_details': {'reason': stop} if stop else None,
                'output': [{'type': 'message', 'content': [block]}]}
    else:
        data = {'id': 'fixture', 'model': model or a.MODELS[provider],
                'stop_reason': 'refusal' if refusal else (stop or 'end_turn'),
                'content': [{'type': 'text', 'text': text}]}
    if usage:
        data['usage'] = {'input_tokens': 10, 'output_tokens': 2}
    return json.dumps(data).encode()


def test_exact_models_sampling_schema_and_frozen_prompt():
    original = dict(old_adapter.MODELS)
    oa, an = [a.build_payload(p, S) for p in a.MODELS]
    assert oa['model'] == 'gpt-5-mini-2025-08-07'
    assert an['model'] == 'claude-haiku-4-5-20251001'
    assert oa['reasoning'] == {'effort': 'medium'}
    assert not ({'temperature', 'top_p', 'seed'} & oa.keys())
    assert an['temperature'] == 1.0 and an['thinking'] == {'type': 'disabled'}
    assert oa['max_output_tokens'] == an['max_tokens'] == 32768
    assert oa['instructions'] == an['system'] == a.PROMPT_PATH.read_text()
    assert oa['input'] == an['messages'][0]['content'] == S
    assert oa['text']['format']['schema'] == an['output_config']['format']['schema'] == action_schema()
    assert dict(old_adapter.MODELS) == original and old_adapter.MODELS['openai'] == 'gpt-5-mini'
    with pytest.raises(TypeError): a.MODELS['openai'] = 'other'


def test_full_record_keeps_decimal_typed_tree_array_and_uncertainty():
    full = {field: None for field in a.FIELDS}
    full.update(summary=' unchanged ', dimensions_estimate={'width': Decimal('123456789.123456789000'),
                'depth': Decimal('1E+2'), 'height': 1, 'uncertain': True},
                materials=['stone', 'unknown'], uncertainties=['maybe', 'not sure'])
    original = _serialize_exact(full)
    for provider in a.MODELS:
        payload = a.build_payload(provider, original)
        actual = payload['input'] if provider == 'openai' else payload['messages'][0]['content']
        assert actual == original
        assert typed_tree(strict_saved_json(actual)) == typed_tree(full)
        a.validate_payload(provider, payload)


@pytest.mark.parametrize('text', [
    '{}', '[]', '{"summary":""}', '{"summary":"  "}', '{"summary":1}',
    '{"summary":"x","scene_id":"building_001"}', '{"summary":"x","GT":{}}',
    '{"summary":"x","condition":"FULL_RECORD"}', '{"summary":"x","repeat":1}',
    '{"summary":"x","provider":"openai"}', '{"summary":"x","summary":"y"}',
    '{"summary":NaN}', '{ "summary": "x" }', '{"summary":"x"}\n',
])
def test_only_exact_canonical_s1_or_f8_input(text):
    with pytest.raises(a.ProviderError, match='configuration_failure'):
        a.build_payload('openai', text)


@pytest.mark.parametrize('provider', a.MODELS)
@pytest.mark.parametrize('change', ['model', 'cap', 'prompt', 'schema', 'tools', 'sampling', 'bool_type', 'conversation'])
def test_payload_allowlist_refuses_any_contract_change(provider, change):
    payload = a.build_payload(provider, S)
    if change == 'model': payload['model'] = 'gpt-5-mini' if provider == 'openai' else 'claude-haiku-4-5'
    elif change == 'cap': payload['max_output_tokens' if provider == 'openai' else 'max_tokens'] = 32767
    elif change == 'prompt': payload['instructions' if provider == 'openai' else 'system'] += ' improvement'
    elif change == 'schema':
        fmt = payload['text' if provider == 'openai' else 'output_config']['format']
        fmt['schema']['additionalProperties'] = True
    elif change == 'tools': payload['tools'] = []
    elif change == 'sampling':
        if provider == 'openai': payload['temperature'] = 0
        else: payload['temperature'] = .5
    elif change == 'bool_type':
        if provider == 'openai': payload['stream'] = 0
        else: payload['temperature'] = True
    else:
        if provider == 'openai': payload['previous_response_id'] = 'fixture'
        else: payload['messages'].append({'role':'assistant', 'content':'prior'})
    with pytest.raises(a.ProviderError, match='configuration_failure'): a.validate_payload(provider, payload)


class ForbiddenEnvironment(dict):
    def get(self, *args):
        raise AssertionError('credential lookup before authorization')


@pytest.mark.parametrize('provider', a.MODELS)
def test_gate_precedes_credential_read_and_transport(provider, monkeypatch):
    monkeypatch.setattr(providers.os, 'environ', ForbiddenEnvironment())
    calls=[]
    payload=a.build_payload(provider,S)
    for options in ({}, {'api_allowed':True}, {'permission':True}, {'api_allowed':1,'permission':True}):
        with pytest.raises(a.ProviderError, match='api_disabled'):
            a.send_request(provider,payload,transport=lambda *x:calls.append(x),**options)
    payload['model']='wrong'
    with pytest.raises(a.ProviderError,match='configuration_failure'):
        a.send_request(provider,payload,api_allowed=True,permission=True,transport=lambda *x:calls.append(x))
    assert calls==[]


@pytest.mark.parametrize('provider', a.MODELS)
def test_one_fake_post_after_explicit_fixture_flags_no_real_key_lookup(provider, monkeypatch):
    # Replace mapping rather than inspecting or temporarily reading existing keys.
    monkeypatch.setattr(providers.os,'environ',{providers.KEY_NAMES[provider]:'fixture-key-not-real'})
    calls=[];payload=a.build_payload(provider,S)
    def transport(request,timeout):
        calls.append((request.full_url,request.get_method(),json.loads(request.data),timeout))
        assert request.get_header('Authorization')=='Bearer fixture-key-not-real' if provider=='openai' else request.get_header('X-api-key')=='fixture-key-not-real'
        return envelope(provider)
    raw=a.send_request(provider,payload,api_allowed=True,permission=True,transport=transport)
    assert a.parse_response(provider,raw).complete
    assert calls==[(providers.ENDPOINTS[provider],'POST',payload,600.)]


@pytest.mark.parametrize('status,kind',[(401,'auth_failure'),(403,'auth_failure'),(520,'http_failure'),(429,'rate_limit'),(404,'model_unavailable'),(400,'schema_unsupported')])
def test_http_error_body_saved_separate_from_safe_error_no_retry(status,kind,monkeypatch):
    body=b'{"error":{"message":"json_schema PRIVATE_SENTINEL"}}';calls=[]
    class Opener:
        def open(self,request,timeout):
            calls.append(1)
            raise urllib.error.HTTPError('https://fixture.invalid/private',status,'bad',{},io.BytesIO(body))
    monkeypatch.setattr(a.urllib.request,'build_opener',lambda *x:Opener())
    with pytest.raises(a.TransportError) as exc: a.http_transport(object(),1)
    assert exc.value.kind==kind and exc.value.http_status==status and exc.value.status_code==status
    assert exc.value.http_response_received and exc.value.raw_response==body and exc.value.raw_complete
    assert 'PRIVATE_SENTINEL' not in str(exc.value) and 'fixture.invalid' not in repr(exc.value)
    assert len(calls)==1


def test_timeout_and_redirect_never_retry_or_claim_response_body(monkeypatch):
    class Opener:
        def open(self,*args,**kwargs): raise socket.timeout('PRIVATE_SENTINEL')
    monkeypatch.setattr(a.urllib.request,'build_opener',lambda *x:Opener())
    with pytest.raises(a.TransportError) as exc:a.http_transport(object(),1)
    assert exc.value.kind=='timeout' and not exc.value.http_response_received and exc.value.raw_response is None
    assert 'PRIVATE_SENTINEL' not in str(exc.value)
    with pytest.raises(providers.ProviderError):
        providers._NoRedirect().redirect_request(None,None,302,'',{},'https://fixture.invalid')


@pytest.mark.parametrize('provider', a.MODELS)
def test_response_snapshot_exactness_usage_retained_empty_is_valid(provider):
    response=a.parse_response(provider,envelope(provider))
    assert response.complete and a.validate_output(response.text)==[]
    bad=a.parse_response(provider,envelope(provider,model='undocumented-model'))
    assert not bad.complete and bad.failure_kind=='response_model_mismatch'
    assert bad.usage=={'input_tokens':10,'output_tokens':2}
    missing=json.loads(envelope(provider));missing.pop('model')
    assert a.parse_response(provider,json.dumps(missing).encode()).failure_kind=='response_model_mismatch'


@pytest.mark.parametrize('provider', a.MODELS)
def test_refusal_truncation_and_incomplete_distinct(provider):
    refused=a.parse_response(provider,envelope(provider,refusal=True))
    assert not refused.complete and refused.failure_kind=='refusal'
    truncated=a.parse_response(provider,envelope(provider,stop='max_output_tokens' if provider=='openai' else 'max_tokens'))
    assert not truncated.complete and truncated.failure_kind=='truncation'
    incomplete=a.parse_response(provider,envelope(provider,stop='unknown_stop'))
    assert not incomplete.complete and incomplete.failure_kind=='incomplete_response'


@pytest.mark.parametrize('provider', a.MODELS)
def test_unrequested_response_features_fail_closed_but_retain_usage(provider):
    data=json.loads(envelope(provider))
    if provider=='openai':
        data['output'].append({'type':'web_search_call','id':'fixture-tool'})
    else:
        data['content'].append({'type':'thinking','thinking':'fixture','signature':'fixture'})
    response=a.parse_response(provider,json.dumps(data).encode())
    assert not response.complete and response.failure_kind=='unexpected_response_feature'
    assert response.usage=={'input_tokens':10,'output_tokens':2}
    assert response.model==a.MODELS[provider]


def test_duplicate_envelope_and_output_schema_rejected_without_repair():
    with pytest.raises(a.ProviderError,match='response_parse_failure'):
        a.parse_response('openai',b'{"model":"a","model":"b"}')
    for text in ('[]','```json\n{"operations":[]}\n```','{"operations":[],"extra":1}',
                 '{"operations":[{"op":"set","x":true,"y":0,"z":0,"block":"stone"}]}'):
        with pytest.raises(ValidationError):a.validate_output(text)


@pytest.mark.parametrize('provider',a.MODELS)
def test_unknown_usage_never_zero_and_known_usage_charged_full(provider):
    for usage in (None,{}, {'input_tokens':True,'output_tokens':2},{'input_tokens':1,'output_tokens':-1}):
        result=a.classify_usage(provider,usage,PRICING[provider])
        assert not result['usage_known'] and result['cost_usd'] is None and result['input_tokens'] is None
    usage={'input_tokens':100,'output_tokens':50}
    if provider=='openai':
        usage['output_tokens_details']={'reasoning_tokens':40}
        usage['input_tokens_details']={'cached_tokens':80}
    else:usage['cache_read_input_tokens']=20
    result=a.classify_usage(provider,usage,PRICING[provider])
    assert result['usage_known'] and result['output_tokens']==50
    assert result['input_tokens']==(100 if provider=='openai' else 120)
    assert result['cost_usd']==(.000125 if provider=='openai' else .00037)


def test_unexpected_cache_write_and_tools_make_usage_unknown():
    for usage in ({'input_tokens':1,'output_tokens':1,'cache_creation_input_tokens':1},
                  {'input_tokens':1,'output_tokens':1,'server_tool_use':{'web_search_requests':1}}):
        assert not a.classify_usage('anthropic',usage,PRICING['anthropic'])['usage_known']
