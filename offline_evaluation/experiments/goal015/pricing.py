"""Official-document GET capture and separately reviewed same-contract evidence.

Capture never asserts semantic verification. Sealing requires a supplied reviewed
contract and a review note; a changed contract cannot silently refresh this plan.
No credentials, account endpoint, model listing, token count or generation.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request

from .runtime import descriptor, read_json, require, _save, verify_plan, path_safe

HOSTS=frozenset(('developers.openai.com','platform.claude.com'))
MAX_BYTES=16*1024**2


def official(url):
    parsed=urllib.parse.urlsplit(url)
    require(parsed.scheme=='https' and parsed.hostname in HOSTS and not parsed.username
            and not parsed.password and parsed.port in (None,443),'official public documentation HTTPS required')
    return url


class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        official(newurl)
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def capture(plan_path,output_dir):
    plan=verify_plan(plan_path);previous=read_json(plan['pricing_evidence']['path'])
    root=path_safe(output_dir)
    require(root.is_relative_to(Path(plan['canonical_root'])) and not root.exists(),'new directory inside canonical run required')
    root.mkdir(parents=True);documents=[]
    opener=urllib.request.build_opener(OfficialRedirect())
    for i,source in enumerate(previous['documents']):
        url=official(source['url'])
        with opener.open(urllib.request.Request(url,headers={'User-Agent':'Goal015-public-docs/1.0'}),timeout=60) as response:
            official(response.url);require(response.status==200,'documentation HTTP200 required')
            raw=response.read(MAX_BYTES+1);require(len(raw)<=MAX_BYTES,'documentation body exceeds bound')
            path=root/f'document-{i:02d}.html'
            with path.open('xb') as stream:stream.write(raw)
            documents.append(dict(descriptor(path),url=url,final_url=response.url,http_status=response.status,
                                  retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
    result={'schema':'goal015_unreviewed_document_capture_v1','review_status':'REQUIRES_HUMAN_CONTRACT_REVIEW',
            'plan_id':plan['plan_id'],'documents':documents,'research_api_requests_sent':0,
            'model_account_availability':'NOT_RUN','key_queries':0}
    _save(root/'capture.json',result,exclusive=True)
    return {'capture':descriptor(root/'capture.json'),'review_status':result['review_status']}


def seal(plan_path,capture_path,reviewed_contract_path,review_note,output_path):
    plan=verify_plan(plan_path);captured=read_json(capture_path);contract=read_json(reviewed_contract_path)
    require(captured['plan_id']==plan['plan_id'],'capture belongs to another plan')
    require(contract==plan['pricing_contract'],'changed price/model/contract requires new explicit approval')
    require(isinstance(review_note,str) and len(review_note.strip())>=20,'specific official-document review note required')
    previous=read_json(plan['pricing_evidence']['path']);documents=captured['documents']
    require(sorted(d['url'] for d in documents)==sorted(d['url'] for d in previous['documents']),'all original official sources must be reviewed')
    for doc in documents:
        official(doc['url']);official(doc['final_url'])
        require(descriptor(doc['path'])=={k:doc[k] for k in ('path','bytes','sha256')},'captured document changed')
        age=(datetime.now(timezone.utc)-datetime.fromisoformat(doc['retrieved_at_utc'])).total_seconds()
        require(0<=age<=86400,'capture too old or future dated')
    stamp=min(d['retrieved_at_utc'] for d in documents)
    result={'schema':'goal015_reviewed_pricing_evidence_v1','review_status':'REVIEWED_OFFICIAL_DOCUMENTS',
            'retrieved_at_utc':stamp,'pricing_contract':contract,'documents':documents,'review_note':review_note,
            'parent_capture':descriptor(capture_path),'reviewed_contract':descriptor(reviewed_contract_path),
            'research_api_requests_sent':0,'model_account_availability':'NOT_RUN','key_queries':0,
            'price_or_contract_change':False,'same_plan_id':plan['plan_id']}
    path=path_safe(output_path)
    require(path.is_relative_to(Path(plan['canonical_root'])),'new evidence stays in canonical run')
    _save(path,result,exclusive=True)
    return descriptor(path)
