"""Acquire only public pinned scorer resources into a fresh dedicated directory.

All HTTP response bodies (including package metadata/indexes) are counted.
Installation is a separate offline operation using the resulting hash lock.
"""
from __future__ import annotations
import argparse,ast,email,hashlib,html.parser,json,os,platform,re,shutil,sys,time,urllib.request,urllib.error,urllib.parse,zipfile
from pathlib import Path
from datetime import datetime,timezone
from packaging.tags import sys_tags
from packaging.utils import parse_wheel_filename,canonicalize_name
from packaging.requirements import Requirement
from packaging.version import Version, InvalidVersion
from packaging.specifiers import SpecifierSet

AUTHOR='1036465276513621f77f1c2208d742e4a430781f'
CLIP='d05afc436d78f1c48dc0dbf8e5980a9d471f35f6'
WEIGHT_SHA='40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
WEIGHT_URL='https://openaipublic.azureedge.net/clip/models/'+WEIGHT_SHA+'/ViT-B-32.pt'
ALLOWED={'raw.githubusercontent.com','openaipublic.azureedge.net','download.pytorch.org','download-r2.pytorch.org','pypi.org','files.pythonhosted.org'}
LIMIT=4*1024**3
PINNED={'torch':'2.9.1+cpu','torchvision':'0.24.1+cpu','numpy':'1.26.4','pillow':'11.3.0','ftfy':'6.3.1','regex':'2025.9.18','packaging':'25.0','tqdm':'4.67.1','pytest':'8.4.2'}

def now(): return datetime.now(timezone.utc).isoformat()
def check_url(url):
 p=urllib.parse.urlsplit(url)
 if p.scheme!='https' or p.hostname not in ALLOWED or p.username or p.password or p.query: raise ValueError('unapproved public distribution URL')

def safe_path(path,root=None):
 path=Path(path)
 if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute acquisition path without traversal required')
 if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('acquisition symlink path refused')
 if root is not None and not path.is_relative_to(root):raise ValueError('acquisition destination escaped dedicated root')
 return path

def received_status(url,expected):
 if expected:return 'VERIFIED'
 p=urllib.parse.urlsplit(url)
 if p.hostname=='raw.githubusercontent.com' and re.fullmatch(r'/(jmhessel/clipscore/'+AUTHOR+r'|openai/CLIP/'+CLIP+r')/.+',p.path):
  return 'RECEIVED_HTTPS_REVISION_BOUND'
 return 'RECEIVED_HTTPS_HASHED_METADATA'

def license_members(archive):
 names=[n for n in archive.namelist() if not n.endswith('/') and any(word in Path(n).name.lower() for word in ['license','licence','copying','notice'])]
 return [{'archive_member':n,'sha256':hashlib.sha256(archive.read(n)).hexdigest(),'bytes':len(archive.read(n))} for n in names]

def tree_bytes(root): return sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())

class Fetcher:
 def __init__(self,root):
  root=safe_path(root)
  self.root=root;self.total=0;self.requests=0;self.events=[]
  safe_path(root/'DOWNLOAD_RECEIPT.json',root)
  if (root/'DOWNLOAD_RECEIPT.json').exists():
   saved=json.loads((root/'DOWNLOAD_RECEIPT.json').read_text())
   self.total=saved['response_body_bytes'];self.requests=saved['http_requests'];self.events=saved['events']
  fetcher=self
  class Redirect(urllib.request.HTTPRedirectHandler):
   def redirect_request(self,req,fp,code,msg,headers,newurl):
    check_url(newurl)
    redirected=super().redirect_request(req,fp,code,msg,headers,newurl)
    if redirected is None:return None
    event={'status':code,'from':req.full_url,'to':newurl,'utc':now(),'body_bytes':0}
    fetcher.events[-1]['redirects'].append(event)
    # urllib otherwise drains this body outside the accounting loop.
    for _ in fetcher.read_body(fp,event,'body_bytes',headers=headers):pass
    fetcher.requests+=1
    return redirected
  self.redirect_handler=Redirect()
  self.opener=urllib.request.build_opener(self.redirect_handler)
 def read_body(self,response,counter,field='bytes',headers=None):
  headers=response.headers if headers is None else headers
  size=headers.get('Content-Length');size=int(size) if size is not None else None
  if size is not None and (size<0 or self.total+size>LIMIT):raise RuntimeError('download limit would be exceeded')
  received=0
  while size is None or received<size:
   remaining=LIMIT-self.total
   if remaining<=0:raise RuntimeError('download byte limit reached; unknown-length EOF unverified')
   amount=min(1024**2,remaining,size-received if size is not None else remaining)
   chunk=response.read(amount)
   if not chunk:break
   if len(chunk)>amount:raise RuntimeError('response violated bounded read')
   self.total+=len(chunk);received+=len(chunk);counter[field]=counter.get(field,0)+len(chunk)
   yield chunk
  if size is not None and received!=size:raise RuntimeError('response byte count mismatch')
 def save(self):
  value={'schema':'goal013a_public_acquisition_v1','updated_utc':now(),'http_requests':self.requests,'response_body_bytes':self.total,'download_limit_bytes':LIMIT,'work_area_limit_bytes':8*1024**3,'response_body_accounting':'Application-read final, redirect and HTTP-error bodies; TLS/header/wire overhead not measured. HTTP count is attempts initiated, including redirect hops. Browser documentation reads are separate from resource acquisition.','research_api_requests_sent':0,'events':self.events}
  safe_path(self.root/'DOWNLOAD_RECEIPT.json',self.root).write_text(json.dumps(value,indent=2)+'\n')
 def get(self,url,path,expected=None):
  check_url(url);path=safe_path(path,self.root)
  if path.exists():
   prior=[e for e in self.events if e['url']==url and e['local_path']==str(path) and e['status'] in {'VERIFIED','RECEIVED_HTTPS_REVISION_BOUND','RECEIVED_HTTPS_HASHED_METADATA'}]
   assert len(prior)==1, 'unverified existing acquisition file'
   raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
   assert len(raw)==prior[0]['bytes'] and actual==prior[0]['measured_sha256']
   assert expected is None or actual==expected
   return path
  path.parent.mkdir(parents=True,exist_ok=True)
  if shutil.disk_usage(self.root).free<10*1024**3 or tree_bytes(self.root)>8*1024**3:raise RuntimeError('work area or free disk limit')
  event={'url':url,'local_path':str(path),'started_utc':now(),'redirects':[],'expected_sha256':expected,'bytes':0,'status':'STARTED'}
  self.events.append(event);self.requests+=1;self.save();h=hashlib.sha256()
  try:
   with self.opener.open(urllib.request.Request(url,headers={'User-Agent':'Goal013A-reproducible-public-acquisition','Accept-Encoding':'identity'}),timeout=60) as response:
    event['http_status']=response.status;event['final_url']=response.geturl();check_url(response.geturl())
    size=response.headers.get('Content-Length');event['declared_content_length']=int(size) if size is not None else None
    with path.open('xb') as target:
     for chunk in self.read_body(response,event):
      target.write(chunk);h.update(chunk)
      if shutil.disk_usage(self.root).free<10*1024**3:raise RuntimeError('free disk limit reached')
    event['measured_sha256']=h.hexdigest()
    if size is not None and event['bytes']!=int(size):raise RuntimeError('response byte count mismatch')
    if expected is not None and h.hexdigest()!=expected:raise RuntimeError('artifact SHA256 mismatch')
    event['status']=received_status(url,expected)
  except Exception as exc:
   if isinstance(exc,urllib.error.HTTPError):
    try:
     for _ in self.read_body(exc,event,'http_error_body_bytes'):pass
    except Exception as accounting_error:event['http_error_body_accounting_error']=type(accounting_error).__name__
   event['status']='FAILED';event['error_type']=type(exc).__name__
   event['http_error_status']=getattr(exc,'code',None)
   event['failure_stage']='HTTPS request/body or artifact verification';self.save();raise
  finally:event['finished_utc']=now();self.save()
  print(json.dumps({'received':path.name,'bytes':event['bytes'],'total_download_bytes':self.total}),flush=True)
  return path

class Links(html.parser.HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   self.urls += [v for k,v in attrs if k=='href']

def choose_wheel(entries,name,version):
 tags=list(sys_tags());rank={t:i for i,t in enumerate(tags)};matches=[]
 for filename,url,sha in entries:
  if not filename.endswith('.whl'):continue
  n,v,_,ws=parse_wheel_filename(filename)
  common=set(ws)&set(rank)
  if canonicalize_name(n)==canonicalize_name(name) and v==Version(version) and common:matches.append((min(rank[t] for t in common),filename,url,sha))
 if not matches:raise ValueError('no compatible binary wheel for '+name+'=='+version)
 return sorted(matches)[0][1:]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args();root=args.root
 root=safe_path(root)
 if root.parent!=Path('/root/minecraft-tools') or not re.fullmatch(r'goal013a-\d{8}T\d{6}Z',root.name):raise ValueError('fresh canonical Goal013A tools root required')
 if args.resume:assert root.is_dir() and (root/'DOWNLOAD_RECEIPT.json').is_file()
 else:
  assert not root.exists();root.mkdir(parents=True,exist_ok=False)
  for name in ['sources','models','wheelhouse','metadata','tmp','cache']: (root/name).mkdir()
 f=Fetcher(root)
 for name in ['clipscore.py','LICENSE','README.md']:
  f.get(f'https://raw.githubusercontent.com/jmhessel/clipscore/{AUTHOR}/{name}',root/'sources/clipscore'/name)
 for name in ['clip/__init__.py','clip/clip.py','clip/model.py','clip/simple_tokenizer.py','clip/bpe_simple_vocab_16e6.txt.gz','LICENSE','README.md','model-card.md']:
  f.get(f'https://raw.githubusercontent.com/openai/CLIP/{CLIP}/{name}',root/'sources/openai_clip'/name)
 module=ast.parse((root/'sources/openai_clip/clip/clip.py').read_text())
 models=next(ast.literal_eval(n.value) for n in module.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_MODELS' for t in n.targets))
 assert models['ViT-B/32']==WEIGHT_URL
 f.get(WEIGHT_URL,root/'models/ViT-B-32.pt',WEIGHT_SHA)
 wheels={};required={};queue=[Requirement(n+'=='+v) for n,v in PINNED.items()]
 while queue:
  requirement=queue.pop(0)
  if requirement.marker and not requirement.marker.evaluate({'extra':''}):continue
  name=canonicalize_name(requirement.name);required.setdefault(name,[]).append(str(requirement.specifier))
  if name in wheels:
   assert Version(wheels[name]['version']) in requirement.specifier,(name,required[name]);continue
  version=PINNED.get(name)
  if name in {'torch','torchvision'}:
   index=f.get('https://download.pytorch.org/whl/cpu/'+name+'/',root/'metadata'/(name+'-cpu-index.html'))
   links=Links();links.feed(index.read_text());entries=[]
   for link in links.urls:
    url=urllib.parse.urljoin('https://download.pytorch.org/whl/cpu/'+name+'/',link);parsed=urllib.parse.urlsplit(url)
    expected=urllib.parse.parse_qs(parsed.fragment).get('sha256',[None])[0]
    entries.append((urllib.parse.unquote(Path(parsed.path).name),urllib.parse.urlunsplit(parsed._replace(fragment='')),expected))
  else:
   meta_path=root/'metadata'/(name+'.json')
   metadata=json.loads(f.get('https://pypi.org/pypi/'+name+('/'+version if version else '')+'/json',meta_path).read_text())
   if version is None:
    candidates=[]
    for candidate,files in metadata['releases'].items():
     try:v=Version(candidate)
     except InvalidVersion:continue
     if v.is_prerelease or v.is_devrelease or v not in requirement.specifier:continue
     eligible=[r for r in files if r['packagetype']=='bdist_wheel' and not r.get('yanked')
               and Version(platform.python_version()) in SpecifierSet(r.get('requires_python') or '')]
     if not eligible:continue
     try:choose_wheel([(r['filename'],r['url'],r['digests']['sha256']) for r in eligible],name,candidate)
     except ValueError:continue
     candidates.append(v)
    assert candidates, 'no compatible dependency release'
    version=str(max(candidates))
    if version!=metadata['info']['version']:
     metadata=json.loads(f.get('https://pypi.org/pypi/'+name+'/'+version+'/json',root/'metadata'/(name+'-'+version+'.json')).read_text())
   assert Version(version) in requirement.specifier,(name,version,str(requirement))
   assert Version(platform.python_version()) in SpecifierSet(metadata['info'].get('requires_python') or ''), 'Python compatibility mismatch'
   entries=[(r['filename'],r['url'],r['digests']['sha256']) for r in metadata['urls'] if r['packagetype']=='bdist_wheel' and not r.get('yanked')]
  filename,url,expected=choose_wheel(entries,name,version);assert expected,'published wheel SHA required'
  path=f.get(url,root/'wheelhouse'/filename,expected)
  with zipfile.ZipFile(path) as z:
   names=[n for n in z.namelist() if n.endswith('.dist-info/METADATA') and n.count('/')==1];assert len(names)==1
   m=email.message_from_bytes(z.read(names[0]));requirements=m.get_all('Requires-Dist',[])
   assert canonicalize_name(m['Name'])==name and Version(m['Version'])==Version(version)
   license_info=license_members(z)
  wheels[name]={'version':version,'filename':filename,'url':url,'sha256':expected,'bytes':path.stat().st_size,'requires_dist':requirements,'license_expression':m.get('License-Expression'),'license_metadata':m.get('License'),'license_members':license_info}
  queue.extend(Requirement(r) for r in requirements)
 lock=''.join(f'{name}=={r["version"]} --hash=sha256:{r["sha256"]}\n' for name,r in sorted(wheels.items()))
 safe_path(root/'requirements.lock',root).write_text(lock)
 safe_path(root/'DEPENDENCY_MANIFEST.json',root).write_text(json.dumps({'python':sys.version,'platform':platform.platform(),'selection':'Pinned torch/vision official supported pair and direct dependencies; transitive current stable PyPI wheel satisfying installed constraints; no outcome input.','packages':wheels},indent=2)+'\n')
 print(json.dumps({'status':'ACQUIRED','http_requests':f.requests,'response_body_bytes':f.total,'wheel_count':len(wheels),'work_area_bytes':tree_bytes(root)}),flush=True)

if __name__=='__main__':main()
