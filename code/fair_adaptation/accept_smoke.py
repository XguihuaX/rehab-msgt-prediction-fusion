"""Independent, fail-closed gate for the prespecified eight-cell GPU smoke.
Reads artifacts and CPU checkpoint tensors only; never fits or selects a new model.
A passing receipt is bound to the final freeze and all examined smoke artifacts.
"""
import argparse,csv,hashlib,json,math,sys,traceback
from pathlib import Path
import numpy as np
import torch

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
 if not ok:raise RuntimeError(message)
def state_hash(state):
 h=hashlib.sha256()
 for key,value in sorted(state.items()):
  v=value.detach().cpu().contiguous();h.update(key.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
def finite_probability(values,label):
 x=np.asarray(values,dtype=float);require(x.size>0 and np.isfinite(x).all() and (x>=0).all() and (x<=1).all(),label)
def macro(y,p,people,dataset):
 e=[np.mean((p[people==v]-y[people==v])**2) for v in sorted(set(people))]
 return float(np.mean(np.sqrt(e) if dataset=='U2K' else e))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve();out=root/'SMOKE_ACCEPTANCE.json';evidence={};freeze_sha=None
 def track(p):
  p=Path(p);evidence[str(p.relative_to(root)) if p.is_relative_to(root) else str(p)]=sha(p)
 def load(p):track(p);return read(p)
 try:
  freeze=root/'freeze.json';freeze_sha=sha(freeze);f=load(freeze)
  required={'PROTOCOL.md','env-spec.json','prepared/manifest.json','code/run.py','code/accept_smoke.py','code/dispatch.py','code/analyze.py','code/inference_cost.py','code/prepare.py'}
  required|={str(p.relative_to(root)) for p in (root/'dependencies').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
  require(required<=set(f['files']),'Freeze does not cover required protocol/code/dependencies')
  for rel,h in f['files'].items():require(sha(root/rel)==h,'Freeze mismatch: '+rel);track(root/rel)
  spec=load(root/'env-spec.json');m=load(root/'prepared/manifest.json');smoke=root/'smoke';env=load(smoke/'environment.json');status=load(smoke/'status.json')
  require(not (smoke/'FAILED.json').exists(),'Smoke retains a FAILED marker; explicit recovery required')
  require(status['status']=='completed' and status['completed_cells']==8 and status['freeze_sha256']==freeze_sha,'Smoke not complete under this freeze')
  require(env['freeze_sha256']==freeze_sha,'Smoke environment freeze differs')
  for key in ['torch','numpy']:require(env[key]==spec['packages'][key],key+' smoke version differs')
  require(torch.__version__==spec['packages']['torch'] and np.__version__==spec['packages']['numpy'],'Acceptance runtime differs from specified environment')
  require(Path(sys.executable).resolve()==Path(spec['python']).resolve(),'Acceptance Python executable differs')
  require(torch.cuda.is_available() and torch.cuda.get_device_name(0)==spec['gpu']==env['gpu'],'GPU does not match specified smoke environment')
  require(env['cuda']==torch.version.cuda and env['threads']==spec['cpu_threads'],'CUDA or thread witness mismatch')
  require(spec['precision']=='float32' and spec['deterministic_algorithms'] is True,'Unexpected precision/determinism contract')
  data={}
  for ds,z in m['data'].items():
   p=root/'prepared'/z['path'];require(sha(p)==z['sha256'],'Prepared data SHA mismatch');track(p);d=dict(np.load(p,allow_pickle=False));d['lookup']={str(v):i for i,v in enumerate(d['record_id'])};data[ds]=d
  cells={ds:next(c for c in m['cells'] if c['dataset']==ds and c['outer']==0 and c['budget']==4) for ds in ['U2K','irds']}
  tags={f'{ds}_{model}_o00_n4_s0_{method}' for ds in cells for model in ['msgt','ri_gcn'] for method in ['ft','lp']}
  require({p.name.removesuffix('_selection.json') for p in smoke.glob('*_selection.json')}==tags,'Smoke selection set differs from eight prescribed cells')
  expected_fits=set();checked_fits=set();source_cache={};lp_checks=0
  def checkpoint(path,require_frozen=False):
   p=Path(path);track(p);z=torch.load(p,map_location='cpu',weights_only=False)
   if require_frozen:require(z['freeze_sha256']==freeze_sha,'Checkpoint freeze differs')
   state=z['state_dict'];require(all(torch.isfinite(v).all().item() for v in state.values()),'Nonfinite checkpoint tensors')
   require(all(not v.is_floating_point() or v.dtype==torch.float32 for v in state.values()),'Non-FP32 floating checkpoint')
   return state
  def check_fit(name,contract,source_state,deps):
   nonlocal lp_checks
   expected_fits.add(name);base=smoke/name;rc=load(base/'completed.json');require(rc['freeze_sha256']==freeze_sha,'Fit freeze differs: '+name);require(rc['input_contract']==contract,'Fit request differs: '+name)
   require(rc['epochs']==1 and rc['train_ids']==contract['train_ids'] and rc['predict_ids']==contract['predict_ids'],'Fit role/history metadata differs')
   require(rc['method']==contract['method'] and rc['seed']==0 and rc['learning_rate']==contract['learning_rate'],'Fit metadata differs')
   for key in ['training_seconds','training_peak_allocated_bytes','training_and_prediction_peak_allocated_bytes']:
    require(math.isfinite(rc[key]) and rc[key]>=0,'Nonfinite/negative fit cost')
   require(set(rc['artifacts'])=={'checkpoint.pt','history.json','predictions.npz'},'Unexpected fit artifact inventory')
   for p,h in rc['artifacts'].items():require(sha(base/p)==h,'Fit artifact damaged: '+name+'/'+p);track(base/p)
   for p in [base/'completed.json']+[base/v for v in rc['artifacts']]:require(deps.get(str(p.relative_to(smoke)))==sha(p),'Selection missing fit dependency: '+str(p))
   hist=load(base/'history.json');require(len(hist)==1 and hist[0]['epoch']==1 and math.isfinite(hist[0]['loss']),'Smoke must have exactly one finite-loss epoch')
   with np.load(base/'predictions.npz',allow_pickle=False) as pp:
    require(pp['record_id'].astype(str).tolist()==contract['predict_ids'],'Fit prediction IDs differ');finite_probability(pp['probability'],'Nonfinite/out-of-range fit predictions')
   state=checkpoint(base/'checkpoint.pt',True)
   if contract['method']=='lp':
    prefix='head.' if contract['model']=='msgt' else 'fcn.'
    require(state.keys()==source_state.keys(),'LP/source state keys differ');require(any(k.startswith(prefix) for k in state),'LP head not found')
    for key,value in state.items():
     if not key.startswith(prefix):require(torch.equal(value,source_state[key]),'LP changed nonhead parameter/buffer: '+name+'/'+key)
    lp_checks+=1
   checked_fits.add(name)
   return rc
  for tag in sorted(tags):
   sel=load(smoke/(tag+'_selection.json'));ds=sel['dataset'];model=sel['model'];method=sel['method'];c=cells[ds];d=data[ds];lk=d['lookup']
   require(tag==f'{ds}_{model}_o00_n4_s0_{method}' and sel['outer']==0 and sel['budget']==4 and sel['seed']==0,'Selection identity mismatch')
   require(sel['freeze_sha256']==freeze_sha and sel['selection_participants']==c['train_participants'],'Selection freeze/participants differ')
   require(sel['development_fits']==6 and sel['target_fits']==1,'Incorrect development resource count')
   deps=sel['dependencies']
   for rel,h in deps.items():require(sha(smoke/rel)==h,'Selection dependency damaged: '+rel);track(smoke/rel)
   source_path=Path(sel['source_checkpoint_path']);require(sha(source_path)==sel['source_checkpoint_sha256'],'Source checkpoint damaged');source_state=source_cache.setdefault(str(source_path),checkpoint(source_path,model=='ri_gcn'));sh=state_hash(source_state)
   if model=='msgt':require(sel['source_checkpoint_sha256']==c['source_checkpoints']['0']['sha256'],'Wrong reused MSGT source')
   else:
    name=f'{ds}_ri_gcn_source_o00_s0';predids=sorted(set(d['record_id'].astype(str))-set(c['source_ids']))
    contract={'model':model,'dataset':ds,'train_ids':c['source_ids'],'predict_ids':predids,'seed':0,'method':'source','learning_rate':.001,'epochs':1,'source_state_sha256':None}
    if name not in checked_fits:check_fit(name,contract,None,deps)
   choice_path=smoke/(tag+'_frozen_choice.json');choice=load(choice_path)
   require(deps.get(choice_path.name)==sha(choice_path),'Frozen choice is not bound')
   for key,value in choice.items():require(sel.get(key)==value,'Selection altered frozen choice field: '+key)
   train_ids=c['train_ids'];people=d['participant'][[lk[v] for v in train_ids]].astype(str);y=d['y'][[lk[v] for v in train_ids]];lr_scores=[];oof_by_lr={}
   for lr in m['learning_rates']:
    oof=np.full(len(train_ids),np.nan)
    for fold,valid in enumerate(c['inner_validation_participants']):
     mask=np.isin(people,valid);tr=[v for v,flag in zip(train_ids,mask) if not flag];va=[v for v,flag in zip(train_ids,mask) if flag];name=tag+f'_lr{lr:g}_inner{fold}'
     contract={'model':model,'dataset':ds,'train_ids':tr,'predict_ids':va,'seed':0,'method':method,'learning_rate':lr,'epochs':1,'source_state_sha256':sh};check_fit(name,contract,source_state,deps)
     with np.load(smoke/name/'predictions.npz',allow_pickle=False) as pp:oof[mask]=pp['probability']
    finite_probability(oof,'Incomplete OOF');score=macro(y,oof,people,ds);lr_scores.append({'lr':lr,'score':score});oof_by_lr[lr]=oof
   require(len(sel['lr_scores'])==3,'Wrong number of LR scores')
   for actual,expected in zip(sel['lr_scores'],lr_scores):require(actual['lr']==expected['lr'] and abs(actual['score']-expected['score'])<1e-10,'LR score not reproduced')
   lr=min(lr_scores,key=lambda x:(x['score'],abs(np.log10(x['lr']/.001)),x['lr']))['lr'];require(sel['selected_lr']==lr,'Wrong LR selection')
   devpath=smoke/(tag+'_development.csv');require(deps.get(devpath.name)==sha(devpath),'Development table not bound');dev=list(csv.DictReader(devpath.open()));require([x['record_id'] for x in dev]==train_ids,'Development IDs differ')
   source=np.array([float(x['source']) for x in dev]);finite_probability(source,'Development source predictions invalid');oof=oof_by_lr[lr]
   require(np.allclose([float(x['oof']) for x in dev],oof,atol=1e-12,rtol=0),'Development OOF differs');require(np.array_equal([float(x['y_true']) for x in dev],y),'Development labels differ');require([x['participant'] for x in dev]==people.tolist(),'Development participants differ')
   ar=[{'alpha':a,'score':macro(y,(1-a)*source+a*oof,people,ds)} for a in m['alpha_grid']]
   require(len(sel['alpha_scores'])==5,'Wrong number of alpha scores')
   for actual,expected in zip(sel['alpha_scores'],ar):require(actual['alpha']==expected['alpha'] and abs(actual['score']-expected['score'])<1e-10,'Alpha score not reproduced')
   alpha=min(ar,key=lambda x:(x['score'],abs(x['alpha']-.5),x['alpha']))['alpha'];require(sel['selected_alpha']==alpha,'Wrong alpha selection')
   contract={'model':model,'dataset':ds,'train_ids':train_ids,'predict_ids':c['test_ids'],'seed':0,'method':method,'learning_rate':lr,'epochs':1,'source_state_sha256':sh};check_fit(tag+'_final',contract,source_state,deps)
   pf=smoke/sel['prediction_file'];require(sha(pf)==sel['prediction_sha256'],'Final prediction CSV damaged');track(pf);pred=list(csv.DictReader(pf.open()));require([x['record_id'] for x in pred]==c['test_ids'],'Final test IDs differ')
   idx=[lk[v] for v in c['test_ids']];require([x['participant'] for x in pred]==d['participant'][idx].astype(str).tolist(),'Test participants differ');require(np.array_equal([float(x['y_true']) for x in pred],d['y'][idx]),'Test labels differ')
   with np.load(smoke/(tag+'_final')/'predictions.npz',allow_pickle=False) as pp:require(np.allclose([float(x['direct']) for x in pred],pp['probability'],atol=1e-12,rtol=0),'Final CSV/direct checkpoint prediction differs')
   for key in ['source','direct','fixed_half','dynamic','development_alpha']:finite_probability([float(x[key]) for x in pred],'Invalid output '+key)
   for row in pred:
    s,t=float(row['source']),float(row['direct'])
    for key,value in [('fixed_half',(s+t)/2),('dynamic',(.5*s+4*t)/4.5),('development_alpha',(1-alpha)*s+alpha*t)]:require(abs(float(row[key])-value)<1e-12,'Fusion output differs')
  require(len(expected_fits)==58 and {p.parent.name for p in smoke.glob('*/completed.json')}==expected_fits,'Smoke fit set differs from 58 prescribed fits')
  require(lp_checks==28,'Expected 28 LP nonhead-state checks')
  # Verify no examined file changed during acceptance.
  for rel,h in evidence.items():require(sha(Path(rel) if Path(rel).is_absolute() else root/rel)==h,'Artifact changed during acceptance: '+rel)
  result={'status':'PASS','preflight':'PASS','freeze_sha256':freeze_sha,'environment_ready':True,'scope':'completed frozen eight-cell smoke only; not formal result approval','selections':8,'fits':58,'lp_nonhead_checks':lp_checks,'smoke_environment':env,'checked_sha256':evidence}
 except Exception as exc:
  result={'status':'FAIL','preflight':'FAIL','freeze_sha256':freeze_sha,'environment_ready':False,'error':str(exc),'traceback':traceback.format_exc(),'checked_sha256':evidence}
  tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out);print(json.dumps({'status':'FAIL','error':str(exc)}));return 1
 tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(out);print(json.dumps({k:result[k] for k in ['status','freeze_sha256','environment_ready','selections','fits','lp_nonhead_checks']}));return 0
if __name__=='__main__':sys.exit(main())
