"""Budget-contained two-fold model selection followed by full-budget refit."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import argparse,csv,fcntl,hashlib,importlib.util,json,sys,time,traceback
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):
 p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');q.replace(p)
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def macro(y,p,people,ds):
 v=[]
 for person in sorted(set(people)):
  ix=people==person;e=np.mean((np.asarray(p)[ix]-y[ix])**2);v.append(np.sqrt(e) if ds=='U2K' else e)
 return float(np.mean(v))
def state_sha(state):
 if state is None:return None
 h=hashlib.sha256()
 for k,v in sorted(state.items()):
  vv=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(vv.dtype).encode());h.update(str(tuple(vv.shape)).encode());h.update(vv.numpy().tobytes())
 return h.hexdigest()
def validate_roles(m,data):
 source_u2k=None
 for c in m['cells']:
  d=data[c['dataset']];lk=d['lookup'];assert len(lk)==len(d['record_id'])
  sets=[]
  for role in ['source','train','test']:
   ids=c[role+'_ids'];assert len(ids)==len(set(ids));assert set(ids)<=set(lk)
   people=set(d['participant'][[lk[x] for x in ids]].astype(str));sets.append((set(ids),people))
   if role!='test':assert people==set(c[role+'_part'+'icipants'])
  for i in range(3):
   for j in range(i):assert not sets[i][0]&sets[j][0] and not sets[i][1]&sets[j][1]
  assert len(c['train_participants'])==len(set(c['train_participants']))==c['budget']
  inner=c['inner_validation_participants'];flat=[x for fold in inner for x in fold]
  if c['budget']==1:assert not inner
  else:assert len(inner)==2 and all(inner) and len(flat)==len(set(flat))==c['budget'] and set(flat)==sets[1][1]
  if c['dataset']=='U2K':
   if source_u2k is None:source_u2k=c['source_ids']
   assert c['source_ids']==source_u2k

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--limit',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--model',choices=['msgt','ri_gcn']);a=ap.parse_args();r=a.root;f=json.loads((r/'freeze.json').read_text());fh=sha(r/'freeze.json')
 for p,h in f['files'].items():assert sha(r/p)==h,('freeze mismatch',p)
 m=json.loads((r/'prepared/manifest.json').read_text());out=r/('smoke' if a.smoke else 'results');out.mkdir(exist_ok=True);lock=(out/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 torch.set_num_threads(2);torch.set_num_interop_threads(2);assert torch.cuda.is_available();sys.path.insert(0,str(r/'dependencies'));from model_factory import make_model,set_seed
 def new_model(name,seed):
  if name=='msgt':return make_model('msgt',r/'dependencies',seed)
  from ri_gcn_adapter import make_model as make_ri
  set_seed(seed);return make_ri()
 data={}
 for ds,z in m['data'].items():
  p=r/'prepared'/z['path'];assert sha(p)==z['sha256'];d=dict(np.load(p,allow_pickle=False));assert np.isfinite(d['x']).all() and np.isfinite(d['y']).all();d['lookup']={str(v):i for i,v in enumerate(d['record_id'])};data[ds]=d
 validate_roles(m,data)
 dump(out/'environment.json',{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),'python':sys.version,'cuda':torch.version.cuda,'freeze_sha256':fh,'threads':2})
 completed=0;started=time.time()
 def infer(model,x,batch):
  model.eval();p=[]
  with torch.no_grad():
   for k in range(0,len(x),batch):p.extend(torch.sigmoid(model(torch.from_numpy(x[k:k+batch]).cuda())).cpu().numpy().astype(float).tolist())
  assert len(p)==len(x) and np.isfinite(p).all() and min(p)>=0 and max(p)<=1
  return np.asarray(p)
 def fit(cell,seed,method,lr,train_ids,pred_ids,name,source_state,epochs):
  d=data[cell['dataset']];base=out/name;receipt=base/'completed.json'
  contract={'model':cell['model'],'dataset':cell['dataset'],'train_ids':train_ids,'predict_ids':pred_ids,'seed':seed,'method':method,'learning_rate':lr,'epochs':epochs,'source_state_sha256':state_sha(source_state)}
  if receipt.exists():
   z=json.loads(receipt.read_text());assert z['freeze_sha256']==fh and z['input_contract']==contract
   for p,h in z['artifacts'].items():assert sha(base/p)==h
   pp=np.load(base/'predictions.npz',allow_pickle=False);assert pp['record_id'].astype(str).tolist()==pred_ids
   return pp['probability'],z
  assert not base.exists(),('Incomplete preserved task; explicit recovery required',str(base));base.mkdir();set_seed(seed+1000);model=new_model(cell['model'],seed).cuda()
  if source_state is not None:model.load_state_dict(source_state)
  if method=='lp':
   for p in model.parameters():p.requires_grad_(False)
   for p in model.head.parameters():p.requires_grad_(True)
   before={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k not in {name for name,p in model.named_parameters() if p.requires_grad}}
  set_seed(seed+1000);batch=64 if cell['dataset']=='U2K' else 16;ix=np.array([d['lookup'][v] for v in train_ids]);px=np.array([d['lookup'][v] for v in pred_ids]);assert not set(train_ids)&set(pred_ids)
  loader=DataLoader(TensorDataset(torch.from_numpy(d['x'][ix]),torch.from_numpy(d['y'][ix])),batch_size=batch,shuffle=True,generator=torch.Generator().manual_seed(seed+1000),num_workers=0)
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=m['weight_decay']);hist=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();ts=time.perf_counter()
  for epoch in range(epochs):
   if method=='lp':model.eval();model.head.train()
   else:model.train()
   total=0.
   for x,y in loader:
    opt.zero_grad(set_to_none=True);yp=model(x.cuda());yt=y.cuda();loss=torch.nn.functional.mse_loss(torch.sigmoid(yp),yt) if cell['dataset']=='U2K' else torch.nn.functional.binary_cross_entropy_with_logits(yp,yt);assert torch.isfinite(loss);loss.backward();opt.step();total+=float(loss.detach())*len(y)
   hist.append({'epoch':epoch+1,'loss':total/len(ix)})
  torch.cuda.synchronize();seconds=time.perf_counter()-ts;training_peak=torch.cuda.max_memory_allocated()
  if method=='lp':assert all(torch.equal(v,model.state_dict()[k].cpu()) for k,v in before.items())
  pred=infer(model,d['x'][px],batch);torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'freeze_sha256':fh},base/'checkpoint.pt');np.savez(base/'predictions.npz',record_id=np.array(pred_ids),probability=pred);dump(base/'history.json',hist)
  z={'freeze_sha256':fh,'input_contract':contract,'receipt_path':str(receipt.relative_to(out)),'training_seconds':seconds,'training_peak_allocated_bytes':training_peak,'training_and_prediction_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'parameters':sum(p.numel() for p in model.parameters()),'train_ids':train_ids,'predict_ids':pred_ids,'epochs':epochs,'method':method,'learning_rate':lr,'seed':seed,'artifacts':{p:sha(base/p) for p in ['checkpoint.pt','predictions.npz','history.json']}};dump(receipt,z);del model,opt,loader;return pred,z
 try:
  for cell in [dict(c,model=model_name) for model_name in m['models'] if not a.model or model_name==a.model for c in m['cells'] if not a.smoke or (c['outer']==0 and c['budget']==4)]:
   ds=cell['dataset'];d=data[ds];n=cell['budget'];train_ids=cell['train_ids'];test_ids=cell['test_ids'];ti=np.array([d['lookup'][x] for x in train_ids]);ei=np.array([d['lookup'][x] for x in test_ids]);people=d['participant'][ti].astype(str);testpeople=d['participant'][ei].astype(str)
   assert not set(people)&set(testpeople) and not set(people)&set(cell['source_participants'])
   for seed in ([0] if a.smoke else m['seeds']):
    source_train_seconds=None
    if cell['model']=='msgt':
     cp=cell['source_checkpoints'][str(seed)];assert sha(cp['path'])==cp['sha256']
    else:
     source_name=f'{ds}_ri_gcn_source_o{0 if ds=="U2K" else cell["outer"]:02d}_s{seed}';source_prediction_ids=sorted(set(d['record_id'].astype(str))-set(cell['source_ids']))
     _,source_cost=fit(cell,seed,'source',.001,cell['source_ids'],source_prediction_ids,source_name,None,1 if a.smoke else (100 if ds=='U2K' else 50));source_train_seconds=source_cost['training_seconds'];source_path=out/source_name/'checkpoint.pt';cp={'path':str(source_path),'sha256':sha(source_path)}
    source_state=torch.load(cp['path'],map_location='cpu',weights_only=False)['state_dict'];src=new_model(cell['model'],seed).cuda();src.load_state_dict(source_state);sp=infer(src,d['x'][ti],64 if ds=='U2K' else 16)
    # Source test forward does not read test labels; labels are emitted only after frozen selection.
    st=infer(src,d['x'][ei],64 if ds=='U2K' else 16);del src
    for method in ['ft','lp']:
     name=f'{ds}_{cell["model"]}_o{cell["outer"]:02d}_n{n}_s{seed}_{method}';result=out/(name+'_selection.json')
     if result.exists():
      old=json.loads(result.read_text());assert old['freeze_sha256']==fh and sha(out/old['prediction_file'])==old['prediction_sha256']
      for rel,h in old['dependencies'].items():assert sha(out/rel)==h,('resume dependency changed',rel)
      completed+=1;continue
     dump(out/'status.json',{'status':'running','active':name,'completed_cells':completed,'started':started,'freeze_sha256':fh})
     rows=[];costs=[];fit_receipts=[];oof_by_lr={};epochs=1 if a.smoke else m['epochs']
     if n>1:
      for lr in m['learning_rates']:
       oof=np.full(len(ti),np.nan)
       for fold,valid_people in enumerate(cell['inner_validation_participants']):
        vmask=np.isin(people,valid_people);tr=[x for x,v in zip(train_ids,vmask) if not v];va=[x for x,v in zip(train_ids,vmask) if v];assert tr and va;assert not set(people[~vmask])&set(people[vmask])
        pred,cost=fit(cell,seed,method,lr,tr,va,name+f'_lr{lr:g}_inner{fold}',source_state,epochs);oof[vmask]=pred;costs.append(cost['training_seconds']);fit_receipts.append(cost['receipt_path'])
       assert np.isfinite(oof).all();score=macro(d['y'][ti],oof,people,ds);rows.append({'lr':lr,'score':score});oof_by_lr[lr]=oof
      selected_lr=min(rows,key=lambda x:(x['score'],abs(np.log10(x['lr']/.001)),x['lr']))['lr'];oof=oof_by_lr[selected_lr]
      ar=[{'alpha':alpha,'score':macro(d['y'][ti],(1-alpha)*sp+alpha*oof,people,ds)} for alpha in m['alpha_grid']];selected_alpha=min(ar,key=lambda x:(x['score'],abs(x['alpha']-.5),x['alpha']))['alpha']
      csvout(out/(name+'_development.csv'),[{'record_id':v,'participant':str(p),'y_true':float(y),'source':float(s),'oof':float(t)} for v,p,y,s,t in zip(train_ids,people,d['y'][ti],sp,oof)])
     else:selected_lr=m['n1_learning_rate'];selected_alpha=None;ar=[]
     # Persist the choice before outer test labels are used in any output/analysis.
     choice={'freeze_sha256':fh,'dataset':ds,'model':cell['model'],'outer':cell['outer'],'budget':n,'seed':seed,'method':method,'selected_lr':selected_lr,'selected_alpha':selected_alpha,'lr_scores':rows,'alpha_scores':ar,'selection_participants':cell['train_participants'],'source_checkpoint_sha256':cp['sha256'],'source_checkpoint_path':cp['path'],'source_training_seconds':source_train_seconds};dump(out/(name+'_frozen_choice.json'),choice)
     pred,cost=fit(cell,seed,method,selected_lr,train_ids,test_ids,name+'_final',source_state,epochs)
     pp=[{'record_id':rid,'participant':str(p),'action':str(ac),'y_true':float(y),'source':float(s),'direct':float(t),'fixed_half':float((s+t)/2),'dynamic':float((.5*s+n*t)/(n+.5)),'development_alpha':float((1-selected_alpha)*s+selected_alpha*t) if selected_alpha is not None else ''} for rid,p,ac,y,s,t in zip(test_ids,testpeople,d['action'][ei],d['y'][ei],st,pred)]
     pf=name+'_predictions.csv';csvout(out/pf,pp);fit_receipts.append(cost['receipt_path']);dependencies={}
     if cell['model']=='ri_gcn':fit_receipts.append(source_cost['receipt_path'])
     for rel in fit_receipts:
      rp=out/rel;dependencies[rel]=sha(rp);rc=json.loads(rp.read_text())
      for art,h in rc['artifacts'].items():
       apath=rp.parent/art;assert sha(apath)==h;dependencies[str(apath.relative_to(out))]=h
     for rel in [name+'_frozen_choice.json']+([name+'_development.csv'] if n>1 else []):dependencies[rel]=sha(out/rel)
     choice.update({'dependencies':dependencies});choice.update({'prediction_file':pf,'prediction_sha256':sha(out/pf),'target_fit_seconds':cost['training_seconds'],'development_training_seconds':sum(costs),'total_adaptation_seconds':sum(costs)+cost['training_seconds'],'development_fits':len(costs),'target_fits':1,'test_participants':sorted(set(testpeople))});dump(result,choice);completed+=1;print(json.dumps({'completed':name,'cells':completed,'search_seconds':round(sum(costs),2),'final_seconds':round(cost['training_seconds'],2)}),flush=True)
     if a.limit and completed>=a.limit:dump(out/'status.json',{'status':'partial','completed_cells':completed,'freeze_sha256':fh});return
  for p,h in f['files'].items():assert sha(r/p)==h,('end freeze mismatch',p)
  for ds,z in m['data'].items():assert sha(r/'prepared'/z['path'])==z['sha256']
  for sel in out.glob('*_selection.json'):
   zz=json.loads(sel.read_text());assert zz['freeze_sha256']==fh;assert sha(zz['source_checkpoint_path'])==zz['source_checkpoint_sha256']
   for rel,h in zz['dependencies'].items():assert sha(out/rel)==h,('end dependency changed',rel)
  dump(out/'status.json',{'status':'completed','completed_cells':completed,'wall_seconds':time.time()-started,'freeze_sha256':fh})
 except BaseException as e:
  dump(out/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc(),'active':locals().get('name')});raise
if __name__=='__main__':main()
