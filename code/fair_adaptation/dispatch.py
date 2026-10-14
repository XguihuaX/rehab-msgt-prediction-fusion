"""Finite authorized batch, waits quietly for existing GPU users; no auto retry."""
import argparse,fcntl,json,subprocess,time,sys,traceback,hashlib
from pathlib import Path

def dump(p,x):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2)+'\n');q.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--smoke-only',action='store_true');a=ap.parse_args();r=a.root;tag='smoke_dispatch' if a.smoke_only else 'dispatch';o=r/tag;o.mkdir(exist_ok=True);lock=(o/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (o/'status.json').exists(),'Preserve previous dispatch state; explicit recovery only';started=time.time();dump(o/'status.json',{'status':'waiting_for_resource','started':started,'check_interval_seconds':600})
 try:
  while True:
   gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();mem=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip();busy=bool(gpu) or any(int(x.strip())>=500 for x in mem.splitlines())
   if not busy:break
   if time.time()-started>43200:raise RuntimeError('Existing GPU work still active after bounded 12h wait; no preemption')
   time.sleep(600)
  smoke_stage=('smoke',[sys.executable,str(r/'code/run.py'),'--root',str(r),'--smoke'])
  accept_stage=('accept_smoke',[sys.executable,str(r/'code/accept_smoke.py'),'--root',str(r)])
  stages=[smoke_stage,accept_stage] if a.smoke_only else [smoke_stage,accept_stage,('training',[sys.executable,str(r/'code/run.py'),'--root',str(r)]),('inference_cost',[sys.executable,str(r/'code/inference_cost.py'),'--root',str(r)]),('analysis',[sys.executable,str(r/'code/analyze.py'),'--root',str(r)])]
  if not a.smoke_only:
   ready=json.loads((r/'LAUNCH_AUTHORIZATION.json').read_text());assert ready['preflight']=='PASS';assert ready['freeze_sha256']==hashlib.sha256((r/'freeze.json').read_bytes()).hexdigest()
  completed=[]
  for name,cmd in stages:
   if name=='training':
    witness=json.loads((r/'SMOKE_ACCEPTANCE.json').read_text());assert witness['status']=='PASS' and witness['freeze_sha256']==hashlib.sha256((r/'freeze.json').read_bytes()).hexdigest()
   dump(o/'status.json',{'status':'running','stage':name,'completed':completed,'started':started});ts=time.time()
   with (o/(name+'.log')).open('x') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
   completed.append({'stage':name,'seconds':time.time()-ts})
  dump(o/'status.json',{'status':'completed_pending_review','completed':completed,'wall_seconds':time.time()-started})
 except BaseException as e:dump(o/'FAILED.json',{'error':str(e),'traceback':traceback.format_exc()});dump(o/'status.json',{'status':'failed','error':str(e)});raise
if __name__=='__main__':main()
