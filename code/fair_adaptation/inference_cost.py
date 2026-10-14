"""Prespecified fixed checkpoint timing, including complete dual-model execution."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']='2'
import argparse,json,csv,hashlib,sys,time
from pathlib import Path
import numpy as np
import torch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();r=a.root;o=r/'results';out=r/'inference_cost';out.mkdir(exist_ok=True);assert not (out/'measurements.csv').exists(),'Do not duplicate completed cost measurements'
 freeze=json.loads((r/'freeze.json').read_text())
 for p,h in freeze['files'].items():assert sha(r/p)==h
 assert json.loads((o/'status.json').read_text())['status']=='completed';torch.set_num_threads(2);torch.set_num_interop_threads(2);sys.path.insert(0,str(r/'dependencies'));from model_factory import make_model,set_seed
 from ri_gcn_adapter import make_model as ri_model
 m=json.loads((r/'prepared/manifest.json').read_text());rows=[];raw=[]
 for ds in ['U2K','irds']:
  cell=next(c for c in m['cells'] if c['dataset']==ds and c['outer']==0 and c['budget']==4);data=np.load(r/'prepared'/m['data'][ds]['path'],allow_pickle=False);lk={str(v):i for i,v in enumerate(data['record_id'])}
  for name in m['models']:
   for method in ['ft','lp']:
    tag=f'{ds}_{name}_o00_n4_s0_{method}';choice=json.loads((o/(tag+'_selection.json')).read_text());source_path=Path(choice['source_checkpoint_path']);assert sha(source_path)==choice['source_checkpoint_sha256'];target_path=o/(tag+'_final')/'checkpoint.pt'
    receipt=json.loads((target_path.parent/'completed.json').read_text());assert sha(target_path)==receipt['artifacts']['checkpoint.pt']
    def create(path):
     set_seed(0);model=(make_model('msgt',r/'dependencies',0) if name=='msgt' else ri_model()).cuda();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['state_dict']);model.eval();return model
    target=create(target_path)
    for batch in [1,16]:
     ids=cell['test_ids'][:batch];assert len(ids)==batch;x=torch.from_numpy(data['x'][[lk[v] for v in ids]]).cuda()
     for rule in ['direct','dual_dynamic']:
      source=create(source_path) if rule=='dual_dynamic' else None;alpha=4/4.5
      def call():
       t=torch.sigmoid(target(x));return (1-alpha)*torch.sigmoid(source(x))+alpha*t if source is not None else t
      with torch.no_grad():
       for _ in range(20):y=call()
       assert torch.isfinite(y).all();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();times=[]
       for rep in range(100):
        torch.cuda.synchronize();ts=time.perf_counter();y=call();torch.cuda.synchronize();ms=(time.perf_counter()-ts)*1000;times.append(ms);raw.append({'dataset':ds,'model':name,'method':method,'batch':batch,'rule':rule,'repeat':rep,'milliseconds':ms})
      params=sum(p.numel() for p in target.parameters())+(sum(p.numel() for p in source.parameters()) if source is not None else 0)
      rows.append({'dataset':ds,'model':name,'method':method,'outer':0,'seed':0,'budget':4,'batch':batch,'rule':rule,'p50_ms':float(np.median(times)),'p90_ms':float(np.quantile(times,.9)),'parameters':params,'resident_allocated_bytes':base,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'incremental_peak_allocated_bytes':torch.cuda.max_memory_allocated()-base,'source_checkpoint_sha256':sha(source_path),'target_checkpoint_sha256':sha(target_path),'record_ids':json.dumps(ids),'input_shape':str(list(x.shape))})
      del source;torch.cuda.empty_cache()
     del x
    del target;torch.cuda.empty_cache()
 for fn,data_rows in [('measurements.csv',rows),('raw_timing.csv',raw)]:
  with (out/fn).open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(data_rows[0]));w.writeheader();w.writerows(data_rows)
 (out/'environment.json').write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'dtype':'float32','warmup':20,'repetitions':100,'scope':'GPU resident model+sigmoid+fusion, excludes preprocessing/pose estimation/H2D','freeze_sha256':sha(r/'freeze.json')},indent=2)+'\n');print(json.dumps({'timing_configurations':len(rows),'raw_samples':len(raw)}))
if __name__=='__main__':main()
