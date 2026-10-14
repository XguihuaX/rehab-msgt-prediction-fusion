"""Build immutable role/data packet from existing authoritative inputs; CPU only."""
import argparse,csv,hashlib,importlib.util,json,shutil
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return list(csv.DictReader(Path(p).open()))
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();b=a.base;r=a.root;out=r/'prepared';out.mkdir(exist_ok=True)
 assert not (out/'manifest.json').exists(),'Already prepared: verify rather than overwrite'
 inputs=b/'inputs';reg=inputs/'manifests/registry_v3';perf=inputs/'manifests/performance_v4';source=inputs/'source_checkpoints'
 spec=importlib.util.spec_from_file_location('original',source/'original_runner.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 registry=read(reg/'data_registry_features.csv');lookup={x['record_id']:x for x in registry};resolved=read(perf/'resolved_records.csv');budgets=read(perf/'budget_members.csv');store=m.RecordStore(registry,b/'data/RehabPile');labels=m.LabelStore(reg/'participant_label_files.json',reg)
 cells=[];records={};used=set()
 for repeat in [0,4,8]:
  mid=f'direction-a-performance-manifest-v1:U2K:r{repeat:02d}';rr=[x for x in resolved if x['manifest_id']==mid]
  source_rows=[x for x in rr if x['split_role']=='source_train'];source_people=sorted({x['participant_key'] for x in source_rows});test=[x for x in rr if x['split_role']=='evaluation']
  for n in [1,4,16]:
   people=[x['participant_key'] for x in sorted(budgets,key=lambda x:int(x['budget_draw_rank'])) if x['manifest_id']==mid and int(x['budget_participants'])==n and x['included_in_budget']=='true'];assert len(set(people))==n
   train=[x for x in rr if x['split_role']=='adaptation_pool' and x['participant_key'] in people]
   assert not set(people)&{x['participant_key'] for x in test};assert not set(source_people)&set(people)
   cell={'dataset':'U2K','outer':repeat,'budget':n,'train_participants':people,'source_participants':source_people,'source_ids':[x['record_id'] for x in source_rows],'train_ids':[x['record_id'] for x in train],'test_ids':[x['record_id'] for x in test],'source_checkpoints':{}}
   for seed in [0,1,2]:
    p=source/f'seed{seed}.pt';cell['source_checkpoints'][str(seed)]={'path':str(p),'sha256':sha(p)}
   cells.append(cell);used.update(cell['source_ids']+cell['train_ids']+cell['test_ids'])
 rows=[lookup[x] for x in sorted(used)];x=np.stack([store.get_feature(x['record_id'],'packet_existing_input') for x in rows]);ys=labels.values(rows,'packet_existing_labels')
 np.savez(out/'U2K.npz',x=x,y=np.array([float(x['label_transformed']) for x in ys],np.float32),record_id=np.array([x['record_id'] for x in rows]),participant=np.array([x['participant_key'] for x in rows]),action=np.array([x['action_raw'] for x in rows]))
 cm_path=b/'classification_completion_v1/manifest.json';cm=json.loads(cm_path.read_text());p=b/'classification_completion_v1/prepared/irds_formal_v1.npz';assert sha(p)==cm['records']['irds']['sha256'];d=np.load(p,allow_pickle=False);ids=d['record_id'].astype(str);assert ids.tolist()==[x['record_id'] for x in cm['records']['irds']['records']]
 np.savez(out/'irds.npz',x=d['x'],y=d['y'].astype(np.float32),record_id=ids,participant=d['participant'].astype(str),action=np.array([x['action'] for x in cm['records']['irds']['records']]))
 for fold in range(5):
  for n in [1,4,8]:
   ts=[x for x in cm['tasks'] if x['dataset']=='irds' and x['model']=='msgt' and x['seed']==0 and x['method']=='ft' and x['fold_id']==f'irds_f{fold:02d}' and x['budget']==n];assert len(ts)==1;z=ts[0]
   sc={}
   for seed in [0,1,2]:
    sd=b/'classification_completion_v1/results_v1'/f'irds_f{fold:02d}_msgt_s{seed}_source';receipt=json.loads((sd/'completed.json').read_text());cp=sd/receipt['checkpoint'];assert sha(cp)==receipt['artifacts'][receipt['checkpoint']];sc[str(seed)]={'path':str(cp),'sha256':sha(cp)}
   source_task=next(x for x in cm['tasks'] if x['task_id']==z['source_task_id'])
   cells.append({'dataset':'irds','outer':fold,'budget':n,'train_participants':z['train_participants'],'source_participants':source_task['train_participants'],'source_ids':source_task['train_record_ids'],'train_ids':z['train_record_ids'],'test_ids':z['test_record_ids'],'source_checkpoints':sc})
 for c in cells:
  people=c['train_participants'];assert len(people)==c['budget'];assert not set(people)&set(c['source_participants']);c['inner_validation_participants']=[people[::2],people[1::2]] if c['budget']>1 else []
  dd=np.load(out/(c['dataset']+'.npz'),allow_pickle=False);lk={v:i for i,v in enumerate(dd['record_id'].astype(str))};assert not set(c['train_ids'])&set(c['test_ids']);assert not set(dd['participant'][[lk[x] for x in c['test_ids']]].astype(str))&set(people)
 records={ds:{'path':ds+'.npz','sha256':sha(out/(ds+'.npz'))} for ds in ['U2K','irds']}
 origins={str(p):sha(p) for p in [reg/'data_registry_features.csv',reg/'participant_label_files.json',perf/'resolved_records.csv',perf/'budget_members.csv',cm_path,source/'original_runner.py']}
 dump(out/'manifest.json',{'cells':cells,'data':records,'upstream':origins,'models':['msgt','ri_gcn'],'seeds':[0,1,2],'learning_rates':[.0001,.001,.003],'epochs':50,'weight_decay':.0001,'alpha_grid':[0,.25,.5,.75,1],'n1_learning_rate':.001,'selection':'participant_macro_RMSE_U2K_Brier_irds','inner_folds':2})
 print(json.dumps({'prepared_cells':len(cells),'manifest_sha256':sha(out/'manifest.json'),'data_sha256':records}))
if __name__=='__main__':main()
