"""Frozen fair-adaptation aggregation. No training; incomplete formal batches fail closed.

Usage: python analyze.py --root /path/to/fair_adaptation
       python analyze.py --self-test
Primary-metric descriptive unique-participant cluster intervals preserve repeats.
"""
import argparse,csv,hashlib,itertools,json,math,statistics as st,tempfile
import numpy as np
from collections import defaultdict
from pathlib import Path
RULES=('source','direct','fixed_half','dynamic','development_alpha')
METS={'U2K':('mse','rmse','mae','ccc'),'irds':('brier','auroc','balanced_accuracy')}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def write(p,rr):
 assert rr
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def metrics(y,p,ds):
 assert y and len(y)==len(p) and all(math.isfinite(v) for v in y+p)
 mse=st.mean((a-b)**2 for a,b in zip(y,p))
 if ds=='U2K':
  a,b=st.mean(y),st.mean(p);vy=st.mean((v-a)**2 for v in y);vp=st.mean((v-b)**2 for v in p);cov=st.mean((u-a)*(v-b) for u,v in zip(y,p));den=vy+vp+(a-b)**2
  return dict(mse=mse,rmse=math.sqrt(mse),mae=st.mean(abs(a-b) for a,b in zip(y,p)),ccc=2*cov/den if den else None)
 assert ds=='irds' and set(y)<={0.,1.}
 pos=sum(y);neg=len(y)-pos
 if not pos or not neg:return dict(brier=mse,auroc=None,balanced_accuracy=None)
 # Mann-Whitney AUC, average ranks for equal probability.
 ordered=sorted(zip(p,y));rank_sum=0.;i=0
 while i<len(ordered):
  j=i+1
  while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
  rank_sum+=((i+1+j)/2)*sum(v[1] for v in ordered[i:j]);i=j
 auc=(rank_sum-pos*(pos+1)/2)/(pos*neg)
 tpr=sum(a==1 and b>=.5 for a,b in zip(y,p))/pos;tnr=sum(a==0 and b<.5 for a,b in zip(y,p))/neg
 return dict(brier=mse,auroc=auc,balanced_accuracy=(tpr+tnr)/2)
def cluster_interval(rr,ds,draws=2000):
 # A cluster has one shared bootstrap multiplicity across all of its repeats.
 people=sorted({r['participant'] for r in rr});outers=sorted({r['outer'] for r in rr});pi={p:i for i,p in enumerate(people)};oi={o:i for i,o in enumerate(outers)}
 mat=np.zeros((len(people),len(outers)));valid=np.zeros_like(mat)
 for r in rr:
  if r['benefit'] is not None:
   i,j=pi[r['participant']],oi[r['outer']];assert valid[i,j]==0;mat[i,j]=r['benefit'];valid[i,j]=1
 active=valid.sum(axis=1)>0;mat=mat[active];valid=valid[active];n=len(mat)
 def stat(counts):
  sums=counts@mat;den=counts@valid
  if ds=='U2K':
   ok=(den>0).all(axis=1);return (sums[ok]/den[ok]).mean(axis=1)
  ok=den.sum(axis=1)>0;return sums[ok].sum(axis=1)/den[ok].sum(axis=1)
 base={'valid_unique_participants':n,'total_unique_participants':len(people),'outer_units':len(outers),'bootstrap_draws':0,'bootstrap_seed':20260920,'ci_scope':'descriptive_unique_participant_cluster_preserving_all_repeat_trajectories; overlapping_training_not_modeled'}
 if n==0 or (ds=='U2K' and (len(outers)!=3 or (valid.sum(axis=0)==0).any())):return {**base,'mean_benefit':None,'ci_low':None,'ci_high':None,'status':'NA_missing_valid_participants_or_repeat'}
 point=float(stat(np.ones((1,n)))[0]);rng=np.random.default_rng(20260920);vals=[];attempt=0
 while sum(len(v) for v in vals)<draws:
  counts=rng.multinomial(n,np.full(n,1/n),size=draws);v=stat(counts);vals.append(v);attempt+=draws
  if attempt>100000:raise RuntimeError('Too few valid complete-repeat bootstrap draws')
 v=np.concatenate(vals)[:draws];lo,hi=np.percentile(v,[2.5,97.5]);return {**base,'bootstrap_draws':draws,'mean_benefit':point,'ci_low':float(lo),'ci_high':float(hi),'status':'descriptive_interval'}
def aggregate(root,out,synthetic=False):
 r=Path(root);res=r/'results';files=sorted(res.glob('*_selection.json'));assert len(files)==288,f'Need exactly 288 final selections; found {len(files)}. No final report written.'
 manifest=json.loads((r/'prepared/manifest.json').read_text());freeze=sha(r/'freeze.json')
 expected={(c['dataset'],model,c['outer'],c['budget'],seed,m) for c in manifest['cells'] for model in manifest['models'] for seed in manifest['seeds'] for m in ['ft','lp']};assert len(expected)==288
 keys=set();raw=[];cost=[];sources={};inputs={'prepared/manifest.json':sha(r/'prepared/manifest.json'),'freeze.json':freeze};selection=[];identity={}
 for f in files:
  z=json.loads(f.read_text());k=tuple(z[x] for x in ['dataset','model','outer','budget','seed','method']);assert k in expected and k not in keys;kds,model,outer,n,seed,method=k;keys.add(k)
  
  for rel,h in z['dependencies'].items():
   dep=res/rel;assert dep.resolve().is_relative_to(res.resolve()) and sha(dep)==h;inputs[str(dep.relative_to(r))]=h
  assert z['freeze_sha256']==freeze and z['target_fits']==1 and z['development_fits']==(0 if n==1 else 6)
  assert (z['selected_alpha'] is None)==(n==1)
  assert len(z['lr_scores'])==(0 if n==1 else 3) and len(z['alpha_scores'])==(0 if n==1 else 5)
  assert z['selected_lr'] in ([manifest['n1_learning_rate']] if n==1 else manifest['learning_rates'])
  if n>1:assert z['selected_alpha'] in manifest['alpha_grid']
  for field in ['target_fit_seconds','development_training_seconds','total_adaptation_seconds']:assert math.isfinite(z[field]) and z[field]>=0
  assert abs(z['total_adaptation_seconds']-z['target_fit_seconds']-z['development_training_seconds'])<1e-7
  if n==1:assert z['development_training_seconds']==0
  pf=res/z['prediction_file'];assert pf.parent==res and sha(pf)==z['prediction_sha256'];inputs[str(f.relative_to(r))]=sha(f);inputs[str(pf.relative_to(r))]=sha(pf)
  frozen=f.with_name(f.name.replace('_selection.json','_frozen_choice.json'));choice=json.loads(frozen.read_text());assert all(z[x]==v for x,v in choice.items());inputs[str(frozen.relative_to(r))]=sha(frozen)
  rows=list(csv.DictReader(pf.open()));assert rows and len(rows)==len({v['record_id'] for v in rows})
  assert sorted({v['participant'] for v in rows})==z['test_participants'];assert not set(z['test_participants'])&set(z['selection_participants'])
  ident=tuple(sorted((v['record_id'],v['participant'],v['y_true']) for v in rows));ik=(kds,outer)
  if ik in identity:assert identity[ik]==ident
  else:identity[ik]=ident
  pg=defaultdict(list)
  for v in rows:
   y=float(v['y_true']);s=float(v['source']);t=float(v['direct']);assert all(math.isfinite(x) and 0<=x<=1 for x in [y,s,t])
   assert abs(float(v['fixed_half'])-(s+t)/2)<1e-10 and abs(float(v['dynamic'])-(.5*s+n*t)/(n+.5))<1e-10
   if n==1:assert v['development_alpha']==''
   else:assert abs(float(v['development_alpha'])-((1-z['selected_alpha'])*s+z['selected_alpha']*t))<1e-10
   pg[v['participant']].append(v)
  base=dict(zip(['dataset','model','outer','budget','seed','method'],k))
  for person,pr in sorted(pg.items()):
   for rule in RULES:
    values={met:None for met in METS[kds]} if rule=='development_alpha' and n==1 else metrics([float(v['y_true']) for v in pr],[float(v[rule]) for v in pr],kds)
    for met,val in values.items():raw.append({**base,'participant':person,'rule':rule,'metric':met,'value':val,'records':len(pr),'status':'budget_one_no_development_labels' if rule=='development_alpha' and n==1 else 'single_class_or_undefined_metric' if val is None else 'available'})
  cost.append({**base,**{x:z[x] for x in ['development_fits','target_fits','development_training_seconds','target_fit_seconds','total_adaptation_seconds','source_checkpoint_sha256']}})
  selection.append({**base,'selected_lr':z['selected_lr'],'selected_alpha':z['selected_alpha'],'lr_scores_json':json.dumps(z['lr_scores']),'alpha_scores_json':json.dumps(z['alpha_scores'])})
  sk=z['source_checkpoint_sha256'];src={'dataset':kds,'model':model,'source_sha256':sk,'source_checkpoint_path':z['source_checkpoint_path'],'source_training_seconds':z['source_training_seconds'],'cost_scope':'new_source_train_loop' if model=='ri_gcn' else 'historical_source_timing_not_in_selection_receipt','reused_final_cells':1}
  if sk in sources:
   assert all(sources[sk][x]==src[x] for x in ['dataset','model','source_training_seconds']);sources[sk]['reused_final_cells']+=1
  else:sources[sk]=src
 assert keys==expected and sum(v['development_fits'] for v in cost)==1152
 assert len([x for x in sources.values() if x['model']=='ri_gcn'])==18
 # Every IRDS person occurs in exactly one outer fold; repeated U2K appearances kept.
 ir=defaultdict(set)
 for (ds,o),ident in identity.items():
  if ds=='irds':
   for _,person,_ in ident:ir[person].add(o)
 assert len(ir)==29 and all(len(v)==1 for v in ir.values())
 sg=defaultdict(list)
 fields=['dataset','model','outer','budget','method','participant','rule','metric']
 for v in raw:sg[tuple(v[x] for x in fields)].append(v)
 units=[]
 for k,rr in sorted(sg.items()):
  assert {v['seed'] for v in rr}==set(manifest['seeds']);vals=[v['value'] for v in rr if v['value'] is not None];units.append({**dict(zip(fields,k)),'value':st.mean(vals) if vals else None,'valid_seeds':len(vals),'expected_seeds':len(rr)})
 mg=defaultdict(list)
 for v in units:mg[tuple(v[x] for x in ['dataset','model','budget','method','rule','metric'])].append(v)
 summary=[]
 for k,rr in sorted(mg.items()):
  byouter=defaultdict(list)
  for v in rr:
   if v['value'] is not None:byouter[v['outer']].append(v['value'])
  vals=[v['value'] for v in rr if v['value'] is not None];score=st.mean(st.mean(v) for v in byouter.values()) if k[0]=='U2K' and byouter else st.mean(vals) if vals else None
  summary.append({**dict(zip(['dataset','model','budget','method','rule','metric'],k)),'value':score,'valid_unique_participants':len({v['participant'] for v in rr if v['value'] is not None}),'total_unique_participants':len({v['participant'] for v in rr}),'valid_outer_units':len(byouter),'aggregation':'seed_then_person_within_repeat_then_equal_repeat' if k[0]=='U2K' else 'seed_then_equal_unique_person'})
 # Pair within participant/repeat; then keep unique participant mean as descriptive companion.
 pairgroups=defaultdict(dict)
 for v in units:pairgroups[tuple(v[x] for x in ['dataset','model','outer','budget','method','participant','metric'])][v['rule']]=v['value']
 paired=[]
 for k,v in sorted(pairgroups.items()):
  for candidate,reference in itertools.combinations(RULES,2):
   a,b=v[candidate],v[reference];sign=1 if k[-1] in ['ccc','auroc','balanced_accuracy'] else -1
   paired.append({**dict(zip(['dataset','model','outer','budget','method','participant','metric'],k)),'candidate':candidate,'reference':reference,'candidate_value':a,'reference_value':b,'benefit':sign*(a-b) if a is not None and b is not None else None})
 pg=defaultdict(list)
 for v in paired:pg[tuple(v[x] for x in ['dataset','model','budget','method','participant','metric','candidate','reference'])].append(v)
 persons=[]
 for k,rr in sorted(pg.items()):
  vals=[v['benefit'] for v in rr if v['benefit'] is not None];persons.append({**dict(zip(['dataset','model','budget','method','participant','metric','candidate','reference'],k)),'benefit':st.mean(vals) if vals else None,'observed_outer_units':len(rr),'valid_outer_units':len(vals)})
 # Matched cross-model and FT/LP differences, avoiding performance winner selection.
 cross=defaultdict(dict)
 for v in units:cross[tuple(v[x] for x in ['dataset','outer','budget','participant','rule','metric'])][v['model'],v['method']]=v['value']
 crossrows=[]
 for k,vals in sorted(cross.items()):
  for a,b in [(('ri_gcn',m),('msgt',m)) for m in ['ft','lp']]+[((model,'ft'),(model,'lp')) for model in ['msgt','ri_gcn']]:
   av,bv=vals[a],vals[b];sign=1 if k[-1] in ['ccc','auroc','balanced_accuracy'] else -1
   crossrows.append({**dict(zip(['dataset','outer','budget','participant','rule','metric'],k)),'candidate_model':a[0],'candidate_method':a[1],'reference_model':b[0],'reference_method':b[1],'benefit':sign*(av-bv) if av is not None and bv is not None else None})
 cig=defaultdict(list)
 for v in paired:
  if v['metric']==('rmse' if v['dataset']=='U2K' else 'brier'):
   key=tuple(v[x] for x in ['dataset','model','budget','method','metric','candidate','reference']);cig[('rule',)+key].append(v)
 for v in crossrows:
  if v['metric']==('rmse' if v['dataset']=='U2K' else 'brier'):
   key=(v['dataset'],v['candidate_model']+'_'+v['candidate_method']+' vs '+v['reference_model']+'_'+v['reference_method'],v['budget'],'paired',v['metric'],v['rule'],v['rule']);cig[('model_or_method',)+key].append(v)
 intervals=[]
 for k,rr in sorted(cig.items()):
  intervals.append({**dict(zip(['comparison_type','dataset','model_or_comparison','budget','method','metric','candidate','reference'],k)),**cluster_interval(rr,k[1])})
 cg=defaultdict(list)
 for v in cost:cg[tuple(v[x] for x in ['dataset','model','budget','method'])].append(v)
 costsummary=[]
 for k,rr in sorted(cg.items()):
  row=dict(zip(['dataset','model','budget','method'],k));row['final_adaptation_cells']=len(rr)
  for field in ['development_training_seconds','target_fit_seconds','total_adaptation_seconds']:
   row[field+'_mean']=st.mean(v[field] for v in rr);row[field+'_sum']=sum(v[field] for v in rr)
  costsummary.append(row)
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 for name,rr in [('participant_seed_metrics',raw),('participant_seedmean',units),('summary',summary),('paired_outer_participant',paired),('paired_unique_participant',persons),('paired_model_method',crossrows),('adaptation_cost',cost),('adaptation_cost_summary',costsummary),('primary_paired_intervals',intervals),('source_cost',list(sources.values())),('selection',selection)]:write(out/(name+'.csv'),rr)
 lines=['# 公平适配内部验证：'+('仅合成测试，非正式结果' if synthetic else '正式完成批次'),'', '本轮使用研究者曾观察过的既有数据与标签，属于新增内部验证，不替代旧协议结果。完整列出两模型、FT/LP、预算和五规则，不按测试成绩挑选赢家。','', '|任务|模型|方法|预算|规则|主指标|数值|有效唯一参与者|','|---|---|---|---:|---|---|---:|---:|']
 for v in summary:
  if v['metric']==('rmse' if v['dataset']=='U2K' else 'brier'):lines.append('|'+ '|'.join(str(v[x]) if v[x] is not None else 'NA' for x in ['dataset','model','method','budget','rule','metric','value','valid_unique_participants'])+'|')
 lines+=['','U2K先人内计算各seed指标，再seed平均、repeat内参与者平均，最后三个repeat等权。IRDS先人内各seed指标，再seed与唯一参与者平均。单类AUROC/BA为NA；每项保留有效分母。n=1开发选权重为NA，不用其他规则替代。','', '逐outer参与者配对表保留完整重复轨迹；唯一参与者汇总仅作描述，不替代U2K的repeat等权主结果。主指标RMSE/Brier使用2000次唯一参与者cluster bootstrap，固定随机种子20260920；同一参与者在所有repeat共享抽中次数，U2K每repeat内按次数加权后三个repeat等权，IRDS按唯一参与者次数加权。若某repeat无有效观察，该draw重抽；NA比较不计算区间。95%分位区间描述已观察队列差异，不覆盖训练随机性，也不证明重叠训练折独立。未重采样预算、seed或fold。其他指标不新增CI。','', '所有十种规则配对、RI-GCN与MSGT、FT与LP配对均保留；收益为正表示表中candidate优于reference。primary_paired_intervals.csv包含主指标差、区间、有效人数和ci_scope。','', '成本表分别给出六次内拟合搜索总秒（n=1为0）、最终拟合秒及两者之和，并保存逐格与汇总。源checkpoint按SHA去重，不重复加到每次适配。原MSGT源计时在selection收据中缺失，因此此处为NA，应单独引用已标明范围的历史证据；不是同场重测。推理成本由独立计时工件提供，不由训练秒数推算。','', f'校验完成：{len(files)}个最终selection、1152次内拟合、288次最终拟合、18个RI-GCN源checkpoint。依赖工件、冻结选择和预测SHA均校验并记录于VALIDATION.json；新增版本仍待独立终审。']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');dump(out/'VALIDATION.json',{'status':'synthetic_only' if synthetic else 'local_validation_pass_pending_independent_review','selections':len(files),'inner_fits':1152,'final_fits':288,'source_checkpoints':len(sources),'inputs_sha256':inputs,'freeze_sha256':freeze,'ci':'2000_unique_participant_cluster_draws_seed20260920_primary_metrics_only'})
 return summary
def self_test():
 assert metrics([0.,1.],[0.,1.],'irds')=={'brier':0.,'auroc':1.,'balanced_accuracy':1.}
 assert metrics([0.,1.],[.5,.5],'irds')['auroc']==.5
 assert metrics([0.,0.],[.2,.3],'irds')['balanced_accuracy'] is None
 assert abs(metrics([0.,1.],[0.,1.],'U2K')['ccc']-1)<1e-12
 with tempfile.TemporaryDirectory(prefix='fair_analysis_synthetic_') as tmp:
  r=Path(tmp);(r/'prepared').mkdir();(r/'results').mkdir();dump(r/'freeze.json',{'synthetic':True});fh=sha(r/'freeze.json')
  actual_path=Path(__file__).resolve().parent.parent/'prepared/manifest.json';actual=json.loads(actual_path.read_text())
  assert {c['dataset'] for c in actual['cells']}=={'U2K','irds'}
  cells=[{k:c[k] for k in ['dataset','outer','budget']} for c in actual['cells']]
  manifest={k:actual[k] for k in ['models','seeds','n1_learning_rate','learning_rates','alpha_grid']};manifest['cells']=cells;dump(r/'prepared/manifest.json',manifest)
  for c,model,seed,method in itertools.product(cells,manifest['models'],manifest['seeds'],['ft','lp']):
   ds,o,n=c['dataset'],c['outer'],c['budget'];name=f'{ds}_{model}_{o}_{n}_{seed}_{method}';people=[f'P{i}' for i in (range(o*6,min((o+1)*6,29)) if ds=='irds' else range(3))];pr=[]
   for person,y in itertools.product(people,[0.,1.]):
    s=.2+.6*y;t=.1+.8*y;pr.append(dict(record_id=f'{person}_{int(y)}',participant=person,action='A',y_true=y,source=s,direct=t,fixed_half=(s+t)/2,dynamic=(.5*s+n*t)/(n+.5),development_alpha=(s+t)/2 if n>1 else ''))
   pf=name+'_predictions.csv';write(r/'results'/pf,pr);choice={'freeze_sha256':fh,'dataset':ds,'model':model,'outer':o,'budget':n,'seed':seed,'method':method,'selected_lr':.001,'selected_alpha':.5 if n>1 else None,'lr_scores':[dict(lr=x,score=0) for x in manifest['learning_rates']] if n>1 else [],'alpha_scores':[dict(alpha=x,score=0) for x in manifest['alpha_grid']] if n>1 else [],'selection_participants':['train'],'source_checkpoint_sha256':f'{ds}_{model}_{o if ds=="irds" else 0}_{seed}','source_checkpoint_path':'synthetic','source_training_seconds':2. if model=='ri_gcn' else None}
   dump(r/'results'/(name+'_frozen_choice.json'),choice);z={**choice,'dependencies':{name+'_frozen_choice.json':sha(r/'results'/(name+'_frozen_choice.json'))},'prediction_file':pf,'prediction_sha256':sha(r/'results'/pf),'target_fit_seconds':1.,'development_training_seconds':6. if n>1 else 0.,'total_adaptation_seconds':7. if n>1 else 1.,'development_fits':6 if n>1 else 0,'target_fits':1,'test_participants':sorted(people)};dump(r/'results'/(name+'_selection.json'),z)
  s=aggregate(r,r/'analysis',synthetic=True)
  assert all(abs(v['value']-.1)<1e-12 for v in s if v['dataset']=='U2K' and v['rule']=='direct' and v['metric']=='rmse')
  assert all(v['value'] is None for v in s if v['budget']==1 and v['rule']=='development_alpha')
  assert all(v['valid_unique_participants']==29 for v in s if v['dataset']=='irds' and v['rule']=='direct')
  # Deliberately incomplete batch must not create output.
  next((r/'results').glob('*_selection.json')).unlink()
  try:aggregate(r,r/'must_not_exist',synthetic=True)
  except AssertionError:pass
  else:raise AssertionError('incomplete batch accepted')
  assert not (r/'must_not_exist').exists()
 test=[dict(participant=p,outer=o,benefit=b) for p,bs in [('A',[1.,2.,3.]),('B',[3.,4.,5.])] for o,b in enumerate(bs)]
 ci=cluster_interval(test,'U2K');assert ci['mean_benefit']==3. and ci['ci_low']==2. and ci['ci_high']==4.
 sparse=[dict(participant=p,outer=o,benefit=b) for p,o,b in [('A',0,0.),('A',1,10.),('A',2,20.),('B',0,2.),('B',2,24.)]]
 ci=cluster_interval(sparse,'U2K');assert ci['mean_benefit']==11. and ci['ci_low']==10. and ci['ci_high']==11.
 assert cluster_interval([dict(participant='A',outer=o,benefit=None) for o in range(3)],'U2K')['ci_low'] is None
 print(json.dumps({'self_test':'PASS','formal_results_read':False,'synthetic_final_cells':288,'checks':['metric_formulas','tied_auc','single_class_NA','full_schema_aggregation','n1_NA','29_IRDS_unique_participants','incomplete_batch_rejected','actual_manifest_task_set','sparse_repeat_weighting_and_rejection','all_NA_interval','cluster_bootstrap_same_counts_across_repeats']}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
 if a.self_test:self_test()
 else:
  ap.error('--root is required unless --self-test') if a.root is None else None
  aggregate(a.root,a.out or a.root/'analysis');print('Complete batch aggregated; independent review pending.')
