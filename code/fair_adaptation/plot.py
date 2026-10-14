"""Local plotting of completed fair-adaptation tables; never starts training.
Use --root FAIR_ROOT. --self-test builds marked synthetic previews in a temporary
folder from the actual prepared manifest, validates them and deletes them.
"""
import argparse,csv,hashlib,itertools,json,math,tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
RULES=['source','direct','fixed_half','dynamic','development_alpha']
COLORS=['#777777','#0072B2','#E69F00','#D55E00','#009E73']
MARKERS=['x','o','s','D','^']
LABELS=['Source','Direct','Fixed 0.5','Dynamic','Development alpha']
DATASETS=['U2K','irds'];MODELS=['msgt','ri_gcn'];METHODS=['ft','lp']
PRIMARY={'U2K':'rmse','irds':'brier'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'savefig.bbox':'tight','axes.grid':True,'grid.alpha':.15})
def read(p):return list(csv.DictReader(Path(p).open()))
def number(x):return float(x) if x not in ('',None,'NA','nan') else float('nan')
def generate(root,out=None,synthetic=False):
 r=Path(root);paths=[r/'analysis'/f for f in ['summary.csv','adaptation_cost_summary.csv','primary_paired_intervals.csv']]+[r/'inference_cost/measurements.csv']
 for p in paths:assert p.exists(),f'Missing completed analysis input: {p}; no figures written'
 summary,cost,intervals,inference=map(read,paths)
 manifest=json.loads((r/'prepared/manifest.json').read_text());budgets={ds:sorted({c['budget'] for c in manifest['cells'] if c['dataset']==ds}) for ds in DATASETS}
 lookup={(v['dataset'],v['model'],v['method'],int(v['budget']),v['rule'],v['metric']):number(v['value']) for v in summary}
 costs={(v['dataset'],v['model'],v['method'],int(v['budget'])):v for v in cost}
 timed={(v['dataset'],v['model'],v['method'],int(v['batch']),v['rule']):v for v in inference}
 for ds,mo,me in itertools.product(DATASETS,MODELS,METHODS):
  for n in budgets[ds]:
   for rule in RULES:
    value=lookup[ds,mo,me,n,rule,PRIMARY[ds]]
    assert (math.isnan(value) if rule=='development_alpha' and n==1 else math.isfinite(value)),(ds,mo,me,n,rule)
   assert (ds,mo,me,n) in costs
  for batch,rule in itertools.product([1,16],['direct','dual_dynamic']):
   t=timed[ds,mo,me,batch,rule];assert int(t['outer'])==0 and int(t['seed'])==0 and int(t['budget'])==4
   assert number(t['p90_ms'])>=number(t['p50_ms'])>=0
 assert len(timed)==32
 out=Path(out) if out else r/'figures';out.mkdir(parents=True,exist_ok=True);art=[]
 def save(fig,name,footnote):
  fig.text(.5,.005,footnote,ha='center',fontsize=8)
  if synthetic:fig.text(.5,.995,'SYNTHETIC LAYOUT TEST — NOT RESEARCH RESULTS',ha='center',va='top',color='#AA0000',fontsize=11)
  for ext in ['pdf','png']:
   p=out/(name+'.'+ext);fig.savefig(p,dpi=300);art.append(p)
  plt.close(fig)
 combos=list(itertools.product(MODELS,METHODS))
 fig,axes=plt.subplots(2,4,figsize=(13.5,6),sharey='row')
 for i,ds in enumerate(DATASETS):
  for j,(mo,me) in enumerate(combos):
   ax=axes[i,j];ns=budgets[ds]
   for rule,c,m,label in zip(RULES,COLORS,MARKERS,LABELS):ax.plot(range(len(ns)),[lookup[ds,mo,me,n,rule,PRIMARY[ds]] for n in ns],color=c,marker=m,label=label)
   ax.set(title=f'{ds} | {mo.upper()} {me.upper()}',xticks=range(len(ns)),xticklabels=ns,xlabel='Target participant budget')
   if j==0:ax.set_ylabel(PRIMARY[ds].upper()+' (lower is better)')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=5);fig.tight_layout(rect=(0,.12,1,.96));save(fig,'budget_primary','Development alpha is unavailable at n=1. Different datasets retain their original metrics and aggregation.')
 # Costs shown separately: search+refit and final refit only. Source is a horizontal accuracy reference, not a zero-cost claim.
 for field,label,name in [('total_adaptation_seconds_mean','Mean search + final-refit seconds','accuracy_total_adaptation_cost'),('target_fit_seconds_mean','Mean final-refit seconds (search excluded)','accuracy_final_refit_cost')]:
  fig,axes=plt.subplots(2,4,figsize=(13.5,6),sharey='row')
  for i,ds in enumerate(DATASETS):
   for j,(mo,me) in enumerate(combos):
    ax=axes[i,j];ns=budgets[ds];xx=[number(costs[ds,mo,me,n][field]) for n in ns]
    sourcevalues=[lookup[ds,mo,me,n,'source',PRIMARY[ds]] for n in ns];assert max(sourcevalues)-min(sourcevalues)<1e-8
    ax.axhline(sourcevalues[0],color=COLORS[0],ls=':',label='Source accuracy')
    for rule,c,m,lab in zip(RULES[1:],COLORS[1:],MARKERS[1:],LABELS[1:]):
     yy=[lookup[ds,mo,me,n,rule,PRIMARY[ds]] for n in ns];ax.plot(xx,yy,color=c,marker=m,label=lab)
     if rule=='direct':
      for x,y,n in zip(xx,yy,ns):ax.annotate(f'n={n}',(x,y),xytext=(3,5),textcoords='offset points',fontsize=7)
    ax.set(title=f'{ds} | {mo.upper()} {me.upper()}',xlabel=label);ax.set_xscale('log');ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x,pos:f'{x:g}'));ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    if j==0:ax.set_ylabel(PRIMARY[ds].upper()+' (lower is better)')
  fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=5);fig.tight_layout(rect=(0,.12,1,.96));save(fig,name,'One-time source training excluded. Fusion rules share the selected endpoint; source line is an accuracy reference only.')
 # Four transparent Dynamic comparisons; all ten numerical pairs remain in CSV.
 ci={(v['dataset'],v['model_or_comparison'],v['method'],int(v['budget']),v['candidate'],v['reference']):v for v in intervals if v['comparison_type']=='rule'}
 fig,axes=plt.subplots(2,4,figsize=(13.5,6),sharey='row')
 for i,ds in enumerate(DATASETS):
  for j,(mo,me) in enumerate(combos):
   ax=axes[i,j];ns=budgets[ds]
   for ri,ref in enumerate(['source','direct','fixed_half','development_alpha']):
    for ni,n in enumerate(ns):
     k=(ds,mo,me,n,'dynamic',ref);sign=1
     if k not in ci:k=(ds,mo,me,n,ref,'dynamic');sign=-1
     v=ci[k];mean=sign*number(v['mean_benefit']);lo,hi=sorted([sign*number(v['ci_low']),sign*number(v['ci_high'])])
     if not math.isfinite(mean):continue
     x=ni+(ri-1.5)*.12;color=COLORS[RULES.index(ref)]
     ax.plot([x,x],[lo,hi],color=color,linewidth=1.4);ax.plot(x,mean,marker=MARKERS[RULES.index(ref)],color=color,label='vs '+LABELS[RULES.index(ref)] if ni==(1 if ref=='development_alpha' else 0) else None)
   ax.axhline(0,color='black',lw=.6);ax.set(title=f'{ds} | {mo.upper()} {me.upper()}',xticks=range(len(ns)),xticklabels=ns,xlabel='Target participant budget')
   if j==0:ax.set_ylabel('Dynamic benefit (positive is better)')
 fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.035),ncol=4);fig.tight_layout(rect=(0,.12,1,.96));save(fig,'dynamic_paired_intervals','Descriptive 95% participant-cluster intervals; repeated trajectories preserved. Not training-uncertainty intervals.')
 # Physical inference costs; p90 is shown as a distinct quantile, never a confidence interval.
 fig,axes=plt.subplots(2,3,figsize=(12,6))
 for i,ds in enumerate(DATASETS):
  for col,quantity in enumerate(['latency','parameters','peak']):
   ax=axes[i,col]
   for kk,(batch,rule) in enumerate(itertools.product([1,16],['direct','dual_dynamic'])):
    if quantity=='parameters' and batch==16:continue
    xx=np.arange(4)+(kk-1.5)*.18;vv=[]
    for mo,me in combos:
     z=timed[ds,mo,me,batch,rule];vv.append(number(z['p50_ms']) if quantity=='latency' else number(z['parameters'])/1e6 if quantity=='parameters' else number(z['peak_allocated_bytes'])/1024**2)
    lab=('Single' if rule=='direct' else 'Dual')+(f', B={batch}' if quantity!='parameters' else '')
    ax.bar(xx,vv,width=.17,label=lab,color=['#56B4E9','#0072B2','#F0C76B','#D55E00'][kk])
    if quantity=='latency':
     hi=[number(timed[ds,mo,me,batch,rule]['p90_ms']) for mo,me in combos];ax.scatter(xx,hi,marker='_',color='black',s=45,zorder=3)
   ax.set(title=ds,xticks=range(4),xticklabels=[f'{mo}\n{me.upper()}' for mo,me in combos],ylabel={'latency':'Latency p50 (ms); black mark = p90','parameters':'Resident model parameters (millions)','peak':'Peak allocated GPU memory (MiB)'}[quantity]);ax.set_ylim(bottom=0)
   if i==0:ax.legend(fontsize=7,ncol=2)
 fig.tight_layout(rect=(0,.07,1,.96));save(fig,'inference_complete_cost','Fixed outer0 / seed0 / n4 checkpoints. GPU-resident forward + sigmoid + fusion; preprocessing, pose extraction and H2D excluded.')
 fig,axes=plt.subplots(2,2,figsize=(10,6),sharey='row')
 for i,ds in enumerate(DATASETS):
  for j,batch in enumerate([1,16]):
   ax=axes[i,j]
   for kk,(mo,me) in enumerate(combos):
    xx=[number(timed[ds,mo,me,batch,rr]['p50_ms']) for rr in ['direct','dual_dynamic']];yy=[lookup[ds,mo,me,4,rr,PRIMARY[ds]] for rr in ['direct','dynamic']]
    ax.plot(xx,yy,color=COLORS[kk+1],linewidth=1,label=f'{mo} {me.upper()}');ax.scatter(xx[0],yy[0],color=COLORS[kk+1],marker='o');ax.scatter(xx[1],yy[1],color=COLORS[kk+1],marker='D')
   ax.set(title=f'{ds} | batch={batch}',xlabel='Representative-checkpoint p50 latency (ms)',xlim=(0,None));ax.legend(fontsize=7)
   if j==0:ax.set_ylabel(PRIMARY[ds].upper()+' at n=4 (cohort aggregate)')
 fig.tight_layout(rect=(0,.1,1,.96));save(fig,'accuracy_inference_cost','Circle: direct; diamond: complete Dynamic. Fixed checkpoint cost joined to cohort-aggregate accuracy; other fusion rules were not timed.')
 provenance={'synthetic':synthetic,'inputs_sha256':{str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'figures':[p.name for p in art],'source_cost_excluded':True,'inference_join':'representative_checkpoint_cost_vs_cohort_aggregate_n4_accuracy','n1_development':'NA_no_imputation'}
 (out/'PLOT_MANIFEST.json').write_text(json.dumps(provenance,indent=2)+'\n');return art
def self_test(preview_dir=None):
 import shutil
 actual=Path(__file__).resolve().parent.parent/'prepared/manifest.json';m=json.loads(actual.read_text());assert {c['dataset'] for c in m['cells']}==set(DATASETS)
 def write(path,rows):
  with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with tempfile.TemporaryDirectory(prefix='fair_plot_synthetic_') as tmp:
  r=Path(tmp)
  for p in ['prepared','analysis','inference_cost']:(r/p).mkdir()
  (r/'prepared/manifest.json').write_text(json.dumps(m));summary=[];cost=[];ci=[];timing=[]
  for ds,mo,me in itertools.product(DATASETS,MODELS,METHODS):
   ns=sorted({c['budget'] for c in m['cells'] if c['dataset']==ds})
   for n in ns:
    vals={rule:(.4 if rule=='source' else .12+.09/(n**.5)+.01*ri) for ri,rule in enumerate(RULES)}
    vals['development_alpha']=None if n==1 else vals['development_alpha']
    for rule,value in vals.items():summary.append(dict(dataset=ds,model=mo,method=me,budget=n,rule=rule,metric=PRIMARY[ds],value=value))
    final=n*(2 if mo=='ri_gcn' else 1)*(1 if me=='ft' else .5);search=0 if n==1 else final*3
    cost.append(dict(dataset=ds,model=mo,method=me,budget=n,target_fit_seconds_mean=final,development_training_seconds_mean=search,total_adaptation_seconds_mean=final+search))
    for a,b in itertools.combinations(RULES,2):
     d=vals[b]-vals[a] if vals[a] is not None and vals[b] is not None else None
     ci.append(dict(comparison_type='rule',dataset=ds,model_or_comparison=mo,method=me,budget=n,candidate=a,reference=b,mean_benefit=d,ci_low=d-.01 if d is not None else None,ci_high=d+.01 if d is not None else None))
   for batch,rule in itertools.product([1,16],['direct','dual_dynamic']):
    factor=(2 if mo=='ri_gcn' else 1)*(2 if rule=='dual_dynamic' else 1);p50=factor*(1+batch/8)
    timing.append(dict(dataset=ds,model=mo,method=me,outer=0,seed=0,budget=4,batch=batch,rule=rule,p50_ms=p50,p90_ms=p50*1.15,parameters=factor*500000,peak_allocated_bytes=factor*batch*1024**2*5))
  for p,rows in [('analysis/summary.csv',summary),('analysis/adaptation_cost_summary.csv',cost),('analysis/primary_paired_intervals.csv',ci),('inference_cost/measurements.csv',timing)]:write(r/p,rows)
  art=generate(r,synthetic=True);assert len(art)==12
  from PIL import Image
  for p in art:
   if p.suffix=='.pdf':assert p.read_bytes().startswith(b'%PDF')
   else:
    with Image.open(p) as im:assert im.width>2000 and im.height>1000
  if preview_dir:
   dest=Path(preview_dir).resolve();assert str(dest).startswith('/tmp/') or str(dest).startswith('/private/tmp/'),'Synthetic previews must remain in temporary storage';shutil.copytree(r/'figures',dest,dirs_exist_ok=True)
 print(json.dumps({'self_test':'PASS','manifest_source':str(actual),'formal_results_used':False,'synthetic_preview_files':12,'retained_temporary_preview':str(preview_dir) if preview_dir else None}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--self-test',action='store_true');ap.add_argument('--preview-dir',type=Path,help='Optional marked synthetic preview copy under /tmp only');a=ap.parse_args()
 if a.self_test:self_test(a.preview_dir)
 elif a.root is None:ap.error('--root required unless --self-test')
 else:print(json.dumps({'figures':[str(p) for p in generate(a.root,a.out)]}))
