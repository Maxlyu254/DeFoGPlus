#!/usr/bin/env python3
"""Audit all three QM9 batches, retaining incomplete curves separately."""
import argparse
import json
from pathlib import Path
import statistics as st

from analyze_batch1 import EPOCHS, METRICS, close, count_audit, read, rows, write_csv

VARIANTS=['B0','B1','A0','A1','M0','M1']


def analyze(roots, output):
    output.mkdir(parents=True,exist_ok=True)
    finals,curves,learning,partial,fingerprints=[],[],[],[],[]
    seen=set()
    for batch,root in enumerate(roots,1):
        launch=read(root/'launch.json')
        for run in sorted((root/'runs').iterdir()):
            if not (run/'manifest.json').is_file():
                continue
            manifest=read(run/'manifest.json')
            variant,seed=manifest['configuration'],manifest['training_seed']
            assert (variant,seed) not in seen
            seen.add((variant,seed))
            assert manifest['git_commit']==launch['git_commit']
            fingerprints.append((manifest['canonical_training_reference_sha256'],
                manifest['validation_indices_sha256'],json.dumps(manifest['statistics'],sort_keys=True)))
            train=rows(run/'metrics/train.jsonl')
            curve=rows(run/'metrics/curve.jsonl')
            assert [r['completed_epoch'] for r in train]==list(range(1,len(train)+1))
            assert train[-1]['optimizer_step']==len(train)*96
            assert [r['completed_epoch'] for r in curve]==[e for e in EPOCHS if e<=len(train)]
            complete=(run/'metrics/summary.json').is_file() and read(run/'metrics/summary.json')['status']=='completed'
            early=[]
            for r in curve:
                count_audit(r)
                assert r['identity_policy']=='strict-connected-isomeric-canonical-no-correction'
                assert r['sampling_steps']==500
                expected=10000 if batch>=2 and r['completed_epoch']>=400 else 2000
                assert r['attempts']==expected
                if expected==10000:
                    subset={k[len('comparison_'):]:v for k,v in r.items() if k.startswith('comparison_')}
                    count_audit(subset)
                    assert subset['attempts']==2000
                if variant.startswith('A'):
                    assert r['atom_change_fraction']==r['composition_change_fraction']==0
                item=dict(configuration=variant,seed=seed,batch=batch,completed_run=complete,
                          epoch=r['completed_epoch'],full_attempts=expected)
                for metric in METRICS:
                    v=r.get('comparison_'+metric,r[metric])
                    item[metric+'_2k_percent']=None if v is None else 100*v
                    item[metric+'_full_percent']=None if r[metric] is None else 100*r[metric]
                curves.append(item)
                if r['completed_epoch']<=100:
                    early.append(r)
            auc=sum((b['completed_epoch']-a['completed_epoch'])*(a['validity']+b['validity'])/2
                    for a,b in zip(early,early[1:]))
            speed=dict(configuration=variant,seed=seed,completed_run=complete,validity_auc_0_100_percent=auc)
            for e in [5,10,25,50,100]:
                speed['validity_epoch'+str(e)]=100*next(r['validity'] for r in early if r['completed_epoch']==e)
            learning.append(speed)
            if not complete:
                index=read(run/'checkpoint-index.json')
                partial.append(dict(configuration=variant,seed=seed,logged_epochs=len(train),
                                    latest_checkpoint=index['latest'],last_curve_epoch=curve[-1]['completed_epoch']))
                continue
            summary=read(run/'metrics/summary.json')
            assert summary['completed_epochs']==len(train)==1000
            final=summary['final_results'][summary['checkpoint_roles']['final']]
            assert final['completed_epoch']==1000
            assert [r['sampling_seed'] for r in final['repeats']]==[21000,21001,21002]
            record=dict(configuration=variant,seed=seed,batch=batch,training_hours=sum(r['seconds'] for r in train)/3600)
            for r in final['repeats']:
                count_audit(r)
                assert r['attempts']==10000 and r['completed_epoch']==1000
                assert r['checkpoint_weights_sha256']==curve[-1]['checkpoint_weights_sha256']
            for metric in METRICS:
                vals=[r[metric] for r in final['repeats']]
                close(st.mean(vals),final['aggregate'][metric]['mean'])
                close(st.stdev(vals),final['aggregate'][metric]['sample_std'])
                record[metric+'_percent']=100*st.mean(vals)
                record[metric+'_sampling_sd_pp']=100*st.stdev(vals)
            finals.append(record)
    assert len(set(fingerprints))==1
    assert len(finals)==17 and len(partial)==1
    groups={}
    for variant in VARIANTS:
        data=[r for r in finals if r['configuration']==variant]
        groups[variant]=dict(seeds=[r['seed'] for r in data])
        for m in METRICS:
            vals=[r[m+'_percent'] for r in data]
            groups[variant][m]=dict(mean_percent=st.mean(vals),training_seed_sd_pp=st.stdev(vals))
    contrasts={}
    for variant,control in [('B1','B0'),('A1','A0'),('M1','M0'),('A0','B0'),('M0','B0'),('M1','B0')]:
        paired=[]
        for r in [r for r in finals if r['configuration']==variant]:
            b=next(b for b in finals if b['configuration']==control and b['seed']==r['seed'])
            speed=lambda v: next(s['validity_auc_0_100_percent'] for s in learning if s['configuration']==v and s['seed']==r['seed'])
            paired.append(dict(seed=r['seed'],**{m+'_pp':r[m+'_percent']-b[m+'_percent'] for m in METRICS},
                               validity_auc_pp=speed(variant)-speed(control)))
        contrasts[variant+' minus '+control]=paired
    result=dict(audit='passed',complete_runs=len(finals),partial_runs=partial,
                groups=groups,paired_contrasts=contrasts,batch3_budget=read(roots[2]/'budget-ledger.json'),
                note='Final repeats are averaged within seed. Seed0 exploratory; no training-seed significance claim.')
    (output/'assessment.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv(output/'final-per-seed.csv',finals)
    write_csv(output/'learning-curves.csv',curves)
    write_csv(output/'learning-speed.csv',learning)
    print(json.dumps(result,indent=2))
    return finals,curves


def plots(finals,curves,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=dict(zip(VARIANTS,['#3366aa','#66aadd','#bb6600','#eeaa44','#663399','#aa77bb']))
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for variant in VARIANTS:
        data=[r for r in curves if r['configuration']==variant and r['completed_run']]
        for ax,m,lo,hi in [(axes[0,0],'validity_2k_percent',0,100),
                           (axes[0,1],'validity_2k_percent',100,1000),
                           (axes[1,0],'vun_2k_percent',0,1000)]:
            epochs=sorted({r['epoch'] for r in data if lo<=r['epoch']<=hi})
            values=[[r[m] for r in data if r['epoch']==e] for e in epochs]
            mean=[st.mean(v) for v in values]
            sd=[st.stdev(v) for v in values]
            ax.plot(epochs,mean,color=colors[variant],label=variant)
            ax.fill_between(epochs,[a-b for a,b in zip(mean,sd)],[a+b for a,b in zip(mean,sd)],color=colors[variant],alpha=.12)
        i=VARIANTS.index(variant)
        for r in [r for r in finals if r['configuration']==variant]:
            x=i+(r['seed']-1)*.16
            axes[1,1].errorbar(x,r['validity_percent'],yerr=r['validity_sampling_sd_pp'],fmt='o',color=colors[variant],capsize=3)
            axes[1,1].annotate(str(r['seed']),(x,r['validity_percent']),xytext=(3,3),textcoords='offset points')
    axes[0,0].set(title='Early validity: mean ± training-seed SD',ylabel='Strict validity (%)')
    axes[0,0].legend(ncol=3,fontsize=8)
    axes[0,1].set(title='Late validity: matched 2k pools',ylabel='Strict validity (%)',ylim=(96,100))
    axes[1,0].set(title='VUN: matched 2k pools, mean ± training-seed SD',ylabel='VUN (%)')
    axes[1,0].set_xscale('symlog',linthresh=1)
    axes[1,1].set(title='Epoch1000: per-seed mean ± sampling SD (3 × 10k)',ylabel='Strict validity (%)',
                  xticks=range(6),xticklabels=VARIANTS)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xlabel('Completed epoch')
    fig.suptitle('QM9: 17 completed runs; B1 seeds0,2, other models seeds0,1,2. Partial B1 seed1 omitted.')
    for ext in ['png','pdf']:
        fig.savefig(output/('learning-curves.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs=3,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    args=p.parse_args()
    f,c=analyze(args.roots,args.output)
    if args.plots:
        plots(f,c,args.output)
