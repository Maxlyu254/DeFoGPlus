#!/usr/bin/env python3
"""Audit available batch2 results alongside seed0, without model execution."""
import argparse
import json
from pathlib import Path
import statistics as st

from analyze_batch1 import EPOCHS, METRICS, close, count_audit, read, rows, write_csv


def analyze(batch1, batch2, output):
    output.mkdir(parents=True, exist_ok=True)
    launch = read(batch2/'launch.json')
    finals, curves, learning, fingerprints = [], [], [], []
    for root, batch in [(batch1, 'batch1'), (batch2, 'batch2')]:
        for path in sorted((root/'runs').glob('*/metrics/summary.json')):
            run = path.parent.parent
            summary, manifest = read(path), read(run/'manifest.json')
            variant, seed = summary['configuration'], summary['training_seed']
            if variant not in ['B0', 'A0', 'M0', 'M1']:
                continue
            if batch == 'batch2':
                assert manifest['git_commit'] == launch['git_commit']
                assert summary['best_epoch'] is None
                assert list(summary['checkpoint_roles']) == ['final']
            assert summary['status'] == 'completed' and summary['completed_epochs'] == 1000
            assert manifest['training_seed'] == seed and manifest['configuration'] == variant
            fingerprints.append((manifest['canonical_training_reference_sha256'],
                                 manifest['validation_indices_sha256'],
                                 json.dumps(manifest['statistics'], sort_keys=True)))
            train, curve = rows(run/'metrics/train.jsonl'), rows(run/'metrics/curve.jsonl')
            assert len(train) == 1000 and train[-1]['optimizer_step'] == 96000
            assert [r['completed_epoch'] for r in train] == list(range(1,1001))
            assert [r['completed_epoch'] for r in curve] == EPOCHS
            result = summary['final_results'][summary['checkpoint_roles']['final']]
            assert result['completed_epoch'] == 1000
            assert [r['sampling_seed'] for r in result['repeats']] == [21000,21001,21002]
            for r in result['repeats']:
                count_audit(r)
                assert r['attempts'] == 10000 and r['completed_epoch'] == 1000
                assert r['identity_policy'] == 'strict-connected-isomeric-canonical-no-correction'
                assert r['sampling_steps'] == 500
                assert r['checkpoint_weights_sha256'] == curve[-1]['checkpoint_weights_sha256']
            record = dict(configuration=variant, seed=seed, batch=batch)
            for metric in METRICS:
                vals = [r[metric] for r in result['repeats']]
                close(st.mean(vals), result['aggregate'][metric]['mean'])
                close(st.stdev(vals), result['aggregate'][metric]['sample_std'])
                record[metric+'_percent'] = 100*st.mean(vals)
                record[metric+'_sampling_sd_pp'] = 100*st.stdev(vals)
            for key in ['invalid_count','disconnected_count','atom_change_fraction','composition_change_fraction']:
                record[key] = st.mean(r[key] for r in result['repeats'])
            record['training_hours'] = sum(r['seconds'] for r in train)/3600
            finals.append(record)
            for r in curve:
                count_audit(r)
                expected = 10000 if batch == 'batch2' and r['completed_epoch'] >= 400 else 2000
                assert r['attempts'] == expected
                if variant == 'A0':
                    assert r['atom_change_fraction'] == r['composition_change_fraction'] == 0
                item = dict(configuration=variant, seed=seed, batch=batch,
                            epoch=r['completed_epoch'], attempts=r['attempts'])
                for metric in METRICS:
                    # Pool size affects uniqueness and VUN; use the saved 2k subset
                    # for cross-batch curves. Full-pool validity is also retained.
                    value = r.get('comparison_'+metric, r[metric])
                    item[metric+'_2k_percent'] = None if value is None else 100*value
                    item[metric+'_full_percent'] = None if r[metric] is None else 100*r[metric]
                if expected == 10000:
                    subset = {k[len('comparison_'):]:v for k,v in r.items() if k.startswith('comparison_')}
                    count_audit(subset)
                    assert subset['attempts'] == 2000
                curves.append(item)
            early = [r for r in curve if r['completed_epoch'] <= 100]
            auc = sum((b['completed_epoch']-a['completed_epoch'])*(a['validity']+b['validity'])/2
                      for a,b in zip(early, early[1:]))
            item = dict(configuration=variant, seed=seed, validity_auc_0_100_percent=auc)
            for epoch in [5,10,25,50,100]:
                item['validity_epoch'+str(epoch)] = 100*next(r['validity'] for r in curve if r['completed_epoch']==epoch)
            for threshold in [90,95,98]:
                item['first_observed_'+str(threshold)] = next((r['completed_epoch'] for r in curve if r['validity']>=threshold/100),None)
            learning.append(item)
    assert len(set(fingerprints)) == 1, 'Data/reference/cache provenance differs'
    assert {(r['configuration'],r['seed']) for r in finals if r['batch']=='batch2'} == {
        ('B0',1),('B0',2),('A0',1),('M0',1)}
    groups = {}
    for variant in ['B0','A0','M0','M1']:
        records = [r for r in finals if r['configuration']==variant]
        groups[variant] = dict(seeds=[r['seed'] for r in records])
        for metric in METRICS:
            vals = [r[metric+'_percent'] for r in records]
            groups[variant][metric] = dict(mean_percent=st.mean(vals),
                training_seed_sd_pp=st.stdev(vals) if len(vals)>1 else None)
    contrasts = {}
    for variant in ['A0','M0','M1']:
        records=[]
        for r in [r for r in finals if r['configuration']==variant]:
            b = next(b for b in finals if b['configuration']=='B0' and b['seed']==r['seed'])
            records.append(dict(seed=r['seed'], **{m+'_pp':r[m+'_percent']-b[m+'_percent'] for m in METRICS}))
        contrasts[variant+' minus B0'] = records
    assessment = dict(audit='passed', note='Sampling repeats are averaged within a training seed; seed0 is exploratory.',
        completed_batch2_runs=4, failed_before_training=['M0-seed2','M1-seed1','M1-seed2'],
        groups=groups, paired_contrasts=contrasts, budget=read(batch2/'budget-ledger.json'))
    (output/'assessment.json').write_text(json.dumps(assessment,indent=2)+'\n')
    write_csv(output/'final-comparison.csv', finals)
    write_csv(output/'learning-curves.csv', curves)
    write_csv(output/'learning-speed.csv', learning)
    print(json.dumps(assessment, indent=2))
    return finals, curves


def plots(finals, curves, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'B0':'#3366aa','A0':'#bb6600','M0':'#663399','M1':'#aa77bb'}
    fig, axes = plt.subplots(2,2,figsize=(12,8), constrained_layout=True)
    for variant in colors:
        for seed in sorted({r['seed'] for r in curves if r['configuration']==variant}):
            data=[r for r in curves if r['configuration']==variant and r['seed']==seed]
            label=f'{variant} seed{seed}'
            style=['--','-',':'][seed]
            for ax, metric, lo, hi in [(axes[0,0],'validity_2k_percent',0,100),
                                      (axes[0,1],'validity_full_percent',400,1000),
                                      (axes[1,0],'vun_2k_percent',0,1000)]:
                sub=[r for r in data if lo<=r['epoch']<=hi]
                ax.plot([r['epoch'] for r in sub], [r[metric] for r in sub],
                        color=colors[variant],linestyle=style,marker='.',label=label)
    axes[0,0].set(title='Early validity (2k attempts)',ylabel='Strict validity (%)',ylim=(0,102))
    axes[0,1].set(title='Late validity (seed0: 2k; new seeds: 10k)',ylabel='Strict validity (%)',ylim=(97.5,100))
    axes[1,0].set(title='Matched 2k VUN through training',ylabel='VUN (%)')
    axes[1,0].set_xscale('symlog',linthresh=1)
    for i,v in enumerate(colors):
        data=[r for r in finals if r['configuration']==v]
        for r in data:
            axes[1,1].errorbar(i+(r['seed']-1)*.16,r['validity_percent'],
                yerr=r['validity_sampling_sd_pp'],fmt='o',color=colors[v],capsize=3)
            axes[1,1].annotate(str(r['seed']), (i+(r['seed']-1)*.16,r['validity_percent']),xytext=(4,3),textcoords='offset points')
    axes[1,1].set(title='Epoch1000: per-seed mean ± sampling SD (3 × 10k)',
                  ylabel='Strict validity (%)',xticks=range(4),xticklabels=list(colors))
    for ax in axes.flat:
        ax.grid(alpha=.2)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:
        ax.set_xlabel('Completed epoch')
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('QM9 batch2 partial completion with batch1 seed0; missing runs omitted')
    for ext in ['png','pdf']:
        fig.savefig(output/('learning-curves.'+ext),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch1',type=Path,required=True)
    p.add_argument('--batch2',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plots',action='store_true')
    args=p.parse_args()
    f,c=analyze(args.batch1,args.batch2,args.output)
    if args.plots:
        plots(f,c,args.output)
