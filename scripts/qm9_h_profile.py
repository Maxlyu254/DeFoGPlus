#!/usr/bin/env python3
"""Explicit-H resource profiling; never launches full research training."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import statistics
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from rdkit import RDLogger
from torch_geometric.data import Batch

from experiments.batch1.checks import smoke_invariants
from experiments.batch1.data import components, statistics_for, build_model
from experiments.batch1.evaluation import generate, synchronize
from experiments.batch1.h_metrics import evaluate_h
from experiments.batch1.metrics import canonical_reference, legacy_evaluate
from experiments.batch1.run import make_config, signature, revision
from experiments.batch1.storage import atomic_json, append_json, export_csv, RunLog, restore_rng


def config(variant, smoke=False):
    return make_config(SimpleNamespace(protocol='h_profile', data_dir=os.environ['BATCH1_DATA'],
                       seed=0, smoke=smoke, wandb_mode='disabled' if smoke else 'offline'), variant)


def load(variant, smoke=False, create=False):
    cfg = config(variant, smoke)
    data, infos, smiles = components(cfg)
    stats, metadata = statistics_for(cfg, data, infos, os.environ['BATCH1_CACHE'], create=create)
    return cfg, data, infos, smiles, stats, metadata


class EpochTimer(pl.Callback):
    def __init__(self, data, log, rows):
        self.data, self.log, self.rows = data, log, rows

    def on_train_start(self, trainer, model):
        if model.batch1_loaded:
            restore_rng(model.batch1_loaded['rng'])
            assert trainer.current_epoch == self.data.completed_epochs

    def on_train_epoch_start(self, trainer, model):
        synchronize(model)
        self.started = time.perf_counter()
        self.graphs = 0
        if model.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.graphs += batch.num_graphs

    def on_train_epoch_end(self, trainer, model):
        synchronize(model)
        epoch = trainer.current_epoch+1
        model.batch1_completed_epochs = self.data.completed_epochs = epoch
        metrics = model.train_loss.log_epoch_metrics()
        row = dict(completed_epoch=epoch, optimizer_step=trainer.global_step,
                   seconds=time.perf_counter()-self.started, graphs=self.graphs,
                   node_ce=float(metrics['train_epoch/x_CE']), edge_ce=float(metrics['train_epoch/E_CE']),
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if model.device.type=='cuda' else 0,
                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved() if model.device.type=='cuda' else 0,
                   warmup=epoch<=2, resume_check=epoch==8)
        self.rows.append(self.log.write('train', **row))
        export_csv(self.log.root/'metrics/train.csv', self.rows)
        print(json.dumps(row), flush=True)


def trainer(cfg, callback, epochs, smoke=False, accumulation=1):
    return pl.Trainer(accelerator='cpu' if smoke else 'gpu', devices=1,
        precision='32-true', max_epochs=epochs, callbacks=[callback], logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        gradient_clip_val=cfg.train.clip_grad, num_sanity_val_steps=0, limit_val_batches=0,
        reload_dataloaders_every_n_epochs=1, accumulate_grad_batches=accumulation,
        limit_train_batches=1 if smoke else 1.0)


def initialize(cfg, data, infos, smiles, stats, metadata):
    model = build_model(cfg, data, infos, smiles, stats)
    model.batch1_signature = signature(cfg, metadata)
    model.batch1_provenance = dict(git_commit=revision(), statistics=metadata, profiling=True)
    return model


def fit_resume(cfg, data, infos, smiles, stats, metadata, directory, smoke=False, accumulation=1):
    directory.mkdir(parents=True, exist_ok=True)
    model = initialize(cfg, data, infos, smiles, stats, metadata)
    if smoke:
        smoke_invariants(model, data)
    manifest = dict(logical_run_id='Hprofile-'+cfg.augmentation.id, git_commit=revision(),
                    statistics=metadata, profiling=True, effective_batch_size=1024,
                    physical_batch_size=cfg.train.batch_size, gradient_accumulation=accumulation)
    atomic_json(directory/'manifest.json', manifest)
    OmegaConf.save(cfg, directory/'resolved-config.yaml')
    log = RunLog(directory, cfg, manifest)
    rows = []
    callback = EpochTimer(data, log, rows)
    first_epochs = 1 if smoke else 7
    fit = trainer(cfg, callback, first_epochs, smoke, accumulation)
    started = time.perf_counter()
    fit.fit(model, datamodule=data)
    first_fit_seconds = time.perf_counter()-started
    checkpoint = directory/'resume-check.ckpt'
    started = time.perf_counter()
    fit.save_checkpoint(str(checkpoint))
    save_seconds = time.perf_counter()-started
    expected_step = fit.global_step
    model._trainer = None
    del fit, model
    gc.collect()
    if not smoke:
        torch.cuda.empty_cache()
    started = time.perf_counter()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    load_seconds = time.perf_counter()-started
    assert saved['batch1']['completed_epochs'] == first_epochs
    assert saved['global_step'] == expected_step and saved['optimizer_states']
    del saved
    model = initialize(cfg, data, infos, smiles, stats, metadata)
    data.completed_epochs = first_epochs
    resumed = trainer(cfg, callback, first_epochs+1, smoke, accumulation)
    resumed.fit(model, datamodule=data, ckpt_path=str(checkpoint))
    assert resumed.global_step > expected_step
    assert model.batch1_completed_epochs == first_epochs+1
    if smoke:
        samples, _, _, _ = generate(model, 4, 4, 71000)
        evaluate_h(samples, infos.atom_decoder, canonical_reference(smiles['train']))
        log.finish({'status':'completed'})
        return None
    return model, log, rows, dict(first_fit_wall_seconds=first_fit_seconds,
        checkpoint_save_seconds=save_seconds, checkpoint_load_seconds=load_seconds,
        checkpoint_bytes=checkpoint.stat().st_size, resumed_epoch=first_epochs+1,
        resumed_optimizer_step=resumed.global_step)


def prepare(root):
    cfg, data, infos, smiles, stats, metadata = load('B0', create=True)
    audit = dict(decoder=infos.atom_decoder, max_nodes=infos.max_n_nodes, statistics=metadata, splits={})
    assert infos.atom_decoder == ['H','C','N','O','F'] and infos.max_n_nodes == 29
    assert cfg.model.n_layers == 9 and cfg.sample.sample_steps == 500
    for split, dataset, filename in [('train',data.train_dataset,'proc_tr_no_h.pt'),
            ('val',data.val_dataset,'proc_val_no_h.pt'),('test',data.test_dataset,'proc_test_no_h.pt')]:
        previous = torch.load(Path(cfg.dataset.datadir)/'processed'/filename, weights_only=False)[0]
        ids = dataset._data.idx
        assert torch.equal(ids, previous.idx), 'Molecule split changed: '+split
        audit['splits'][split] = dict(count=len(dataset), molecule_ids_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest())
    # Build original explicit-H graph targets, then apply exactly our strict audit.
    audit_samples=[]
    for graph in data.train_dataset:
        edges=torch.zeros(graph.num_nodes,graph.num_nodes,dtype=torch.long)
        edges[graph.edge_index[0],graph.edge_index[1]]=graph.edge_attr.argmax(-1)
        audit_samples.append((graph.x.argmax(-1),edges))
    reference=canonical_reference(smiles['train'])
    audit['training_targets']=evaluate_h(audit_samples, infos.atom_decoder, reference)
    atomic_json(root/'data-audit.json', audit)
    del audit_samples, data, infos, smiles, stats
    gc.collect()
    for variant in ['B0','M1']:
        cfg, data, infos, smiles, stats, metadata = load(variant, smoke=True)
        fit_resume(cfg,data,infos,smiles,stats,metadata,root/'cpu-preflight'/variant,smoke=True)
        del data, infos, smiles, stats
        gc.collect()
    atomic_json(root/'cpu-preflight/READY.json',dict(status='passed',variants=['B0','M1']))
    (root/'READY').write_text('explicit-H splits/audit and B0/M1 train/sample/resume passed\n')


def probe(root, variant, kind, size):
    directory=root/'profiles'/variant
    directory.mkdir(parents=True,exist_ok=True)
    output=directory/f'probe-{kind}-{size}.json'
    cfg,data,infos,smiles,stats,metadata=load(variant)
    model=initialize(cfg,data,infos,smiles,stats,metadata).cuda()
    row=dict(kind=kind,batch_size=size,status='ok',max_nodes=29)
    try:
        torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter()
        if kind=='train':
            sizes=data.train_dataset.slices['x'][1:]-data.train_dataset.slices['x'][:-1]
            index=int(torch.argmax(sizes))
            assert int(sizes[index])==29
            batch=Batch.from_data_list([data.train_dataset[index]]*size).cuda()
            optimizer=model.configure_optimizers()
            # Include initialized AMSGrad optimizer state and second-step activations.
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss=model.training_step(batch,1)['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.train.clip_grad)
                optimizer.step()
        else:
            model.eval()
            model.cfg.sample.sample_steps=2
            with torch.no_grad():
                model.sample_conditioned(torch.full((size,),29,device='cuda',dtype=torch.long))
        torch.cuda.synchronize()
        row.update(seconds=time.perf_counter()-started,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
    except torch.cuda.OutOfMemoryError:
        row['status']='oom'
    atomic_json(output,row)
    print(json.dumps(row),flush=True)


def select(root):
    results={}
    for variant in ['B0','M1']:
        results[variant]={}
        for kind in ['train','sample']:
            rows=[json.loads(p.read_text()) for p in (root/'profiles'/variant).glob(f'probe-{kind}-*.json')]
            safe=[r['batch_size'] for r in rows if r['status']=='ok' and
                  r['peak_gpu_reserved_bytes'] <= .8*r['total_gpu_bytes']]
            if not safe:
                raise RuntimeError(f'No safe {kind} batch for {variant}; inspect probes')
            results[variant][kind]=max(safe)
    common=min(r['train'] for r in results.values())
    atomic_json(root/'selected-batches.json',dict(physical_batch_size=common,
        effective_batch_size=1024,gradient_accumulation=1024//common,
        inference_batches={v:r['sample'] for v,r in results.items()},memory_ceiling_fraction=.8))


def profile(root,variant):
    directory=root/'profiles'/variant
    selected=json.loads((root/'selected-batches.json').read_text())
    cfg,data,infos,smiles,stats,metadata=load(variant)
    cfg.train.batch_size=selected['physical_batch_size']
    cfg.batch1.inference_batch_size=selected['inference_batches'][variant]
    model,log,rows,io=fit_resume(cfg,data,infos,smiles,stats,metadata,directory,
                                 accumulation=selected['gradient_accumulation'])
    reference=canonical_reference(smiles['train'])
    pools=[]
    for count in [2000,10000]:
        started=time.perf_counter()
        samples,initials,traces,seconds=generate(model,count,cfg.batch1.inference_batch_size,81000+count)
        pool=directory/'samples'/str(count)
        pool.mkdir(parents=True,exist_ok=True)
        serialization=time.perf_counter()
        with (pool/'graphs.pkl').open('wb') as handle:
            pickle.dump(dict(samples=samples,initial_atoms=initials),handle)
        serialization=time.perf_counter()-serialization
        result=evaluate_h(samples,infos.atom_decoder,reference)
        result.update(completed_epoch=8,count=count,generation_seconds=seconds,
                      serialization_seconds=serialization,sampling_steps=500,
                      inference_batch_cap=cfg.batch1.inference_batch_size,
                      note='Resource profile after eight epochs; not a model-quality comparison.')
        result['legacy']=legacy_evaluate(samples,infos,smiles['train'],cfg,pool/'legacy')
        result['pool_wall_seconds']=time.perf_counter()-started
        atomic_json(pool/'result.json',result)
        log.write('curve',**{k:v for k,v in result.items() if not isinstance(v,dict)})
        pools.append(result)
        del samples,initials,traces
    epoch_seconds=statistics.median(r['seconds'] for r in rows if not r['warmup'] and not r['resume_check'])
    # Cost the inherited evaluation schedule, including its larger late pools.
    schedule=list(cfg.batch1.evaluation_epochs)
    def projection(epochs):
        curve_count=sum(e<=epochs for e in schedule)
        curve_seconds=sum(pools[1 if e>=cfg.batch1.late_curve_from else 0]['pool_wall_seconds']
                          for e in schedule if e<=epochs)
        return dict(training_hours=epoch_seconds*epochs/3600,
                    training_plus_curve_and_final_hours=(epoch_seconds*epochs+
                        curve_seconds+3*pools[1]['pool_wall_seconds'])/3600,
                    curve_pools=curve_count,final_10k_pools=3,
                    note='Estimate: 2k curves, 10k curves from epoch 400, three final 10k pools; excludes validation CE, startup and checkpoint overhead.')
    summary=dict(status='completed',variant=variant,completed_epochs=8,git_commit=revision(),
        selected_batches=selected,median_epoch_seconds=epoch_seconds,checkpoint=io,pools=pools,
        projections={str(e):projection(e) for e in [200,1000]},gpu=torch.cuda.get_device_name(),
        note='Explicit-H profiling only; effective batch 1024 via accumulation when needed.')
    atomic_json(directory/'summary.json',summary)
    if log.wandb:
        log.wandb.log({'profile/median_epoch_seconds':epoch_seconds,
                       'profile/physical_batch_size':cfg.train.batch_size,
                       'profile/checkpoint_bytes':io['checkpoint_bytes']})
    log.finish({'status':'completed','profiling':True})
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','probe','select','profile'])
    parser.add_argument('--variant',choices=['B0','M1'])
    parser.add_argument('--kind',choices=['train','sample'])
    parser.add_argument('--batch-size',type=int)
    args=parser.parse_args()
    root=Path(os.environ['BATCH1_ROOT']).resolve()
    os.chdir(root)
    RDLogger.DisableLog('rdApp.*')
    torch.set_num_threads(4)
    pl.seed_everything(0,workers=True)
    if args.mode=='prepare': prepare(root)
    elif args.mode=='select': select(root)
    elif args.mode=='probe': probe(root,args.variant,args.kind,args.batch_size)
    else: profile(root,args.variant)
