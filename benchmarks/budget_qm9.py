"""Measure the unmodified QM9 model; no prior/architecture changes.

Uses an isolated source snapshot and direct molecular setup equivalent to main.py.
Unused full-training-set reference-metric evaluation is intentionally omitted:
it is not used by optimization, the sampler, or the strict metric below. Cold
dataset preparation and cached startup are measured separately.
"""
import argparse
import contextlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import resource
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
IMPORT_START = time.perf_counter()
import numpy as np
import torch
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import Chem, RDLogger

from datasets.qm9_dataset import QM9DataModule, QM9infos, get_smiles
from graph_discrete_flow_model import GraphDiscreteFlowModel
from metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from models.extra_features import ExtraFeatures
from models.extra_features_molecular import ExtraMolecularFeatures
from analysis.rdkit_functions import BasicMolecularMetrics, build_molecule
IMPORT_SECONDS = time.perf_counter() - IMPORT_START


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def emit(output, event, **fields):
    record = {"event": event, "unix_time": time.time(), **fields}
    with (output / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(record, default=str, allow_nan=False) + "\n")
    print("BENCHMARK " + json.dumps(record, default=str), flush=True)
    return record


def config(data_dir, workers):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[
            "+experiment=qm9_no_h", "dataset=qm9", "general.name=budget_qm9",
            "general.wandb=disabled", "general.gpus=1", "train.batch_size=1024",
            f"train.num_workers={workers}", f"dataset.datadir={data_dir}",
            "sample.sample_steps=500", "sample.time_distortion=polydec",
            "sample.eta=0", "sample.omega=0", "sample.search=False",
            "general.chains_to_save=0", "general.samples_to_save=0",
            "general.final_model_chains_to_save=0", "general.final_model_samples_to_save=0",
        ])
    return cfg


def data_setup(cfg):
    data = QM9DataModule(cfg)
    infos = QM9infos(data, cfg)
    smiles = get_smiles(cfg, data, infos, evaluate_datasets=False)
    return data, infos, smiles


def build(cfg, output, tag):
    start = time.perf_counter()
    data, infos, smiles = data_setup(cfg)
    extra = ExtraFeatures(cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=infos)
    molecular = ExtraMolecularFeatures(infos)
    infos.compute_input_output_dims(data, extra, molecular)
    infos.ref_metrics = {"val": {}, "test": {}}
    model = GraphDiscreteFlowModel(
        cfg=cfg, dataset_infos=infos,
        train_metrics=TrainMolecularMetricsDiscrete(infos),
        sampling_metrics=SamplingMolecularMetrics(infos, smiles, cfg),
        visualization_tools=None, extra_features=extra,
        domain_features=molecular, test_labels=None,
    )
    emit(output, "cached_model_setup", tag=tag, seconds=time.perf_counter()-start,
         train_graphs=len(data.train_dataset), parameters=sum(p.numel() for p in model.parameters()))
    return model, data, infos, smiles


def strict_metrics(samples, decoder, reference):
    """Whole, connected, sanitized molecules; no charge/fragment correction."""
    accepted = []
    invalid = disconnected = 0
    start = time.perf_counter()
    for atoms, edges in samples:
        try:
            molecule = build_molecule(atoms, edges, decoder)
            Chem.SanitizeMol(molecule)
            if len(Chem.GetMolFrags(molecule)) != 1:
                disconnected += 1
                continue
            accepted.append(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
        except (ValueError, RuntimeError):
            invalid += 1
    chemistry_seconds = time.perf_counter() - start
    start = time.perf_counter()
    unique = set(accepted)
    unique_seconds = time.perf_counter() - start
    start = time.perf_counter()
    novel = unique - reference
    novelty_seconds = time.perf_counter() - start
    total, valid_count, unique_count, novel_count = len(samples), len(accepted), len(unique), len(novel)
    return dict(attempts=total, valid_count=valid_count, unique_count=unique_count,
                novel_count=novel_count, invalid_count=invalid, disconnected_count=disconnected,
                validity=valid_count/total if total else 0,
                uniqueness=unique_count/valid_count if valid_count else 0,
                novelty=novel_count/unique_count if unique_count else None,
                vun=novel_count/total if total else 0,
                chemistry_seconds=chemistry_seconds, uniqueness_seconds=unique_seconds,
                novelty_seconds=novelty_seconds)


def metric_self_test():
    def sample(n, pairs):
        edges = torch.zeros(n, n, dtype=torch.long)
        for i, j in pairs:
            edges[i, j] = edges[j, i] = 1
        return [torch.zeros(n, dtype=torch.long), edges]
    samples = [sample(1, []), sample(1, []), sample(2, [(0, 1)]),
               sample(2, []), sample(6, [(0, i) for i in range(1, 6)])]
    result = strict_metrics(samples, ["C"], {"C"})
    assert (result["valid_count"], result["unique_count"], result["novel_count"]) == (3, 2, 1), result
    assert abs(result["vun"] - 0.2) < 1e-12
    assert abs(result["validity"]*result["uniqueness"]*result["novelty"]-result["vun"]) < 1e-12


def preflight(args, output):
    """Exercise model/framework compatibility before reserving a GPU.

    Only this smoke check uses a four-molecule batch and two sampling steps.
    The GPU benchmark builds fresh models with the full reference settings.
    """
    torch.set_num_threads(4)
    pl.seed_everything(args.seed, workers=True)
    cfg = config(args.data_dir, 0)
    cfg.train.batch_size = 4
    cfg.sample.sample_steps = 2
    model, data, infos, smiles = build(cfg, output, "cpu_preflight")
    trainer = pl.Trainer(
        accelerator="cpu", devices=1, precision="32-true", max_epochs=1,
        limit_train_batches=1, limit_val_batches=0, num_sanity_val_steps=0,
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data)
    checkpoint = output / "preflight.ckpt"
    trainer.save_checkpoint(str(checkpoint))
    restored, _, _, _ = build(cfg, output, "cpu_preflight_reload")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    restored.load_state_dict(state["state_dict"])
    restored.eval()
    with torch.no_grad():
        samples, _ = restored.sample_batch(
            batch_id=0, batch_size=4, keep_chain=0, number_chain_steps=1,
            save_final=0, num_nodes=4, save_visualization=False,
        )
    assert len(samples) == 4
    metrics = strict_metrics(samples, infos.atom_decoder, set(smiles["train"]))
    emit(output, "preflight_passed", optimizer_steps=trainer.global_step,
         generated=len(samples), checkpoint_bytes=checkpoint.stat().st_size,
         smoke_only_batch_size=4, smoke_only_sample_steps=2, metrics=metrics)
    (output / "READY").write_text("CPU train/sample/checkpoint preflight passed\n")


class EpochTimer(pl.Callback):
    def __init__(self, output, tag, integrated=None):
        self.output, self.tag, self.integrated = output, tag, integrated
        self.rows = []

    def on_train_epoch_start(self, trainer, model):
        sync()
        torch.cuda.reset_peak_memory_stats()
        self.start = time.perf_counter()
        self.examples = self.batches = 0

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.examples += batch.num_graphs
        self.batches += 1

    def on_train_epoch_end(self, trainer, model):
        sync()
        elapsed = time.perf_counter() - self.start
        row = emit(self.output, "train_epoch", tag=self.tag,
                   completed_epoch=trainer.current_epoch+1, global_step=trainer.global_step,
                   seconds=elapsed, examples=self.examples, batches=self.batches,
                   examples_per_second=self.examples/elapsed,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                   host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        self.rows.append(row)
        if self.integrated:
            # Preserve RNG state so inserted evaluation does not change future training noise.
            py_state, np_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    self.integrated(model, trainer.current_epoch+1)
            finally:
                random.setstate(py_state)
                np.random.set_state(np_state)
                model.train()
            start = time.perf_counter()
            trainer.save_checkpoint(str(self.output / f"integrated_epoch{trainer.current_epoch+1}.ckpt"))
            emit(self.output, "integrated_checkpoint", seconds=time.perf_counter()-start,
                 completed_epoch=trainer.current_epoch+1)


def trainer_for(cfg, epochs, callback):
    return pl.Trainer(
        accelerator="gpu", devices=1, precision="32-true",
        strategy="ddp_find_unused_parameters_true", max_epochs=epochs,
        gradient_clip_val=cfg.train.clip_grad, callbacks=[callback], logger=False,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=0, limit_val_batches=0,
        log_every_n_steps=50, deterministic=False,
    )


def train_profile(args, output, workers):
    tag = f"workers{workers}"
    pl.seed_everything(args.seed, workers=True)
    cfg = config(args.data_dir, workers)
    model, data, infos, smiles = build(cfg, output, tag)
    OmegaConf.save(cfg, output / f"resolved_{tag}.yaml")
    callback = EpochTimer(output, tag)
    trainer = trainer_for(cfg, args.epochs, callback)
    start = time.perf_counter()
    trainer.fit(model, datamodule=data)
    sync()
    fit_seconds = time.perf_counter()-start
    checkpoint = output / f"{tag}.ckpt"
    start = time.perf_counter()
    trainer.save_checkpoint(str(checkpoint))
    emit(output, "checkpoint_save", tag=tag, seconds=time.perf_counter()-start,
         bytes=checkpoint.stat().st_size, path=str(checkpoint))
    steady = [row["seconds"] for row in callback.rows[2:]]
    result = emit(output, "train_summary", tag=tag, fit_seconds=fit_seconds,
                  epochs=len(callback.rows), warmup_epochs=2,
                  steady_epoch_median_seconds=statistics.median(steady),
                  steady_epoch_p90_seconds=float(np.percentile(steady, 90)),
                  checkpoint=str(checkpoint))
    model.cpu()
    del trainer, model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    return result


def generate(model, count, batch_size, seed):
    pl.seed_everything(seed, workers=True)
    model.eval()
    model.to("cuda")
    # Reuse identical graph-size draws for comparisons at different inference batches.
    sizes = model.node_dist.sample_n(count, model.device)
    sync()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    samples = []
    with torch.no_grad():
        for offset in range(0, count, batch_size):
            n = min(batch_size, count-offset)
            generated, _ = model.sample_batch(
                batch_id=offset, batch_size=n, keep_chain=0, number_chain_steps=1,
                save_final=0, num_nodes=sizes[offset:offset+n], save_visualization=False,
            )
            samples.extend(generated)
    sync()
    return samples, time.perf_counter()-start, torch.cuda.max_memory_allocated()


def evaluate_case(model, infos, smiles, output, tag, count, batch_size, seed, legacy=False):
    directory = output / tag
    directory.mkdir(exist_ok=True)
    samples, seconds, peak = generate(model, count, batch_size, seed)
    emit(output, "generation", tag=tag, attempts=count, batch_size=batch_size,
         seed=seed, steps=500, seconds=seconds, molecules_per_second=count/seconds,
         peak_gpu_allocated_bytes=peak, checkpoint_age=model.current_epoch)
    start = time.perf_counter()
    with (directory / "samples.pkl").open("wb") as handle:
        pickle.dump(samples, handle)
    emit(output, "serialization", tag=tag, seconds=time.perf_counter()-start,
         bytes=(directory / "samples.pkl").stat().st_size)
    start = time.perf_counter()
    reference = set(smiles["train"])
    emit(output, "reference_set", tag=tag, seconds=time.perf_counter()-start, count=len(reference))
    result = strict_metrics(samples, infos.atom_decoder, reference)
    emit(output, "strict_metrics", tag=tag, **result)
    (directory / "strict_metrics.json").write_text(json.dumps(result, indent=2))
    if legacy:
        start = time.perf_counter()
        cwd = Path.cwd()
        try:
            os.chdir(directory)
            with Path("legacy_stdout.log").open("w") as log, contextlib.redirect_stdout(log):
                evaluator = BasicMolecularMetrics(infos, smiles["train"], model.cfg)
                results = evaluator.evaluate(samples, test=False)
            values = results[0]
            emit(output, "legacy_metrics", tag=tag, seconds=time.perf_counter()-start,
                 validity=values[0], relaxed_validity=values[1],
                 uniqueness=values[2], novelty=values[3],
                 relaxed_vun=values[1]*values[2]*values[3] if values[3] >= 0 else None)
        finally:
            os.chdir(cwd)
    return seconds


def run(args, output):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required; refusing a CPU fallback")
    gpu = torch.cuda.get_device_name()
    if "A100" not in gpu:
        raise RuntimeError(f"Expected a full A100, received {gpu}")
    torch.set_num_threads(4)
    metric_self_test()
    emit(output, "environment", gpu=gpu, gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
         import_seconds=IMPORT_SECONDS, torch=torch.__version__, cuda=torch.version.cuda,
         packages={name: importlib.metadata.version(name) for name in
                   ["numpy", "rdkit", "pytorch-lightning", "torch-geometric", "torchmetrics"]},
         notes=["model/prior unchanged", "unused full-dataset reference metrics omitted",
                "strict connected metrics added; legacy metrics separately labeled",
                "epoch timer ends before module epoch-end logging"])
    results = []
    for workers in [0, 4]:
        results.append(train_profile(args, output, workers))
    selected = min(results, key=lambda row: row["steady_epoch_median_seconds"])
    workers = int(selected["tag"].removeprefix("workers"))
    cfg = config(args.data_dir, workers)
    model, data, infos, smiles = build(cfg, output, "evaluation")
    start = time.perf_counter()
    state = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"])
    emit(output, "checkpoint_load", seconds=time.perf_counter()-start,
         path=selected["checkpoint"], completed_epochs=args.epochs)
    evaluate_case(model, infos, smiles, output, "warmup128", 128, 128, args.seed+100)
    t2k = evaluate_case(model, infos, smiles, output, "eval2000_rep0", 2000, 2048,
                       args.seed+101, legacy=True)
    t10k = evaluate_case(model, infos, smiles, output, "eval10000", 10000, 2048,
                        args.seed+102, legacy=True)
    for repeat in [1, 2]:
        evaluate_case(model, infos, smiles, output, f"eval2000_rep{repeat}",
                      2000, 2048, args.seed+101+repeat)
    for batch in [256, 512, 1024]:
        evaluate_case(model, infos, smiles, output, f"batch{batch}",
                      2000, batch, args.seed+101)
    model.cpu()
    del model, state
    torch.cuda.empty_cache()
    resumed, data, infos, smiles = build(cfg, output, "integrated_resume")
    def integrated(model, epoch):
        evaluate_case(model, infos, smiles, output, f"integrated{epoch}",
                      2000, 2048, args.seed+200+epoch)
    callback = EpochTimer(output, "integrated_resume", integrated)
    trainer = trainer_for(cfg, args.epochs+2, callback)
    start = time.perf_counter()
    trainer.fit(resumed, datamodule=data, ckpt_path=selected["checkpoint"])
    sync()
    emit(output, "integrated_summary", seconds=time.perf_counter()-start,
         epochs=len(callback.rows), resumed_from=selected["checkpoint"])
    if len(callback.rows) != 2:
        raise RuntimeError("Checkpoint resume did not run exactly two additional epochs")
    summary = dict(status="completed", training=results, selected_workers=workers,
                   generation2000_seconds=t2k, generation10000_seconds=t10k,
                   training_only_1000_epochs_gpu_hours=
                   1000*selected["steady_epoch_median_seconds"]/3600,
                   note="Add startup, CPU metrics, 12 curve evaluations, final repeats and I/O to budget.")
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    emit(output, "completed", **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "preflight", "run", "self-test"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    RDLogger.DisableLog("rdApp.*")
    start = time.perf_counter()
    try:
        if args.mode == "self-test":
            metric_self_test()
            emit(output, "self_test_passed")
        elif args.mode == "preflight":
            preflight(args, output)
        elif args.mode == "prepare":
            pl.seed_everything(args.seed, workers=True)
            data, infos, smiles = data_setup(config(args.data_dir, 0))
            metric_self_test()
            emit(output, "data_prepared", seconds=time.perf_counter()-start,
                 import_seconds=IMPORT_SECONDS, train=len(data.train_dataset),
                 val=len(data.val_dataset), test=len(data.test_dataset),
                 train_smiles=len(smiles["train"]))
            (output / "READY").write_text("QM9 prepared and strict-metric self-test passed\n")
        else:
            run(args, output)
    except Exception:
        emit(output, "failed", seconds=time.perf_counter()-start, traceback=traceback.format_exc())
        (output / "FAILED").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
