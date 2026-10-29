"""Prepare, preflight, or execute one configuration. Never submits Slurm jobs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rdkit import RDLogger

from experiments.batch1.controller import RunController
from experiments.batch1.checks import smoke_invariants
from experiments.batch1.data import components, statistics_for, build_model
from experiments.batch1.metrics import canonical_reference
from experiments.batch1.statistics import sha256
from experiments.batch1.storage import atomic_json, RunLog

ROOT = Path(__file__).resolve().parents[3]
CONFIGURATIONS = ["B0", "B1", "A0", "A1", "M0", "M1"]


def make_config(args, variant):
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT/"configs")):
        cfg = compose(config_name=getattr(args, "protocol", "batch1"), overrides=[f"augmentation={variant}"])
    cfg.dataset.datadir = str(Path(args.data_dir).resolve())
    cfg.train.seed = args.seed
    if args.smoke:
        cfg.train.n_epochs = 2
        cfg.train.batch_size = 4
        cfg.train.num_workers = 0
        cfg.sample.sample_steps = 2
        cfg.batch1.evaluation_epochs = [0, 1, 2]
        cfg.batch1.curve_samples = 4
        cfg.batch1.final_samples = 4
        cfg.batch1.final_repeats = 1
        cfg.batch1.inference_batch_size = 4
        cfg.batch1.validation_samples = 4
        cfg.batch1.validation_batch_size = 4
        cfg.batch1.validation_times = [0.5]
        cfg.batch1.diagnostic_samples = 4
        cfg.batch1.wandb_mode = "disabled"
        cfg.batch1.recovery_every = 1
        if "late_curve_samples" in cfg.batch1:
            cfg.batch1.late_curve_from = 1
            cfg.batch1.late_curve_samples = 6
    if args.wandb_mode:
        cfg.batch1.wandb_mode = args.wandb_mode
    return cfg


def revision():
    marker = ROOT/"SOURCE_COMMIT"
    if marker.exists():
        return marker.read_text().strip()
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


def signature(cfg, stats_metadata):
    # Paths, runtime logging and time-limited continuations are not model semantics.
    content = dict(augmentation=OmegaConf.to_container(cfg.augmentation),
                   model=OmegaConf.to_container(cfg.model),
                   train=OmegaConf.to_container(cfg.train),
                   sample=OmegaConf.to_container(cfg.sample),
                   protocol={k: v for k, v in OmegaConf.to_container(cfg.batch1).items()
                             if not k.startswith("wandb_")},
                   statistics_sha256=stats_metadata["sha256"])
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def train(args, variant, root, resume=None, stop_after=None):
    cfg = make_config(args, variant)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Original RRWP utilities emit timer.dat in the current directory. Keep
    # that and any legacy evaluator artifacts with this run, outside Git.
    os.chdir(root)
    if (root/"manifest.json").exists() and resume is None:
        raise FileExistsError(f"Existing run {root}; use --resume or a new run ID")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    if args.device == "cuda" and "A100" not in torch.cuda.get_device_name():
        raise RuntimeError("Batch 1 GPU budgets are for a full A100")
    pl.seed_everything(args.seed, workers=True)
    torch.set_num_threads(args.threads)
    data, infos, smiles = components(cfg)
    stats, metadata = statistics_for(cfg, data, infos, args.cache_dir)
    model = build_model(cfg, data, infos, smiles, stats)
    if args.smoke:
        smoke_invariants(model, data)
    model.batch1_signature = signature(cfg, metadata)
    reference = canonical_reference(smiles["train"])
    generator = torch.Generator().manual_seed(cfg.batch1.validation_seed)
    indices = torch.randperm(len(data.val_dataset), generator=generator)[:cfg.batch1.validation_samples].tolist()
    validation_hash = hashlib.sha256(json.dumps(indices).encode()).hexdigest()
    manifest = dict(schema=1, logical_run_id=root.name, configuration=variant, training_seed=args.seed,
                    git_commit=revision(), statistics=metadata, model_signature=model.batch1_signature,
                    lock_sha256=sha256(ROOT/"pixi.lock"), validation_indices_sha256=validation_hash,
                    canonical_training_reference_sha256=hashlib.sha256("\n".join(sorted(reference)).encode()).hexdigest(),
                    canonical_training_reference_count=len(reference),
                    validation_indices=indices, runtime_device=args.device,
                    gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                    parameters=sum(p.numel() for p in model.parameters()),
                    versions={x: importlib.metadata.version(x) for x in
                              ["torch", "numpy", "rdkit", "pytorch-lightning", "torch-geometric", "wandb"]},
                    note="Unused full-training-set molecular reference ratios omitted; original model/prior retained for B0.")
    model.batch1_provenance = manifest
    checkpoint_path = None
    if resume:
        if resume == "latest":
            index = json.loads((root/"checkpoint-index.json").read_text())
            checkpoint_path = root/"checkpoints"/index["latest"]
            expected = index.get("sha256", {}).get(index["latest"])
            if expected is None or sha256(checkpoint_path) != expected:
                raise ValueError("Latest recovery checkpoint checksum mismatch")
        else:
            checkpoint_path = Path(resume).resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["batch1"]["signature"] != model.batch1_signature:
            raise ValueError("Resume configuration/statistics differ from checkpoint")
        data.completed_epochs = checkpoint["batch1"]["completed_epochs"]
        del checkpoint
    if (root/"manifest.json").exists():
        original = json.loads((root/"manifest.json").read_text())
        if original["model_signature"] != manifest["model_signature"]:
            raise ValueError("Existing run manifest mismatch")
    else:
        atomic_json(root/"manifest.json", manifest)
        OmegaConf.save(cfg, root/"resolved-config.yaml")
    log = RunLog(root, cfg, manifest)
    atomic_json(root/"attempts"/f"{log.attempt}.json", {**manifest, "resume": str(checkpoint_path) if resume else None})
    archive = Path(args.archive_root)/root.name if args.archive_root else None
    controller = RunController(cfg, root, log, infos, smiles, reference, indices, archive)
    controller.data = data
    controller.resume_checkpoint = checkpoint_path
    controller.evaluation_device = args.device
    max_epochs = min(stop_after or cfg.train.n_epochs, cfg.train.n_epochs)
    trainer = pl.Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu", devices=1,
        strategy="ddp_find_unused_parameters_true" if args.device == "cuda" else "auto",
        precision="32-true", max_epochs=max_epochs, callbacks=[controller], logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        gradient_clip_val=cfg.train.clip_grad, num_sanity_val_steps=0, limit_val_batches=0,
        reload_dataloaders_every_n_epochs=1,
        limit_train_batches=1 if args.smoke else 1.0)
    try:
        trainer.fit(model, datamodule=data, ckpt_path=str(checkpoint_path) if checkpoint_path else None)
        summary = json.loads((root/"metrics"/"summary.json").read_text())
        log.finish({"status": summary["status"]})
        return summary
    except BaseException:
        atomic_json(root/"metrics"/"failure.json", dict(status="failed", attempt=log.attempt,
                    unix_time=time.time(), traceback=traceback.format_exc()))
        log.finish({"status": "failed"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "preflight", "train"])
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="B0")
    parser.add_argument("--protocol", choices=["batch1", "batch2"], default="batch1")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--run-dir")
    parser.add_argument("--archive-root")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", help="latest or an explicit checkpoint")
    parser.add_argument("--stop-after-epochs", type=int)
    parser.add_argument("--wandb-mode", choices=["offline", "disabled"])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    for name in ["data_dir", "cache_dir", "run_dir", "archive_root"]:
        value = getattr(args, name)
        if value:
            setattr(args, name, str(Path(value).resolve()))
    RDLogger.DisableLog("rdApp.*")
    if args.mode == "prepare":
        cfg = make_config(args, "B0")
        data, infos, _ = components(cfg)
        _, metadata = statistics_for(cfg, data, infos, args.cache_dir, create=True)
        print(json.dumps(metadata, indent=2))
    elif args.mode == "preflight":
        if not args.run_dir:
            parser.error("--run-dir is required")
        args.smoke, args.device = True, "cpu"
        args.wandb_mode = "disabled"
        variants = ["B0", "A0", "M0", "M1"] if args.protocol == "batch2" else CONFIGURATIONS
        for variant in variants:
            root = Path(args.run_dir)/variant
            first = train(args, variant, root, stop_after=1)
            assert first["status"] == "paused" and first["completed_epochs"] == 1
            second = train(args, variant, root, resume="latest")
            assert second["status"] == "completed" and second["completed_epochs"] == 2
            assert len(second["final_results"]) >= 1
            if args.protocol == "batch2":
                assert set(second["checkpoint_roles"]) == {"final"}
                assert second["best_epoch"] is None
            index = json.loads((root/"checkpoint-index.json").read_text())
            checkpoint = torch.load(root/"checkpoints"/index["final"], map_location="cpu", weights_only=False)
            assert checkpoint["global_step"] == 2
            assert checkpoint["batch1"]["completed_epochs"] == 2
            assert "rng" in checkpoint["batch1"]
            curves = [json.loads(line) for line in (root/"metrics"/"curve.jsonl").read_text().splitlines()]
            assert [row["completed_epoch"] for row in curves] == [0, 1, 2]
            if args.protocol == "batch2":
                assert all(row["attempts"] == 6 and row["comparison_attempts"] == 4
                           for row in curves[1:])
            if variant in ["A0", "A1"]:
                assert all(row["atom_change_fraction"] == 0 for row in curves)
                assert all(row["composition_change_fraction"] == 0 for row in curves)
            for result in second["final_results"].values():
                assert all(row["completed_epoch"] == result["completed_epoch"] for row in result["repeats"])
            del checkpoint
        atomic_json(Path(args.run_dir)/"READY.json", dict(status="passed", configurations=variants, protocol=args.protocol))
    else:
        if not args.run_dir:
            parser.error("--run-dir is required")
        if args.device == "cpu" and not args.smoke:
            parser.error("Full experiments require CUDA; CPU is limited to explicit smoke tests")
        summary = train(args, args.configuration, args.run_dir,
                        resume=args.resume, stop_after=args.stop_after_epochs)
        if summary["status"] != "completed":
            raise SystemExit(75)


if __name__ == "__main__":
    main()
