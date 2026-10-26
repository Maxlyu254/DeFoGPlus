"""Epoch evaluations, resumable retention, final reporting, and local backups."""
import json
import os
from pathlib import Path
import resource
import signal
import statistics
import time

import pytorch_lightning as pl
import torch

from experiments.batch1.evaluation import evaluate_pool, validation_loss, synchronize
from experiments.batch1.storage import atomic_json, backup_selected, export_csv, restore_rng
from experiments.batch1.statistics import sha256


class RunController(pl.Callback):
    def __init__(self, cfg, root, log, infos, smiles, reference, indices, archive=None):
        self.cfg, self.root, self.run_log = cfg, Path(root), log
        self.infos, self.smiles, self.reference = infos, smiles, reference
        self.indices, self.archive = indices, archive
        self.completed = 0
        self.best_epoch, self.best_vun = None, -1
        self.index = dict(snapshots={}, latest=None, previous=None, best=None, final=None)
        self.train_rows, self.curve_rows = [], []
        self.stop_requested = False

    def state_dict(self):
        return dict(completed=self.completed, best_epoch=self.best_epoch, best_vun=self.best_vun,
                    index=self.index, train_rows=self.train_rows, curve_rows=self.curve_rows)

    def load_state_dict(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def on_fit_start(self, trainer, model):
        def stop(signum, frame):
            self.stop_requested = True
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, stop)
        signal.signal(signal.SIGTERM, stop)

    def on_train_start(self, trainer, model):
        if model.batch1_loaded:
            # The checksum is intentionally external to the serialized file,
            # so the callback state inside it cannot contain its own hash.
            path = self.resume_checkpoint
            self.index.setdefault("sha256", {})[path.name] = sha256(path)
            restore_rng(model.batch1_loaded["rng"])
            self.run_log.write("events", event="resume", completed_epoch=self.completed)
        else:
            self.evaluate_epoch(model, 0)
            self.save(trainer, model, 0, permanent=True)
        if trainer.current_epoch != self.completed:
            raise RuntimeError(f"Resume epoch mismatch: framework={trainer.current_epoch}, recorded={self.completed}")

    def on_train_epoch_start(self, trainer, model):
        synchronize(model)
        self.started = time.perf_counter()
        self.examples = self.batches = 0
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.examples += batch.num_graphs
        self.batches += 1

    def evaluate_epoch(self, model, epoch):
        losses = validation_loss(model, self.data, self.indices, self.cfg.batch1.validation_seed)
        self.run_log.write("validation", completed_epoch=epoch, **losses)
        result = evaluate_pool(
            model, self.infos, self.smiles["train"], self.reference, self.root,
            f"curve-epoch{epoch:04d}-{self.run_log.attempt}", self.cfg.batch1.curve_samples,
            self.cfg.batch1.curve_seed, epoch)
        self.curve_rows.append(self.run_log.write("curve", **result))
        if epoch>0 and result["vun"]>self.best_vun:
            self.best_epoch, self.best_vun = epoch, result["vun"]
        export_csv(self.root/"metrics"/"curve.csv", self.curve_rows)

    def on_train_epoch_end(self, trainer, model):
        synchronize(model)
        elapsed = time.perf_counter()-self.started
        self.completed = trainer.current_epoch+1
        model.batch1_completed_epochs = self.completed
        self.data.completed_epochs = self.completed
        losses = model.train_loss.log_epoch_metrics()
        def scalar(value):
            v = float(value)
            return v if torch.isfinite(torch.tensor(v)) and v>=0 else None
        row = dict(completed_epoch=self.completed, optimizer_step=trainer.global_step,
                   seconds=elapsed, graphs=self.examples, batches=self.batches,
                   graphs_per_second=self.examples/elapsed, learning_rate=trainer.optimizers[0].param_groups[0]["lr"],
                   node_ce=None if model.atoms_mode == "fixed" else scalar(losses["train_epoch/x_CE"]),
                   edge_ce=scalar(losses["train_epoch/E_CE"]),
                   host_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                   peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if model.device.type == "cuda" else 0)
        row["cumulative_training_seconds"] = sum(r["seconds"] for r in self.train_rows)+elapsed
        row["allocated_job_gpu_hours"] = (
            (time.time()-float(os.environ.get("BATCH1_JOB_START_UNIX", time.time())))/3600
            if model.device.type == "cuda" else 0)
        self.train_rows.append(self.run_log.write("train", **row))
        if self.completed in self.cfg.batch1.evaluation_epochs:
            self.evaluate_epoch(model, self.completed)
        if (self.completed in self.cfg.batch1.evaluation_epochs or
                self.completed % self.cfg.batch1.recovery_every == 0 or self.stop_requested or
                self.completed == self.cfg.train.n_epochs):
            self.save(trainer, model, self.completed,
                      permanent=self.completed in self.cfg.batch1.evaluation_epochs)
        if self.stop_requested:
            if not self.index["latest"].startswith(f"epoch{self.completed:04d}-"):
                self.save(trainer, model, self.completed, permanent=False)
            trainer.should_stop = True

    def save(self, trainer, model, epoch, permanent):
        directory = self.root/"checkpoints"
        directory.mkdir(exist_ok=True)
        name = f"epoch{epoch:04d}-{self.run_log.attempt}.ckpt"
        path = directory/name
        previous = self.index["latest"]
        self.index["previous"] = previous
        self.index["latest"] = name
        if permanent:
            self.index["snapshots"][str(epoch)] = name
        if self.best_epoch is not None:
            self.index["best"] = self.index["snapshots"].get(str(self.best_epoch))
        if epoch == self.cfg.train.n_epochs:
            self.index["final"] = name
        model.batch1_completed_epochs = epoch
        temporary = path.with_suffix(".ckpt.tmp")
        start = time.perf_counter()
        trainer.save_checkpoint(str(temporary))
        temporary.replace(path)
        self.index.setdefault("sha256", {})[name] = sha256(path)
        atomic_json(self.root/"checkpoint-index.json", self.index)
        self.run_log.write("events", event="checkpoint", completed_epoch=epoch, file=name,
                       sha256=self.index["sha256"][name], bytes=path.stat().st_size,
                       seconds=time.perf_counter()-start)
        backup_selected(self.root, self.archive)
        keep = set(self.index["snapshots"].values()) | {
            self.index[k] for k in ["latest", "previous", "best", "final"] if self.index[k]}
        for old in directory.glob("epoch*.ckpt"):
            if old.name not in keep:
                old.unlink()

    def on_fit_end(self, trainer, model):
        export_csv(self.root/"metrics"/"train.csv", self.train_rows)
        if self.completed < self.cfg.train.n_epochs:
            summary = dict(status="paused", completed_epochs=self.completed,
                           latest=self.index["latest"], reason="scheduler signal or trainer stop")
            atomic_json(self.root/"metrics"/"summary.json", summary)
            backup_selected(self.root, self.archive)
            return
        # Trainer teardown may have moved the model to CPU. The explicit final
        # evaluation device is the same one requested for training.
        model.to(self.evaluation_device)
        roles = {"final": self.index["final"], "best": self.index["best"]}
        all_results = {}
        for name in dict.fromkeys(roles.values()):
            if name is None:
                continue
            checkpoint = torch.load(self.root/"checkpoints"/name, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])
            epoch = checkpoint["batch1"]["completed_epochs"]
            rows = []
            for repeat in range(self.cfg.batch1.final_repeats):
                result = evaluate_pool(
                    model, self.infos, self.smiles["train"], self.reference, self.root,
                    f"final-epoch{epoch:04d}-rep{repeat}-{self.run_log.attempt}",
                    self.cfg.batch1.final_samples, self.cfg.batch1.final_seed+repeat, epoch)
                result.update(repeat=repeat, checkpoint=name)
                rows.append(self.run_log.write("final", **result))
            aggregate = {}
            for metric in ["validity", "uniqueness", "novelty", "vun"]:
                values = [r[metric] for r in rows if r[metric] is not None]
                aggregate[metric] = dict(mean=statistics.mean(values) if values else None,
                                         sample_std=statistics.stdev(values) if len(values)>1 else None)
            all_results[name] = dict(completed_epoch=epoch, repeats=rows, aggregate=aggregate)
        summary = dict(status="completed", configuration=self.cfg.augmentation.id,
                       training_seed=self.cfg.train.seed, completed_epochs=self.completed,
                       best_epoch=self.best_epoch, best_validation_vun=self.best_vun,
                       checkpoint_roles=roles, final_results=all_results,
                       note="One training seed; repetitions measure sampling variability only.")
        atomic_json(self.root/"metrics"/"summary.json", summary)
        backup_selected(self.root, self.archive)
        dashboard_summary = dict(status="completed", best_epoch=self.best_epoch,
                                 best_validation_vun=self.best_vun)
        for role, checkpoint_name in roles.items():
            if checkpoint_name in all_results:
                for metric, values in all_results[checkpoint_name]["aggregate"].items():
                    for statistic, value in values.items():
                        if value is not None:
                            dashboard_summary[f"{role}/{metric}/{statistic}"] = value
        self.run_log.finish(dashboard_summary)
