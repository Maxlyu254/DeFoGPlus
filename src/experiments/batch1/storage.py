"""Atomic artifacts, bounded recovery copies, and small allowlisted telemetry."""
import contextlib
import csv
import json
import os
from pathlib import Path
import random
import shutil
import time
import uuid

import numpy as np
import torch


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(data, allow_nan=False)+"\n")
        handle.flush()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextlib.contextmanager
def isolated_rng(seed):
    previous = rng_state()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        restore_rng(previous)


def export_csv(path, rows):
    scalar_rows = [{k: v for k, v in r.items() if v is None or isinstance(v, (str, int, float, bool))}
                   for r in rows]
    fields = sorted({key for row in scalar_rows for key in row})
    temporary = Path(path).with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(scalar_rows)
    temporary.replace(path)


class RunLog:
    def __init__(self, root, cfg, manifest, wandb_mode=None):
        self.root = Path(root).resolve()
        self.attempt = time.strftime("%Y%m%dT%H%M%S", time.gmtime())+"-"+uuid.uuid4().hex[:8]
        self.cfg, self.manifest = cfg, manifest
        self.wandb = None
        mode = wandb_mode or cfg.batch1.wandb_mode
        if mode != "disabled":
            import wandb
            from omegaconf import OmegaConf
            self.wandb = wandb.init(
                entity=cfg.batch1.wandb_entity, project=cfg.batch1.wandb_project,
                group=manifest["logical_run_id"], name=f"{cfg.augmentation.id}-seed{cfg.train.seed}-{self.attempt}",
                id=uuid.uuid4().hex[:12], mode=mode, dir=str(self.root),
                config={"resolved": OmegaConf.to_container(cfg, resolve=True),
                        "provenance": manifest, "attempt": self.attempt},
                tags=[cfg.augmentation.id, "qm9", "batch1"],
                settings=wandb.Settings(_disable_stats=True, console="off"),
                save_code=False)
            for namespace in ["train/*", "validation/*", "curve/*"]:
                self.wandb.define_metric(namespace, step_metric="completed_epoch")
            self.wandb.define_metric("completed_epoch")
        self.write("events", event="attempt_start")

    def write(self, stream, **row):
        record = dict(unix_time=time.time(), logical_run_id=self.manifest["logical_run_id"],
                      attempt=self.attempt, **row)
        append_json(self.root/"metrics"/f"{stream}.jsonl", record)
        if self.wandb and stream in ["train", "validation", "curve", "final"]:
            values = {f"{stream}/{key}": value for key, value in row.items()
                      if value is not None and isinstance(value, (int, float))}
            if "completed_epoch" in row:
                values["completed_epoch"] = row["completed_epoch"]
            self.wandb.log(values)
        return record

    def finish(self, summary=None):
        if self.wandb:
            if summary is not None:
                self.wandb.summary.update(summary)
            self.wandb.finish()
            self.wandb = None


def backup_selected(root, archive):
    """Back up owned run artifacts; retain only named selected/recovery models."""
    if archive is None:
        return
    root, archive = Path(root), Path(archive)
    archive.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/"manifest.json").read_text())
    if "statistics" in manifest:
        statistics_path = Path(manifest["statistics"]["path"])
        destination = archive.parent/"prior-cache"
        destination.mkdir(exist_ok=True)
        for source in [statistics_path, statistics_path.with_suffix(".json")]:
            target = destination/source.name
            if source.exists() and not target.exists():
                temporary = target.with_suffix(target.suffix+".tmp")
                shutil.copy2(source, temporary)
                temporary.replace(target)
    for name in ["manifest.json", "resolved-config.yaml", "checkpoint-index.json"]:
        source = root/name
        if source.exists():
            shutil.copy2(source, archive/name)
    shutil.copytree(root/"metrics", archive/"metrics", dirs_exist_ok=True)
    index_file = root/"checkpoint-index.json"
    if not index_file.exists():
        return
    index = json.loads(index_file.read_text())
    names = {index.get(key) for key in ["latest", "previous", "best", "final"]}-{None}
    destination = archive/"checkpoints"
    destination.mkdir(exist_ok=True)
    for name in names:
        source = root/"checkpoints"/name
        target = destination/name
        if source.exists() and (not target.exists() or target.stat().st_size != source.stat().st_size):
            temporary = target.with_suffix(".tmp")
            shutil.copy2(source, temporary)
            temporary.replace(target)
    for old in destination.glob("epoch*.ckpt"):
        if old.name not in names:
            old.unlink()
