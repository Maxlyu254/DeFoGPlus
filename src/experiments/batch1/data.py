"""Dataset loading, reproducible epoch shuffling, and fingerprinted prior caches."""
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from datasets.qm9_dataset import QM9DataModule, QM9infos, get_smiles
from models.extra_features import ExtraFeatures
from models.extra_features_molecular import ExtraMolecularFeatures
from metrics.molecular_metrics import SamplingMolecularMetrics
from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from experiments.batch1.model import Batch1Model
from experiments.batch1.statistics import fit_statistics, save_statistics, load_statistics, sha256


class EpochQM9(QM9DataModule):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.completed_epochs = 0

    def train_dataloader(self):
        generator = torch.Generator().manual_seed(self.cfg.train.seed+100003*self.completed_epochs)
        return DataLoader(self.train_dataset, batch_size=self.cfg.train.batch_size,
                          shuffle=True, num_workers=self.cfg.train.num_workers,
                          pin_memory=self.cfg.dataset.pin_memory, generator=generator)


def components(cfg):
    data = EpochQM9(cfg)
    infos = QM9infos(data, cfg)
    smiles = get_smiles(cfg, data, infos, evaluate_datasets=False)
    return data, infos, smiles


def fingerprints(data):
    return {name: sha256(dataset.processed_paths[dataset.file_idx])
            for name, dataset in [("train", data.train_dataset), ("val", data.val_dataset),
                                  ("test", data.test_dataset)]}


def statistics_for(cfg, data, infos, cache_root, create=False):
    hashes = fingerprints(data)
    key_fields = dict(schema=1, train_sha256=hashes["train"], remove_h=cfg.dataset.remove_h,
                      aromatic=cfg.dataset.aromatic, degree_pseudoatoms=cfg.batch1.degree_pseudoatoms,
                      density_pseudographs=cfg.batch1.density_pseudographs,
                      statistics_code_sha256=sha256(Path(__file__).with_name("statistics.py")))
    key = hashlib.sha256(json.dumps(key_fields, sort_keys=True).encode()).hexdigest()
    path = Path(cache_root)/f"statistics-{key}.pt"
    if not path.exists():
        if not create:
            raise FileNotFoundError(f"Run batch1 prepare first; missing {path}")
        result = fit_statistics(data.train_dataset, infos.num_atom_types, infos.max_n_nodes,
                                cfg.batch1.degree_pseudoatoms, cfg.batch1.density_pseudographs)
        save_statistics(result, path, key)
    result = load_statistics(path, key)
    return result, dict(path=str(path), sha256=sha256(path), dataset_hashes=hashes, cache_key=key)


def build_model(cfg, data, infos, smiles, statistics):
    extra = ExtraFeatures(cfg.model.extra_features, cfg.model.rrwp_steps, dataset_info=infos)
    molecular = ExtraMolecularFeatures(infos)
    infos.compute_input_output_dims(data, extra, molecular)
    infos.ref_metrics = {"val": {}, "test": {}}
    return Batch1Model(cfg, statistics,
                       dataset_infos=infos, train_metrics=TrainMolecularMetricsDiscrete(infos),
                       sampling_metrics=SamplingMolecularMetrics(infos, smiles, cfg),
                       visualization_tools=None, extra_features=extra,
                       domain_features=molecular, test_labels=None)
