"""Generation and validation that never consume the optimizer's RNG stream."""
import hashlib
import json
from pathlib import Path
import pickle
import time

import torch
from torch_geometric.loader import DataLoader

from experiments.batch1.metrics import evaluate, legacy_evaluate
from experiments.batch1.storage import atomic_json, isolated_rng


def synchronize(model):
    if model.device.type == "cuda":
        torch.cuda.synchronize(model.device)


def weight_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def generate(model, count, batch_size, seed):
    previous_training = model.training
    model.eval()
    samples, initials, traces = [], [], []
    try:
        with isolated_rng(seed):
            # Size draws use their own stream, common to all six configurations.
            with isolated_rng(seed+1):
                sizes = model.node_dist.sample_n(count, model.device)
            references = None
            if model.atoms_mode != "marginal":
                with isolated_rng(seed+2):
                    mask = torch.arange(model.dataset_info.max_n_nodes, device=model.device)[None] < sizes[:, None]
                    references = model.prior_bank.reference(mask)
            synchronize(model)
            start = time.perf_counter()
            for offset in range(0, count, batch_size):
                ref = references[offset:offset+batch_size] if references is not None else None
                current, init, trace = model.sample_conditioned(
                    sizes[offset:offset+batch_size], reference=ref, trace=offset == 0)
                samples.extend(current)
                initials.extend(init)
                traces.extend(trace)
            synchronize(model)
            elapsed = time.perf_counter()-start
        return samples, initials, traces, elapsed
    finally:
        model.train(previous_training)


def evaluate_pool(model, infos, training_smiles, reference, root, tag, count, seed,
                  completed_epoch, legacy=True, comparison_count=None):
    directory = Path(root)/"samples"/tag
    directory.mkdir(parents=True, exist_ok=True)
    samples, initials, traces, generation_seconds = generate(
        model, count, model.cfg.batch1.inference_batch_size, seed)
    start = time.perf_counter()
    temporary = directory/"graphs.pkl.tmp"
    with temporary.open("wb") as handle:
        pickle.dump(dict(samples=samples, initial_atoms=initials), handle)
    temporary.replace(directory/"graphs.pkl")
    serialization_seconds = time.perf_counter()-start
    result, diagnostics, compositions, identities = evaluate(samples, infos.atom_decoder, reference, initials)
    if comparison_count is not None:
        subset, _, _, _ = evaluate(samples[:comparison_count], infos.atom_decoder,
                                   reference, initials[:comparison_count])
        atomic_json(directory/"comparison-subset.json", subset)
        result.update({"comparison_"+key: value for key, value in subset.items()})
    result.update(completed_epoch=int(completed_epoch), sampling_seed=seed,
                  checkpoint_weights_sha256=weight_fingerprint(model),
                  generation_seconds=generation_seconds, serialization_seconds=serialization_seconds,
                  sampling_steps=model.cfg.sample.sample_steps,
                  inference_batch_cap=model.cfg.batch1.inference_batch_size,
                  identity_policy="strict-connected-isomeric-canonical-no-correction")
    atomic_json(directory/"strict.json", result)
    atomic_json(directory/"diagnostics.json", diagnostics)
    atomic_json(directory/"composition-metrics.json", compositions)
    atomic_json(directory/"trajectory.json", traces)
    (directory/"unique.smi").write_text("\n".join(identities)+"\n")
    if legacy:
        old = legacy_evaluate(samples, infos, training_smiles, model.cfg, directory/"legacy")
        atomic_json(directory/"legacy.json", old)
        result.update({"legacy_"+key: value for key, value in old.items()})
    for key in ["atom_change_fraction", "composition_change_fraction", "normalized_composition_distance"]:
        result[key] = diagnostics[key]
    return result


@torch.no_grad()
def validation_loss(model, data, indices, seed):
    was_training = model.training
    model.eval()
    result = {}
    try:
        with isolated_rng(seed):
            subset = [data.val_dataset[i] for i in indices]
            for time_point in model.cfg.batch1.validation_times:
                rows = []
                for batch in DataLoader(subset, batch_size=model.cfg.batch1.validation_batch_size):
                    rows.append(model.reconstruct_loss(batch.to(model.device), time_point))
                for metric in ["node_ce", "edge_ce"]:
                    values = [r[metric] for r in rows if r[metric] is not None]
                    result[f"{metric}_t{time_point}"] = sum(values)/len(values) if values else None
    finally:
        model.train(was_training)
    return result
