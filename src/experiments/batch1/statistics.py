"""Train-only sufficient statistics and empirical composition sampling banks."""
import hashlib
import json
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

SCHEMA = 1


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_statistics(dataset, atom_classes, max_nodes, degree_pseudoatoms=100,
                   density_pseudographs=20):
    """Fit only the supplied training dataset; val/test datasets are not inputs."""
    counts = torch.zeros(max_nodes + 1, dtype=torch.long)
    pairs = torch.zeros(max_nodes + 1, dtype=torch.float64)
    edges = torch.zeros_like(pairs)
    atoms = torch.zeros(max_nodes + 1, atom_classes, dtype=torch.float64)
    degrees = torch.zeros_like(atoms)
    profiles = [[] for _ in range(max_nodes + 1)]
    split_ids = []
    for graph in dataset:
        labels = graph.x.argmax(-1).long()
        n = labels.numel()
        if not 1 <= n <= max_nodes:
            raise ValueError(f"Unsupported training graph size {n}")
        degree = torch.bincount(graph.edge_index[0], minlength=n).double()
        counts[n] += 1
        pairs[n] += n * (n-1) / 2
        edges[n] += graph.edge_index.shape[1] / 2
        atoms[n] += torch.bincount(labels, minlength=atom_classes)
        degrees[n].scatter_add_(0, labels, degree)
        padded = torch.zeros(max_nodes, dtype=torch.uint8)
        padded[:n] = labels.to(torch.uint8)
        profiles[n].append(padded)
        split_ids.append(int(graph.idx.item()))
    if not counts.sum() or not pairs.sum():
        raise ValueError("Training data do not contain usable atom/bond statistics")
    global_q = edges.sum() / pairs.sum()
    global_mu = degrees.sum(0) / atoms.sum(0).clamp_min(1)
    mu = (degrees + degree_pseudoatoms * global_mu) / (atoms + degree_pseudoatoms)
    pseudo_pairs = counts.new_tensor([n*(n-1)/2 for n in range(max_nodes+1)]).double()
    pseudo_pairs *= density_pseudographs
    q = (edges + pseudo_pairs * global_q) / (pairs + pseudo_pairs).clamp_min(1)
    q[0:2] = 0
    offsets = torch.zeros_like(counts)
    offset = 0
    for n in range(max_nodes+1):
        offsets[n] = offset
        offset += counts[n]
    bank = torch.stack([p for group in profiles for p in group])
    atom_prob = atoms.sum(0) / atoms.sum()
    result = dict(schema=SCHEMA, counts=counts, offsets=offsets, references=bank,
                  q=q.float(), mu=mu.float(), atom_prob=atom_prob.float(),
                  split_ids=torch.tensor(split_ids), global_q=float(global_q),
                  degree_pseudoatoms=degree_pseudoatoms,
                  density_pseudographs=density_pseudographs,
                  atom_classes=atom_classes, max_nodes=max_nodes)
    return result


def save_statistics(result, path, dataset_fingerprint):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {**result, "dataset_fingerprint": dataset_fingerprint}
    temporary = path.with_suffix(".tmp")
    torch.save(result, temporary)
    temporary.replace(path)
    metadata = dict(schema=SCHEMA, dataset_fingerprint=dataset_fingerprint,
                    statistics_sha256=sha256(path), training_graphs=int(result["counts"].sum()),
                    size_counts=result["counts"].tolist(), q=result["q"].tolist(),
                    mu=result["mu"].tolist(), degree_pseudoatoms=result["degree_pseudoatoms"],
                    density_pseudographs=result["density_pseudographs"],
                    split_ids_sha256=hashlib.sha256(result["split_ids"].numpy().tobytes()).hexdigest())
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")
    return metadata


def load_statistics(path, expected_dataset=None):
    result = torch.load(path, map_location="cpu", weights_only=False)
    if result["schema"] != SCHEMA:
        raise ValueError("Unknown statistics schema; regenerate explicitly")
    if expected_dataset is not None and result["dataset_fingerprint"] != expected_dataset:
        raise ValueError("Statistics/data fingerprint mismatch")
    return result
