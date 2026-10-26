"""Consistent molecular identities and counts, independent of W&B."""
import contextlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import time

from rdkit import Chem
import torch

from analysis.rdkit_functions import BasicMolecularMetrics, build_molecule


def canonical_reference(smiles):
    result = set()
    for value in smiles:
        if not isinstance(value, str):
            continue
        molecule = Chem.MolFromSmiles(value)
        if molecule is not None:
            result.add(Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True))
    return result


def fractions(attempts, valid, unique, novel):
    return dict(attempts=attempts, valid_count=valid, unique_count=unique, novel_count=novel,
                validity=valid/attempts if attempts else 0,
                uniqueness=unique/valid if valid else 0,
                novelty=novel/unique if unique else None,
                vun=novel/attempts if attempts else 0)


def evaluate(samples, decoder, reference, initials=None):
    start = time.perf_counter()
    accepted = []
    invalid = disconnected = valence_failures = 0
    degree_sum = torch.zeros(len(decoder), dtype=torch.double)
    bond_sum = torch.zeros_like(degree_sum)
    atom_count = torch.zeros_like(degree_sum)
    degree_histogram = [Counter() for _ in decoder]
    bond_order_histogram = [Counter() for _ in decoder]
    compositions = Counter()
    initial_compositions = Counter()
    groups = defaultdict(lambda: {"attempts": 0, "valid": 0, "identities": set()})
    changes = composition_changes = changed_graphs = 0
    total_atoms = 0
    order = torch.tensor([0., 1., 2., 3., 1.5])
    for index, (atoms, edges) in enumerate(samples):
        atoms, edges = atoms.cpu(), edges.cpu()
        counts = torch.bincount(atoms, minlength=len(decoder))
        key = ",".join(map(str, counts.tolist()))
        compositions[key] += 1
        groups[key]["attempts"] += 1
        atom_count += counts
        degree = (edges>0).sum(-1)
        bond_orders = order[edges].sum(-1)
        degree_sum.scatter_add_(0, atoms, degree.double())
        bond_sum.scatter_add_(0, atoms, bond_orders.double())
        for atom_type, neighbors, bond_order in zip(atoms.tolist(), degree.tolist(), bond_orders.tolist()):
            degree_histogram[atom_type][str(neighbors)] += 1
            bond_order_histogram[atom_type][str(bond_order)] += 1
        if initials is not None:
            initial = initials[index].cpu()
            initial_counts = torch.bincount(initial, minlength=len(decoder))
            initial_compositions[",".join(map(str, initial_counts.tolist()))] += 1
            changes += int((atoms != initial).sum())
            distance = int((counts-initial_counts).abs().sum())
            composition_changes += distance
            changed_graphs += int(distance>0)
            total_atoms += len(atoms)
        try:
            molecule = build_molecule(atoms, edges, decoder)
            Chem.SanitizeMol(molecule)
            if len(Chem.GetMolFrags(molecule)) != 1:
                disconnected += 1
                continue
            identity = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            accepted.append(identity)
            groups[key]["valid"] += 1
            groups[key]["identities"].add(identity)
        except Chem.AtomValenceException:
            invalid += 1
            valence_failures += 1
        except (ValueError, RuntimeError):
            invalid += 1
    chemistry_seconds = time.perf_counter()-start
    start = time.perf_counter()
    unique = set(accepted)
    dedup_seconds = time.perf_counter()-start
    start = time.perf_counter()
    novel = unique-reference
    novelty_seconds = time.perf_counter()-start
    result = fractions(len(samples), len(accepted), len(unique), len(novel))
    result.update(invalid_count=invalid, disconnected_count=disconnected,
                  valence_failures=valence_failures, chemistry_seconds=chemistry_seconds,
                  dedup_seconds=dedup_seconds, novelty_seconds=novelty_seconds)
    diagnostics = dict(
        atom_counts=atom_count.tolist(),
        mean_degree_by_atom=(degree_sum/atom_count.clamp_min(1)).tolist(),
        mean_bond_order_sum_by_atom=(bond_sum/atom_count.clamp_min(1)).tolist(),
        degree_histogram_by_atom=[dict(values) for values in degree_histogram],
        bond_order_histogram_by_atom=[dict(values) for values in bond_order_histogram],
        composition_histogram=dict(compositions), initial_composition_histogram=dict(initial_compositions),
        atom_change_fraction=changes/total_atoms if total_atoms else None,
        composition_change_fraction=changed_graphs/len(samples) if initials is not None and samples else None,
        normalized_composition_distance=composition_changes/(2*total_atoms) if total_atoms else None)
    group_rows = []
    for key, group in sorted(groups.items()):
        ids = group["identities"]
        group_rows.append({"composition": key, **fractions(group["attempts"], group["valid"], len(ids), len(ids-reference))})
    return result, diagnostics, group_rows, sorted(unique)


def legacy_evaluate(samples, infos, training_smiles, cfg, directory):
    """Use the repository evaluator unchanged, including its relaxed policy."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    start = time.perf_counter()
    try:
        os.chdir(directory)
        with Path("stdout.log").open("w") as log, contextlib.redirect_stdout(log):
            metrics = BasicMolecularMetrics(infos, training_smiles, cfg).evaluate(samples, test=False)[0]
    finally:
        os.chdir(original)
    v, rv, u, n = metrics
    return dict(sanitization_validity=v, relaxed_validity=rv, uniqueness=u, novelty=n,
                vun=rv*u*n if n>=0 else 0, seconds=time.perf_counter()-start)
