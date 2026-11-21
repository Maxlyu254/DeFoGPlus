"""Explicit-H audit used by the H profiler, without changing no-H metrics."""
import time
from rdkit import Chem
from analysis.rdkit_functions import build_molecule
from experiments.batch1.metrics import fractions


def evaluate_h(samples,decoder,reference):
    started=time.perf_counter()
    accepted=[]
    invalid=disconnected=incomplete_h=radical=0
    for atoms,edges in samples:
        try:
            mol=build_molecule(atoms.cpu(),edges.cpu(),decoder)
            Chem.SanitizeMol(mol)
            if len(Chem.GetMolFrags(mol))!=1:
                disconnected+=1
                continue
            if any(a.GetNumImplicitHs() for a in mol.GetAtoms()):
                incomplete_h+=1
                continue
            if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                radical+=1
                continue
            accepted.append(Chem.MolToSmiles(Chem.RemoveHs(mol),canonical=True,isomericSmiles=True))
        except (ValueError,RuntimeError):
            invalid+=1
    unique=set(accepted)
    result=fractions(len(samples),len(accepted),len(unique),len(unique-reference))
    result.update(invalid_count=invalid,disconnected_count=disconnected,
                  implicit_h_completion_count=incomplete_h,radical_count=radical,
                  chemistry_seconds=time.perf_counter()-started,
                  identity_policy='explicit-H-complete-connected-closed-shell; canonical identity after RemoveHs')
    return result
